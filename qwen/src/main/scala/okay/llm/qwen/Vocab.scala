package okay.llm.qwen

import okay.llm.compute.QLinear

/**
 * THE VOCABULARY IN BLOCKS BY LANGUAGE (okay-inlay specs/core.md §8, the vocabulary): the embedding is the largest
 * tensor of a small model (78 of 244 MB in the cut Qwen3-0.6B) and most of its rows are tokens of languages a page
 * never meets. The rows are split into blocks by the script of the token's letters — common (bytes, digits, marks),
 * latin-en (the Latin tokens English uses), latin-rest, cyrillic, cjk, other — each block a file; a page loads the
 * blocks of its languages (English + Russian: 29 MB) and fetches another when a token of it appears.
 */
object Vocab:
  val blocks: Vector[String] = Vector("common", "latin-en", "latin-rest", "cyrillic", "cjk", "other")

  /** every token's block: by the script of its letters; the Latin tokens `englishUses` (those met when an English
   * word list is encoded) are latin-en, the other Latin ones latin-rest */
  def of(tok: Tokenizer, englishUses: Int => Boolean): Array[String] =
    Array.tabulate(tok.size) { id =>
      val letters = Tokenizer.codePoints(tok.decode(Seq(id))).filter(Character.isLetter)
      if letters.isEmpty || tok.isSpecial(id) then "common"   // (the special tokens — the chat's marks — are common)
      else
        val scripts = letters.map(Character.UnicodeScript.of).distinct
        if scripts.length > 1 then "other"
        else scripts.head match
          case Character.UnicodeScript.LATIN => if englishUses(id) then "latin-en" else "latin-rest"
          case Character.UnicodeScript.CYRILLIC => "cyrillic"
          case Character.UnicodeScript.HAN | Character.UnicodeScript.HIRAGANA | Character.UnicodeScript.KATAKANA |
            Character.UnicodeScript.HANGUL => "cjk"
          case _ => "other"
    }

  /** the tokens met when the words are encoded — alone, after a space, lower-cased, capitalized, and with the
   * usual endings (a word list has no plurals or tenses: " orders" was missing) */
  def uses(tok: Tokenizer, words: Iterator[String]): Int => Boolean =
    val used = scala.collection.mutable.BitSet()
    val endings = Seq("", "s", "es", "ed", "ing", "ly", "er", "ers", "est", "'s")
    for w <- words; base <- Seq(w.toLowerCase, w.capitalize).distinct; e <- endings; form <- Seq(base + e, " " + base + e) do
      used ++= tok.encode(form)
    used.contains

  /** A BLOCK OF THE EMBEDDING as a safetensors file: `ids` (U32, the tokens' ids in the order of the rows) and the
   * rows' `weight`, `scales` and `biases` under `embed` (the file's dtypes kept) */
  def blockFile(st: Safetensors, ids: Vector[Int]): Array[Byte] =
    val w = st.entries("model.embed_tokens.weight"); val s = st.entries("model.embed_tokens.scales")
    val b = st.entries("model.embed_tokens.biases")
    def rowsOf(e: st.Entry, name: String): Array[Byte] =
      val all = st.raw(name); val rowBytes = all.length / e.shape(0)
      val out = new Array[Byte](ids.size * rowBytes)
      ids.zipWithIndex.foreach((id, k) => System.arraycopy(all, id * rowBytes, out, k * rowBytes, rowBytes))
      out
    val idBytes = java.nio.ByteBuffer.allocate(ids.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    ids.foreach(idBytes.putInt); 
    Safetensors.write(Vector(
      ("ids", "U32", Vector(ids.size), idBytes.array()),
      ("embed.weight", w.dtype, Vector(ids.size, w.shape(1)), rowsOf(w, "model.embed_tokens.weight")),
      ("embed.scales", s.dtype, Vector(ids.size, s.shape(1)), rowsOf(s, "model.embed_tokens.scales")),
      ("embed.biases", b.dtype, Vector(ids.size, b.shape(1)), rowsOf(b, "model.embed_tokens.biases"))))

  /** a block file read: its ids and its rows */
  def readBlock(st: Safetensors): (Array[Int], QLinear) = (st.words("ids"), st.linear("embed"))

/**
 * THE EMBEDDING'S ROWS, wherever they are: the whole (`Rows.Full`, a QLinear over the vocabulary), or the blocks
 * loaded so far (`Rows.Blocks`), more added as tokens of them appear. A row by a token's id; the logits of a state
 * against every loaded row (the others minus infinity: a token with no row is never chosen, and the log-softmax is
 * over the loaded).
 */
trait Rows:
  def vocab: Int
  def has(id: Int): Boolean
  /** W[id][i] */
  def at(id: Int, i: Int): Float
  /** the logits over the vocabulary: minus infinity where no row is loaded */
  def logits(x: Array[Float]): Array[Float]
  /** the loaded rows as one weight for a device, and the id of each of its rows */
  def packed: (QLinear, Array[Int])

object Rows:
  final class Full(q: QLinear) extends Rows:
    def vocab: Int = q.out
    def has(id: Int): Boolean = id >= 0 && id < q.out
    def at(id: Int, i: Int): Float = q.at(id, i)
    def logits(x: Array[Float]): Array[Float] = q(x)
    def packed: (QLinear, Array[Int]) = (q, Array.tabulate(q.out)(identity))

  /** the blocks loaded: each its ids and its rows; `add` one more */
  final class Blocks(val vocab: Int, in: Int) extends Rows:
    private val where = new Array[Int](vocab); java.util.Arrays.fill(where, -1)   // the block of a token, -1 none
    private val row = new Array[Int](vocab)
    private val parts = scala.collection.mutable.ArrayBuffer.empty[QLinear]
    def loaded: Int = parts.size
    def add(ids: Array[Int], q: QLinear): Unit =
      require(q.in == in, s"a block of ${q.in} columns for $in")
      val k = parts.size; parts += q
      ids.zipWithIndex.foreach((id, r) => { where(id) = k; row(id) = r })
    def has(id: Int): Boolean = id >= 0 && id < vocab && where(id) >= 0
    def at(id: Int, i: Int): Float =
      if !has(id) then throw new NoSuchElementException(s"no embedding row loaded for the token $id") else parts(where(id)).at(row(id), i)
    def logits(x: Array[Float]): Array[Float] =
      val out = Array.fill(vocab)(Float.NegativeInfinity)
      val byPart = parts.map(_(x))
      var id = 0
      while id < vocab do { if where(id) >= 0 then out(id) = byPart(where(id))(row(id)); id += 1 }
      out
    def packed: (QLinear, Array[Int]) =
      val ids = (0 until vocab).filter(has).toArray
      val perRow = in / 8; val groups = in / 64
      val words = new Array[Int](ids.length * perRow); val scales = new Array[Float](ids.length * groups); val biases = new Array[Float](ids.length * groups)
      ids.zipWithIndex.foreach { (id, k) =>
        val q = parts(where(id)); val r = row(id)
        System.arraycopy(q.words, r * perRow, words, k * perRow, perRow)
        System.arraycopy(q.scales, r * groups, scales, k * groups, groups)
        System.arraycopy(q.biases, r * groups, biases, k * groups, groups)
      }
      (QLinear(ids.length, in, words, scales, biases), ids)
