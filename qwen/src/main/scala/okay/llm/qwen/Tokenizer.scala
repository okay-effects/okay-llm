package okay.llm.qwen

import okay.effects.json.Json

import scala.collection.mutable

/** A BYTE-LEVEL BPE TOKENIZER read from a HuggingFace tokenizer.json (Qwen's, GPT-2's family): added tokens matched
 * first, the rest split by the pre-tokenizer, each piece's bytes as characters, merged by rank.
 *
 * The pre-tokenizer is Qwen's regex written out by hand — `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|
 * \p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, its alternatives tried in order at each position: the
 * look-ahead `(?!\S)` has no RE2 form, and Scala Native's regex is RE2's. Text between added tokens is normalized as
 * the file says: NFC (`Nfc`), or not at all */
final class Tokenizer private (
    vocab: java.util.HashMap[String, Integer],
    ranks: java.util.HashMap[String, Integer],
    pieces: Array[String],
    added: Map[String, Int],
    special: Set[Int],
    nfc: Boolean):

  private val addedByFirst: Map[Char, Seq[String]] =
    added.keys.toSeq.groupBy(_.head).view.mapValues(_.sortBy(-_.length)).toMap
  private val cache = mutable.HashMap.empty[String, Array[Int]]

  def size: Int = pieces.length
  def id(token: String): Int = added.getOrElse(token, Option(vocab.get(token)).map(_.toInt).getOrElse(-1))
  def isSpecial(id: Int): Boolean = special(id)

  def encode(text: String): Array[Int] =
    val out = mutable.ArrayBuilder.make[Int]
    var i = 0
    var from = 0
    while i < text.length do
      addedAt(text, i) match
        case Some(t) =>
          if from < i then plain(text.substring(from, i), out)
          out += added(t)
          i += t.length
          from = i
        case None => i += 1
    if from < text.length then plain(text.substring(from), out)
    out.result()

  private def addedAt(text: String, i: Int): Option[String] =
    addedByFirst.get(text.charAt(i)).flatMap(_.find(text.startsWith(_, i)))

  private def plain(text: String, out: mutable.ArrayBuilder[Int]): Unit =
    for piece <- Tokenizer.pretokenize(if nfc then Nfc(text) else text) do
      out ++= cache.getOrElseUpdate(piece, bpe(Tokenizer.toByteChars(piece)))

  /** the piece's byte characters merged, the lowest rank first, until no pair has one */
  private def bpe(word: String): Array[Int] =
    val syms = mutable.ArrayBuffer.from(word.map(_.toString))
    var done = syms.length < 2
    while !done do
      var best = Int.MaxValue
      var at = -1
      var i = 0
      while i < syms.length - 1 do
        val r = ranks.get(syms(i) + " " + syms(i + 1))
        if r != null && r < best then { best = r; at = i }
        i += 1
      if at < 0 then done = true
      else
        syms(at) = syms(at) + syms(at + 1)
        syms.remove(at + 1)
        done = syms.length < 2
    syms.map(s => Option(vocab.get(s)).map(_.toInt).getOrElse(sys.error(s"no token for $s"))).toArray

  /** the bytes of tokens: a piece's characters back to bytes, an added token's text as UTF-8 */
  def bytes(ids: Seq[Int]): Array[Byte] =
    val b = mutable.ArrayBuilder.make[Byte]
    for id <- ids do
      val p = pieces(id)
      if added.contains(p) then b ++= p.getBytes("UTF-8")
      else p.foreach(c => b += Tokenizer.byteOf(c))
    b.result()

  def decode(ids: Seq[Int]): String = String(bytes(ids), "UTF-8")

  /** text as tokens come: the bytes so far, given out as far as they are whole UTF-8 */
  final class Stream:
    private val pending = mutable.ArrayBuffer.empty[Byte]
    def push(id: Int): String =
      pending ++= bytes(Seq(id))
      val whole = Tokenizer.wholeUtf8(pending)
      val s = String(pending.take(whole).toArray, "UTF-8")
      pending.remove(0, whole)
      s

object Tokenizer:
  def fromJson(json: Json): Tokenizer =
    val model = json("model")
    require(model("type").string == "BPE", s"a BPE model expected: ${model("type")}")
    val vocab = java.util.HashMap[String, Integer]()
    var top = 0
    for (k, v) <- model("vocab").record do
      vocab.put(k, v.int)
      top = top max v.int
    val ranks = java.util.HashMap[String, Integer]()
    for (m, r) <- model("merges").list.zipWithIndex do
      val key = m match
        case Json.Arr(Vector(a, b)) => a.string + " " + b.string
        case s => s.string
      ranks.put(key, r)
    val added = json("added_tokens").list.map(t => t("content").string -> t("id").int).toMap
    val special = json("added_tokens").list.filter(_.field("special").exists(_.boolean)).map(_("id").int).toSet
    val pieces = new Array[String]((top max added.values.max) + 1)
    vocab.forEach((k, v) => pieces(v) = k)
    for (k, v) <- added do pieces(v) = k
    val nfc = json("normalizer") match
      case Json.Null => false
      case n if n("type").string == "NFC" => true
      case n => sys.error(s"a normalizer not supported: $n")
    new Tokenizer(vocab, ranks, pieces, added, special, nfc)

  // GPT-2's byte characters: the printable bytes as themselves, the rest from 256 on
  private val byteChar: Array[Char] =
    val keep = (33 to 126) ++ (161 to 172) ++ (174 to 255)
    val a = new Array[Char](256)
    var n = 0
    for b <- 0 until 256 do
      if keep.contains(b) then a(b) = b.toChar
      else { a(b) = (256 + n).toChar; n += 1 }
    a
  private val charByte: Map[Char, Byte] = byteChar.zipWithIndex.map((c, b) => c -> b.toByte).toMap
  private def byteOf(c: Char): Byte = charByte(c)
  def toByteChars(s: String): String = s.getBytes("UTF-8").map(b => byteChar(b & 0xff)).mkString

  /** how many of the bytes make whole UTF-8 characters */
  private def wholeUtf8(b: collection.IndexedSeq[Byte]): Int =
    var i = 0
    var whole = 0
    while i < b.length do
      val c = b(i) & 0xff
      val n =
        if c < 0x80 then 1 else if c >> 5 == 6 then 2 else if c >> 4 == 14 then 3 else if c >> 3 == 30 then 4 else 1
      if i + n <= b.length then { i += n; whole = i } else i = b.length
    whole

  // the regex's classes: Unicode letters, numbers, White_Space
  private def isL(c: Int): Boolean = Character.isLetter(c)
  private def isN(c: Int): Boolean =
    val t = Character.getType(c)
    t == Character.DECIMAL_DIGIT_NUMBER || t == Character.LETTER_NUMBER || t == Character.OTHER_NUMBER
  private def isS(c: Int): Boolean =
    (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
      c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000
  private def isNl(c: Int): Boolean = c == '\r' || c == '\n'

  /** the text in the regex's pieces */
  def pretokenize(text: String): Vector[String] =
    val cps = text.codePoints().toArray
    val n = cps.length
    val out = Vector.newBuilder[String]
    def run(from: Int, p: Int => Boolean): Int = { var j = from; while j < n && p(cps(j)) do j += 1; j }
    var i = 0
    while i < n do
      val c = cps(i)
      val end =
        // 's 't 're 've 'm 'll 'd, any case
        val contraction =
          if c != '\'' || i + 1 >= n then -1
          else
            val a = Character.toLowerCase(cps(i + 1))
            val b = if i + 2 < n then Character.toLowerCase(cps(i + 2)) else -1
            if a == 's' || a == 't' || a == 'm' || a == 'd' then i + 2
            else if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') then i + 3
            else -1
        if contraction > 0 then contraction
        // [^\r\n\p{L}\p{N}]?\p{L}+
        else if isL(c) then run(i, isL)
        else if !isNl(c) && !isN(c) && i + 1 < n && isL(cps(i + 1)) then run(i + 1, isL)
        // \p{N}
        else if isN(c) then i + 1
        else
          // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
          val k = if c == ' ' then i + 1 else i
          def punct(x: Int) = !isS(x) && !isL(x) && !isN(x)
          if k < n && punct(cps(k)) then run(run(k, punct), isNl)
          else
            // whitespace: `\s*[\r\n]+`, `\s+(?!\S)`, `\s+`
            val e = run(i, isS)
            var lastNl = -1
            var j = i
            while j < e do { if isNl(cps(j)) then lastNl = j; j += 1 }
            if lastNl >= 0 then lastNl + 1
            else if e == n then e
            else if e - i >= 2 then e - 1
            else e
      out += String(cps, i, end - i)
      i = end
    out.result()
