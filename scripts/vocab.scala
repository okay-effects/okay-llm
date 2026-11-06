//> using scala 3.3.8
//> using dep io.github.okay-effects::okay-llm-qwen:0.1.0
//> using dep io.github.okay-effects::okay-effects-json-value:0.1.1
//> using repository https://okay-effects.github.io/okay-simple/maven

import okay.llm.qwen.Tokenizer

/** THE VOCABULARY BY SCRIPT (okay-inlay specs/core.md §8, the vocabulary): every token of a tokenizer.json decoded and
 * put under the script of its letters — the blocks an embedding would be split into, their rows and bytes at 4 bits
 * (1024 columns: 512 bytes of words + 32 of scales and biases a row). `scala-cli run scripts/vocab.scala -- <model dir>` */
@main def vocab(dir: String): Unit =
  val tok = Tokenizer.fromJson(okay.effects.json.Json.fromText(java.nio.file.Files.readString(java.nio.file.Path.of(dir, "tokenizer.json"))))
  def block(id: Int): String =
    val s = tok.decode(Seq(id))
    val letters = s.codePoints().toArray.filter(Character.isLetter)
    if letters.isEmpty then "common (bytes, digits, marks, spaces)"
    else
      val scripts = letters.map(c => Character.UnicodeScript.of(c)).distinct
      if scripts.length > 1 then "mixed"
      else scripts.head match
        case Character.UnicodeScript.LATIN => "latin"
        case Character.UnicodeScript.CYRILLIC => "cyrillic"
        case Character.UnicodeScript.HAN | Character.UnicodeScript.HIRAGANA | Character.UnicodeScript.KATAKANA | Character.UnicodeScript.HANGUL => "cjk"
        case Character.UnicodeScript.ARABIC => "arabic"
        case other => "other (" + other.toString.toLowerCase + ")"
  val counts = (0 until tok.size).groupBy(block).view.mapValues(_.size).toVector.sortBy(-_._2)
  val rowBytes = 512 + 32
  println(f"${tok.size} tokens, ${tok.size * rowBytes / 1048576.0}%.1f MB of embedding at 4 bits")
  for (b, n) <- counts do println(f"  $b%-42s ${n}%7d rows  ${n * rowBytes / 1048576.0}%6.1f MB  ${n * 100.0 / tok.size}%5.1f%%")
  val others = counts.filter(_._1.startsWith("other")).map(_._2).sum
  println(f"  (all the others together: $others rows, ${others * rowBytes / 1048576.0}%.1f MB)")

/** THE LATIN BLOCK BY USE: the tokens English needs — those that appear when the system's word list (/usr/share/dict/words,
 * each word alone, after a space, capitalized) is encoded — against the rest of the Latin tokens (other languages' pieces).
 * `scala-cli run scripts/vocab.scala --main-class english -- <model dir> [words file]` */
@main def english(dir: String, words: String = "/usr/share/dict/words"): Unit =
  val tok = Tokenizer.fromJson(okay.effects.json.Json.fromText(java.nio.file.Files.readString(java.nio.file.Path.of(dir, "tokenizer.json"))))
  val ws = scala.io.Source.fromFile(words).getLines().toVector
  val used = scala.collection.mutable.BitSet()
  for w <- ws; form <- Seq(w, " " + w, w.toLowerCase, " " + w.toLowerCase, w.capitalize, " " + w.capitalize) do used ++= tok.encode(form)
  def latin(id: Int): Boolean =
    val letters = tok.decode(Seq(id)).codePoints().toArray.filter(Character.isLetter)
    letters.nonEmpty && letters.forall(c => Character.UnicodeScript.of(c) == Character.UnicodeScript.LATIN)
  val latinIds = (0 until tok.size).filter(latin)
  val en = latinIds.count(used); val rest = latinIds.size - en
  val rowBytes = 544
  println(f"${ws.size} words; Latin tokens English uses: $en rows (${en * rowBytes / 1048576.0}%.1f MB); the rest of the Latin: $rest rows (${rest * rowBytes / 1048576.0}%.1f MB)")
  println(f"a page for English + Russian: common 10074 + Latin-English $en + Cyrillic 4149 = ${10074 + en + 4149} rows, ${(10074 + en + 4149) * rowBytes / 1048576.0}%.1f MB of 78.7")
