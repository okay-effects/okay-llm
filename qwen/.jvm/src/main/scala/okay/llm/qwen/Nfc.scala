package okay.llm.qwen

/** UNICODE NORMALIZATION FORM C, on both platforms (Scala Native's library has no `java.text.Normalizer`): each code
 * point fully decomposed, combining marks put in canonical order, then composed back where a primary composite
 * exists — Hangul by its formula. Unicode's data from the resource mlx/llm/nfc.txt (scripts/gen-nfc.py): on Scala
 * Native the app embeds resources — `nativeConfig ~= (_.withEmbedResources(true))` */
object Nfc:
  private val tables: Vector[Array[Int]] =
    val in = getClass.getResourceAsStream("/okay/llm/qwen/nfc.txt")
    if in == null then throw IllegalStateException(
      "okay/llm/qwen/nfc.txt is not among the resources: on Scala Native, build with nativeConfig.withEmbedResources(true)")
    try
      String(in.readAllBytes(), "UTF-8").linesIterator.filterNot(_.startsWith("#")).map(line =>
        line.split(' ').filter(_.nonEmpty).map(Integer.parseInt(_, 36))).toVector
    finally in.close()
  private val decomposition: java.util.HashMap[Int, Array[Int]] =
    val m = java.util.HashMap[Int, Array[Int]]()
    val d = tables(0)
    var i = 0
    while i < d.length do
      val n = d(i + 1)
      m.put(d(i), d.slice(i + 2, i + 2 + n))
      i += 2 + n
    m
  private val ccc: java.util.HashMap[Int, Int] =
    val m = java.util.HashMap[Int, Int]()
    val c = tables(1)
    var i = 0
    while i < c.length do { m.put(c(i), c(i + 1)); i += 2 }
    m
  private val composite: java.util.HashMap[Long, Int] =
    val m = java.util.HashMap[Long, Int]()
    val p = tables(2)
    var i = 0
    while i < p.length do { m.put(pair(p(i), p(i + 1)), p(i + 2)); i += 3 }
    m
  private def pair(a: Int, b: Int): Long = (a.toLong << 21) | b
  private def classOf(c: Int): Int = ccc.getOrDefault(c, 0)

  // Hangul syllables: L·V·T jamo by arithmetic
  private val SBase = 0xac00; private val LBase = 0x1100; private val VBase = 0x1161; private val TBase = 0x11a7
  private val LCount = 19; private val VCount = 21; private val TCount = 28
  private val NCount = VCount * TCount; private val SCount = LCount * NCount

  def apply(s: String): String =
    // below U+0300 nothing decomposes that would not compose back, and nothing combines
    var quick = true
    var k = 0
    while quick && k < s.length do { if s.charAt(k) >= 0x300 then quick = false; k += 1 }
    if quick then s else compose(order(decompose(s)))

  private def decompose(s: String): Array[Int] =
    val out = scala.collection.mutable.ArrayBuilder.make[Int]
    s.codePoints().forEach { c =>
      val si = c - SBase
      if si >= 0 && si < SCount then
        out += LBase + si / NCount
        out += VBase + (si % NCount) / TCount
        if si % TCount != 0 then out += TBase + si % TCount
      else
        val d = decomposition.get(c)
        if d == null then out += c else out ++= d
    }
    out.result()

  /** runs of combining marks sorted by class, stably */
  private def order(cs: Array[Int]): Array[Int] =
    var i = 1
    while i < cs.length do
      val c = classOf(cs(i))
      if c != 0 then
        var j = i
        while j > 0 && classOf(cs(j - 1)) > c do
          val t = cs(j); cs(j) = cs(j - 1); cs(j - 1) = t
          j -= 1
      i += 1
    cs

  private def compose(cs: Array[Int]): String =
    if cs.isEmpty then return ""
    val out = scala.collection.mutable.ArrayBuffer.empty[Int]
    var starter = 0                       // the index in out of the last starter, or -1
    var last = -1                         // the class of the last character put after it
    out += cs(0)
    if classOf(cs(0)) != 0 then starter = -1
    var i = 1
    while i < cs.length do
      val c = cs(i)
      val cc = classOf(c)
      val composed =
        if starter < 0 then -1
        else
          val s = out(starter)
          val blocked = last != -1 && (last == 0 || last >= cc)
          if blocked then -1
          else
            val li = s - LBase
            val vi = c - VBase
            val si = s - SBase
            if li >= 0 && li < LCount && vi >= 0 && vi < VCount then SBase + (li * VCount + vi) * TCount
            else if si >= 0 && si < SCount && si % TCount == 0 && c > TBase && c < TBase + TCount then s + (c - TBase)
            else composite.getOrDefault(pair(s, c), -1)
      if composed >= 0 then out(starter) = composed
      else
        out += c
        if cc == 0 then { starter = out.length - 1; last = -1 } else last = cc
      i += 1
    val sb = new java.lang.StringBuilder
    out.foreach(sb.appendCodePoint)
    sb.toString
