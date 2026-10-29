package okay.llm

import scala.collection.immutable.VectorMap

/** JSON, read and written: a model's config.json and tokenizer.json (11 MB: one pass over a char array), an API's
 * bodies. An object keeps its fields in the order written */
enum Json:
  case Obj(entries: VectorMap[String, Json])
  case Arr(values: Vector[Json])
  case Str(value: String)
  case Num(value: Double)
  case Bool(value: Boolean)
  case Null

  def apply(key: String): Json = this match
    case Obj(f) => f.getOrElse(key, Null)
    case _ => Null
  def get(key: String): Option[Json] = this match
    case Obj(f) => f.get(key).filter(_ != Null)
    case _ => None
  def str: String = this match { case Str(s) => s; case other => sys.error(s"not a string: $other") }
  def int: Int = this match { case Num(n) => n.toInt; case other => sys.error(s"not a number: $other") }
  def double: Double = this match { case Num(n) => n; case other => sys.error(s"not a number: $other") }
  def bool: Boolean = this match { case Bool(b) => b; case other => sys.error(s"not a boolean: $other") }
  def items: Vector[Json] = this match { case Arr(xs) => xs; case other => sys.error(s"not an array: $other") }
  def fields: VectorMap[String, Json] = this match { case Obj(f) => f; case other => sys.error(s"not an object: $other") }
  /** compact JSON text */
  def render: String =
    val sb = new java.lang.StringBuilder
    Json.write(this, sb)
    sb.toString

object Json:
  def parse(text: String): Json = Reader(text.toCharArray).document()
  def obj(fields: (String, Json)*): Json = Obj(VectorMap.from(fields))
  def arr(items: Json*): Json = Arr(items.toVector)

  private def write(j: Json, sb: java.lang.StringBuilder): Unit = j match
    case Obj(f) =>
      sb.append('{')
      var first = true
      for (k, v) <- f do
        if !first then sb.append(',')
        first = false
        quote(k, sb); sb.append(':'); write(v, sb)
      sb.append('}')
    case Arr(xs) =>
      sb.append('[')
      var first = true
      for v <- xs do
        if !first then sb.append(',')
        first = false
        write(v, sb)
      sb.append(']')
    case Str(s) => quote(s, sb)
    case Num(n) => if n == math.rint(n) && math.abs(n) < 1e15 then sb.append(n.toLong) else sb.append(n)
    case Bool(b) => sb.append(b)
    case Null => sb.append("null")

  /** a string quoted as JSON: `"`, `\` and the control characters escaped, the rest as it is */
  def quote(s: String, sb: java.lang.StringBuilder): Unit =
    sb.append('"')
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      c match
        case '"' => sb.append("\\\"")
        case '\\' => sb.append("\\\\")
        case '\n' => sb.append("\\n")
        case '\r' => sb.append("\\r")
        case '\t' => sb.append("\\t")
        case '\b' => sb.append("\\b")
        case '\f' => sb.append("\\f")
        case _ if c < ' ' => sb.append(f"\\u${c.toInt}%04x")
        case _ => sb.append(c)
      i += 1
    sb.append('"')

  private final class Reader(cs: Array[Char]):
    private var i = 0
    private val sb = new java.lang.StringBuilder

    def document(): Json =
      val v = value()
      space()
      if i != cs.length then fail("text after the value")
      v

    private def fail(what: String): Nothing = throw IllegalArgumentException(s"JSON at $i: $what")
    private def space(): Unit =
      while i < cs.length && (cs(i) == ' ' || cs(i) == '\n' || cs(i) == '\r' || cs(i) == '\t') do i += 1
    private def expect(c: Char): Unit =
      space()
      if i >= cs.length || cs(i) != c then fail(s"'$c' expected")
      i += 1

    private def value(): Json =
      space()
      if i >= cs.length then fail("a value expected")
      cs(i) match
        case '{' => obj()
        case '[' => arr()
        case '"' => Str(string())
        case 't' => word("true", Bool(true))
        case 'f' => word("false", Bool(false))
        case 'n' => word("null", Null)
        case _ => num()

    private def word(w: String, v: Json): Json =
      if !cs.slice(i, i + w.length).sameElements(w) then fail(s"$w expected")
      i += w.length
      v

    private def obj(): Json =
      i += 1
      val m = VectorMap.newBuilder[String, Json]
      space()
      if cs(i) == '}' then { i += 1; return Obj(VectorMap.empty) }
      var more = true
      while more do
        space()
        val k = string()
        expect(':')
        m += k -> value()
        space()
        if cs(i) == ',' then i += 1 else { expect('}'); more = false }
      Obj(m.result())

    private def arr(): Json =
      i += 1
      val b = Vector.newBuilder[Json]
      space()
      if cs(i) == ']' then { i += 1; return Arr(Vector.empty) }
      var more = true
      while more do
        b += value()
        space()
        if cs(i) == ',' then i += 1 else { expect(']'); more = false }
      Arr(b.result())

    private def string(): String =
      if cs(i) != '"' then fail("a string expected")
      i += 1
      sb.setLength(0)
      while cs(i) != '"' do
        val c = cs(i)
        if c == '\\' then
          i += 1
          cs(i) match
            case 'n' => sb.append('\n')
            case 't' => sb.append('\t')
            case 'r' => sb.append('\r')
            case 'b' => sb.append('\b')
            case 'f' => sb.append('\f')
            case 'u' =>
              sb.append(Integer.parseInt(String(cs, i + 1, 4), 16).toChar)
              i += 4
            case other => sb.append(other)
        else sb.append(c)
        i += 1
      i += 1
      sb.toString

    private def num(): Json =
      val start = i
      while i < cs.length && "+-0123456789.eE".indexOf(cs(i).toInt) >= 0 do i += 1
      if start == i then fail("a value expected")
      Num(String(cs, start, i - start).toDouble)
