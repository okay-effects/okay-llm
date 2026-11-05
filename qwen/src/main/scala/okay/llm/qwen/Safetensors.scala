package okay.llm.qwen

import okay.effects.json.Json
import okay.llm.compute.QLinear

/**
 * SAFETENSORS READ IN PLAIN SCALA (the JVM and the browser alike): the header (8 bytes of length, JSON), then the
 * tensors' bytes. MLX's 4-bit layout: a linear's `weight` U32 [out, in / 8] (eight 4-bit values a word, the lowest
 * bits first), `scales` and `biases` BF16 [out, in / 64]; norms BF16 [d]. Values decoded to float32 on demand.
 */
final class Safetensors(bytes: Array[Byte]):
  private val headerLength: Long =
    java.nio.ByteBuffer.wrap(bytes, 0, 8).order(java.nio.ByteOrder.LITTLE_ENDIAN).getLong
  private val base = 8 + headerLength.toInt
  final case class Entry(dtype: String, shape: Vector[Int], from: Int, until: Int)
  val entries: Map[String, Entry] =
    Json.fromText(new String(bytes, 8, headerLength.toInt, "UTF-8")) match
      case Json.Obj(fields) => fields.collect { case (name, Json.Obj(fs)) if name != "__metadata__" =>
        val dtype = fs.collectFirst { case ("dtype", Json.Str(s)) => s }.getOrElse("")
        val shape = fs.collectFirst { case ("shape", Json.Arr(xs)) => xs.map(_.asInstanceOf[Json.Num].n.toInt).toVector }.getOrElse(Vector.empty)
        val (from, until) = fs.collectFirst { case ("data_offsets", Json.Arr(Vector(Json.Num(a), Json.Num(b)))) => (a.toInt, b.toInt) }.getOrElse((0, 0))
        name -> Entry(dtype, shape, from, until) }.toMap
      case _ => Map.empty
  def has(name: String): Boolean = entries.contains(name)

  /** a BF16 tensor as float32: the 16 bits are the float's upper half */
  def floats(name: String): Array[Float] =
    val e = entries(name)
    val n = (e.until - e.from) / 2
    val b = java.nio.ByteBuffer.wrap(bytes, base + e.from, e.until - e.from).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    e.dtype match
      case "BF16" => Array.tabulate(n)(_ => java.lang.Float.intBitsToFloat((b.getShort & 0xffff) << 16))
      case "F16" => Array.tabulate(n)(_ => Safetensors.halfToFloat(b.getShort & 0xffff))
      case "F32" => Array.tabulate(n / 2)(_ => b.getFloat)
      case other => sys.error(s"$name: $other not read")
  /** a U32 tensor as ints */
  def words(name: String): Array[Int] =
    val e = entries(name)
    val b = java.nio.ByteBuffer.wrap(bytes, base + e.from, e.until - e.from).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    Array.tabulate((e.until - e.from) / 4)(_ => b.getInt)

  def linear(prefix: String): QLinear =
    val e = entries(prefix + ".weight")
    QLinear(e.shape(0), e.shape(1) * 8, words(prefix + ".weight"), floats(prefix + ".scales"), floats(prefix + ".biases"))

object Safetensors:
  def halfToFloat(h: Int): Float =
    val s = (h >> 15) & 1; val e = (h >> 10) & 0x1f; val m = h & 0x3ff
    val f = if e == 0 then (if m == 0 then 0f else (m / 1024f) * math.pow(2, -14).toFloat)
      else if e == 31 then (if m == 0 then Float.PositiveInfinity else Float.NaN)
      else (1f + m / 1024f) * math.pow(2, e - 15).toFloat
    if s == 1 then -f else f
