package okay.llm.qwen

import okay.effects.json.Json
import okay.llm.compute.QLinear

/**
 * SAFETENSORS READ IN PLAIN SCALA (the JVM and the browser alike): the header (8 bytes of length, JSON), then the
 * tensors' bytes. MLX's 4-bit layout: a linear's `weight` U32 [out, in / 8] (eight 4-bit values a word, the lowest
 * bits first), `scales` and `biases` BF16 [out, in / 64]; norms BF16 [d]. Values decoded to float32 on demand.
 */
/** TENSORS BY NAME, wherever they are: one safetensors file, or several (`Tensors.Many`) — the weights published as
 * files of a few layers each, read one by one, so a page never holds the whole of them twice */
trait Tensors:
  def has(name: String): Boolean
  /** a BF16, F16 or F32 tensor as float32 */
  def floats(name: String): Array[Float]
  /** a U32 tensor as ints */
  def words(name: String): Array[Int]
  /** the shape of a tensor */
  def shape(name: String): Vector[Int]
  def linear(prefix: String): QLinear =
    val s = shape(prefix + ".weight")
    QLinear(s(0), s(1) * 8, words(prefix + ".weight"), floats(prefix + ".scales"), floats(prefix + ".biases"))

object Tensors:
  /** several files as one: a name found in the first that has it */
  final class Many(parts: Vector[Tensors]) extends Tensors:
    private def in(name: String): Tensors = parts.find(_.has(name)).getOrElse(throw new java.util.NoSuchElementException(s"no tensor $name"))
    def has(name: String): Boolean = parts.exists(_.has(name))
    def floats(name: String): Array[Float] = in(name).floats(name)
    def words(name: String): Array[Int] = in(name).words(name)
    def shape(name: String): Vector[Int] = in(name).shape(name)

final class Safetensors(buffer: java.nio.ByteBuffer) extends Tensors:
  /** the bytes from `from` for `length`, little-endian — a view, no copy (in a browser the fetched ArrayBuffer itself) */
  private def view(from: Int, length: Int): java.nio.ByteBuffer =
    val b = buffer.duplicate()
    b.position(from); b.limit(from + length)
    b.slice().order(java.nio.ByteOrder.LITTLE_ENDIAN)
  private val headerLength: Long = view(0, 8).getLong
  private val base = 8 + headerLength.toInt
  final case class Entry(dtype: String, shape: Vector[Int], from: Int, until: Int)
  val entries: Map[String, Entry] =
    Json.fromText({ val h = new Array[Byte](headerLength.toInt); view(8, h.length).get(h); new String(h, "UTF-8") }) match
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
    val b = view(base + e.from, e.until - e.from)
    e.dtype match
      case "BF16" => Array.tabulate(n)(_ => java.lang.Float.intBitsToFloat((b.getShort & 0xffff) << 16))
      case "F16" => Array.tabulate(n)(_ => Safetensors.halfToFloat(b.getShort & 0xffff))
      case "F32" => Array.tabulate(n / 2)(_ => b.getFloat)
      case other => sys.error(s"$name: $other not read")
  /** a U32 tensor as ints */
  def words(name: String): Array[Int] =
    val e = entries(name)
    val b = view(base + e.from, e.until - e.from)
    Array.tabulate((e.until - e.from) / 4)(_ => b.getInt)

  def shape(name: String): Vector[Int] = entries(name).shape
  /** the raw bytes of a tensor, as written: to write it elsewhere */
  def raw(name: String): Array[Byte] =
    val e = entries(name); val out = new Array[Byte](e.until - e.from); view(base + e.from, out.length).get(out); out

object Safetensors:
  /** THE FILE WRITTEN: the entries given (name, dtype, shape, bytes) — a header of their offsets, then the bytes; what
   * `split` makes for a page */
  def write(entries: Vector[(String, String, Vector[Int], Array[Byte])]): Array[Byte] =
    var at = 0
    val fields = entries.map { (name, dtype, shape, bytes) =>
      val e = Json.obj("dtype" -> Json.Str(dtype), "shape" -> Json.Arr(shape.map(n => Json.Num(n))),
        "data_offsets" -> Json.Arr(Vector(Json.Num(at), Json.Num(at + bytes.length))))
      at += bytes.length
      name -> e
    }
    val header = Json.obj(fields*).render.getBytes("UTF-8")
    val pad = (8 - header.length % 8) % 8
    val out = java.nio.ByteBuffer.allocate(8 + header.length + pad + at).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    out.putLong(header.length + pad).put(header).put(Array.fill[Byte](pad)(0x20))   // the pad is spaces, as the format has it
    entries.foreach((_, _, _, b) => out.put(b))
    out.array()

  /** THE FILE SPLIT for a page: files of at most `limit` bytes, whole tensors in the file's order — the embedding and
   * the norm first, then the layers; each a safetensors file of its own (`Tensors.Many` reads them as one) */
  def split(st: Safetensors, limit: Int): Vector[Array[Byte]] =
    val names = st.entries.keys.toVector.sortBy { n =>
      val layer = "model\\.layers\\.(\\d+)\\..*".r
      n match { case layer(i) => (1, i.toInt, n); case _ => (0, 0, n) } }
    val files = Vector.newBuilder[Array[Byte]]
    var current = Vector.newBuilder[(String, String, Vector[Int], Array[Byte])]; var size = 0
    for n <- names do
      val e = st.entries(n); val bytes = st.raw(n)
      if size > 0 && size + bytes.length > limit then { files += write(current.result()); current = Vector.newBuilder; size = 0 }
      current += ((n, e.dtype, e.shape, bytes)); size += bytes.length
    if size > 0 then files += write(current.result())
    files.result()

  def apply(bytes: Array[Byte]): Safetensors = new Safetensors(java.nio.ByteBuffer.wrap(bytes))
  def halfToFloat(h: Int): Float =
    val s = (h >> 15) & 1; val e = (h >> 10) & 0x1f; val m = h & 0x3ff
    val f = if e == 0 then (if m == 0 then 0f else (m / 1024f) * math.pow(2, -14).toFloat)
      else if e == 31 then (if m == 0 then Float.PositiveInfinity else Float.NaN)
      else (1f + m / 1024f) * math.pow(2, e - 15).toFloat
    if s == 1 then -f else f
