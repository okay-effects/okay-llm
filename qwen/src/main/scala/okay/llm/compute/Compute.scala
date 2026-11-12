package okay.llm.compute

import scala.concurrent.Future

/**
 * A DEVICE OF TENSORS, as a transformer needs one (BACKLOG "The browser engine"): a tensor is rows × cols of float32
 * on the device, a weight a 4-bit matrix in MLX's layout (`QLinear`); the ops are the few a Qwen3 layer is made of.
 * A program over `Compute` (`okay.llm.qwen.Qwen3Compute`) runs the same on every handler: the CPU (`OnCpu`, plain
 * Scala, the reference), WebGPU (`OnWebGpu`) and WebGL2 (`OnWebGl2`) in the browser. Reading back is the one
 * asynchronous step (`rows`); a tensor is released by hand (`release`) — the device's memory is not the garbage
 * collector's. A tensor may be released right after the op that reads it is issued: a handler keeps it until that
 * op has run.
 */
trait Compute:
  /** Release a dedicated model device before another model takes its place. */
  def close(): Unit = ()
  /** Transfer a tensor into long-lived prefix storage. The handler may consume the input. */
  def cache(x: T): T = x
  type T
  type W
  def api: String
  /** a tensor of `rows` × `cols` from its values, row-major */
  def tensor(rows: Int, cols: Int, data: Array[Float]): T
  /** a 4-bit weight [out, in] as `QLinear` holds it */
  def quantized(q: QLinear): W
  /** y[r] = W x[r]: every row of `in` against the weight [out, in], to `out` */
  def linear(x: T, w: W): T
  /** every row RMS-normed with the weight `w` (1 × cols) */
  def rmsNorm(x: T, w: T, eps: Float): T
  /** every head of every row (cols = heads · headDim) RMS-normed by `w` (1 × headDim), then RoPE by the row's
   * position — `pos0` plus the row's number (a suffix after a cached prefix): the halves rotated, the frequencies
   * theta^(-2i/headDim) */
  def headNormRope(x: T, w: T, heads: Int, headDim: Int, eps: Float, theta: Float, pos0: Int = 0): T
  /** causal attention: q rows × (heads · headDim) at positions `pos0` plus the row, k and v rows × (kvHeads · headDim)
   * at positions 0 until k.rows (k.rows = pos0 + q.rows); the head h reads the kv head h / (heads / kvHeads); the
   * scores scaled by 1 / sqrt(headDim) */
  def attention(q: T, k: T, v: T, heads: Int, kvHeads: Int, headDim: Int, pos0: Int = 0): T
  /** the rows of `a` then the rows of `b` (the same cols) */
  def concat(a: T, b: T): T
  def add(a: T, b: T): T
  /** gate · sigmoid(gate) · up */
  def swiglu(gate: T, up: T): T
  /** the rows from `from` until `until` */
  def slice(x: T, from: Int, until: Int): T
  /** the rows `at` read back */
  def rows(x: T, at: Seq[Int]): Future[Vector[Array[Float]]]
  def release(x: T): Unit

/** A QUANTIZED LINEAR: W [out, in] = scales · q + biases by groups of 64 along `in`; the 4-bit values eight a word,
 * the lowest bits first (MLX's layout: `weight` U32 [out, in / 8], `scales` and `biases` [out, in / 64]) */
final case class QLinear(out: Int, in: Int, words: Array[Int], scales: Array[Float], biases: Array[Float]):
  val groups: Int = in / 64
  /** W[o][i] decoded */
  def at(o: Int, i: Int): Float =
    val word = words(o * (in / 8) + i / 8)
    val q = (word >>> ((i % 8) * 4)) & 0xf
    val g = o * groups + i / 64
    scales(g) * q + biases(g)
  /** y = W x for one vector x (length `in`): the row's groups decoded as they go */
  def apply(x: Array[Float]): Array[Float] =
    val y = new Array[Float](out)
    val perRow = in / 8
    var o = 0
    while o < out do
      var acc = 0f
      var i = 0
      while i < in do
        val g = o * groups + i / 64
        val s = scales(g); val bq = biases(g)
        var k = 0
        while k < 64 do
          val word = words(o * perRow + (i + k) / 8)
          val q = (word >>> (((i + k) % 8) * 4)) & 0xf
          acc += (s * q + bq) * x(i + k)
          k += 1
        i += 64
      y(o) = acc
      o += 1
    y
