package okay.llm.compute

import mlx.{Arr, Dtype, Scope}
import mlx.Mlx

import scala.concurrent.Future

/**
 * `Compute` ON APPLE'S GPU through mlx-scala: the fourth handler of the one forward (`Qwen3Compute`), beside the
 * CPU, WebGPU and WebGL2 — so the JVM runs the SAME program as the browser, and mlx-scala's own forward is no longer
 * a second implementation the laws must bridge. Thin: every op is one of MLX's (quantized matmul in MLX's own
 * layout, rms_norm, rope, matmul + softmax for the attention with a causal mask of the positions). Tensors are
 * float32 arrays owned by the handler's scope; `release` frees one, `close` all.
 */
final class OnMlx extends Compute:
  type T = Arr
  type W = (Arr, Arr, Arr)
  private given s: Scope = new Scope
  def api: String = "mlx"

  def tensor(rows: Int, cols: Int, data: Array[Float]): T = Mlx.floats(data, rows, cols)
  def quantized(q: QLinear): W =
    (Mlx.astype(Mlx.ints(q.words, q.out, q.in / 8), Dtype.UInt32), Mlx.floats(q.scales, q.out, q.groups), Mlx.floats(q.biases, q.out, q.groups))
  def linear(x: T, w: W): T = Mlx.quantizedMatmul(x, w._1, w._2, w._3, transpose = true, groupSize = 64, bits = 4)
  // THE INTERMEDIATES OF AN OP FREED AT ONCE: MLX arrays are lazy and reference each other, so our handle on a
  // reshape or a transpose can go the moment the op's result holds it — else a reading of a hundred questions grows
  // without bound (the bench was killed at 137)
  private def freeing[A](f: scala.collection.mutable.ArrayBuffer[Arr] => A): A =
    val temps = scala.collection.mutable.ArrayBuffer.empty[Arr]
    try f(temps) finally temps.foreach(s.release)
  def rmsNorm(x: T, w: T, eps: Float): T = freeing { t =>
    val w1 = Mlx.reshape(w, w.dim(1)); t += w1
    Mlx.rmsNorm(x, w1, eps) }
  def headNormRope(x: T, w: T, heads: Int, headDim: Int, eps: Float, theta: Float, pos0: Int): T = freeing { t =>
    val rows = x.dim(0)
    val h = Mlx.reshape(x, rows, heads, headDim); val w1 = Mlx.reshape(w, headDim); t ++= Seq(h, w1)
    val normed = Mlx.rmsNorm(h, w1, eps); t += normed
    val byHead = Mlx.transpose(normed, 1, 0, 2); t += byHead // heads × rows × headDim: the sequence axis before the dims, as rope wants
    val roped = Mlx.rope(byHead, headDim, traditional = false, base = theta, scale = 1f, offset = pos0); t += roped
    val back = Mlx.transpose(roped, 1, 0, 2); t += back
    Mlx.reshape(back, rows, heads * headDim) }
  def attention(q: T, k: T, v: T, heads: Int, kvHeads: Int, headDim: Int, pos0: Int): T = freeing { t =>
    val rows = q.dim(0); val kRows = k.dim(0); val rep = heads / kvHeads
    def grouped(x: Arr, n: Int, hs: Int, groups: Int, per: Int): Arr =
      val a = Mlx.reshape(x, n, hs, headDim); val b = Mlx.transpose(a, 1, 0, 2); t ++= Seq(a, b)
      val c = Mlx.reshape(b, groups, per, n, headDim); t += c; c
    // the heads grouped by their kv head: q [kvHeads, rep, rows, headDim] against k, v [kvHeads, 1, kRows, headDim]
    val qh = grouped(q, rows, heads, kvHeads, rep); val kh = grouped(k, kRows, kvHeads, kvHeads, 1); val vh = grouped(v, kRows, kvHeads, kvHeads, 1)
    val kt = Mlx.transpose(kh, 0, 1, 3, 2); val raw = Mlx.matmul(qh, kt); val scale = Mlx.scalar(1f / math.sqrt(headDim.toDouble).toFloat)
    val scores = Mlx.multiply(raw, scale); val masked = Mlx.add(scores, causal(rows, kRows, pos0)); val probs = Mlx.softmax(masked, -1)
    val out = Mlx.matmul(probs, vh) // kvHeads × rep × rows × headDim
    val o1 = Mlx.reshape(out, heads, rows, headDim); val o2 = Mlx.transpose(o1, 1, 0, 2)
    t ++= Seq(kt, raw, scale, scores, masked, probs, out, o1, o2)
    Mlx.reshape(o2, rows, heads * headDim) }
  // causal: the query at pos0 + i sees the keys at positions up to it; one mask per shape, kept (a reading asks the
  // same shapes again and again)
  private val masks = scala.collection.mutable.Map.empty[(Int, Int, Int), Arr]
  private def causal(rows: Int, kRows: Int, pos0: Int): Arr = masks.getOrElseUpdate((rows, kRows, pos0),
    Mlx.floats(Array.tabulate(rows * kRows)(i => if i % kRows <= pos0 + i / kRows then 0f else Float.NegativeInfinity), rows, kRows))
  def concat(a: T, b: T): T = Mlx.concatenate(Seq(a, b), 0)
  def add(a: T, b: T): T = Mlx.add(a, b)
  def swiglu(gate: T, up: T): T = freeing { t => val g = Mlx.silu(gate); t += g; Mlx.multiply(g, up) }
  def slice(x: T, from: Int, until: Int): T = Mlx.slice(x, Seq(from, 0), Seq(until, x.dim(1)))
  def rows(x: T, at: Seq[Int]): Future[Vector[Array[Float]]] = freeing { t =>
    val cols = x.dim(1); val f = Mlx.astype(x, Dtype.Float32); t += f
    val all = f.eval().toFloats
    Future.successful(at.toVector.map(i => all.slice(i * cols, (i + 1) * cols))) }
  def release(x: T): Unit = s.release(x)
  /** every tensor freed */
  override def close(): Unit = s.close()
