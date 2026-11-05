package okay.llm.compute

import scala.concurrent.Future

/** THE DEVICE AS PLAIN SCALA: the reference every other handler is checked against (slow: a prompt of hundreds of
 * tokens is minutes), the fallback where a browser has neither WebGPU nor WebGL2 */
final class OnCpu extends Compute:
  final class T(val rows: Int, val cols: Int, val data: Array[Float])
  type W = QLinear
  val api = "CPU"
  def tensor(rows: Int, cols: Int, data: Array[Float]): T = T(rows, cols, data)
  def quantized(q: QLinear): W = q
  private def row(x: T, r: Int): Array[Float] = x.data.slice(r * x.cols, (r + 1) * x.cols)
  private def byRows(x: T, cols: Int)(f: Int => Array[Float]): T =
    val out = new Array[Float](x.rows * cols)
    for r <- 0 until x.rows do f(r).copyToArray(out, r * cols)
    T(x.rows, cols, out)
  def linear(x: T, w: W): T = byRows(x, w.out)(r => w(row(x, r)))
  private def norm(x: Array[Float], off: Int, n: Int, w: Array[Float], eps: Float): Array[Float] =
    var ss = 0f; var i = 0
    while i < n do { ss += x(off + i) * x(off + i); i += 1 }
    val s = (1.0 / math.sqrt(ss / n + eps)).toFloat
    Array.tabulate(n)(k => x(off + k) * s * w(k))
  def rmsNorm(x: T, w: T, eps: Float): T = byRows(x, x.cols)(r => norm(x.data, r * x.cols, x.cols, w.data, eps))
  def headNormRope(x: T, w: T, heads: Int, headDim: Int, eps: Float, theta: Float): T =
    val half = headDim / 2
    byRows(x, x.cols) { r =>
      val out = new Array[Float](x.cols)
      for hd <- 0 until heads do
        val v = norm(x.data, r * x.cols + hd * headDim, headDim, w.data, eps)
        for i <- 0 until half do
          val a = r * math.pow(theta, -2.0 * i / headDim)
          val c = math.cos(a).toFloat; val s = math.sin(a).toFloat
          out(hd * headDim + i) = v(i) * c - v(half + i) * s
          out(hd * headDim + half + i) = v(half + i) * c + v(i) * s
      out
    }
  def attention(q: T, k: T, v: T, heads: Int, kvHeads: Int, headDim: Int): T =
    val scale = (1.0 / math.sqrt(headDim)).toFloat
    byRows(q, heads * headDim) { t =>
      val out = new Array[Float](heads * headDim)
      for hd <- 0 until heads do
        val kv = hd / (heads / kvHeads)
        val scores = Array.tabulate(t + 1) { j =>
          var s = 0f; var i = 0
          while i < headDim do { s += q.data(t * q.cols + hd * headDim + i) * k.data(j * k.cols + kv * headDim + i); i += 1 }
          s * scale
        }
        val m = scores.max; val e = scores.map(s => math.exp((s - m).toDouble).toFloat); val z = e.sum
        for j <- 0 to t do
          val w = e(j) / z; var i = 0
          while i < headDim do { out(hd * headDim + i) += w * v.data(j * v.cols + kv * headDim + i); i += 1 }
      out
    }
  def add(a: T, b: T): T = T(a.rows, a.cols, Array.tabulate(a.data.length)(i => a.data(i) + b.data(i)))
  def swiglu(g: T, u: T): T =
    T(g.rows, g.cols, Array.tabulate(g.data.length)(i => g.data(i) / (1f + math.exp(-g.data(i)).toFloat) * u.data(i)))
  def slice(x: T, from: Int, until: Int): T = T(until - from, x.cols, x.data.slice(from * x.cols, until * x.cols))
  def rows(x: T, at: Seq[Int]): Future[Vector[Array[Float]]] = Future.successful(at.toVector.map(row(x, _)))
  def release(x: T): Unit = ()
