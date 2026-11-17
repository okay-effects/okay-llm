package okay.llm.compute

import scala.concurrent.{ExecutionContext, Future}

/** THE WEIGHTS ON THE DEVICE BEFORE THE MODEL IS MADE, A FILE AT A TIME: `put` sends a weight to `base` at once and
 * gives back a token (a `QLinear` of empty arrays) that `quantized` later answers with the weight already there. A page
 * fetches its weight files one by one, puts each file's weights and lets the file go — the model (`Qwen3Compute`) is
 * made from the tokens afterwards, so the files are never all in memory together (a phone's tab: specs/phone.md of
 * okay-desk, 1.1 GB of a WebKit page that the iPhone's Safari killed). Every op is the base's own. */
final class Preloaded(val base: Compute) extends Compute:
  type T = base.T
  type W = base.W
  private val ready = new java.util.IdentityHashMap[Array[Int], base.W]()

  /** the weight on the device now; the token to give `quantized` for it later */
  def put(q: QLinear): QLinear =
    val w = base.quantized(q)
    val token = QLinear(q.out, q.in, new Array[Int](0), new Array[Float](0), new Array[Float](0))
    ready.put(token.words, w)
    token
  /** a weight the device already holds (sent by its own fast path): its token */
  def putMade(out: Int, in: Int, w: base.W): QLinear =
    val token = QLinear(out, in, new Array[Int](0), new Array[Float](0), new Array[Float](0))
    ready.put(token.words, w)
    token
  /** the weights put and not yet taken */
  def waiting: Int = ready.size

  def quantized(q: QLinear): W =
    val w = ready.remove(q.words)
    if w != null then w else base.quantized(q)

  def api: String = base.api
  override def close(): Unit = base.close()
  override def cache(x: T): T = base.cache(x)
  def tensor(rows: Int, cols: Int, data: Array[Float]): T = base.tensor(rows, cols, data)
  def linear(x: T, w: W): T = base.linear(x, w)
  def rmsNorm(x: T, w: T, eps: Float): T = base.rmsNorm(x, w, eps)
  def headNormRope(x: T, w: T, heads: Int, headDim: Int, eps: Float, theta: Float, pos0: Int = 0): T =
    base.headNormRope(x, w, heads, headDim, eps, theta, pos0)
  def attention(q: T, k: T, v: T, heads: Int, kvHeads: Int, headDim: Int, pos0: Int = 0): T =
    base.attention(q, k, v, heads, kvHeads, headDim, pos0)
  // the packed ops and the log-probability: the base's own (a default here would copy a segment at a time)
  override def headNormRopeSegments(x: T, w: T, heads: Int, headDim: Int, eps: Float, theta: Float, pos0: Int,
      starts: Array[Int], rows: Int): T = base.headNormRopeSegments(x, w, heads, headDim, eps, theta, pos0, starts, rows)
  override def attentionSegments(q: T, kPre: T, vPre: T, k: T, v: T, heads: Int, kvHeads: Int, headDim: Int,
      pos0: Int, starts: Array[Int], rows: Int): T =
    base.attentionSegments(q, kPre, vPre, k, v, heads, kvHeads, headDim, pos0, starts, rows)
  override def logProbs(x: T, w: W, n: Int, columns: Array[Int])(using ExecutionContext): Future[Array[Float]] =
    base.logProbs(x, w, n, columns)
  def concat(a: T, b: T): T = base.concat(a, b)
  def add(a: T, b: T): T = base.add(a, b)
  def swiglu(gate: T, up: T): T = base.swiglu(gate, up)
  def slice(x: T, from: Int, until: Int): T = base.slice(x, from, until)
  def rows(x: T, at: Seq[Int]): Future[Vector[Array[Float]]] = base.rows(x, at)
  def release(x: T): Unit = base.release(x)

object Preloaded:
  /** THE TENSORS OF A MODEL WHOSE WEIGHTS WERE PUT: the linears as tokens, the small tensors (norms) as copies —
   * the files they came from already let go */
  final class Kept extends okay.llm.qwen.Tensors:
    private val linears = scala.collection.mutable.HashMap.empty[String, QLinear]
    private val small = scala.collection.mutable.HashMap.empty[String, Array[Float]]
    private val shapes = scala.collection.mutable.HashMap.empty[String, Vector[Int]]
    // A LINEAR CUT BETWEEN TWO FILES (its scales in one, its words in the next): its parts kept until all three came
    private val parts = scala.collection.mutable.HashMap.empty[String, scala.collection.mutable.HashMap[String, Any]]
    /** every tensor of one file: its linears put on `device` (as soon as their three parts are here), the rest copied */
    def add(device: Preloaded, file: okay.llm.qwen.Safetensors): Unit =
      for (name, e) <- file.entries do
        shapes(name) = e.shape
        val dot = name.lastIndexOf('.')
        val (prefix, part) = (name.take(dot), name.drop(dot + 1))
        val linearPart = part == "scales" || part == "biases" || part == "weight" && e.dtype == "U32"
        if !linearPart then small(name) = file.floats(name)
        else
          val got = parts.getOrElseUpdate(prefix, scala.collection.mutable.HashMap.empty)
          got(part) = if part == "weight" then file.words(name) else file.floats(name)
          if got.size == 3 then
            val s = shapes(prefix + ".weight")
            linears(prefix) = device.put(QLinear(s(0), s(1) * 8, got("weight").asInstanceOf[Array[Int]],
              got("scales").asInstanceOf[Array[Float]], got("biases").asInstanceOf[Array[Float]]))
            parts.remove(prefix)
    /** a linear already on the device (its token), a small tensor, a shape: for a loader with its own reading */
    def keepLinear(prefix: String, token: QLinear): Unit = linears(prefix) = token
    def keepSmall(name: String, values: Array[Float]): Unit = small(name) = values
    def keepShape(name: String, s: Vector[Int]): Unit = shapes(name) = s
    /** every linear complete (none waiting for a part from a file not given) */
    def complete: Boolean = parts.isEmpty
    def has(name: String): Boolean = shapes.contains(name)
    def floats(name: String): Array[Float] = small.getOrElse(name, sys.error(s"$name: not kept"))
    def words(name: String): Array[Int] = sys.error(s"$name: its words are on the device")
    def shape(name: String): Vector[Int] = shapes(name)
    override def linear(prefix: String): QLinear = linears.getOrElse(prefix, sys.error(s"$prefix: not put"))
