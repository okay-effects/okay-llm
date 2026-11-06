package okay.llm.qwen

import okay.llm.compute.{Compute, QLinear}

import scala.concurrent.{ExecutionContext, Future}

/**
 * QWEN3'S FORWARD AS A PROGRAM OVER `Compute`: the same words as `Qwen3Forward` — the embedding gathered on the CPU
 * (one row a token), then for each layer the norm, q k v, the heads normed and rotated, attention, the output
 * projection and the residual, the norm, gate and up, SwiGLU, down and the residual; the final norm; the logits of
 * the last row against the tied head. The weights go to the device once, at construction. What a skill needs: the
 * residual after a layer at some positions (`layer`, `at`), and the next token's logits.
 */
/** `headOnCpu`: the logits against the tied head computed on the CPU from the last row read back, the embedding's
 * 78 MB not sent to the device — a phone's WebGL2 has no room for it; slower an ask (150M multiplies in JS) */
/** `embed`: the embedding's rows — the whole from the file, or blocks by language (`Rows.Blocks`, more loaded as tokens
 * appear: `rehead()` after) */
final class Qwen3Compute(val c: Compute, st: Tensors, val config: Qwen3Config, headOnCpu: Boolean = false,
    embedGiven: Option[Rows] = None):
  import config.*
  /** a layer's weights on the device */
  final case class Layer(inNorm: c.T, postNorm: c.T, q: c.W, k: c.W, v: c.W, o: c.W, qNorm: c.T, kNorm: c.T,
    gate: c.W, up: c.W, down: c.W)
  // THE FILE IS READ ONCE, HERE, AND NOT KEPT: the weights live on the device from now on (a phone's browser has no
  // room for the file beside them — 244 MB held by this object was a tab killed); the embedding stays on the CPU
  // for the gather
  val (blocks: Vector[Layer], embedding: Rows) =
    def lin(p: String): c.W = c.quantized(st.linear(p))
    def vec(name: String): c.T = { val a = st.floats(name); c.tensor(1, a.length, a) }
    val layersRead = Vector.tabulate(layers) { i =>
      val p = s"model.layers.$i."
      Layer(vec(p + "input_layernorm.weight"), vec(p + "post_attention_layernorm.weight"),
        lin(p + "self_attn.q_proj"), lin(p + "self_attn.k_proj"), lin(p + "self_attn.v_proj"), lin(p + "self_attn.o_proj"),
        vec(p + "self_attn.q_norm.weight"), vec(p + "self_attn.k_norm.weight"),
        lin(p + "mlp.gate_proj"), lin(p + "mlp.up_proj"), lin(p + "mlp.down_proj"))
    }
    (layersRead, embedGiven.getOrElse(Rows.Full(st.linear("model.embed_tokens"))))
  // the head: the loaded rows on the device, and the id of each; made again when rows are added
  private var head: Option[(c.W, Array[Int])] = None
  def rehead(): Unit = if !headOnCpu then { val (q, ids) = embedding.packed; head = Some((c.quantized(q), ids)) }
  rehead()
  private val norm: c.T = { val a = st.floats("model.norm.weight"); c.tensor(1, a.length, a) }

  /** A PREFIX ALREADY RUN: its length, every layer's k and v on the device (the rows 0 until n), the residual after
   * `layer` for those rows — so a sequence that begins with it runs only the rest. Freed by `free` */
  final class Cache(val n: Int, val k: Vector[c.T], val v: Vector[c.T], val h: c.T, val layer: Int):
    def free(): Unit = { k.foreach(c.release); v.foreach(c.release); c.release(h) }

  /** the residual after `layer` at the rows `at` (none when `layer` is 0), and the next token's logits if asked */
  def forward(ids: Array[Int], layer: Int, at: Seq[Int], withLogits: Boolean)(using ExecutionContext)
      : Future[(Vector[Array[Float]], Array[Float])] =
    run(ids, layer, at, withLogits, None, keep = false).map((s, l, _) => (s, l))

  /** the same, after a cached prefix (`ids` begins with it: only the rest is run), and — `keep` — the whole of `ids`
   * cached for the next call; a cache given and not kept stays as it was */
  def run(ids: Array[Int], layer: Int, at: Seq[Int], withLogits: Boolean, cache: Option[Cache], keep: Boolean)(
      using ExecutionContext): Future[(Vector[Array[Float]], Array[Float], Option[Cache])] =
    val n0 = cache.fold(0)(_.n)
    require(n0 <= ids.length, "the cached prefix is longer than the sequence")
    require(cache.forall(_.layer == layer) || at.forall(_ >= n0), "the cache holds another layer's states")
    val suffix = ids.drop(n0); val n = suffix.length
    val (before, after) = at.partition(_ < n0)
    val fromCache: Future[Vector[Array[Float]]] =
      if before.isEmpty then Future.successful(Vector.empty) else c.rows(cache.get.h, before)
    if n == 0 then
      // nothing new: the states from the cache; no logits (a prefix's last token has none worth asking)
      fromCache.map(s => (s, Array.empty[Float], if keep then cache else None))
    else
      var h = c.tensor(n, hidden, Array.tabulate(n * hidden)(k => embedding.at(suffix(k / hidden), k % hidden)))
      var states: Future[Vector[Array[Float]]] = Future.successful(Vector.empty)
      val ks = Vector.newBuilder[c.T]; val vs = Vector.newBuilder[c.T]
      var kept: Option[c.T] = None
      for (l, li) <- blocks.zipWithIndex do
        val x = c.rmsNorm(h, l.inNorm, eps)
        val q0 = c.linear(x, l.q); val k0 = c.linear(x, l.k); val v = c.linear(x, l.v); c.release(x)
        val q = c.headNormRope(q0, l.qNorm, heads, headDim, eps, ropeTheta, n0); c.release(q0)
        val k = c.headNormRope(k0, l.kNorm, kvHeads, headDim, eps, ropeTheta, n0); c.release(k0)
        // the keys and values of the whole sequence: the prefix's from the cache, then the new ones
        val (kAll, vAll) = cache.fold((k, v))(cc => { val ka = c.concat(cc.k(li), k); val va = c.concat(cc.v(li), v); c.release(k); c.release(v); (ka, va) })
        val a = c.attention(q, kAll, vAll, heads, kvHeads, headDim, n0); c.release(q)
        if keep then { ks += kAll; vs += vAll } else { c.release(kAll); c.release(vAll) }
        val o = c.linear(a, l.o); c.release(a)
        val h1 = c.add(h, o); c.release(h); c.release(o)
        val y = c.rmsNorm(h1, l.postNorm, eps)
        val g = c.linear(y, l.gate); val u = c.linear(y, l.up); c.release(y)
        val act = c.swiglu(g, u); c.release(g); c.release(u)
        val d = c.linear(act, l.down); c.release(act)
        h = c.add(h1, d); c.release(h1); c.release(d)
        if li + 1 == layer then
          states = c.rows(h, after.map(_ - n0))
          if keep then kept = Some(cache.fold(c.slice(h, 0, n))(cc => c.concat(cc.h, h)))
      val fin = c.rmsNorm(h, norm, eps); c.release(h)
      val logits =
        if withLogits then head match
          case Some((hd, ids)) =>
            val last = c.slice(fin, n - 1, n); val lg = c.linear(last, hd); c.release(last)
            val read = c.rows(lg, Seq(0)); c.release(lg)
            read.map { rows =>
              val out = Array.fill(vocab)(Float.NegativeInfinity)
              ids.zipWithIndex.foreach((id, k) => out(id) = rows.head(k)); out }
          case None => c.rows(fin, Seq(n - 1)).map(rows => embedding.logits(rows.head))
        else Future.successful(Array.empty[Float])
      c.release(fin)
      val next = if keep then Some(Cache(ids.length, ks.result(), vs.result(), kept.getOrElse(c.tensor(0, hidden, Array.empty)), layer)) else None
      for cached <- fromCache; s <- states; l <- logits yield (cached ++ s, l, next)
