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
final class Qwen3Compute(val c: Compute, st: Safetensors, val config: Qwen3Config):
  import config.*
  private def lin(p: String): c.W = c.quantized(st.linear(p))
  private def vec(name: String): c.T = { val a = st.floats(name); c.tensor(1, a.length, a) }
  final class Layer(i: Int):
    val p = s"model.layers.$i."
    val inNorm = vec(p + "input_layernorm.weight"); val postNorm = vec(p + "post_attention_layernorm.weight")
    val q = lin(p + "self_attn.q_proj"); val k = lin(p + "self_attn.k_proj")
    val v = lin(p + "self_attn.v_proj"); val o = lin(p + "self_attn.o_proj")
    val qNorm = vec(p + "self_attn.q_norm.weight"); val kNorm = vec(p + "self_attn.k_norm.weight")
    val gate = lin(p + "mlp.gate_proj"); val up = lin(p + "mlp.up_proj"); val down = lin(p + "mlp.down_proj")
  val blocks: Vector[Layer] = Vector.tabulate(layers)(Layer(_))
  private val embedding: QLinear = st.linear("model.embed_tokens")
  private val head: c.W = c.quantized(embedding)
  private val norm: c.T = vec("model.norm.weight")

  /** the residual after `layer` at the rows `at` (none when `layer` is 0), and the next token's logits if asked */
  def forward(ids: Array[Int], layer: Int, at: Seq[Int], withLogits: Boolean)(using ExecutionContext)
      : Future[(Vector[Array[Float]], Array[Float])] =
    val n = ids.length
    var h = c.tensor(n, hidden, Array.tabulate(n * hidden)(k => embedding.at(ids(k / hidden), k % hidden)))
    var states: Future[Vector[Array[Float]]] = Future.successful(Vector.empty)
    for (l, li) <- blocks.zipWithIndex do
      val x = c.rmsNorm(h, l.inNorm, eps)
      val q0 = c.linear(x, l.q); val k0 = c.linear(x, l.k); val v = c.linear(x, l.v); c.release(x)
      val q = c.headNormRope(q0, l.qNorm, heads, headDim, eps, ropeTheta); c.release(q0)
      val k = c.headNormRope(k0, l.kNorm, kvHeads, headDim, eps, ropeTheta); c.release(k0)
      val a = c.attention(q, k, v, heads, kvHeads, headDim); c.release(q); c.release(k); c.release(v)
      val o = c.linear(a, l.o); c.release(a)
      val h1 = c.add(h, o); c.release(h); c.release(o)
      val y = c.rmsNorm(h1, l.postNorm, eps)
      val g = c.linear(y, l.gate); val u = c.linear(y, l.up); c.release(y)
      val act = c.swiglu(g, u); c.release(g); c.release(u)
      val d = c.linear(act, l.down); c.release(act)
      h = c.add(h1, d); c.release(h1); c.release(d)
      if li + 1 == layer then states = c.rows(h, at)
    val fin = c.rmsNorm(h, norm, eps); c.release(h)
    val logits =
      if withLogits then
        val last = c.slice(fin, n - 1, n); val lg = c.linear(last, head); c.release(last)
        val read = c.rows(lg, Seq(0)); c.release(lg)
        read.map(_.head)
      else Future.successful(Array.empty[Float])
    c.release(fin)
    for s <- states; l <- logits yield (s, l)
