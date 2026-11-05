package okay.llm.qwen

/**
 * QWEN3'S FORWARD IN PLAIN SCALA (specs/data.md §15 step 2): the reference handler — one code for the JVM (the law
 * against mlx-scala) and the browser (the fallback under WebGPU and WebGL2). The model as `Cut` leaves it: the
 * embedding (quantized, tied to the head), L layers of attention (GQA, q and k normed per head, RoPE) and a SwiGLU
 * MLP, the final norm. What a skill needs: the residual after a layer at positions, and the last position's logits.
 * Slow (a prompt of 500 tokens is minutes on a CPU) — the device handlers are the point; this is the truth they
 * are checked against.
 */
final case class Qwen3Config(hidden: Int, layers: Int, heads: Int, kvHeads: Int, headDim: Int, intermediate: Int,
    vocab: Int, ropeTheta: Float, eps: Float):
  def kvDim: Int = kvHeads * headDim

object Qwen3Config:
  /** from config.json's text */
  def parse(text: String): Qwen3Config =
    def int(k: String) = s""""$k"\\s*:\\s*(\\d+)""".r.findFirstMatchIn(text).map(_.group(1).toInt).getOrElse(sys.error(s"config: $k"))
    def num(k: String, d: Float) = s""""$k"\\s*:\\s*([0-9.eE+-]+)""".r.findFirstMatchIn(text).map(_.group(1).toFloat).getOrElse(d)
    Qwen3Config(int("hidden_size"), int("num_hidden_layers"), int("num_attention_heads"), int("num_key_value_heads"),
      int("head_dim"), int("intermediate_size"), int("vocab_size"), num("rope_theta", 1000000f), num("rms_norm_eps", 1e-6f))

final class Qwen3Forward(st: Safetensors, val config: Qwen3Config):
  import config.*
  final class Layer(i: Int):
    val p = s"model.layers.$i."
    val inNorm = st.floats(p + "input_layernorm.weight"); val postNorm = st.floats(p + "post_attention_layernorm.weight")
    val q = st.linear(p + "self_attn.q_proj"); val k = st.linear(p + "self_attn.k_proj")
    val v = st.linear(p + "self_attn.v_proj"); val o = st.linear(p + "self_attn.o_proj")
    val qNorm = st.floats(p + "self_attn.q_norm.weight"); val kNorm = st.floats(p + "self_attn.k_norm.weight")
    val gate = st.linear(p + "mlp.gate_proj"); val up = st.linear(p + "mlp.up_proj"); val down = st.linear(p + "mlp.down_proj")
  val blocks: Vector[Layer] = Vector.tabulate(layers)(Layer(_))
  val embed: QLinear = st.linear("model.embed_tokens")
  val norm: Array[Float] = st.floats("model.norm.weight")

  private def rmsNorm(x: Array[Float], w: Array[Float]): Array[Float] =
    var ss = 0f; var i = 0
    while i < x.length do { ss += x(i) * x(i); i += 1 }
    val r = (1.0 / math.sqrt(ss / x.length + eps)).toFloat
    Array.tabulate(x.length)(k => x(k) * r * w(k))
  /** RoPE on one head's vector, the halves rotated (as mlx_lm's traditional = false) */
  private def rope(x: Array[Float], off: Int, pos: Int): Unit =
    val half = headDim / 2
    var i = 0
    while i < half do
      val freq = math.pow(ropeTheta, -2.0 * i / headDim)
      val a = pos * freq
      val c = math.cos(a).toFloat; val s = math.sin(a).toFloat
      val x1 = x(off + i); val x2 = x(off + half + i)
      x(off + i) = x1 * c - x2 * s
      x(off + half + i) = x2 * c + x1 * s
      i += 1

  /** the residual after each layer in `want` at every position, and the final-normed last state */
  def run(ids: Array[Int], want: Set[Int]): (Map[Int, Vector[Array[Float]]], Array[Float]) =
    val n = ids.length
    var h: Vector[Array[Float]] = ids.toVector.map(t => Array.tabulate(hidden)(i => embed.at(t, i)))
    val got = scala.collection.mutable.Map.empty[Int, Vector[Array[Float]]]
    val scale = (1.0 / math.sqrt(headDim)).toFloat
    for (l, li) <- blocks.zipWithIndex do
      val xs = h.map(rmsNorm(_, l.inNorm))
      val qs = xs.map(l.q.apply); val ks = xs.map(l.k.apply); val vs = xs.map(l.v.apply)
      // q and k normed per head, then RoPE by position
      for t <- 0 until n do
        for hd <- 0 until heads do
          val nq = rmsNorm(qs(t).slice(hd * headDim, (hd + 1) * headDim), l.qNorm)
          nq.copyToArray(qs(t), hd * headDim); rope(qs(t), hd * headDim, t)
        for hd <- 0 until kvHeads do
          val nk = rmsNorm(ks(t).slice(hd * headDim, (hd + 1) * headDim), l.kNorm)
          nk.copyToArray(ks(t), hd * headDim); rope(ks(t), hd * headDim, t)
      // causal attention, GQA: q head hd reads kv head hd / (heads / kvHeads)
      val att = Vector.tabulate(n) { t =>
        val out = new Array[Float](heads * headDim)
        for hd <- 0 until heads do
          val kv = hd / (heads / kvHeads)
          val scores = Array.tabulate(t + 1) { j =>
            var s = 0f; var i = 0
            while i < headDim do { s += qs(t)(hd * headDim + i) * ks(j)(kv * headDim + i); i += 1 }
            s * scale
          }
          val m = scores.max; val e = scores.map(s => math.exp((s - m).toDouble).toFloat); val z = e.sum
          for j <- 0 to t do
            val w = e(j) / z; var i = 0
            while i < headDim do { out(hd * headDim + i) += w * vs(j)(kv * headDim + i); i += 1 }
        out
      }
      h = h.zip(att).map((x, a) => { val o = l.o(a); Array.tabulate(hidden)(i => x(i) + o(i)) })
      val ys = h.map(rmsNorm(_, l.postNorm))
      h = h.zip(ys).map { (x, y) =>
        val g = l.gate(y); val u = l.up(y)
        val act = Array.tabulate(g.length)(i => g(i) / (1f + math.exp(-g(i)).toFloat) * u(i))
        val d = l.down(act)
        Array.tabulate(hidden)(i => x(i) + d(i))
      }
      if want(li + 1) then got(li + 1) = h
    (got.toMap, rmsNorm(h.last, norm))

  /** the logits of the next token: the final-normed last state against every embedding row (tied head) */
  def logits(normedLast: Array[Float]): Array[Float] =
    val out = new Array[Float](vocab)
    var t = 0
    while t < vocab do
      var s = 0f; var i = 0
      while i < hidden do { s += embed.at(t, i) * normedLast(i); i += 1 }
      out(t) = s; t += 1
    out
