package okay.llm.qwen

/** THE ARCHITECTURE AS DATA, what config.json says of it: the sizes, the rope base, the norm's epsilon. What the FILE
 * says decides the rest (`Qwen3Compute`): a layer's attention or MLP present or not, the head tied to the embedding
 * or its own `lm_head` */
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
