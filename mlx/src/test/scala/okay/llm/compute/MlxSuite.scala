package okay.llm.compute

import okay.llm.qwen.{Qwen3Compute, Qwen3Config, Safetensors}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Await
import scala.concurrent.duration.*

/** THE LAW OF THE FOURTH HANDLER: `Qwen3Compute` on MLX says what it says on the CPU — the states after a layer at
 * several positions and the logits, and the cached prefix then the rest. On `$QWEN_DIR`; skipped without one. */
class MlxSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
  val ids = Array(151644, 872, 198, 4340, 1657, 6978, 614, 3146, 284, 9625, 30, 151645, 198, 151644, 77091, 198)
  def load() =
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    (st, Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json"))))

  test("Qwen3Compute on OnMlx == on OnCpu: states and logits") {
    assume(dir.isDefined, "QWEN_DIR not set")
    val (st, config) = load(); val layer = math.min(config.layers, 19); val at = Seq(0, 5, ids.length - 1)
    val (cpuStates, cpuLogits) = Await.result(Qwen3Compute(OnCpu(), st, config).forward(ids, layer, at, withLogits = true), 10.minutes)
    val mlx = OnMlx()
    val (gpuStates, gpuLogits) = Await.result(Qwen3Compute(mlx, st, config).forward(ids, layer, at, withLogits = true), 10.minutes)
    val diff = cpuStates.zip(gpuStates).map((a, b) => a.zip(b).map((x, y) => math.abs(x - y)).max)
    val scale = cpuStates.map(_.map(math.abs).max)
    val ldiff = cpuLogits.zip(gpuLogits).map((a, b) => math.abs(a - b)).max
    println(s"mlx vs cpu: states' max abs diff ${diff.mkString(" ")} (scale ${scale.mkString(" ")}); logits' $ldiff; " +
      s"next ${cpuLogits.indexOf(cpuLogits.max)} vs ${gpuLogits.indexOf(gpuLogits.max)}")
    assert(diff.zip(scale).forall((d, s) => d < 1e-2f * s), diff)
    assertEquals(cpuLogits.indexOf(cpuLogits.max), gpuLogits.indexOf(gpuLogits.max))
    mlx.close()
  }

  test("on OnMlx a cached prefix then the rest == the whole at once") {
    assume(dir.isDefined, "QWEN_DIR not set")
    val (st, config) = load(); val layer = math.min(config.layers, 19); val at = Seq(0, 5, ids.length - 1)
    val mlx = OnMlx(); val m = Qwen3Compute(mlx, st, config)
    val (whole, logits) = Await.result(m.forward(ids, layer, at, withLogits = true), 10.minutes)
    val n = 10
    val (_, _, cache) = Await.result(m.run(ids.take(n), layer, Nil, withLogits = false, None, keep = true), 10.minutes)
    val (rest, restLogits, _) = Await.result(m.run(ids, layer, at, withLogits = true, cache, keep = false), 10.minutes)
    val diff = whole.zip(rest).map((a, b) => a.zip(b).map((x, y) => math.abs(x - y)).max)
    println(s"cached vs whole on mlx: ${diff.mkString(" ")}; logits ${logits.zip(restLogits).map((a, b) => math.abs(a - b)).max}")
    assert(diff.forall(_ < 1e-2f), diff)
    mlx.close()
  }
