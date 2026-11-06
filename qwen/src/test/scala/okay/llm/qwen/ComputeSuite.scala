package okay.llm.qwen

import okay.llm.compute.OnCpu

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Await
import scala.concurrent.duration.*

/** THE LAW OF THE PROGRAM OVER `Compute`: on the CPU handler it says what `Qwen3Forward` says — the states after a
 * layer and the logits. On a model from `$QWEN_DIR` (config.json, model.safetensors); skipped without one. */
class ComputeSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  test("Qwen3Compute on OnCpu == Qwen3Forward") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    val config = Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json")))
    val ids = Array(151644, 872, 198, 4340, 1657, 6978, 614, 3146, 284, 9625, 30, 151645, 198, 151644, 77091, 198)
    val layer = math.min(config.layers, 19)
    val plain = Qwen3Forward(st, config)
    val (states, last) = plain.run(ids, Set(layer))
    val logits = plain.logits(last)
    val over = Qwen3Compute(OnCpu(), st, config)
    val (got, gotLogits) = Await.result(over.forward(ids, layer, Seq(0, ids.length - 1), withLogits = true), 10.minutes)
    val diff = Seq(0, ids.length - 1).zip(got).map((i, g) => states(layer)(i).zip(g).map((a, b) => math.abs(a - b)).max)
    val ldiff = logits.zip(gotLogits).map((a, b) => math.abs(a - b)).max
    println(s"states' max abs diff ${diff.mkString(" ")}; logits' $ldiff; next ${logits.indexOf(logits.max)} vs ${gotLogits.indexOf(gotLogits.max)}")
    assert(diff.forall(_ < 1e-2f), diff); assert(ldiff < 1e-2f, ldiff)
    assertEquals(logits.indexOf(logits.max), gotLogits.indexOf(gotLogits.max))
  }

/** THE CACHE'S LAW: a prefix run and kept, then the rest after it — the same states and logits as the whole run at
 * once, exactly, on the CPU handler */
class CacheSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  test("a cached prefix then the rest == the whole at once") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    val config = Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json")))
    val ids = Array(151644, 872, 198, 4340, 1657, 6978, 614, 3146, 284, 9625, 30, 151645, 198, 151644, 77091, 198)
    val layer = math.min(config.layers, 19)
    val m = Qwen3Compute(OnCpu(), st, config)
    val at = Seq(0, 5, ids.length - 1)
    val (whole, logits) = Await.result(m.forward(ids, layer, at, withLogits = true), 10.minutes)
    val (_, _, cache) = Await.result(m.run(ids.take(12), layer, Nil, withLogits = false, None, keep = true), 10.minutes)
    val (got, gotLogits, _) = Await.result(m.run(ids, layer, at, withLogits = true, cache, keep = false), 10.minutes)
    assertEquals(got.size, whole.size)
    val diff = whole.zip(got).map((a, b) => a.zip(b).map((x, y) => math.abs(x - y)).max)
    val ldiff = logits.zip(gotLogits).map((x, y) => math.abs(x - y)).max
    println(s"cache: states' max abs diff ${diff.mkString(" ")}; logits' $ldiff")
    assert(diff.forall(_ < 1e-3f), diff); assert(ldiff < 1e-3f, ldiff)
    // nothing new after the cached whole: the states from the cache alone
    val (_, _, all) = Await.result(m.run(ids, layer, Nil, withLogits = false, None, keep = true), 10.minutes)
    val (same, _, _) = Await.result(m.run(ids, layer, at, withLogits = false, all, keep = false), 10.minutes)
    assert(whole.zip(same).forall((a, b) => a.sameElements(b)))
  }
