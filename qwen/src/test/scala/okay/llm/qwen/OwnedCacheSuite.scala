package okay.llm.qwen

import okay.llm.compute.{Compute, OnCpu, QLinear}
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

/** Tiny real forwards verify cache ownership without downloading a model or skipping the laws. */
class OwnedCacheSuite extends munit.FunSuite:
  final class Watched extends Compute:
    val cpu = OnCpu()
    type T = cpu.T
    type W = cpu.W
    export cpu.{api, tensor, quantized, linear, rmsNorm, headNormRope, attention, concat, add, swiglu, slice}
    private val released = scala.collection.mutable.Set.empty[T]
    def release(t: T): Unit =
      assert(!released(t), "a tensor was released twice")
      released += t
    def rows(t: T, at: Seq[Int]): Future[Vector[Array[Float]]] =
      assert(at.nonEmpty, "an empty state request must not cause a device readback")
      assert(!released(t), "readback from a released tensor")
      cpu.rows(t, at)

  private val config = Qwen3Config(64, 2, 1, 1, 64, 64, 8, 1000000f, 1e-6f)
  private val weights = new Tensors:
    def has(name: String): Boolean = !name.startsWith("lm_head.")
    def floats(name: String): Array[Float] = Array.fill(64)(1f)
    def words(name: String): Array[Int] = throw UnsupportedOperationException(name)
    def shape(name: String): Vector[Int] = throw UnsupportedOperationException(name)
    override def linear(prefix: String): QLinear =
      val out = if prefix == "model.embed_tokens" then 8 else 64
      val packed = Array.fill(out * 8)(0)
      for row <- 0 until out do packed(row * 8 + row / 8) = 1 << ((row % 8) * 4)
      QLinear(out, 64, packed, Array.fill(out)(1f), Array.fill(out)(0f))

  test("consuming a prefix frees it once and produces the same successor as the whole forward"):
    val c = Watched()
    val model = Qwen3Compute(c, weights, config)
    val ids = Array(0, 1, 2, 3)
    for
      (whole, gold, _) <- model.run(ids, 2, Seq(0, 3), true, None, false)
      (_, _, prefix) <- model.run(ids.take(2), 2, Nil, true, None, true)
      (got, logits, next) <- model.run(ids, 2, Seq(0, 3), true, prefix, true, consumeCache = true)
      _ =
        assert(gold.zip(logits).forall((a, b) => math.abs(a - b) < 1e-4f))
        assert(whole.zip(got).forall((a, b) => a.zip(b).forall((x, y) => math.abs(x - y) < 1e-4f)))
        prefix.get.free()
        prefix.get.free()
        intercept[IllegalArgumentException](model.run(ids, 2, Nil, true, prefix, false))
      (cached, _, _) <- model.run(ids, 2, Seq(0, 3), false, next, false)
    yield
      assert(got.zip(cached).forall((a, b) => a.sameElements(b)))
      next.get.free()
      next.get.free()

  test("the original non-consuming overload keeps a prefix available for multiple branches"):
    val model = Qwen3Compute(Watched(), weights, config)
    for
      (_, _, prefix) <- model.run(Array(0, 1), 2, Nil, true, None, true)
      (_, first, _) <- model.run(Array(0, 1, 2), 2, Nil, true, prefix, false)
      (_, second, _) <- model.run(Array(0, 1, 2), 2, Nil, true, prefix, false)
    yield
      assert(first.sameElements(second))
      prefix.get.free()
