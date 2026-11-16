package okay.llm.read

import okay.llm.compute.{OnCpu, QLinear}
import okay.llm.qwen.Tensors
import scala.concurrent.ExecutionContext.Implicits.global

class AdaptedSuite extends munit.FunSuite:
  private def q(out: Int, f: (Int, Int) => Int): QLinear =
    val words = Array.tabulate(out * 8) { at =>
      (0 until 8).foldLeft(0)((word, n) => word | (f(at / 8, (at % 8) * 8 + n) << (4 * n)))
    }
    QLinear(out, 64, words, Array.fill(out)(1f), Array.fill(out)(0f))
  private def tensors(values: Map[String, QLinear]): Tensors = new Tensors:
    def has(name: String): Boolean = values.keys.exists(p => name == p + ".weight")
    def floats(name: String): Array[Float] = throw IllegalArgumentException(name)
    def words(name: String): Array[Int] = throw IllegalArgumentException(name)
    def shape(name: String): Vector[Int] = throw IllegalArgumentException(name)
    override def linear(prefix: String): QLinear = values(prefix)

  test("separate low-rank factors are applied without changing the backbone or retaining source buffers"):
    val base = q(2, (_, _) => 1)
    val a = q(64, (row, _) => if row == 0 then 1 else 0)
    val b = q(2, (row, col) => if col == 0 then row + 2 else 0)
    val zeroA = q(64, (_, _) => 0)
    val zeroB = q(2, (_, _) => 0)
    val c = Adapted(OnCpu(), tensors(Map("test" -> base)), tensors(Map(
      "test.a" -> a, "test.ar" -> zeroA, "test.b" -> b, "test.br" -> zeroB)))
    val weight = c.quantized(c.tensors.linear("test"))
    c.releaseSources()
    intercept[IllegalStateException](c.tensors.has("test.weight"))
    val x = c.tensor(1, 64, Array.fill(64)(1f))
    val y = c.linear(x, weight)
    c.rows(y, Vector(0)).map { rows =>
      assertEquals(rows.head.toVector, Vector(192f, 256f))
      c.release(x); c.release(y); c.close()
    }
