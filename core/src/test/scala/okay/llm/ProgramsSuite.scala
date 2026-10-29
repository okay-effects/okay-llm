package okay.llm

import okay.scala3.*
import okay.scala3.std.*

/** the app's programs under handlers of their own: no GPU, no model, no disk */
class ProgramsSuite extends munit.FunSuite:
  type App[X] = (Llm + Console + Clock + Log)[X]
  private def runApp[A](p: A ! App): (Vector[String], A) =
    val (_, (_, (lines, a))) = run(Log.collect()(Clock.virtual(0)(Console.collect(Llm.fake(_ => "Paris.")(p)))))
    (lines, a)

  test("ask prints the answer and its speed"):
    val (lines, f) = runApp(Programs.ask[App]("the capital of France?", 16))
    assertEquals(f.reply.content, "Paris.")
    assertEquals(lines.head, "Paris.")
    assert(lines(1).startsWith("-- prompt"), lines(1))

  test("bench prints a line per prompt size"):
    val (lines, _) = runApp(Programs.bench[App]("fake", "fn main() {}", Seq(100, 1000), 3, 32))
    assertEquals(lines.size, 2)
    assert(lines.forall(_.startsWith("fake\tprompt")), lines.toString)
