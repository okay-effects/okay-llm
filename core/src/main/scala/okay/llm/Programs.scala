package okay.llm

import okay.effects.json.Json

import okay.scala3.*
import okay.scala3.std.*


/** THE APP'S PROGRAMS: written with effects alone — what answers them is the handlers' */
object Programs:
  /** each of `xs` in order */
  def each[A, R[_]](xs: Seq[A])(f: A => Unit ! R): Unit ! R =
    xs.foldLeft(unit[R])((p, x) => p.andThen(f(x)))

  def user(text: String): Ask = Ask(Vector(Json.obj("role" -> Json.Str("user"), "content" -> Json.Str(text))))

  /** a question asked, the answer printed with its speed */
  def ask[R[_]: Has[Llm] : Has[Console]](question: String, maxTokens: Int): Finished ! R =
    for
      f <- Llm.complete(user(question).copy(maxTokens = maxTokens))
      _ <- Console.line(f.reply.content)
      _ <- Console.line(f"-- prompt ${f.timing.promptTokens} tok, ${f.timing.prefillPerSecond}%.0f tok/s; " +
        f"${f.timing.tokens} tok at ${f.timing.decodePerSecond}%.1f tok/s")
    yield f

  private def median(xs: Seq[Double]): Double = xs.sorted.apply(xs.size / 2)

  /** THE BENCHMARK: for each prompt size, `runs` generations of `maxTokens` — a different prompt each run, so no
   * cache holds one — their medians printed: prefill and decode tokens a second. The source text is cut to size */
  def bench[R[_]: Has[Llm] : Has[Console] : Has[Clock] : Has[Log]](
      label: String, source: String, sizes: Seq[Int], runs: Int, maxTokens: Int): Unit ! R =
    def prompt(chars: Int, run: Int): String =
      val body = (source * (chars / source.length + 1)).take(chars)
      s"[run $run]\nHere is some Rust code:\n```rust\n$body\n```\nExplain what this code does, in detail."
    each(sizes)(chars =>
      for
        t0 <- Clock.now
        fs <- (1 to runs).foldLeft(pure[Vector[Finished], R](Vector.empty))((acc, r) =>
          acc.flatMap(v => Llm.complete(user(prompt(chars, r)).copy(maxTokens = maxTokens)).map(v :+ _)))
        t1 <- Clock.now
        _ <- Log.debug(s"$chars chars: $runs runs in ${t1 - t0} ms")
        ts = fs.map(_.timing)
        prefill = median(ts.map(_.prefillPerSecond))
        decode = median(ts.map(_.decodePerSecond))
        _ <- Console.line(f"$label\tprompt ${ts.head.promptTokens}%6d\tprefill $prefill%7.0f tok/s\t" +
          f"decode $decode%6.1f tok/s\tgen ${ts.head.tokens}")
      yield ())
