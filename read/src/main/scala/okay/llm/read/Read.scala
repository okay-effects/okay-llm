package okay.llm.read

import okay.scala3.*
import okay.scala3.std.Async

import scala.concurrent.Future

/**
 * WHAT A SKILL ASKS OF A MODEL, AS AN EFFECT (okay-desk specs/data.md §15, step 4; here for every app since
 * 2026-11-16): the residual after the skill's layer at
 * positions of a token sequence, the next token's logits (ask), the tokenizer, the chat prompt — a skill is a
 * program over `Read`, data; a handler answers it: in place on the JVM (`Read.sync`: mlx-scala's `Reading`), by
 * futures in the browser (`Read.async`: a `Qwen3Compute` on WebGPU or WebGL2, the program waiting through `Async`,
 * its continuation a value of the machine — no replay). A skill never sees weights.
 */
enum Read[X]:
  case States(ids: Array[Int], at: Vector[Int]) extends Read[Vector[Array[Float]]]
  case Logits(ids: Array[Int]) extends Read[Array[Float]]
  case Encode(text: String) extends Read[Array[Int]]
  case Decode(ids: Seq[Int]) extends Read[String]
  case Prompt(system: String, user: String) extends Read[Array[Int]]
  /** a step said: what the skill decided, or is deciding now — the page shows it while the question is read */
  case Note(words: String) extends Read[Unit]
  /** THE OPTIONS WEIGHED, ALL AT ONCE (okay-desk specs/frames.md §2): the mean log-probability of each option's tokens
   * after `context` — one operation for a choice's every option, so a handler may weigh them together (the weights
   * read once for all: decoding is bound by memory, not by arithmetic) */
  case Score(context: Array[Int], options: Vector[Array[Int]]) extends Read[Vector[Double]]

object Read:
  /** the residual after the skill's layer at positions `at` of `ids`: one vector each */
  def states[R[_]](ids: Array[Int], at: Vector[Int])(using In[Read, R]): Vector[Array[Float]] ! R =
    perform[Read](States(ids, at))
  /** the logits of the next token after `ids` */
  def logits[R[_]](ids: Array[Int])(using In[Read, R]): Array[Float] ! R = perform[Read](Logits(ids))
  def encode[R[_]](text: String)(using In[Read, R]): Array[Int] ! R = perform[Read](Encode(text))
  def decode[R[_]](ids: Seq[Int])(using In[Read, R]): String ! R = perform[Read](Decode(ids))
  /** the chat prompt for a system text and a user text, thinking off */
  def prompt[R[_]](system: String, user: String)(using In[Read, R]): Array[Int] ! R =
    perform[Read](Prompt(system, user))
  /** a step said (shown on the page; nothing on the JVM) */
  def note[R[_]](words: String)(using In[Read, R]): Unit ! R = perform[Read](Note(words))
  /** the options weighed after the context: each the mean log-probability of its tokens */
  def score[R[_]](context: Array[Int], options: Vector[Array[Int]])(using In[Read, R]): Vector[Double] ! R =
    perform[Read](Score(context, options))

  /** the options weighed by next-token logits alone, one forward a token: for a backend with no checkpoint of its
   * state (correct, slow) */
  def scoreByLogits(context: Array[Int], options: Vector[Array[Int]], logits: Array[Int] => Array[Float])
      : Vector[Double] =
    options.map { o =>
      o.indices.map { j =>
        val l = logits(context ++ o.take(j)); val m = l.max
        val z = l.iterator.map(x => math.exp((x - m).toDouble)).sum
        (l(o(j)) - m).toDouble - math.log(z)
      }.sum / math.max(1, o.length)
    }

  /** the programs of `f` over `xs`, one after another, their values */
  def each[A, B, R[_]](xs: Vector[A])(f: A => B ! R): Vector[B] ! R =
    xs.foldLeft(pure[Vector[B], R](Vector.empty))((acc, a) => acc.flatMap(v => f(a).map(v :+ _)))

  /** answered in place by a `Reading` — the JVM's mlx-scala */
  def sync[R[_], A](r: Reading): Handler[Read, R, A, A] =
    Handler.tail[Read, R, A, A](pure)([X] => (op: Read[X]) => r.answer(op))
  /** a program of `Read` alone, run on a `Reading` */
  def run[A](p: A ! Read, r: Reading): A = okay.scala3.run(handle(p, sync[Pure, A](r)))

  /** answered by futures — the browser's device: the program waits on each through `Async` */
  def async[R[_], A](f: [X] => Read[X] => Future[X])(using In[Async, R]): Handler[Read, R, A, A] =
    Handler.general[Read, R, A, A](pure)([X] => (op: Read[X], k: X => A ! R) => Async.fromFuture(f(op)).flatMap(k))
  /** a program of `Read` over `Async`, driven to a future */
  def future[A](p: A ! (Async + Read), f: [X] => Read[X] => Future[X])(using scala.concurrent.ExecutionContext)
      : Future[A] = Async.toFuture(handle(p, async[Async, A](f)))
