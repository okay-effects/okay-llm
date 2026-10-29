package okay.llm

import java.nio.file.{Files as NioFiles, Path}
import scala.collection.mutable
import okay.scala3.*

/** what is asked of a model: a conversation (messages as a chat template takes them: role, content, tool calls), the
 * tools it may call, how many tokens at most, whether it thinks first */
final case class Ask(messages: Vector[Json], tools: Vector[Json] = Vector.empty, maxTokens: Int = 1024,
    thinking: Boolean = false)

object Ask:
  /** system messages wherever they are (a client may send one mid-conversation: Claude Code does) put together, in
   * order, as the first message — the only place most chat templates take one (Qwen3.6's raises otherwise) */
  def systemFirst(messages: Vector[Json]): Vector[Json] =
    val (system, rest) = messages.partition(_("role") == Json.Str("system"))
    if system.isEmpty || (system.length == 1 && messages.head("role") == Json.Str("system")) then messages
    else
      val text = system.map(_("content")).collect { case Json.Str(t) => t }.filter(_.nonEmpty).mkString("\n\n")
      Json.obj("role" -> Json.Str("system"), "content" -> Json.Str(text)) +: rest

/** a generation going on */
final case class Gen(id: Long)

/** what a generation measured: the prompt's tokens and the time to the first new one (the prefill), the tokens made
 * and their time after the first (the decode) */
final case class Timing(promptTokens: Int, prefillNanos: Long, tokens: Int, decodeNanos: Long):
  def prefillPerSecond: Double = promptTokens / (prefillNanos / 1e9)
  def decodePerSecond: Double = if tokens <= 1 then 0 else (tokens - 1) / (decodeNanos / 1e9)

/** a generation done: its turn read, what it measured, whether the model ended it (else: the limit) */
final case class Finished(reply: Reply, timing: Timing, byStop: Boolean)

/** THE LANGUAGE MODEL, as a program sees it: its name, a prompt's size, a generation started, its text piece by
 * piece, its end. Which model, on which hardware, is the handler's — a module of its own (okay-mlx: Apple's MLX) */
enum Llm[X]:
  case Name() extends Llm[String]
  case Count(ask: Ask) extends Llm[Int]
  case Start(ask: Ask) extends Llm[Gen]
  case Next(gen: Gen) extends Llm[Option[String]]
  case Finish(gen: Gen) extends Llm[Finished]

object Llm:
  def name[R[_]](using In[Llm, R]): String ! R = perform[Llm](Name())
  def count[R[_]](ask: Ask)(using In[Llm, R]): Int ! R = perform[Llm](Count(ask))
  def start[R[_]](ask: Ask)(using In[Llm, R]): Gen ! R = perform[Llm](Start(ask))
  /** the next piece of text, or none: the generation is over */
  def next[R[_]](gen: Gen)(using In[Llm, R]): Option[String] ! R = perform[Llm](Next(gen))
  def finish[R[_]](gen: Gen)(using In[Llm, R]): Finished ! R = perform[Llm](Finish(gen))

  /** a generation to its end */
  def complete[R[_]](ask: Ask)(using In[Llm, R]): Finished ! R =
    for
      g <- start(ask)
      _ <- loop[Unit, Unit, R](())(_ => next(g).map(p => if p.isDefined then Left(()) else Right(())))
      f <- finish(g)
    yield f

  /** A MODEL OF YOUR OWN: what a hardware module implements — a prompt's size, a generation as an iterator of text
   * pieces — and the handler made of it */
  trait Backend:
    def name: String
    def count(ask: Ask): Int
    def start(ask: Ask): Generation
  /** a generation going on: its pieces; at its end, its whole text and what it measured */
  trait Generation:
    def next(): Option[String]
    def text: String
    def timing: Timing
    def byStop: Boolean
    def close(): Unit

  def on[R[_], Rest[_], A](backend: Backend)(p: A ! R)(using Split[Llm, R, Rest]): A ! Rest =
    val running = mutable.HashMap.empty[Long, (Generation, Ask)]
    var ids = 0L
    handle(p, Handler.tail[Llm, Rest, A, A](pure)([X] => (op: Llm[X]) => op match
      case Name() => backend.name
      case Count(ask) => backend.count(ask)
      case Start(ask) =>
        ids += 1
        running(ids) = (backend.start(ask), ask)
        Gen(ids)
      case Next(gen) => running(gen.id)._1.next()
      case Finish(gen) =>
        val (g, ask) = running.remove(gen.id).get
        g.close()
        Finished(Reply.parse(g.text, ask.thinking, ask.tools), g.timing, g.byStop)))

  /** answers given by a function of the last message's text, piece by piece (by word), timings made up: the
   * programs without a model */
  def fake[R[_], Rest[_], A](answer: String => String)(p: A ! R)(using Split[Llm, R, Rest]): A ! Rest =
    on(Fake(answer))(p)

  private final class Fake(answer: String => String) extends Backend:
    def name = "fake"
    private def lastText(ask: Ask) =
      ask.messages.lastOption.map(_("content")).collect { case Json.Str(s) => s }.getOrElse("")
    def count(ask: Ask) = ask.messages.map(m => m("content") match { case Json.Str(s) => s.length / 4 + 4; case _ => 4 })
      .sum
    def start(ask: Ask) = new Generation:
      private var rest = words(answer(lastText(ask)))
      private val sb = new StringBuilder
      def next() = rest match
        case piece :: more => rest = more; sb.append(piece); Some(piece)
        case Nil => None
      def text = sb.toString
      def timing =
        val n = sb.length / 4 + 1
        Timing(10, 1_000_000L, n, n * 10_000_000L)
      def byStop = true
      def close() = ()
    /** the text in pieces a word each, its space kept after it (no look-behind: Scala Native's regex is RE2's) */
    private def words(s: String): List[String] =
      val out = List.newBuilder[String]
      var from = 0
      for i <- s.indices if s.charAt(i) == ' ' do { out += s.substring(from, i + 1); from = i + 1 }
      if from < s.length then out += s.substring(from)
      out.result()

/** FILES: a text read */
enum Files[X]:
  case Read(path: Path) extends Files[String]

object Files:
  def read[R[_]](path: Path)(using In[Files, R]): String ! R = perform[Files](Read(path))

  def disk[R[_], Rest[_], A](p: A ! R)(using Split[Files, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Files, Rest, A, A](pure)([X] => (op: Files[X]) => op match
      case Read(path) => NioFiles.readString(path)))

  /** files given: a map of paths to texts */
  def of[R[_], Rest[_], A](files: Map[Path, String])(p: A ! R)(using Split[Files, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Files, Rest, A, A](pure)([X] => (op: Files[X]) => op match
      case Read(path) => files.getOrElse(path, throw java.io.FileNotFoundException(path.toString))))

/** THE CONSOLE: a line out */
enum Console[X]:
  case Line(text: String) extends Console[Unit]

object Console:
  def line[R[_]](text: String)(using In[Console, R]): Unit ! R = perform[Console](Line(text))

  def stdout[R[_], Rest[_], A](p: A ! R)(using Split[Console, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Console, Rest, A, A](pure)([X] => (op: Console[X]) => op match
      case Line(text) => println(text)))

  /** the lines, in order, and the value */
  def collect[R[_], Rest[_], A](p: A ! R)(using Split[Console, R, Rest]): (Vector[String], A) ! Rest =
    handle(p, Handler.stateful[Console, Rest, A, (Vector[String], A), Vector[String]](Vector.empty)(
      (lines, a) => pure((lines, a)))(
      [X] => (op: Console[X], cell: Cell[Vector[String]]) => op match
        case Line(text) => cell.set(cell.get :+ text)))

/** AN HTTP REQUEST, read */
final case class Request(method: String, path: String, headers: Map[String, String], body: String)

/** AN HTTP RESPONSE, as the program makes it: whole, or streamed — opened, sent piece by piece, closed */
enum Http[X]:
  case Respond(status: Int, contentType: String, body: String) extends Http[Unit]
  case Open(status: Int, contentType: String) extends Http[Unit]
  case Send(chunk: String) extends Http[Unit]
  case Close() extends Http[Unit]

object Http:
  def respond[R[_]](status: Int, contentType: String, body: String)(using In[Http, R]): Unit ! R =
    perform[Http](Respond(status, contentType, body))
  def json[R[_]](status: Int, body: Json)(using In[Http, R]): Unit ! R =
    respond(status, "application/json", body.render)
  def open[R[_]](status: Int, contentType: String)(using In[Http, R]): Unit ! R =
    perform[Http](Open(status, contentType))
  def send[R[_]](chunk: String)(using In[Http, R]): Unit ! R = perform[Http](Send(chunk))
  def close[R[_]](using In[Http, R]): Unit ! R = perform[Http](Close())

  private def reason(status: Int) = status match
    case 200 => "OK"; case 400 => "Bad Request"; case 404 => "Not Found"; case 500 => "Internal Server Error"
    case _ => "Status"

  /** to a socket's stream: a whole response with its length; a streamed one closed by closing the connection */
  def socket[R[_], Rest[_], A](out: java.io.OutputStream)(p: A ! R)(using Split[Http, R, Rest]): A ! Rest =
    def head(status: Int, contentType: String, extra: String) =
      out.write((s"HTTP/1.1 $status ${reason(status)}\r\nContent-Type: $contentType\r\n$extra" +
        "Connection: close\r\n\r\n").getBytes("UTF-8"))
    handle(p, Handler.tail[Http, Rest, A, A](pure)([X] => (op: Http[X]) => op match
      case Respond(status, contentType, body) =>
        val bytes = body.getBytes("UTF-8")
        head(status, contentType, s"Content-Length: ${bytes.length}\r\n")
        out.write(bytes)
        out.flush()
      case Open(status, contentType) =>
        head(status, contentType, "Cache-Control: no-cache\r\n")
        out.flush()
      case Send(chunk) =>
        out.write(chunk.getBytes("UTF-8"))
        out.flush()
      case Close() => out.flush()))

  /** what was sent: the status, the content type, the body (pieces joined) */
  def collect[R[_], Rest[_], A](p: A ! R)(using Split[Http, R, Rest]): ((Int, String, String), A) ! Rest =
    handle(p, Handler.stateful[Http, Rest, A, ((Int, String, String), A), (Int, String, String)]((0, "", ""))(
      (r, a) => pure((r, a)))(
      [X] => (op: Http[X], cell: Cell[(Int, String, String)]) => op match
        case Respond(status, contentType, body) => cell.set((status, contentType, body))
        case Open(status, contentType) => cell.set((status, contentType, ""))
        case Send(chunk) => val (s, c, b) = cell.get; cell.set((s, c, b + chunk))
        case Close() => ()))
