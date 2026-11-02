package okay.llm

import java.io.{BufferedInputStream, InputStream}
import java.net.{Socket, URI}
import java.nio.charset.StandardCharsets.UTF_8
import okay.effects.json.Json
import scala.collection.mutable

/** A MODEL BEHIND A SERVER that speaks OpenAI's chat completions — okay-mlx's gateway, LM Studio, Ollama, vLLM — as
 * an `Llm.Backend`: the request sent streamed, the text given piece by piece as it comes, the tool calls as the server
 * read them, the timings measured here (the server's counts when it gives them). `http://` alone: a gateway on
 * this machine or the network. One connection a generation, closed at its end */
final class HttpBackend(base: String, val name: String) extends Llm.Backend:
  private val uri = URI.create(base.stripSuffix("/"))
  require(uri.getScheme == "http", s"http:// alone: $base")
  private val host = uri.getHost
  private val port = if uri.getPort > 0 then uri.getPort else 80
  private val prefix = Option(uri.getPath).getOrElse("").stripSuffix("/")

  /** a guess, not the model's tokenizer: a quarter of the text's characters (the server's count comes with a turn) */
  def count(ask: Ask): Int = (ask.messages.map(_.render.length).sum + ask.tools.map(_.render.length).sum) / 4

  def start(ask: Ask): Llm.Generation = HttpBackend.Streamed(host, port, s"$prefix/chat/completions",
    HttpBackend.body(ask, ask.model.getOrElse(name)))

object HttpBackend:
  private def s(t: String) = Json.Str(t)

  /** an `Ask` as OpenAI's request: its messages (a tool call's arguments as text, as the API has them), tools,
   * sampling, thinking (`chat_template_kwargs`), streamed with its usage */
  def body(ask: Ask, model: String): Json =
    val messages = ask.messages.map {
      case o: Json.Obj => Json.Obj(o.fields.map {
        case ("tool_calls", calls) => "tool_calls" -> Json.Arr(calls.list.zipWithIndex.map { (c, k) =>
          val f = c("function")
          val args = f.field("arguments").map { case Json.Str(t) => t; case a => a.render }.getOrElse("{}")
          Json.obj("id" -> c.field("id").getOrElse(s(s"call_$k")), "type" -> s("function"),
            "function" -> Json.obj("name" -> f("name"), "arguments" -> s(args)))
        })
        case other => other
      })
      case other => other
    }
    val sampling = ask.sampling
    Json.Obj(Vector("model" -> s(model), "messages" -> Json.Arr(messages), "max_tokens" -> Json.num(ask.maxTokens),
      "stream" -> Json.Bool(true), "stream_options" -> Json.obj("include_usage" -> Json.Bool(true)),
      "chat_template_kwargs" -> Json.obj("enable_thinking" -> Json.Bool(ask.thinking))) ++
      (if ask.tools.isEmpty then Vector.empty else Vector("tools" -> Json.Arr(ask.tools))) ++
      sampling.temperature.map("temperature" -> Json.num(_)) ++ sampling.topP.map("top_p" -> Json.num(_)) ++
      sampling.topK.map("top_k" -> Json.num(_)) ++ sampling.seed.map("seed" -> Json.num(_)) ++
      sampling.presencePenalty.map("presence_penalty" -> Json.num(_)) ++
      sampling.frequencyPenalty.map("frequency_penalty" -> Json.num(_)))

  /** a response's body as lines: by its length, chunked, or to the connection's close */
  private final class Lines(in: InputStream, chunked: Boolean):
    private var left = if chunked then 0L else Long.MaxValue
    private def byte(): Int =
      if chunked && left == 0 then
        val size = rawLine()
        if size == null then return -1
        val n = java.lang.Long.parseLong(size.trim.takeWhile(_ != ';'), 16)
        if n == 0 then return -1
        left = n
      val b = in.read()
      if b >= 0 then
        left -= 1
        if chunked && left == 0 then { in.read(); in.read() }   // the chunk's CRLF
      b
    private def rawLine(): String | Null =
      val buf = java.io.ByteArrayOutputStream()
      var b = in.read()
      if b < 0 then return null
      while b >= 0 && b != '\n' do { if b != '\r' then buf.write(b); b = in.read() }
      String(buf.toByteArray, UTF_8)
    def line(): String | Null =
      val buf = java.io.ByteArrayOutputStream()
      var b = byte()
      if b < 0 then return null
      while b >= 0 && b != '\n' do { if b != '\r' then buf.write(b); b = byte() }
      String(buf.toByteArray, UTF_8)

  /** a generation over one connection: the request written, the stream's `data:` lines read as they come */
  private final class Streamed(host: String, port: Int, path: String, request: Json) extends Llm.Generation:
    private val t0 = System.nanoTime()
    private var firstAt = 0L
    private var endAt = 0L
    private val socket = Socket(host, port)
    private val text0 = new StringBuilder
    private val reasoning = new StringBuilder
    private val calls = mutable.TreeMap.empty[Int, (String, StringBuilder)]
    private var finish = ""
    private var promptTokens = 0
    private var completionTokens = -1
    private var cachedTokens = 0
    private var pieces = 0
    private var done = false
    private val lines =
      val payload = request.render.getBytes(UTF_8)
      val out = socket.getOutputStream
      out.write((s"POST $path HTTP/1.1\r\nHost: $host:$port\r\nContent-Type: application/json\r\n" +
        s"Accept: text/event-stream\r\nContent-Length: ${payload.length}\r\nConnection: close\r\n\r\n").getBytes(UTF_8))
      out.write(payload)
      out.flush()
      val in = BufferedInputStream(socket.getInputStream)
      val head = Lines(in, chunked = false)
      val status = Option(head.line()).getOrElse("")
      var headers = Map.empty[String, String]
      var h = head.line()
      while h != null && h.nonEmpty do
        val i = h.indexOf(':')
        if i > 0 then headers += h.take(i).trim.toLowerCase -> h.drop(i + 1).trim
        h = head.line()
      val lines = Lines(in, headers.get("transfer-encoding").exists(_.toLowerCase.contains("chunked")))
      if !status.split(' ').lift(1).exists(_.startsWith("2")) then
        val body = Iterator.continually(lines.line()).takeWhile(_ != null).mkString("\n")
        socket.close()
        throw RuntimeException(s"$status: ${body.take(500)}")
      lines

    /** the next `data:` line's delta read in; its text, if any */
    private def step(): Option[Option[String]] =
      val l = lines.line()
      if l == null then None
      else if !l.startsWith("data:") then Some(None)
      else
        val data = l.drop(5).trim
        if data == "[DONE]" then None
        else
          val j = Json.fromText(data)
          j.field("usage").filter(_ != Json.Null).foreach { u =>
            u.field("prompt_tokens").foreach(n => promptTokens = n.int)
            u.field("completion_tokens").foreach(n => completionTokens = n.int)
            u.field("prompt_tokens_details").flatMap(_.field("cached_tokens")).foreach(n => cachedTokens = n.int)
          }
          val choice = j.field("choices").map(_.list).getOrElse(Vector.empty).headOption
          choice.flatMap(_.field("finish_reason")).foreach { case Json.Str(r) => finish = r; case _ => () }
          val delta = choice.flatMap(_.field("delta"))
          delta.flatMap(_.field("reasoning_content")).foreach { case Json.Str(t) => reasoning ++= t; case _ => () }
          delta.flatMap(_.field("tool_calls")).foreach(_.list.foreach { c =>
            val k = c.field("index").map(_.int).getOrElse(calls.size)
            val f = c.field("function")
            val (n0, args) = calls.getOrElse(k, ("", new StringBuilder))
            val n = f.flatMap(_.field("name")).collect { case Json.Str(t) => t }.getOrElse(n0)
            f.flatMap(_.field("arguments")).foreach { case Json.Str(t) => args ++= t; case a => args ++= a.render }
            calls(k) = (n, args)
          })
          val t = delta.flatMap(_.field("content")).collect { case Json.Str(t) if t.nonEmpty => t }
          if firstAt == 0 && (t.nonEmpty || reasoning.nonEmpty || calls.nonEmpty) then firstAt = System.nanoTime()
          t.foreach { x => text0 ++= x; pieces += 1 }
          Some(t)

    def next(): Option[String] =
      if done then None
      else
        var got: Option[String] = None
        var more = true
        while more && got.isEmpty do
          step() match
            case None => more = false
            case Some(t) => got = t
        if got.isEmpty then
          done = true
          endAt = System.nanoTime()
        got

    def text: String = text0.toString
    def byStop: Boolean = finish == "stop" || finish == "tool_calls"
    def timing: Timing =
      val first = if firstAt == 0 then System.nanoTime() else firstAt
      val end = if endAt == 0 then System.nanoTime() else endAt
      Timing(promptTokens, first - t0, if completionTokens >= 0 then completionTokens else pieces, end - first,
        cachedTokens)
    def close(): Unit =
      done = true
      if endAt == 0 then endAt = System.nanoTime()
      socket.close()

    /** the turn as the server read it: its thinking, its text, its tool calls (arguments parsed); a server that gave
     * none but left them in the text, read from the text */
    override def reply(thinking: Boolean, tools: Seq[Json]): Reply =
      if calls.isEmpty && (text.contains("<tool_call>") || text.contains("<think>")) then
        Reply.parse(text, thinking, tools)
      else Reply(reasoning.toString.trim, text.trim, calls.values.toVector.map { (n, a) =>
        val raw = a.toString
        ToolCall(n, if raw.isEmpty then Json.obj() else try Json.fromText(raw) catch
          case _: IllegalArgumentException => Json.Str(raw))
      })
