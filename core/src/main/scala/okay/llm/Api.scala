package okay.llm



/** A TURN AS IT STREAMS: the model's text, piece by piece, as thinking, then text; a tool call held back (the whole
 * turn is read at its end). A tail that may begin a tag waits for the next piece */
final class Streamer(thinking: Boolean):
  import Streamer.*
  private var mode = if thinking then Mode.Thinking else Mode.Text
  private val pending = new StringBuilder
  private var started = false

  def push(piece: String): Vector[Part] =
    pending.append(piece)
    val out = Vector.newBuilder[Part]
    var more = true
    while more do
      mode match
        case Mode.Thinking =>
          val s = pending.toString
          val at = s.indexOf("</think>")
          if at >= 0 then
            emit(out, Part.Thinking(s.substring(0, at).replace("<think>", "")))
            pending.delete(0, at + "</think>".length)
            mode = Mode.Text
            started = false
          else
            val keep = partial(s, "</think>")
            emit(out, Part.Thinking(s.substring(0, s.length - keep).replace("<think>", "")))
            pending.delete(0, s.length - keep)
            more = false
        case Mode.Text =>
          val s = pending.toString
          val at = s.indexOf("<tool_call>")
          if at >= 0 then
            emit(out, Part.Text(s.substring(0, at)))
            pending.clear()
            mode = Mode.Tools
          else
            val keep = partial(s, "<tool_call>")
            emit(out, Part.Text(s.substring(0, s.length - keep)))
            pending.delete(0, s.length - keep)
          more = false
        case Mode.Tools =>
          pending.clear()
          more = false
    out.result()

  /** what was held back, at the end */
  def flush(): Vector[Part] =
    val out = Vector.newBuilder[Part]
    mode match
      case Mode.Thinking => emit(out, Part.Thinking(pending.toString))
      case Mode.Text => emit(out, Part.Text(pending.toString))
      case Mode.Tools =>
    pending.clear()
    out.result()

  /** a part's leading whitespace dropped until something else comes */
  private def emit(out: collection.mutable.Builder[Part, ?], p: Part): Unit =
    val t = if started then p.text else p.text.dropWhile(_.isWhitespace)
    if t.nonEmpty then
      started = true
      out += (p match { case Part.Thinking(_) => Part.Thinking(t); case Part.Text(_) => Part.Text(t) })

object Streamer:
  private enum Mode:
    case Thinking, Text, Tools
  enum Part:
    case Thinking(value: String)
    case Text(value: String)
    def text: String = this match { case Thinking(v) => v; case Text(v) => v }
  /** how long a tail of `s` begins `tag` */
  private def partial(s: String, tag: String): Int =
    (math.min(tag.length - 1, s.length) to 1 by -1).find(n => tag.startsWith(s.substring(s.length - n))).getOrElse(0)

/** content as text: a string, or parts' texts joined */
private def text(c: Json): String = c match
  case Json.Str(s) => s
  case Json.Arr(parts) => parts.flatMap(p => p.get("text").collect { case Json.Str(t) => t }).mkString
  case Json.Null => ""
  case other => other.render

private def str(s: String): Json = Json.Str(s)
private def num(n: Long): Json = Json.Num(n.toDouble)

/** OPENAI'S CHAT COMPLETIONS: a request as an `Ask`, a turn as a response or as chunks of a stream */
object OpenAi:
  def ask(body: Json): Ask =
    val messages = body("messages").items.map { m =>
      val fields = Vector.newBuilder[(String, Json)]
      fields += "role" -> m("role")
      fields += "content" -> str(text(m("content")))
      m.get("tool_calls").foreach(calls => fields += "tool_calls" -> Json.Arr(calls.items.map { c =>
        val f = c("function")
        val args = f("arguments") match
          case Json.Str(s) => try Json.parse(s) catch case _: IllegalArgumentException => Json.Str(s)
          case a => a
        Json.obj("type" -> str("function"), "function" -> Json.obj("name" -> f("name"), "arguments" -> args))
      }))
      m.get("reasoning_content").foreach(r => fields += "reasoning_content" -> r)
      Json.obj(fields.result()*)
    }
    val max = body.get("max_completion_tokens").orElse(body.get("max_tokens")).map(_.int).getOrElse(1024)
    val thinking = body.get("chat_template_kwargs").flatMap(_.get("enable_thinking")).exists(_ == Json.Bool(true)) ||
      body.get("reasoning_effort").isDefined
    Ask(Ask.systemFirst(messages), body.get("tools").map(_.items).getOrElse(Vector.empty), max, thinking)

  def finishReason(f: Finished): String =
    if f.reply.toolCalls.nonEmpty then "tool_calls" else if f.byStop then "stop" else "length"

  def toolCalls(id: String, calls: Vector[ToolCall]): Json = Json.Arr(calls.zipWithIndex.map((c, k) =>
    Json.obj("index" -> num(k), "id" -> str(s"call_${id}_$k"), "type" -> str("function"),
      "function" -> Json.obj("name" -> str(c.name), "arguments" -> str(c.arguments.render)))))

  def usage(f: Finished): Json =
    Json.obj("prompt_tokens" -> num(f.timing.promptTokens), "completion_tokens" -> num(f.timing.tokens),
      "total_tokens" -> num(f.timing.promptTokens + f.timing.tokens))

  def response(id: String, model: String, created: Long, f: Finished): Json =
    val message = Vector("role" -> str("assistant"), "content" -> str(f.reply.content)) ++
      (if f.reply.reasoning.nonEmpty then Vector("reasoning_content" -> str(f.reply.reasoning)) else Vector.empty) ++
      (if f.reply.toolCalls.nonEmpty then Vector("tool_calls" -> toolCalls(id, f.reply.toolCalls)) else Vector.empty)
    Json.obj("id" -> str(s"chatcmpl-$id"), "object" -> str("chat.completion"), "created" -> num(created),
      "model" -> str(model), "choices" -> Json.arr(Json.obj("index" -> num(0), "message" -> Json.obj(message*),
        "finish_reason" -> str(finishReason(f)))), "usage" -> usage(f))

  def chunk(id: String, model: String, created: Long, delta: Json, finish: Option[String] = None,
      usage: Option[Json] = None): String =
    val choices = if usage.isDefined then Json.arr() else Json.arr(Json.obj("index" -> num(0), "delta" -> delta,
      "finish_reason" -> finish.map(str).getOrElse(Json.Null)))
    val fields = Vector("id" -> str(s"chatcmpl-$id"), "object" -> str("chat.completion.chunk"),
      "created" -> num(created), "model" -> str(model), "choices" -> choices) ++ usage.map("usage" -> _)
    s"data: ${Json.obj(fields*).render}\n\n"

/** ANTHROPIC'S MESSAGES: a request as an `Ask`, a turn as a message or as a stream's events */
object Anthropic:
  def ask(body: Json): Ask =
    val system = body.get("system").map(text).filter(_.nonEmpty)
      .map(s => Json.obj("role" -> str("system"), "content" -> str(s))).toVector
    val turns = body("messages").items.flatMap { m =>
      val role = m("role").str
      m("content") match
        case Json.Str(s) => Vector(Json.obj("role" -> str(role), "content" -> str(s)))
        case Json.Arr(blocks) =>
          val texts = blocks.filter(_("type") == str("text")).map(_("text").str).mkString
          val uses = blocks.filter(_("type") == str("tool_use"))
          val results = blocks.filter(_("type") == str("tool_result"))
            .map(r => Json.obj("role" -> str("tool"), "content" -> str(text(r("content")))))
          val turn =
            if role == "assistant" && uses.nonEmpty then
              Vector(Json.obj("role" -> str(role), "content" -> str(texts), "tool_calls" -> Json.Arr(uses.map(u =>
                Json.obj("type" -> str("function"), "function" ->
                  Json.obj("name" -> u("name"), "arguments" -> u("input")))))))
            else if texts.nonEmpty || results.isEmpty then
              Vector(Json.obj("role" -> str(role), "content" -> str(texts)))
            else Vector.empty
          results ++ turn
        case other => Vector(Json.obj("role" -> str(role), "content" -> str(text(other))))
    }
    val tools = body.get("tools").map(_.items).getOrElse(Vector.empty).map(t => Json.obj("type" -> str("function"),
      "function" -> Json.obj("name" -> t("name"), "description" -> t.get("description").getOrElse(str("")),
        "parameters" -> t.get("input_schema").getOrElse(Json.obj()))))
    val thinking = body.get("thinking").exists(_("type") == str("enabled"))
    Ask(Ask.systemFirst(system ++ turns), tools, body.get("max_tokens").map(_.int).getOrElse(1024), thinking)

  def stopReason(f: Finished): String =
    if f.reply.toolCalls.nonEmpty then "tool_use" else if f.byStop then "end_turn" else "max_tokens"

  def toolUse(id: String, k: Int, c: ToolCall): Json =
    Json.obj("type" -> str("tool_use"), "id" -> str(s"toolu_${id}_$k"), "name" -> str(c.name), "input" -> c.arguments)

  def message(id: String, model: String, f: Finished): Json =
    val content = (if f.reply.reasoning.nonEmpty then
        Vector(Json.obj("type" -> str("thinking"), "thinking" -> str(f.reply.reasoning), "signature" -> str("")))
      else Vector.empty) ++
      (if f.reply.content.nonEmpty || f.reply.toolCalls.isEmpty then
        Vector(Json.obj("type" -> str("text"), "text" -> str(f.reply.content))) else Vector.empty) ++
      f.reply.toolCalls.zipWithIndex.map((c, k) => toolUse(id, k, c))
    Json.obj("id" -> str(s"msg_$id"), "type" -> str("message"), "role" -> str("assistant"), "model" -> str(model),
      "content" -> Json.Arr(content), "stop_reason" -> str(stopReason(f)), "stop_sequence" -> Json.Null,
      "usage" -> Json.obj("input_tokens" -> num(f.timing.promptTokens), "output_tokens" -> num(f.timing.tokens)))

  def event(name: String, data: Json): String = s"event: $name\ndata: ${data.render}\n\n"
