package okay.llm

import okay.effects.json.Json



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
  case Json.Arr(parts) => parts.flatMap(p => p.field("text").collect { case Json.Str(t) => t }).mkString
  case Json.Null => ""
  case other => other.render

private def js(s: String): Json = Json.Str(s)
private def jn(n: Long): Json = Json.num(n)

/** OPENAI'S CHAT COMPLETIONS: a request as an `Ask`, a turn as a response or as chunks of a stream */
object OpenAi:
  def ask(body: Json): Ask =
    val messages = body("messages").list.map { m =>
      val fields = Vector.newBuilder[(String, Json)]
      fields += "role" -> m("role")
      fields += "content" -> js(text(m("content")))
      m.field("tool_calls").foreach(calls => fields += "tool_calls" -> Json.Arr(calls.list.map { c =>
        val f = c("function")
        val args = f("arguments") match
          case Json.Str(s) => try Json.fromText(s) catch case _: IllegalArgumentException => Json.Str(s)
          case a => a
        Json.obj("type" -> js("function"), "function" -> Json.obj("name" -> f("name"), "arguments" -> args))
      }))
      m.field("reasoning_content").foreach(r => fields += "reasoning_content" -> r)
      Json.obj(fields.result()*)
    }
    val max = body.field("max_completion_tokens").orElse(body.field("max_tokens")).map(_.int).getOrElse(1024)
    val thinking =
      body.field("chat_template_kwargs").flatMap(_.field("enable_thinking")).exists(_ == Json.Bool(true)) ||
      body.field("reasoning_effort").isDefined
    Ask(Ask.systemFirst(messages), body.field("tools").map(_.list).getOrElse(Vector.empty), max, thinking)

  def finishReason(f: Finished): String =
    if f.reply.toolCalls.nonEmpty then "tool_calls" else if f.byStop then "stop" else "length"

  def toolCalls(id: String, calls: Vector[ToolCall]): Json = Json.Arr(calls.zipWithIndex.map((c, k) =>
    Json.obj("index" -> jn(k), "id" -> js(s"call_${id}_$k"), "type" -> js("function"),
      "function" -> Json.obj("name" -> js(c.name), "arguments" -> js(c.arguments.render)))))

  def usage(f: Finished): Json =
    Json.obj("prompt_tokens" -> jn(f.timing.promptTokens), "completion_tokens" -> jn(f.timing.tokens),
      "total_tokens" -> jn(f.timing.promptTokens + f.timing.tokens))

  def response(id: String, model: String, created: Long, f: Finished): Json =
    val message = Vector("role" -> js("assistant"), "content" -> js(f.reply.content)) ++
      (if f.reply.reasoning.nonEmpty then Vector("reasoning_content" -> js(f.reply.reasoning)) else Vector.empty) ++
      (if f.reply.toolCalls.nonEmpty then Vector("tool_calls" -> toolCalls(id, f.reply.toolCalls)) else Vector.empty)
    Json.obj("id" -> js(s"chatcmpl-$id"), "object" -> js("chat.completion"), "created" -> jn(created),
      "model" -> js(model), "choices" -> Json.arr(Json.obj("index" -> jn(0), "message" -> Json.obj(message*),
        "finish_reason" -> js(finishReason(f)))), "usage" -> usage(f))

  def chunk(id: String, model: String, created: Long, delta: Json, finish: Option[String] = None,
      usage: Option[Json] = None): String =
    val choices = if usage.isDefined then Json.arr() else Json.arr(Json.obj("index" -> jn(0), "delta" -> delta,
      "finish_reason" -> finish.map(js).getOrElse(Json.Null)))
    val fields = Vector("id" -> js(s"chatcmpl-$id"), "object" -> js("chat.completion.chunk"),
      "created" -> jn(created), "model" -> js(model), "choices" -> choices) ++ usage.map("usage" -> _)
    s"data: ${Json.obj(fields*).render}\n\n"

/** ANTHROPIC'S MESSAGES: a request as an `Ask`, a turn as a message or as a stream's events */
object Anthropic:
  def ask(body: Json): Ask =
    // Claude Code's first system block is its billing header, a hash new each request — not for the model, and a
    // prompt that differs from its first tokens has no prefix to reuse
    val system = body.field("system").map {
      case Json.Arr(blocks) =>
        Json.Arr(blocks.filterNot(b => b("text") match
          case Json.Str(t) => t.startsWith("x-anthropic-billing-header:")
          case _ => false))
      case other => other
    }.map(text).filter(_.nonEmpty).map(s => Json.obj("role" -> js("system"), "content" -> js(s))).toVector
    val turns = body("messages").list.flatMap { m =>
      val role = m("role").string
      m("content") match
        case Json.Str(s) => Vector(Json.obj("role" -> js(role), "content" -> js(s)))
        case Json.Arr(blocks) =>
          val texts = blocks.filter(_("type") == js("text")).map(_("text").string).mkString
          val uses = blocks.filter(_("type") == js("tool_use"))
          val results = blocks.filter(_("type") == js("tool_result"))
            .map(r => Json.obj("role" -> js("tool"), "content" -> js(text(r("content")))))
          val turn =
            if role == "assistant" && uses.nonEmpty then
              Vector(Json.obj("role" -> js(role), "content" -> js(texts), "tool_calls" -> Json.Arr(uses.map(u =>
                Json.obj("type" -> js("function"), "function" ->
                  Json.obj("name" -> u("name"), "arguments" -> u("input")))))))
            else if texts.nonEmpty || results.isEmpty then
              Vector(Json.obj("role" -> js(role), "content" -> js(texts)))
            else Vector.empty
          results ++ turn
        case other => Vector(Json.obj("role" -> js(role), "content" -> js(text(other))))
    }
    val tools = body.field("tools").map(_.list).getOrElse(Vector.empty).map(t => Json.obj("type" -> js("function"),
      "function" -> Json.obj("name" -> t("name"), "description" -> t.field("description").getOrElse(js("")),
        "parameters" -> t.field("input_schema").getOrElse(Json.obj()))))
    val thinking = body.field("thinking").exists(_("type") == js("enabled"))
    Ask(Ask.systemFirst(system ++ turns), tools, body.field("max_tokens").map(_.int).getOrElse(1024), thinking)

  def stopReason(f: Finished): String =
    if f.reply.toolCalls.nonEmpty then "tool_use" else if f.byStop then "end_turn" else "max_tokens"

  def toolUse(id: String, k: Int, c: ToolCall): Json =
    Json.obj("type" -> js("tool_use"), "id" -> js(s"toolu_${id}_$k"), "name" -> js(c.name), "input" -> c.arguments)

  def message(id: String, model: String, f: Finished): Json =
    val content = (if f.reply.reasoning.nonEmpty then
        Vector(Json.obj("type" -> js("thinking"), "thinking" -> js(f.reply.reasoning), "signature" -> js("")))
      else Vector.empty) ++
      (if f.reply.content.nonEmpty || f.reply.toolCalls.isEmpty then
        Vector(Json.obj("type" -> js("text"), "text" -> js(f.reply.content))) else Vector.empty) ++
      f.reply.toolCalls.zipWithIndex.map((c, k) => toolUse(id, k, c))
    Json.obj("id" -> js(s"msg_$id"), "type" -> js("message"), "role" -> js("assistant"), "model" -> js(model),
      "content" -> Json.Arr(content), "stop_reason" -> js(stopReason(f)), "stop_sequence" -> Json.Null,
      "usage" -> Json.obj("input_tokens" -> jn(f.timing.promptTokens), "output_tokens" -> jn(f.timing.tokens)))

  def event(name: String, data: Json): String = s"event: $name\ndata: ${data.render}\n\n"
