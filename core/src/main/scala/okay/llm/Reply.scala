package okay.llm

/** A TOOL CALLED: its name and its arguments, a JSON object */
final case class ToolCall(name: String, arguments: Json)

/** A MODEL'S TURN, read: its thinking, its text, the tools it called */
final case class Reply(reasoning: String, content: String, toolCalls: Vector[ToolCall])

object Reply:
  /** the text a model made: `<think>…</think>` first if any (the prompt may have opened it — `thinking`), then text,
   * then tool calls — Qwen3's `<tool_call>{"name": …, "arguments": …}</tool_call>`, or Qwen3.6's
   * `<tool_call><function=name><parameter=p>value</parameter></function></tool_call>`, whose values are read as the
   * tool's JSON schema says (a string as it is; else JSON when it parses) */
  def parse(text: String, thinking: Boolean = false, tools: Seq[Json] = Nil): Reply =
    val (reasoning, rest) =
      val close = text.indexOf("</think>")
      if close >= 0 && (thinking || text.trim.startsWith("<think>")) then
        (text.substring(0, close).replace("<think>", "").trim, text.substring(close + "</think>".length))
      else if thinking then (text.trim, "")              // still thinking when it stopped
      else ("", text)
    val first = rest.indexOf("<tool_call>")
    if first < 0 then Reply(reasoning, rest.trim, Vector.empty)
    else
      val calls = Vector.newBuilder[ToolCall]
      var at = first
      while at >= 0 do
        val end = rest.indexOf("</tool_call>", at)
        val body = rest.substring(at + "<tool_call>".length, if end < 0 then rest.length else end).trim
        call(body, tools).foreach(calls += _)
        at = if end < 0 then -1 else rest.indexOf("<tool_call>", end)
      Reply(reasoning, rest.substring(0, first).trim, calls.result())

  private def call(body: String, tools: Seq[Json]): Option[ToolCall] =
    if body.startsWith("<function=") then
      val nameEnd = body.indexOf('>')
      val name = body.substring("<function=".length, nameEnd).trim
      val schema = tools.map(t => t.get("function").getOrElse(t)).find(_("name") == Json.Str(name))
        .map(t => t.get("parameters").orElse(t.get("input_schema")).getOrElse(Json.Null)("properties"))
        .getOrElse(Json.Null)
      val params = Vector.newBuilder[(String, Json)]
      var at = body.indexOf("<parameter=", nameEnd)
      while at >= 0 do
        val pEnd = body.indexOf('>', at)
        val pName = body.substring(at + "<parameter=".length, pEnd).trim
        val close = body.indexOf("</parameter>", pEnd)
        var value = body.substring(pEnd + 1, if close < 0 then body.length else close)
        if value.startsWith("\n") then value = value.drop(1)
        if value.endsWith("\n") then value = value.dropRight(1)
        params += pName -> typed(value, schema(pName)("type"))
        at = if close < 0 then -1 else body.indexOf("<parameter=", close)
      Some(ToolCall(name, Json.Obj(scala.collection.immutable.VectorMap.from(params.result()))))
    else
      try
        val j = Json.parse(body)
        val args = j("arguments") match
          case Json.Str(s) => try Json.parse(s) catch case _: IllegalArgumentException => Json.Str(s)
          case Json.Null => Json.obj()
          case a => a
        Some(ToolCall(j("name").str, args))
      catch case _: RuntimeException => None

  private def typed(value: String, kind: Json): Json = kind match
    case Json.Str("string") => Json.Str(value)
    case _ =>
      try Json.parse(value.trim)
      catch case _: IllegalArgumentException => Json.Str(value)
