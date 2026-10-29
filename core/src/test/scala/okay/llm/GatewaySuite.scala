package okay.llm

import okay.effects.json.Json

import okay.scala3.*
import okay.scala3.std.*


/** the gateway's programs under handlers of their own: a fake model, a response collected, virtual time */
class GatewaySuite extends munit.FunSuite:
  type Gw[X] = (Llm + Http + Clock + Fresh + Log)[X]
  private def call(method: String, path: String, body: String, answer: String = "The capital is Paris.")
      : (Int, String, String) =
    val req = Request(method, path, Map.empty, body)
    val (_, (_, (resp, _))) =
      run(Log.collect()(Fresh.from(0)(Clock.virtual(1_700_000_000_000L)(Http.collect(
        Llm.fake(_ => answer)(Gateway.route[Gw](req)))))))
    resp
  private def events(body: String): Vector[Json] =
    body.split("\n\n").toVector.flatMap(_.linesIterator.find(_.startsWith("data: ")).map(_.drop(6)))
      .filter(_ != "[DONE]").map(Json.fromText)

  private val chat = """{"model": "m", "messages": [{"role": "user", "content": "Capital of France?"}]}"""
  private val weather = """{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
    "properties": {"city": {"type": "string"}}}}}"""
  private val toolAnswer = "Checking. <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}" +
    "\n</tool_call>"

  test("OpenAI: a whole response"):
    val (status, _, body) = call("POST", "/v1/chat/completions", chat)
    assertEquals(status, 200)
    val j = Json.fromText(body)
    assertEquals(j("choices").list.head("message")("content").string, "The capital is Paris.")
    assertEquals(j("choices").list.head("finish_reason").string, "stop")
    assertEquals(j("object").string, "chat.completion")

  test("OpenAI: a stream, its deltas the answer, [DONE] last"):
    val (status, kind, body) = call("POST", "/v1/chat/completions",
      chat.dropRight(1) + """, "stream": true, "stream_options": {"include_usage": true}}""")
    assertEquals((status, kind), (200, "text/event-stream"))
    val es = events(body)
    val text = es.flatMap(_("choices").list.headOption).flatMap(_("delta").field("content")).map(_.string).mkString
    assertEquals(text, "The capital is Paris.")
    assertEquals(es.flatMap(_("choices").list.headOption).flatMap(_.field("finish_reason")).map(_.string),
      Vector("stop"))
    assert(es.last.field("usage").isDefined)
    assert(body.endsWith("data: [DONE]\n\n"))

  test("OpenAI: a tool called"):
    val req = chat.dropRight(1) + s""", "tools": [$weather]}"""
    val j = Json.fromText(call("POST", "/v1/chat/completions", req, toolAnswer)._3)
    val choice = j("choices").list.head
    assertEquals(choice("finish_reason").string, "tool_calls")
    assertEquals(choice("message")("content").string, "Checking.")
    val f = choice("message")("tool_calls").list.head("function")
    assertEquals((f("name").string, Json.fromText(f("arguments").string)),
      ("get_weather", Json.fromText("""{"city":"Paris"}""")))
    // streamed: the text before the call, then the call, the reason
    val es = events(call("POST", "/v1/chat/completions", req.dropRight(1) + """, "stream": true}""", toolAnswer)._3)
    val deltas = es.flatMap(_("choices").list.headOption).map(_("delta"))
    assertEquals(deltas.flatMap(_.field("content")).map(_.string).mkString, "Checking. ")
    assert(deltas.exists(_.field("tool_calls").isDefined))

  test("Anthropic: a whole message, and a tool used"):
    val req = """{"model": "m", "max_tokens": 64, "system": "Be brief.",
      "messages": [{"role": "user", "content": [{"type": "text", "text": "Capital of France?"}]}]}"""
    val m = Json.fromText(call("POST", "/v1/messages", req)._3)
    assertEquals(m("content").list.map(b => b("type").string -> b("text").string),
      Vector("text" -> "The capital is Paris."))
    assertEquals(m("stop_reason").string, "end_turn")
    assertEquals(m("model").string, "m")   // as the request named it
    val t = Json.fromText(call("POST", "/v1/messages", req.dropRight(1) +
      """, "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}""", toolAnswer)._3)
    assertEquals(t("stop_reason").string, "tool_use")
    assertEquals(t("content").list.map(_("type").string), Vector("text", "tool_use"))
    assertEquals(t("content").list(1)("input"), Json.fromText("""{"city":"Paris"}"""))

  test("Anthropic: a stream's events in order, thinking in its own block"):
    val req = """{"model": "m", "max_tokens": 64, "stream": true, "thinking": {"type": "enabled"},
      "messages": [{"role": "user", "content": "Capital?"}]}"""
    val es = events(call("POST", "/v1/messages", req, "Hm, France. </think>\n\nParis it is.")._3)
    assertEquals(es.map(_("type").string), Vector("message_start", "content_block_start", "content_block_delta",
      "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta",
      "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"))
    val thinking = es.filter(e => e.field("delta").exists(_("type") == Json.Str("thinking_delta")))
    assertEquals(thinking.map(_("delta")("thinking").string).mkString, "Hm, France. ")
    val text = es.filter(e => e.field("delta").exists(_("type") == Json.Str("text_delta")))
    assertEquals(text.map(_("delta")("text").string).mkString, "Paris it is.")

  test("count_tokens, models, errors"):
    assert(Json.fromText(call("POST", "/v1/messages/count_tokens",
      """{"model": "m", "messages": [{"role": "user", "content": "hello there"}]}""")._3)("input_tokens").int > 0)
    val models = Json.fromText(call("GET", "/v1/models", "")._3)("data").list.head
    assertEquals(models("id").string, "fake")
    assertEquals(models("display_name").string, "fake")   // Anthropic's list, as rozum reads it
    assertEquals(call("POST", "/v1/chat/completions", "{not json")._1, 400)
    assertEquals(call("GET", "/nowhere", "")._1, 404)

  test("the streamer holds a tag's beginning until it is whole"):
    val s = Streamer(thinking = true)
    val parts = "thinking hard</think>\n\nanswer<tool_call>{}".map(_.toString).flatMap(s.push) ++ s.flush()
    val thinking = parts.collect { case Streamer.Part.Thinking(t) => t }.mkString
    val text = parts.collect { case Streamer.Part.Text(t) => t }.mkString
    assertEquals((thinking, text), ("thinking hard", "answer"))

  test("a system message mid-conversation stays where it is, as a user's: the prompt's beginning unchanged"):
    val a = Anthropic.ask(Json.fromText("""{"max_tokens": 8, "system": [{"type": "text", "text": "You are helpful."}],
      "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": [{"type": "text",
      "text": "# Environment"}]}, {"role": "assistant", "content": "hello"}]}"""))
    assertEquals(a.messages.map(m => m("role").string -> m("content").string), Vector(
      "system" -> "You are helpful.", "user" -> "hi", "user" -> "# Environment", "assistant" -> "hello"))

  test("Claude Code's billing header is no part of the prompt"):
    val a = Anthropic.ask(Json.fromText("""{"max_tokens": 8, "system": [{"type": "text",
      "text": "x-anthropic-billing-header: cc_version=2.1.311.69e; cc_entrypoint=sdk-cli;"},
      {"type": "text", "text": "You are an agent."}], "messages": [{"role": "user", "content": "hi"}]}"""))
    assertEquals(a.messages.head("content").string, "You are an agent.")

  test("a request's sampling, each field unset when it says none"):
    val a = OpenAi.ask(Json.fromText("""{"messages": [], "temperature": 0.7, "top_p": 0.8, "seed": 5}"""))
    assertEquals(a.sampling, Sampling(Some(0.7), Some(0.8), None, Some(5L)))
    assertEquals(Anthropic.ask(Json.fromText("""{"messages": [], "top_k": 20}""")).sampling,
      Sampling(None, None, Some(20), None))
    assertEquals(OpenAi.ask(Json.fromText("""{"messages": [], "presence_penalty": 1.5}""")).sampling,
      Sampling(presencePenalty = Some(1.5)))

  test("the model a request names, for a backend of several"):
    assertEquals(Anthropic.ask(Json.fromText("""{"model": "a/b", "messages": []}""")).model, Some("a/b"))
    assertEquals(OpenAi.ask(Json.fromText("""{"messages": []}""")).model, None)
