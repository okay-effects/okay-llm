package okay.llm

import okay.effects.json.Json
import okay.scala3.*
import okay.scala3.std.*

/** a model behind a server, over the wire: the gateway on a port of its own (a fake model inside), the backend
 * asking it — its text piece by piece, its tool call as the server read it, its usage */
class HttpBackendSuite extends munit.FunSuite:
  type Gw[X] = (Llm + Http + Clock + Fresh + Log)[X]
  private val toolAnswer = "Checking. <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}" +
    "\n</tool_call>"
  private val weather = Json.fromText("""{"type": "function", "function": {"name": "get_weather", "parameters":
    {"type": "object", "properties": {"city": {"type": "string"}}}}}""")

  /** the gateway serving a fake model that answers `answer` to the last message, on a free port */
  private def serving(answer: String => String): Int =
    val probe = java.net.ServerSocket(0)
    val port = probe.getLocalPort
    probe.close()
    val t = Thread(() => Server.run(port) { (req, out) =>
      run(Log.collect()(Fresh.from(0)(Clock.system(Http.socket(out)(Llm.fake(answer)(Gateway.route[Gw](req)))))))
      ()
    })
    t.setDaemon(true)
    t.start()
    var up = false
    while !up do
      try { java.net.Socket("127.0.0.1", port).close(); up = true }
      catch case _: java.io.IOException => Thread.sleep(20)
    port

  test("a turn over the wire: the text in pieces, the tool call as the server read it, the usage"):
    val port = serving(_ => toolAnswer)
    val backend = HttpBackend(s"http://127.0.0.1:$port/v1", "fake")
    val ask = Programs.user("Weather in Paris?").copy(tools = Vector(weather), maxTokens = 64)
    val f = run(Llm.on(backend)(Llm.complete[Llm](ask)))
    assertEquals(f.reply.toolCalls, Vector(ToolCall("get_weather", Json.obj("city" -> Json.Str("Paris")))))
    assertEquals(f.reply.content, "Checking.")
    assert(f.byStop)
    assert(f.timing.tokens > 0)

  test("a plain answer; a tool call of the conversation sent back with its arguments as text"):
    val port = serving(_ => "Sunny.")
    val backend = HttpBackend(s"http://127.0.0.1:$port/v1", "fake")
    val history = Vector(Json.obj("role" -> Json.Str("user"), "content" -> Json.Str("Weather?")),
      Json.obj("role" -> Json.Str("assistant"), "content" -> Json.Str(""), "tool_calls" -> Json.arr(Json.obj(
        "type" -> Json.Str("function"), "function" -> Json.obj("name" -> Json.Str("get_weather"),
          "arguments" -> Json.obj("city" -> Json.Str("Paris")))))),
      Json.obj("role" -> Json.Str("tool"), "content" -> Json.Str("sunny")))
    val wire = HttpBackend.body(Ask(history, Vector(weather)), "m")
    assertEquals(wire("messages").list(1)("tool_calls").list.head("function")("arguments"),
      Json.Str("""{"city":"Paris"}"""))
    val f = run(Llm.on(backend)(Llm.complete[Llm](Ask(history, Vector(weather)))))
    assertEquals(f.reply.content, "Sunny.")
    assertEquals(f.reply.toolCalls, Vector.empty)
