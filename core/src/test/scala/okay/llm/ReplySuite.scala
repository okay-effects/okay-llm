package okay.llm

class ReplySuite extends munit.FunSuite:
  private val weather = Json.parse("""{"type": "function", "function": {"name": "get_weather", "parameters":
    {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}""")

  test("text alone; thinking apart"):
    assertEquals(Reply.parse("Paris."), Reply("", "Paris.", Vector.empty))
    assertEquals(Reply.parse("<think>\nhm\n</think>\n\nParis."), Reply("hm", "Paris.", Vector.empty))
    assertEquals(Reply.parse("hm, the capital\n</think>\n\nParis.", thinking = true),
      Reply("hm, the capital", "Paris.", Vector.empty))

  test("Qwen3's tool calls: JSON"):
    val r = Reply.parse("Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": " +
      "\"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": " +
      "\\\"Rome\\\"}\"}\n</tool_call>")
    assertEquals(r.content, "Let me check.")
    assertEquals(r.toolCalls.map(c => c.name -> c.arguments.render),
      Vector("get_weather" -> """{"city":"Paris"}""", "get_weather" -> """{"city":"Rome"}"""))

  test("Qwen3.6's tool calls: XML, the values as the schema types them"):
    val r = Reply.parse("<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n" +
      "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>", tools = Seq(weather))
    assertEquals(r.content, "")
    assertEquals(r.toolCalls, Vector(ToolCall("get_weather", Json.parse("""{"city": "New York", "days": 3}"""))))
