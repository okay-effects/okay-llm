package okay.llm

import okay.effects.json.Json

class AgUiSuite extends munit.FunSuite:
  test("events as AG-UI names them; a text's three, a call's three; SSE framing"):
    val t = AgUi.text("m1", "hi")
    assertEquals(t.map(_("type").string), Vector("TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"))
    assertEquals(AgUi.text("m1", ""), Vector.empty)
    val c = AgUi.toolCall("c1", "read", Json.obj("path" -> Json.Str("a")))
    assertEquals(c(1)("delta").string, """{"path":"a"}""")
    assert(AgUi.sse(t.head).startsWith("data: {") && AgUi.sse(t.head).endsWith("\n\n"))
    val i = AgUi.runInterrupted("t", "r", Vector(AgUi.approval("i1", "write a.txt?", Some("c1"))))
    assertEquals(i("outcome")("interrupts").list.head("toolCallId").string, "c1")

  test("a run's input: the last user words (text or parts), the resume entries"):
    val in = AgUi.RunInput.parse(Json.fromText("""{"threadId": "t", "runId": "r", "messages": [
      {"id": "1", "role": "user", "content": "first"}, {"id": "2", "role": "assistant", "content": "ok"},
      {"id": "3", "role": "user", "content": [{"type": "text", "text": "second"}]}],
      "resume": [{"interruptId": "i1", "status": "resolved", "payload": {"approved": true}}]}"""))
    assertEquals(in.lastUserText, Some("second"))
    assertEquals(in.resume, Vector(AgUi.Resume("i1", true, Json.obj("approved" -> Json.Bool(true)))))
