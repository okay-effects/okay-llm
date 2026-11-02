package okay.llm

import okay.effects.json.Json

/** AG-UI (the Agent–User Interaction protocol, 1.0): a run's events as JSON, each a server-sent event — what any AG-UI
 * client (CopilotKit, its terminal and mobile clients) shows. Shared: okay-agent's server makes them of its program,
 * okay-remote of its sessions. A run that needs a person ends with an interrupt; the next run on the thread answers it
 * (`resume`) */
object AgUi:
  private def s(v: String) = Json.Str(v)
  private def event(kind: String, fields: (String, Json)*): Json =
    Json.Obj(Vector("type" -> s(kind), "timestamp" -> Json.num(System.currentTimeMillis())) ++ fields)

  def runStarted(threadId: String, runId: String): Json = event("RUN_STARTED", "threadId" -> s(threadId),
    "runId" -> s(runId))
  /** ended, done */
  def runFinished(threadId: String, runId: String): Json = event("RUN_FINISHED", "threadId" -> s(threadId),
    "runId" -> s(runId), "outcome" -> Json.obj("type" -> s("success")))
  /** ended, waiting for a person: the interrupts the next run's `resume` answers */
  def runInterrupted(threadId: String, runId: String, interrupts: Vector[Interrupt]): Json = event("RUN_FINISHED",
    "threadId" -> s(threadId), "runId" -> s(runId),
    "outcome" -> Json.obj("type" -> s("interrupt"), "interrupts" -> Json.Arr(interrupts.map(_.json))))
  def runCancelled(threadId: String, runId: String): Json = event("RUN_FINISHED", "threadId" -> s(threadId),
    "runId" -> s(runId), "outcome" -> Json.obj("type" -> s("cancelled")))
  def runError(message: String): Json = event("RUN_ERROR", "message" -> s(message))
  def stepStarted(name: String): Json = event("STEP_STARTED", "stepName" -> s(name))
  def stepFinished(name: String): Json = event("STEP_FINISHED", "stepName" -> s(name))

  /** a message of the model's, whole: start, its text, end (a text never empty: AG-UI's content needs a delta) */
  def text(messageId: String, text: String): Vector[Json] =
    if text.isEmpty then Vector.empty
    else Vector(event("TEXT_MESSAGE_START", "messageId" -> s(messageId), "role" -> s("assistant")),
      event("TEXT_MESSAGE_CONTENT", "messageId" -> s(messageId), "delta" -> s(text)),
      event("TEXT_MESSAGE_END", "messageId" -> s(messageId)))

  /** a call the model made: start, its arguments (JSON text), end */
  def toolCall(id: String, name: String, args: Json, parent: Option[String] = None): Vector[Json] =
    Vector(event("TOOL_CALL_START", (Vector("toolCallId" -> s(id), "toolCallName" -> s(name)) ++
        parent.map("parentMessageId" -> s(_)))*),
      event("TOOL_CALL_ARGS", "toolCallId" -> s(id), "delta" -> s(args.render)),
      event("TOOL_CALL_END", "toolCallId" -> s(id)))
  def toolResult(messageId: String, toolCallId: String, content: String): Json = event("TOOL_CALL_RESULT",
    "messageId" -> s(messageId), "toolCallId" -> s(toolCallId), "content" -> s(content), "role" -> s("tool"))
  def custom(name: String, value: Json): Json = event("CUSTOM", "name" -> s(name), "value" -> value)

  /** an event as SSE puts it on the wire */
  def sse(e: Json): String = s"data: ${e.render}\n\n"

  /** what a run needs from outside: an approval (`responseSchema`: what `resume`'s payload is) */
  final case class Interrupt(id: String, reason: String, message: String, toolCallId: Option[String] = None,
      responseSchema: Option[Json] = None):
    def json: Json = Json.Obj(Vector("id" -> s(id), "reason" -> s(reason), "message" -> s(message)) ++
      toolCallId.map("toolCallId" -> s(_)) ++ responseSchema.map("responseSchema" -> _))

  /** an approval's interrupt: the payload `{"approved": true|false, "reason"?: text}` */
  def approval(id: String, message: String, toolCallId: Option[String]): Interrupt = Interrupt(id, "approval",
    message, toolCallId, Some(Json.obj("type" -> s("object"), "properties" -> Json.obj(
      "approved" -> Json.obj("type" -> s("boolean")), "reason" -> Json.obj("type" -> s("string"))),
      "required" -> Json.arr(s("approved")))))

  /** an answer to an interrupt: resolved (with its payload) or cancelled */
  final case class Resume(interruptId: String, resolved: Boolean, payload: Json)

  /** a run asked for (RunAgentInput): its thread, its id, the conversation as the client keeps it, the answers to the
   * last run's interrupts */
  final case class RunInput(threadId: String, runId: String, messages: Vector[Json], tools: Vector[Json],
      resume: Vector[Resume]):
    /** the person's words: the last user message's text (its text parts, when it has parts) */
    def lastUserText: Option[String] = messages.reverse.find(_("role") == s("user")).map(m => text(m("content")))
    private def text(c: Json): String = c match
      case Json.Str(t) => t
      case Json.Arr(parts) => parts.flatMap(_.field("text").collect { case Json.Str(t) => t }).mkString("\n")
      case _ => ""

  object RunInput:
    def parse(j: Json): RunInput =
      def str(k: String) = j.field(k).collect { case Json.Str(v) => v }.getOrElse("")
      RunInput(str("threadId"), str("runId"), j.field("messages").map(_.list).getOrElse(Vector.empty),
        j.field("tools").map(_.list).getOrElse(Vector.empty),
        j.field("resume").map(_.list).getOrElse(Vector.empty).map(r => Resume(
          r.field("interruptId").collect { case Json.Str(v) => v }.getOrElse(""),
          r.field("status").contains(s("resolved")), r("payload"))))
