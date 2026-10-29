package okay.llm

import okay.effects.json.Json

import okay.scala3.*
import okay.scala3.std.*

import Streamer.Part

/** THE GATEWAY: OpenAI's and Anthropic's APIs over the model — each request a program of effects: the model (`Llm`),
 * the response (`Http`), ids (`Clock`, `Fresh`), the log. Whole responses and streams (server-sent events) */
object Gateway:
  private def s(x: String): Json = Json.Str(x)
  private def n(x: Long): Json = Json.num(x)

  def route[R[_]: Has[Llm] : Has[Http] : Has[Clock] : Has[Fresh] : Has[Log]](req: Request): Unit ! R =
    def body: Either[String, Json] =
      try Right(Json.fromText(req.body)) catch case e: RuntimeException => Left(e.getMessage)
    def bad(message: String): Unit ! R =
      Http.json(400, Json.obj("error" -> Json.obj("type" -> s("invalid_request_error"), "message" -> s(message))))
    val path = req.path.takeWhile(_ != '?')
    for
      _ <- Log.info(s"${req.method} $path")
      _ <- (req.method, path) match
        case ("GET", "/" | "/health") => Http.json(200, Json.obj("status" -> s("ok")))
        case ("GET", "/v1/models") =>
          // both APIs' lists in one: OpenAI's (`object`, `owned_by`), Anthropic's (`type`, `display_name`, …)
          Llm.models.flatMap(ms => Http.json(200, Json.obj("object" -> s("list"), "data" -> Json.arr(ms.map(m =>
            Json.obj("id" -> s(m), "object" -> s("model"), "owned_by" -> s("okay-llm"), "type" -> s("model"),
              "display_name" -> s(m), "created_at" -> s("1970-01-01T00:00:00Z")))*),
            "has_more" -> Json.Bool(false), "first_id" -> s(ms.head), "last_id" -> s(ms.last))))
        case ("POST", "/v1/chat/completions") => body.fold(bad, b => openAi(b))
        case ("POST", "/v1/messages") => body.fold(bad, b => anthropic(b))
        case ("POST", "/v1/messages/count_tokens") =>
          body.fold(bad, b =>
            Llm.count(Anthropic.ask(b)).flatMap(t => Http.json(200, Json.obj("input_tokens" -> n(t)))))
        case _ => Http.json(404, Json.obj("error" -> Json.obj("type" -> s("not_found"), "message" -> s(path))))
    yield ()

  /** a generation's line in the log: the prompt (and how much of it a cache gave), the answer, the speeds */
  private def report(t: Timing): String =
    val cached = if t.cachedTokens > 0 then s" (${t.cachedTokens} cached)" else ""
    f"${t.promptTokens} in$cached, prefill ${t.prefillNanos / 1e6}%.0f ms; ${t.tokens} out, " +
      f"${t.decodePerSecond}%.1f tok/s"

  private def ident[R[_]: Has[Clock] : Has[Fresh]]: (String, Long) ! R =
    for
      t <- Clock.now
      k <- Fresh.next
    yield (s"$t$k", t / 1000)

  /** the generation's pieces, each to `f` with a state carried: the state at the end */
  private def pieces[S, R[_]: Has[Llm]](g: Gen, start: S)(f: (S, String) => S ! R): S ! R =
    loop[S, S, R](start)(st => Llm.next(g).flatMap {
      case Some(p) => f(st, p).map(Left(_))
      case None => pure(Right(st))
    })

  private def sendAll[R[_]: Has[Http]](chunks: Seq[String]): Unit ! R =
    chunks.foldLeft(unit[R])((p, c) => p.andThen(Http.send(c)))

  // ---- OpenAI

  def openAi[R[_]: Has[Llm] : Has[Http] : Has[Clock] : Has[Fresh] : Has[Log]](body: Json): Unit ! R =
    val ask = OpenAi.ask(body)
    val stream = body.field("stream").contains(Json.Bool(true))
    val withUsage = body.field("stream_options").flatMap(_.field("include_usage")).contains(Json.Bool(true))
    for
      ids <- ident
      (id, created) = ids
      // the model as the request named it (a backend of several answers by that one), else the backend's
      model <- ask.model.fold(Llm.name)(pure)
      _ <-
        if !stream then Llm.complete(ask).flatMap(f =>
          Log.info(report(f.timing)).andThen(Http.json(200, OpenAi.response(id, model, created, f))))
        else
          def delta(parts: Vector[Part]): Vector[String] = parts.map {
            case Part.Text(t) => OpenAi.chunk(id, model, created, Json.obj("content" -> s(t)))
            case Part.Thinking(t) => OpenAi.chunk(id, model, created, Json.obj("reasoning_content" -> s(t)))
          }
          val streamer = Streamer(ask.thinking)
          for
            g <- Llm.start(ask)
            _ <- Http.open(200, "text/event-stream")
            _ <- Http.send(OpenAi.chunk(id, model, created, Json.obj("role" -> s("assistant"), "content" -> s(""))))
            _ <- pieces(g, ())((_, p) => sendAll(delta(streamer.push(p))))
            _ <- sendAll(delta(streamer.flush()))
            f <- Llm.finish(g)
            _ <- if f.reply.toolCalls.isEmpty then unit[R] else Http.send(OpenAi.chunk(id, model, created,
              Json.obj("tool_calls" -> OpenAi.toolCalls(id, f.reply.toolCalls))))
            _ <- Http.send(OpenAi.chunk(id, model, created, Json.obj(), Some(OpenAi.finishReason(f))))
            _ <- if withUsage then Http.send(OpenAi.chunk(id, model, created, Json.obj(), None, Some(OpenAi.usage(f))))
              else unit[R]
            _ <- Http.send("data: [DONE]\n\n")
            _ <- Http.close
            _ <- Log.info(report(f.timing))
          yield ()
    yield ()

  // ---- Anthropic

  /** the open content block: its kind and index */
  private final case class Blocks(open: Option[String], index: Int)

  def anthropic[R[_]: Has[Llm] : Has[Http] : Has[Clock] : Has[Fresh] : Has[Log]](body: Json): Unit ! R =
    val ask = Anthropic.ask(body)
    val stream = body.field("stream").contains(Json.Bool(true))
    for
      ids <- ident
      id = ids._1
      // the model as the request named it (a backend of several answers by that one), else the backend's
      model <- ask.model.fold(Llm.name)(pure)
      _ <-
        if !stream then Llm.complete(ask).flatMap(f =>
          Log.info(report(f.timing)).andThen(Http.json(200, Anthropic.message(id, model, f))))
        else
          import Anthropic.event
          def stop(b: Blocks): (Blocks, Vector[String]) = b.open match
            case Some(_) => (Blocks(None, b.index + 1), Vector(event("content_block_stop",
              Json.obj("type" -> s("content_block_stop"), "index" -> n(b.index)))))
            case None => (b, Vector.empty)
          def into(b: Blocks, kind: String): (Blocks, Vector[String]) =
            if b.open.contains(kind) then (b, Vector.empty)
            else
              val (closed, out) = stop(b)
              val block = if kind == "thinking" then Json.obj("type" -> s("thinking"), "thinking" -> s(""))
                else Json.obj("type" -> s("text"), "text" -> s(""))
              (Blocks(Some(kind), closed.index), out :+ event("content_block_start", Json.obj("type" ->
                s("content_block_start"), "index" -> n(closed.index), "content_block" -> block)))
          def parts(b0: Blocks, ps: Vector[Part]): (Blocks, Vector[String]) =
            ps.foldLeft((b0, Vector.empty[String])) { case ((b, out), part) =>
              val (kind, field, deltaType, t) = part match
                case Part.Thinking(t) => ("thinking", "thinking", "thinking_delta", t)
                case Part.Text(t) => ("text", "text", "text_delta", t)
              val (b2, opened) = into(b, kind)
              (b2, out ++ opened :+ event("content_block_delta", Json.obj("type" -> s("content_block_delta"),
                "index" -> n(b2.index), "delta" -> Json.obj("type" -> s(deltaType), field -> s(t)))))
            }
          val streamer = Streamer(ask.thinking)
          for
            input <- Llm.count(ask)
            g <- Llm.start(ask)
            _ <- Http.open(200, "text/event-stream")
            _ <- Http.send(event("message_start", Json.obj("type" -> s("message_start"), "message" -> Json.obj(
              "id" -> s(s"msg_$id"), "type" -> s("message"), "role" -> s("assistant"), "model" -> s(model),
              "content" -> Json.arr(), "stop_reason" -> Json.Null, "stop_sequence" -> Json.Null,
              "usage" -> Json.obj("input_tokens" -> n(input), "output_tokens" -> n(0))))))
            b1 <- pieces(g, Blocks(None, 0)) { (b, p) =>
              val (b2, out) = parts(b, streamer.push(p))
              sendAll(out).as(b2)
            }
            f <- Llm.finish(g)
            (b2, tail) = parts(b1, streamer.flush())
            (b3, tail2) = if b2.open.isEmpty && b2.index == 0 && f.reply.toolCalls.isEmpty then
                val (o, opened) = into(b2, "text"); (o, opened) else (b2, Vector.empty)
            (b4, closed) = stop(b3)
            tools = f.reply.toolCalls.zipWithIndex.flatMap((c, k) => Vector(
              event("content_block_start", Json.obj("type" -> s("content_block_start"), "index" -> n(b4.index + k),
                "content_block" -> Json.obj("type" -> s("tool_use"), "id" -> s(s"toolu_${id}_$k"),
                  "name" -> s(c.name), "input" -> Json.obj()))),
              event("content_block_delta", Json.obj("type" -> s("content_block_delta"), "index" -> n(b4.index + k),
                "delta" -> Json.obj("type" -> s("input_json_delta"), "partial_json" -> s(c.arguments.render)))),
              event("content_block_stop", Json.obj("type" -> s("content_block_stop"), "index" -> n(b4.index + k)))))
            _ <- sendAll(tail ++ tail2 ++ closed ++ tools)
            _ <- Http.send(event("message_delta", Json.obj("type" -> s("message_delta"), "delta" -> Json.obj(
              "stop_reason" -> s(Anthropic.stopReason(f)), "stop_sequence" -> Json.Null),
              "usage" -> Json.obj("output_tokens" -> n(f.timing.tokens)))))
            _ <- Http.send(event("message_stop", Json.obj("type" -> s("message_stop"))))
            _ <- Http.close
            _ <- Log.info(report(f.timing))
          yield ()
    yield ()
