# okay-llm

Language models as effects ([okay-scala](https://github.com/okay-effects/okay-scala)), on no hardware in particular:
the model is an effect, `Llm`, and what answers it is a module of its own — [okay-mlx](https://github.com/okay-effects/okay-mlx)
on Apple's MLX; another backend is another module. On it: a gateway speaking OpenAI's and Anthropic's APIs, and the
programs ask and bench. The JVM and Scala Native; JSON is okay-simple's value alone
(okay-effects-json-value, no dependency).

```scala
def ask[R[_]: Has[Llm] : Has[Console]](question: String, maxTokens: Int): Finished ! R =
  for
    f <- Llm.complete(user(question).copy(maxTokens = maxTokens))
    _ <- Console.line(f.reply.content)
  yield f

run(Console.stdout(Llm.on(backend)(ask[App]("What is a KV cache?", 256))))   // any Llm.Backend
run(Console.collect(Llm.fake(_ => "Paris.")(ask[App]("Capital of France?", 16))))   // no model at all
```

## A backend

Hardware is a `Llm.Backend` — a prompt's size, and a generation as pieces of text — and `Llm.on(backend)` makes the
handler: generations by id, each turn read at its end (`Reply.parse`: thinking, text, tool calls in Qwen3's JSON or
Qwen3.6's XML).

```scala
trait Backend:
  def name: String
  def count(ask: Ask): Int
  def start(ask: Ask): Generation       // next(): Option[String]; text; timing; byStop; close()
```

## Effects

| effect | operations | handlers |
|---|---|---|
| `Llm` | `name`, `count`, `start`, `next` (a piece of text), `finish` (the turn read) | `on(backend)`, `fake` |
| `Http` | `respond`; `open`, `send`, `close` (a stream) | `socket`, `collect` |
| `Files` | `read` | `disk`, `of` |
| `Console` | `line` | `stdout`, `collect` |
| std `Clock`, `Fresh`, `Log` | | `system` / `virtual`, `from`, `to` / `collect` |

## The gateway

`Gateway.route(request)` — programs of those effects; `Server.run(port)` serves them, HTTP/1.1 on
`java.net.ServerSocket` (the JVM and Scala Native alike), a connection a thread: requests at once, a backend
serializing what it must. A request's `temperature`, `top_p`, `top_k`, `seed` reach the backend (`Ask.sampling`).

| endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI: whole or streamed (server-sent events), tool calls, `reasoning_content`, usage |
| `POST /v1/messages` | Anthropic: whole or streamed — `thinking`, `text`, `tool_use` blocks |
| `POST /v1/messages/count_tokens`, `GET /v1/models`, `GET /health` | |

Claude Code's billing-header system block (a hash new each request) is dropped: a prompt that differs from its first
tokens has no prefix for a backend to reuse. A turn streams through a `Streamer`: thinking, then text; a tool call held back to the end. A system message a
client sends mid-conversation (Claude Code does, each turn) stays where it is, as a user's: most chat templates take a
system message first only, and moving it there would change the prompt's beginning each turn.

Checked: every program under fake handlers (13 tests, both platforms); Claude Code against it on okay-mlx — a 13k-token
prompt with its tools, Qwen3.6-35B-A3B calling `Write` to make a file.
