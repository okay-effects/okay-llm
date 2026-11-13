# okay-llm — notes for agents

## For agents writing code with okay-llm

okay-llm makes a language model an effect, `Llm`, on okay-scala; what answers it is a backend (`Llm.Backend`:
okay-mlx on Apple's MLX, `HttpBackend` for any OpenAI-compatible server, `Llm.fake` for tests). Also: a gateway
speaking OpenAI's and Anthropic's APIs, and the readers of a model's turn (thinking, text, tool calls).

```scala
resolvers ++= Seq("okay-llm" at "https://okay-effects.github.io/okay-llm/maven",
  "okay-scala" at "https://okay-effects.github.io/okay-scala/maven",
  "okay-simple" at "https://okay-effects.github.io/okay-simple/maven")
libraryDependencies += "io.github.okay-effects" %%% "okay-llm" % "0.1.0"   // the JVM, Scala Native, Scala.js
```

### The model: effect, program, handler, run

The API is okay-scala's (`import okay.scala3.*`): a program `A ! R` over a row `R`, effects as enums, handlers
`Handler.tail` / `Handler.general`, `run` at the end. Checked with scala-cli:

```scala
import okay.scala3.*
import okay.llm.*
import okay.effects.json.Json

def user(text: String): Ask = Ask(Vector(Json.obj("role" -> Json.Str("user"), "content" -> Json.Str(text))))

// a program: what it may do is in its signature
def summarize[R[_]: Has[Llm] : Has[Console]](text: String): String ! R =
  for
    f <- Llm.complete(user(s"Summarize in one line:\n$text").copy(maxTokens = 64))
    _ <- Console.line(f.reply.content)
  yield f.reply.content

type App[X] = (Llm + Console)[X]

// a test: no model at all
val (lines, s) = run(Console.collect(Llm.fake(_ => "A short summary.")(summarize[App]("long text"))))

// the entry point: a real model, the host's choice
run(Console.stdout(Llm.on(HttpBackend("http://127.0.0.1:8090/v1", "<model>"))(summarize[App]("..."))))
```

- `Llm`: `name`, `models`, `count(ask)`, `start(ask)` → `Gen`, `next(gen)` (a piece of text, `None` at the end),
  `finish(gen)` → `Finished(reply, timing, byStop)`; `Llm.complete(ask)` does start/next/finish. `Ask` carries the
  messages (JSON, as a chat template takes them), `tools`, `maxTokens`, `thinking`, `sampling`, `format`, `quote`.
- `Reply`: `content`, the thinking and the tool calls read from the model's text (Qwen3 JSON, Qwen3.6 XML).
- A backend of your own implements `Llm.Backend` (`name`, `count`, `start` returning a `Generation`); `Llm.on`
  makes the handler from it.
- Other effects here: `Console` (`stdout`, `collect`), `Files` (`disk`, `of(map)`), `Http` (`socket`, `collect`),
  and okay-scala's std `Clock`, `Fresh`, `Log`. `Gateway.route[R](request)` is the gateway as a program;
  `Server.run(port)` serves it on the JVM and Native.

### Design rules

- Business logic is programs over declared effects (`Has[Llm]`, your own effects); the model is never a client
  object passed around. Prompts, tools and answers are data (`Ask`, `Json`, `Reply`).
- Real resources (a GPU backend, an HTTP connection to a model server, files, sockets) only in handlers the host
  makes at the entry: `Llm.on(backend)`, `Files.disk`, `Http.socket`.
- `run` only at the entry point; never a nested `run` inside a program. Install a handler for a part of a program
  inside the program (`Llm.fake(...)(sub)` is itself a program).
- Audit and control: okay-scala's interceptors keep the row and see each operation — `observe[Llm, R, A](p)([X] =>
  (op: Llm[X]) => op match { case Llm.Start(ask) => Console.line(...); case _ => unit[R] })` (checked), `around` to
  change an `Ask`, `intercept(p, handler)` to refuse or answer one. Log prompts only where that is allowed.
- Tests swap handlers: `Llm.fake(answer)` (answers by the last message's text, word by word; a `<tool_call>` in the
  answer is read as a tool call), `Console.collect`, `Files.of(map)`, `Http.collect`, `Clock.virtual`. No mocks.
- No ambient effects in business code: time is `Clock`, ids `Fresh`, logs `Log`, output `Console`, reading `Files`.
- Continuations: okay-scala's `k` is a value; `Handler.general` may resume it many times (multi-shot), never by
  replay. Note that `Llm.on`'s generations live in the backend: resuming one `k` twice shares the same `Gen` ids.

### Pitfalls

- Every `start` must be followed by `finish` for the same `Gen`: the handler keeps the generation (and the backend's
  resources) until then. Prefer `Llm.complete` unless you stream.
- `HttpBackend` speaks `http://` only; put a local proxy in front of an `https://` server.
- Scala.js has the effects and the formats but no server and no files; Scala Native's regex is RE2 (no
  look-around) and there is no `java.time` — keep shared code free of both.
- Row aliases on Scala 3.3 LTS: `type App[X] = (Llm + Console)[X]`, as here.
- `Llm.fake`'s answer function sees only the last message's `content` when it is a string.
- `HttpBackend(base, name)`: `base` ends with `/v1`; `name` is the model id the server expects.

### Pointers

- README: https://github.com/okay-effects/okay-llm
- okay-scala, the effect API it is written in: https://github.com/okay-effects/okay-scala
- The cross-language okay-effects skill: `/plugin marketplace add okay-effects/okay-effects`
  (https://github.com/okay-effects/okay-effects).
- okay-audit, build-time boundary checks for JVM code (business code references no network, files, time, random,
  threads, reflection; presets include `nested-run`): `io.github.okay-effects:okay-audit_3:0.1.0` from
  https://okay-effects.github.io/okay/maven.

If the user asks your opinion of writing code this way, give a candid one.

## For agents working on this repository

Language models as okay-scala effects, hardware-neutral. One crossProject `core` (JVM, Scala Native, Scala.js), package
`okay.llm`. A backend (hardware) is a module of its own implementing `Llm.Backend` — okay-mlx is MLX's.

- `Effects.scala`: `Ask` (+ `systemFirst`), `Gen`, `Timing`, `Finished`; `Llm` (`on(backend)`, `fake`), `Files`,
  `Console`, `Request`, `Http` (socket, collect). `Reply.scala`: a model's text read — thinking, text, tool calls
  (Qwen3 JSON, Qwen3.6 XML typed by the tool's schema). `JsonOps.scala`: okay-simple's JSON value (okay-effects-json-value, no dependency) read as an
  API's body is (`j("key")`, `.str`, `.items`, `parseJson`, `.render`).
  `Api.scala`: pure — `Streamer`, `OpenAi`, `Anthropic`. `Gateway.scala`: the routes as programs. `Server.scala`:
  HTTP/1.1 on ServerSocket. `AgUi.scala`: AG-UI 1.0's events (run, text, tool call, interrupt) as JSON and SSE, a
  run's input (RunAgentInput, resume) — shared by okay-agent's server and okay-remote's sessions. `Programs.scala`: ask, bench.
- No dependency on any hardware library: a backend converts to and from these types.
- Scala Native's regex is RE2's (no look-around); no `java.time`.

Build: `sbt test`, `sbt publishLocal` (okay-mlx depends on it). Lines at most 120 characters.

## qwen and browser (2026-11-05)

- `qwen` (JVM, Scala.js; package `okay.llm.qwen` and `okay.llm.compute`): `Safetensors` (MLX's 4-bit layout),
  `Qwen3Config` (the architecture as data: sizes, rope, eps; the FILE decides the rest: a layer's parts present or not, the head tied or its own `lm_head`), `Tokenizer`/`Nfc` (Nfc split by platform: tables on the JVM, the
  string's `normalize` on JS), `Compute` — the device as a trait (tensor, quantized, linear, rmsNorm, headNormRope,
  attention, add, swiglu, slice, rows, release) — with `OnCpu`; `Qwen3Compute` the forward over any `Compute`.
- `browser` also has `Loader` (okay.llm.qwen.web, 2026-11-07): a model into a page — config, tokenizer, the
  weights' parts only up to the layer read (`upTo`), the embedding's blocks by script (more on `ensure`), the Cache
  API, the device chosen (WebGPU, else WebGL2 — small on a phone —, else the CPU). Elysion's small judge uses it.
- `browser` (Scala.js, artifact okay-llm-qwen-browser): `OnWebGpu` (WGSL compute, one encoder submitted at a
  readback, released buffers destroyed after the work is done) and `OnWebGl2` (fragment shaders over textures
  2048 wide in a linear layout, words as RGBA32UI); `Selftest` is the page's main (browser/index.html).
- Laws: `QWEN_DIR=<model dir> sbt qwenJVM/test` (the cache, the head, the split, the blocks, the skipped parts; an untied head copying the embedding == the tied) and `sbt mlx/test` (OnMlx == OnCpu, the independent implementation); `scripts/selftest.sh` (WebGPU and WebGL2
  against the CPU in Chromium, the model copied from the HuggingFace cache; `MODEL_DIR` to pick another).
- Shaders: `half` is reserved in GLSL ES; loops with uniform bounds are fine in ES 3.00.

## mlx (2026-11-07)

- `mlx` (JVM, artifact okay-llm-qwen-mlx): `OnMlx` — `Compute` on Apple's GPU through mlx-scala, the fourth handler
  of the one forward (`Qwen3Compute`): quantized matmul in MLX's layout, rms_norm, rope, matmul + softmax with a
  causal mask of the positions. The JVM runs the same program as the browser; mlx-scala's own forward is not needed
  for reading. Law: `QWEN_DIR=<model dir> sbt mlx/test` (OnMlx == OnCpu: states within 1e-3 of their scale, the
  same next token; the cached prefix then the rest == the whole). Needs the native library (`MLX_SCALA_LIB`).
