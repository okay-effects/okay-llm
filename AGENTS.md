# okay-llm — notes for agents

Language models as okay-scala effects, hardware-neutral. One crossProject `core` (JVM, Scala Native), package
`okay.llm`. A backend (hardware) is a module of its own implementing `Llm.Backend` — okay-mlx is MLX's.

- `Effects.scala`: `Ask` (+ `systemFirst`), `Gen`, `Timing`, `Finished`; `Llm` (`on(backend)`, `fake`), `Files`,
  `Console`, `Request`, `Http` (socket, collect). `Reply.scala`: a model's text read — thinking, text, tool calls
  (Qwen3 JSON, Qwen3.6 XML typed by the tool's schema). `JsonOps.scala`: okay-simple's JSON (okay-effects-json) read as an
  API's body is (`j("key")`, `.str`, `.items`, `parseJson`, `.render`).
  `Api.scala`: pure — `Streamer`, `OpenAi`, `Anthropic`. `Gateway.scala`: the routes as programs. `Server.scala`:
  HTTP/1.1 on ServerSocket. `Programs.scala`: ask, bench.
- No dependency on any hardware library: a backend converts to and from these types.
- Scala Native's regex is RE2's (no look-around); no `java.time`.

Build: `sbt test`, `sbt publishLocal` (okay-mlx depends on it). Lines at most 120 characters.
