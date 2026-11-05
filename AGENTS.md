# okay-llm — notes for agents

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
  `Qwen3Forward` (plain Scala, the truth), `Tokenizer`/`Nfc` (Nfc split by platform: tables on the JVM, the
  string's `normalize` on JS), `Compute` — the device as a trait (tensor, quantized, linear, rmsNorm, headNormRope,
  attention, add, swiglu, slice, rows, release) — with `OnCpu`; `Qwen3Compute` the forward over any `Compute`.
- `browser` (Scala.js, artifact okay-llm-qwen-browser): `OnWebGpu` (WGSL compute, one encoder submitted at a
  readback, released buffers destroyed after the work is done) and `OnWebGl2` (fragment shaders over textures
  2048 wide in a linear layout, words as RGBA32UI); `Selftest` is the page's main (browser/index.html).
- Laws: `QWEN_DIR=<model dir> sbt qwenJVM/test` (OnCpu == Qwen3Forward); `scripts/selftest.sh` (WebGPU and WebGL2
  against the CPU in Chromium, the model copied from the HuggingFace cache; `MODEL_DIR` to pick another).
- Shaders: `half` is reserved in GLSL ES; loops with uniform bounds are fine in ES 3.00.
