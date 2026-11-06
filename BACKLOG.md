# BACKLOG — okay-llm

Reference below: FreeToken (github.com/FlashML-org/FreeToken, Apache-2.0, read at 628b88f).

- **Tool-call formats of other models** (2026-11-04, when a model needs it): `Reply` reads Qwen3 JSON and Qwen3.6 XML.
  Streaming detectors for Qwen2.5/Qwen3-Coder, GLM-4.7, DeepSeek V3.2/V4.1, Gemma 4, MiniMax M2/M3, gpt-oss (harmony),
  Mistral, Llama 3.2: `python/freetoken/server/function_call_parser.py`; reasoning edge cases (prose before the first
  opener): `server/reasoning_parser.py`.
- **OpenAI Responses API** (for Codex): `server/responses_api.py` there; we speak chat completions and Anthropic only.
- **Speculative stats in usage**: drafted / accepted tokens (their `draft_n`, `draft_n_accepted`), once mlx-scala has MTP.

## The browser engine (operator, 2026-11-05)

- **DONE 2026-11-05**: `Compute` (qwen module, package okay.llm.compute) with `OnCpu`; `OnWebGpu` and `OnWebGl2` in the
  `browser` module; `Qwen3Compute` the forward over it. Laws: `ComputeSuite` (OnCpu == Qwen3Forward, exact) and
  scripts/selftest.sh (WebGPU diff 6e-5, WebGL2 8e-3 on states up to 7942; 16 tokens: CPU 11 s, WebGPU 0.1 s,
  WebGL2 0.4 s on an M4 Max). Open: speed on long prompts (a thread per output element, no tiling; the scores
- **2026-11-06**: the prefix cache — `Qwen3Compute.run(ids, layer, at, logits, cache, keep)`: k and v per layer and the
  residual at the layer kept on the device, a sequence beginning with the cached one runs only the rest (`pos0` in
  RoPE and the causal mask, `concat`); exact on all three (CacheSuite, the self-test). WebGL2: a texture pool (a
  phone's driver freed deleted textures late; the tab died). Safetensors over a ByteBuffer view, the file not kept.
  buffer n·heads·n), f16 where the device has it, the model fetched once into the browser cache.
- **Shared with okay-go-game** (the operator): the same `Compute` layer runs the Go net too — a convolution added
  as one more op; the game's GpuNet / GlNet become clients of it rather than their own kernels.
