# BACKLOG — okay-llm

Reference below: FreeToken (github.com/FlashML-org/FreeToken, Apache-2.0, read at 628b88f).

- **Tool-call formats of other models** (2026-11-04, when a model needs it): `Reply` reads Qwen3 JSON and Qwen3.6 XML.
  Streaming detectors for Qwen2.5/Qwen3-Coder, GLM-4.7, DeepSeek V3.2/V4.1, Gemma 4, MiniMax M2/M3, gpt-oss (harmony),
  Mistral, Llama 3.2: `python/freetoken/server/function_call_parser.py`; reasoning edge cases (prose before the first
  opener): `server/reasoning_parser.py`.
- **OpenAI Responses API** (for Codex): `server/responses_api.py` there; we speak chat completions and Anthropic only.
- **Speculative stats in usage**: drafted / accepted tokens (their `draft_n`, `draft_n_accepted`), once mlx-scala has MTP.
