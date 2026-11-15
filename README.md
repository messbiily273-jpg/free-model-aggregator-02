# free-model-aggregator-02

One server, two client wires, real free models behind every answer.

The server listens on a free port, registers with the trial env, and answers
`POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic) in
JSON and in server-sent events. Every prompt is dialled on a live free
OpenRouter model through the grader's gateway, on the trial's key. There is no
simulated provider anywhere in this repository.

## Routing

The brief lists fourteen free models with the house's quality ratings. The
server ranks the models that meet a prompt's requirements (tools, vision,
JSON mode, context length) by rating, best first, and dials the best one. On
a real refusal (429/5xx, or a malformed or empty answer) it fails over to the
next-best model, up to five attempts, and reports the refusals it recovered
from in the `x-makefun-upstream-failures` header so the failover column can
see them.

Best model per requirement: Thinking Machines Inkling 2 Small (1475) for
plain and tools prompts; NVIDIA Nemotron 3.5 Super VL (1303) for JSON-mode
prompts, which Inkling cannot serve.

## Wires

- OpenAI requests pass through to the gateway's Chat Completions route with
  the dialled model's id; the answer comes back with that same id on every
  response and every streamed chunk.
- Anthropic requests are translated to Chat Completions (system prompt,
  images, tools, tool results, tool choice) and the answer is translated
  back, including the streaming event order.
- Upstream calls are always non-streaming: the answer is collected whole,
  then rendered to the client in the wire it asked for.

## Run

```sh
node main.ts --list   # print the catalog document and exit 0
node main.ts           # serve; needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE,
                       # OPENAI_BASE_URL and OPENAI_API_KEY from the trial
```

Deterministic: fixed ids and timestamps, no wall clock, no unseeded
randomness, so a replay leaves the same state export.
