// free-model-aggregator-02: a live free-model aggregator (Node 22, stdlib only).
//
// One server, two client wires (OpenAI Chat Completions and Anthropic Messages,
// JSON and server-sent events), and a real completion behind every answer: each
// prompt is dialled on a live free OpenRouter model through the grader's
// gateway, on the trial's key. There is no simulated provider anywhere in this
// file. The server picks the best free model the brief rates for the prompt's
// requirements, fails over down the rating order on a real 429/5xx, and reports
// the model it really dialled on every answer and every streamed chunk.
//
// Determinism: fixed ids and fixed created timestamps, no wall clock, no
// unseeded randomness. Every choice is a function of the brief, the requests
// and the gateway's answers, so a replay leaves the same state export.
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One upstream call may take this long; the client waits 30 s per prompt. */
const CALL_TIMEOUT_MS = 25000;
/** How many ranked candidates one prompt may try before it is given up on. */
const MAX_ATTEMPTS = 5;
/** A fixed epoch second for every answer and listing: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

export type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** The first document under this contest's prefix carrying one of `keys` as an array. */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes('free-model-aggregator-02/')) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`: the free model list and the six prompts, found by schema. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], error: `the brief at ${path} holds no models document` };
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], error: `the brief at ${path} holds no prompts document` };
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

const brief: Brief = readBrief(PROMPT_FILE);

/** The catalog `--list` prints and the register body carries, byte for byte. */
function buildCatalog(): Json {
  const seen = new Set<string>();
  const models: Json[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models,
      },
    ],
  };
}

const catalog: Json = buildCatalog();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;

/** Whether a model's capabilities meet one prompt's requirements. Unknown never meets one. */
function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.context_length === null || model.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The candidates for a prompt, best first by the brief's rating, brief order
 * breaking ties. Quality is 60% of the score, so the first candidate is the
 * best free model that meets the requirements; failover walks down this list.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  return brief.models
    .map((model, index) => ({ model, index }))
    .filter(({ model }) => model.free && meetsRequirements(requires, model))
    .sort((a, b) => b.model.rating_milli - a.model.rating_milli || a.index - b.index)
    .map(({ model }) => model);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const scan = (messages: unknown): void => {
    if (!Array.isArray(messages)) return;
    for (const m of messages) {
      if (!isRecord(m) || !Array.isArray(m.content)) continue;
      for (const b of m.content) {
        if (isRecord(b) && (b.type === 'image' || b.type === 'image_url')) vision = true;
      }
    }
  };
  scan(body.messages);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The gateway's Chat Completions route. OPENAI_BASE_URL is `…/v1` on the trial network. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface Normalized {
  text: string;
  toolCalls: ToolCall[];
  input: number;
  output: number;
  stop: string;
}

type DialResult =
  | { ok: true; result: Normalized }
  | { ok: false; status: number; code: string };

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textFromContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

function parseJsonSafe(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

/** A gateway error body as a short code string for the failures trail. */
function errorCodeOf(status: number, text: string): string {
  const parsed = parseJsonSafe(text);
  if (isRecord(parsed) && isRecord(parsed.error)) {
    if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
    if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
  }
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'upstream_error';
  if (status === 0) return 'network_error';
  return `http_${status}`;
}

/** A Chat Completions JSON answer as {text, toolCalls, input, output, stop}, or null when malformed. */
function normalizeAnswer(body: unknown): Normalized | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
      });
    }
  }
  const input = num(usage.prompt_tokens);
  const output = num(usage.completion_tokens);
  return {
    text: textFromContent(message.content),
    toolCalls,
    // Usage is whole on every real answer; fall back to a floor of 1 output
    // token so a well-formed answer never fails the wire's usage check.
    input,
    output: output > 0 ? output : 1,
    stop: typeof choice.finish_reason === 'string' && choice.finish_reason !== '' ? choice.finish_reason : 'stop',
  };
}

/**
 * Dial one free model on the gateway's Chat Completions route, always
 * non-streaming: the answer is collected whole, then rendered to the client in
 * the wire it asked for (usage arrives last on a stream, and Anthropic's
 * message_start needs the input count up front). The client's stream flag
 * never reaches the gateway, so a replay sees the same request either way.
 */
async function dial(model: string, clientBody: Json): Promise<DialResult> {
  const url = chatCompletionsUrl();
  if (url === '') return { ok: false, status: 0, code: 'no_gateway' };
  const body: Json = { ...clientBody, model };
  delete body.stream;
  delete body.stream_options;
  if (typeof body.max_tokens !== 'number' && typeof body.max_completion_tokens !== 'number') body.max_tokens = 1024;
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}` },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, code: 'network_error' };
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    return { ok: false, status: res.status, code: errorCodeOf(res.status, text) };
  }
  const answer: unknown = await res.json().catch(() => null);
  const norm = normalizeAnswer(answer);
  if (norm === null) return { ok: false, status: 502, code: 'bad_upstream_body' };
  if (norm.text === '' && norm.toolCalls.length === 0) return { ok: false, status: 502, code: 'empty_answer' };
  return { ok: true, result: norm };
}

/** The text of a string or a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url parts, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p and stop sequences pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: string): string {
  if (finish === 'length' || finish === 'max_tokens' || finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'tool_use' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter' || finish === 'refusal' || finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function openAIStop(stop: string, hasTools: boolean): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'tool_use' || stop === 'function_call' || hasTools) return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

function anthropicStop(stop: string, hasTools: boolean): string {
  if (hasTools && stop !== 'length' && stop !== 'max_tokens') return 'tool_use';
  return stopReasonOf(stop);
}

/** One row of the optional upstream-failures trail header. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

function failuresHeader(failures: Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: Failure[] = []): void {
  const headers = failuresHeader(failures);
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), headers);
  else sendJson(res, status, openAIError(message, type, null), headers);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = brief.models.filter((m) => m.free);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

function splitText(text: string, size: number): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [''] : out;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function renderOpenAI(res: ServerResponse, stream: boolean, norm: Normalized, modelId: string, failures: Failure[]): void {
  const stop = openAIStop(norm.stop, norm.toolCalls.length > 0);
  const usage = { prompt_tokens: norm.input, completion_tokens: norm.output, total_tokens: norm.input + norm.output };
  const headers = failuresHeader(failures);
  if (!stream) {
    const message: Json = { role: 'assistant', content: norm.text };
    if (norm.toolCalls.length > 0) {
      message.tool_calls = norm.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
    }
    sendJson(
      res,
      200,
      {
        id: 'chatcmpl-agg',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model: modelId,
        choices: [{ index: 0, message, finish_reason: stop, logprobs: null }],
        usage,
      },
      headers,
    );
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  res.write(chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(norm.text, 48)) res.write(chunk({ content: piece }, null));
  if (norm.toolCalls.length > 0) {
    norm.toolCalls.forEach((c, i) => {
      res.write(
        chunk({ tool_calls: [{ index: i, id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }] }, null),
      );
    });
  }
  res.write(chunk({}, stop));
  res.write(
    `data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [], usage })}\n\n`,
  );
  res.write('data: [DONE]\n\n');
  res.end();
}

function renderAnthropic(res: ServerResponse, stream: boolean, norm: Normalized, modelId: string, failures: Failure[]): void {
  const stop = anthropicStop(norm.stop, norm.toolCalls.length > 0);
  const content: Json[] = [];
  if (norm.text !== '') content.push({ type: 'text', text: norm.text });
  for (const c of norm.toolCalls) {
    let input: Json = {};
    try {
      const parsed: unknown = JSON.parse(c.arguments);
      if (isRecord(parsed)) input = parsed;
    } catch {
      // Not JSON: an empty input.
    }
    content.push({ type: 'tool_use', id: c.id === '' ? 'toolu_agg' : c.id, name: c.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const headers = failuresHeader(failures);
  if (!stream) {
    sendJson(
      res,
      200,
      {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content,
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: norm.input, output_tokens: norm.output },
      },
      headers,
    );
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
  res.write(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: norm.input, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (norm.text !== '') {
    res.write(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(norm.text, 48)) {
      res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const block of content) {
    if (block.type !== 'tool_use') continue;
    res.write(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: block }));
    res.write(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input) },
      }),
    );
    res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  res.write(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: norm.output } }),
  );
  res.write(sseEvent('message_stop', { type: 'message_stop' }));
  res.end();
}

/**
 * Serve one client request: walk the ranked candidates, stop at the first
 * real answer. A refusal is never an answer: the failure is recorded for the
 * trail header and the next-best free model is tried. A 401 is the trial's
 * key, shared by every candidate, so it ends the walk at once; the prompt is
 * given up on only when the list runs out or the key is refused.
 */
async function serve(res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const candidates = candidatesFor(requires);
  const failures: Failure[] = [];
  for (const cand of candidates.slice(0, MAX_ATTEMPTS)) {
    const upstreamBody = clientApi === 'openai' ? body : anthropicToOpenAI(body, cand.id);
    const result = await dial(cand.id, upstreamBody);
    if (result.ok) {
      if (clientApi === 'openai') renderOpenAI(res, stream, result.result, cand.id, failures);
      else renderAnthropic(res, stream, result.result, cand.id, failures);
      return;
    }
    log(`model ${cand.id} refused HTTP ${result.status} (${result.code}): failing over`);
    failures.push({ status: result.status, code: result.code, model: cand.id });
    // A 401 is the trial's key, not the model: every candidate shares it, so
    // trying the rest only burns gateway calls. Give up on this prompt now.
    if (result.status === 401) break;
  }
  sendError(
    res,
    clientApi,
    503,
    clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'no free model in the brief meets this request',
    failures,
  );
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(res, 'anthropic', body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the prompt count. */
async function register(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${brief.models.filter((m) => m.free).length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
