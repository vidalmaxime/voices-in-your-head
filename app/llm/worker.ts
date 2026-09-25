import { createModelLoadProfile } from "./model-loading";
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  DynamicCache,
  TextStreamer,
  Tensor,
  InterruptableStoppingCriteria,
  StoppingCriteria,
  StoppingCriteriaList,
  env,
} from "@huggingface/transformers";

import { assetUrl } from "../base-path";
import { analyzeThoughtCompletion, buildCompletionMessages } from "./completion-logic";
import type { ContextTurn } from "./conversation-context";
import { normalizeResponseTokenLimit } from "./response-settings";
import { CompletionQueue } from "./completion-queue";
import { getCompletionCachePrefix } from "./completion-prefix";

export interface WorkerMessage {
  type: "check" | "load" | "generate" | "generateCompletion" | "generateBranches" | "interrupt" | "reset" | "setModel" | "keepWarm";
  data?: {
    messages: Array<{ role: string; content: string }>;
    reasonEnabled?: boolean;
    partialText?: string;
    context?: ContextTurn[];
    numBranches?: number;
    maxResponseTokens?: number;
    usePrefixCache?: boolean;
    /** Echoed on start/update/complete so the page can route speculative runs. */
    requestId?: number;
    /** Overrides the completion model for a benchmark run. */
    completionModel?: { modelId?: string; dtype?: string; externalData?: boolean; fewShot?: boolean; local?: boolean; maxResponseTokens?: number };
    /** Page clock (epoch ms) when the request was posted, for the profile. */
    postedAtEpoch?: number;
    /** Stop decoding at the first sentence end (default true). */
    earlyStop?: boolean;
  };
}

/** Where the time went in one completion, in ms from the worker receiving it. */
export interface CompletionTimings {
  /** Page post to worker receipt (epoch clocks), when the page sent one. */
  queuedMs: number | null;
  /** Prompt rendered and tokenized, prefix cache applied. */
  inputsMs: number;
  /** First sampled token. */
  firstTokenMs: number | null;
  /** Each token after the first, as absolute marks. */
  tokenMs: number[];
  /** `generate` returned. */
  generateMs: number;
  /** Output decoded and the complete message posted. */
  completeMs: number;
  stopReason: "eos" | "finished" | "max" | "interrupted";
}

export interface WorkerResponse {
  status: "error" | "loading" | "initiate" | "progress" | "done" | "ready" | "start" | "update" | "complete" | "branchComplete";
  data?: string;
  output?: string;
  tps?: number;
  numTokens?: number;
  firstTokenMs?: number;
  prefixCacheUsed?: boolean;
  state?: "thinking" | "answering";
  file?: string;
  progress?: number;
  total?: number;
  branchId?: number;
  branchComplete?: boolean;
  timings?: CompletionTimings;
}

declare const navigator: Navigator & { gpu?: { requestAdapter(): Promise<unknown> } };

async function check() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU is not supported (no adapter found)");
    }
  } catch (e) {
    self.postMessage({
      status: "error",
      data: String(e),
    });
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyModel = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyTokenizer = any;

// The completion model is swappable so candidates can be compared through the
// debug harness without a rebuild (see README.md, "Debugging and measurements"). onnx-community exports are external-data by default; small
// single-file exports set externalData=false.
export const DEFAULT_COMPLETION_MODEL = {
  modelId: "onnx-community/granite-4.0-1b-ONNX",
  dtype: "q4",
  externalData: true,
  /** Finetuned task models are trained without the few-shot examples. */
  fewShot: true,
  /** Served from public/models/ instead of the Hugging Face Hub. */
  local: false,
  /** Output cap for requests that do not set one; matches the SFT target length. */
  maxResponseTokens: 16,
};

type CompletionModelConfig = {
  modelId: string;
  dtype: string;
  externalData: boolean;
  fewShot: boolean;
  local: boolean;
  maxResponseTokens: number;
};

/**
 * Removes the training-only `{% generation %}` / `{% endgeneration %}` markers
 * from a chat template. The LFM2.5 template writes them with whitespace
 * control (`{%- generation -%}`), which the jinja copy bundled in
 * transformers.js 4.2.0 does not strip, so compiling it throws
 * "Unknown statement type: generation". Same rewrite as @huggingface/jinja
 * 0.5.9; a no-op for templates without the markers.
 */
function stripGenerationMarkers(tokenizer: AnyTokenizer) {
  if (typeof tokenizer.chat_template === "string") {
    tokenizer.chat_template = tokenizer.chat_template.replace(
      /(\s*){%(-?)\s*(?:end)?generation\s*(-?)%}(\s*)/g,
      (_: string, before: string, lstrip: string, rstrip: string, after: string) =>
        (lstrip ? "" : before) + (rstrip ? "" : after),
    );
  }
  return tokenizer;
}

class TextGenerationPipeline {
  static config: CompletionModelConfig = { ...DEFAULT_COMPLETION_MODEL };
  static tokenizer: Promise<AnyTokenizer> | null = null;
  static model: Promise<AnyModel> | null = null;

  /** Selects the model for the next load. Ignored once a model is resident. */
  static configure(config?: Partial<CompletionModelConfig>) {
    if (!config || this.model) return;
    this.config = { ...DEFAULT_COMPLETION_MODEL, ...config };
  }

  /** Disposes the resident model so a different one can load without leaking GPU memory. */
  static async reset() {
    const model = this.model;
    this.model = null;
    this.tokenizer = null;
    try {
      const resolved = await model;
      await (resolved as { dispose?: () => Promise<void> } | null)?.dispose?.();
    } catch {
      // A model that never finished loading has nothing to dispose.
    }
  }

  static async getInstance(progress_callback?: (x: unknown) => void) {
    const { modelId, dtype, externalData, local } = this.config;
    // A personal finetune lives in public/models/ and must never be fetched
    // from (or leak its name to) the Hub; hub models keep the remote path.
    env.allowLocalModels = local;
    env.allowRemoteModels = !local;
    // The Cache API cannot parse relative request keys from this worker's
    // blob: base either, so local mode relies on ordinary HTTP caching.
    env.useBrowserCache = !local;
    if (local) {
      env.localModelPath = assetUrl("/models/");
      // Transformers.js only takes its local-file path for non-absolute URLs,
      // but this worker runs from a blob: base in dev, where fetch cannot
      // resolve a root-relative path. Resolve against the page origin here.
      env.fetch = (input: RequestInfo | URL, init?: RequestInit) =>
        fetch(
          typeof input === "string" && input.startsWith("/")
            ? new URL(input, self.location.origin)
            : input,
          init,
        );
    }
    this.tokenizer ??= AutoTokenizer.from_pretrained(modelId, {
      progress_callback,
    }).then(stripGenerationMarkers);

    this.model ??= AutoModelForCausalLM.from_pretrained(modelId, {
      dtype,
      device: "webgpu",
      use_external_data_format: externalData,
      progress_callback,
    } as Record<string, unknown>);

    return Promise.all([this.tokenizer, this.model]);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();

/**
 * Stops decoding once the sanitizer would keep nothing more: the first
 * sentence has ended or a role label began. The text spoken is identical to
 * letting the run hit EOS or the token cap; only the discarded tokens are
 * never generated. Applies the same cleanup as the sanitizer so a repeated
 * boundary word with a period is not mistaken for an end.
 */
class CompletionFinishedCriteria extends StoppingCriteria {
  triggered = false;

  constructor(
    private readonly tokenizer: AnyTokenizer,
    private readonly fragment: string,
    private readonly promptLength: number,
    private readonly minNewTokens: number,
  ) {
    super();
  }

  // The runtime passes bigint ids; the type declaration says number.
  _call(input_ids: number[][] | bigint[][]) {
    return (input_ids as Array<Array<number | bigint>>).map((ids) => {
      const generated = ids.slice(this.promptLength);
      if (generated.length < this.minNewTokens) return false;
      const text = this.tokenizer.decode(generated, { skip_special_tokens: true }) as string;
      const finished = analyzeThoughtCompletion(text, this.fragment).finished;
      if (finished) this.triggered = true;
      return finished;
    });
  }
}

function nowEpoch() {
  return performance.timeOrigin + performance.now();
}

// Completions run one at a time. A request that arrives while an interrupted
// run is still winding down waits for it instead of sharing the session, and
// the interrupt stays in force until that run has exited.
const completionQueue = new CompletionQueue();

let past_key_values_cache: unknown = null;
let completionPrefixCache: InstanceType<typeof DynamicCache> | null = null;
let completionPrefixTokenIds: bigint[] = [];
let completionPrefixPromise: Promise<boolean> | null = null;
let completionPrefixUnavailable = false;

function getCompletionInputs(tokenizer: AnyTokenizer, partialText: string, context?: ContextTurn[]) {
  const messages = buildCompletionMessages(partialText, {
    includeExamples: TextGenerationPipeline.config.fewShot,
    context,
  });
  return tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: false,
  } as Record<string, unknown>) as Record<string, Tensor>;
}

function getInputTokenIds(inputs: Record<string, Tensor>) {
  return Array.from(inputs.input_ids.data as BigInt64Array);
}

function presentToPastName(name: string) {
  return name
    .replace("present_ssm", "past_ssm")
    .replace("present_conv", "past_conv")
    .replace("present_recurrent", "past_recurrent")
    .replace("present", "past_key_values");
}

async function buildCompletionPrefixCache(tokenizer: AnyTokenizer, model: AnyModel) {
  if (completionPrefixCache) return true;
  if (completionPrefixUnavailable) return false;
  if (completionPrefixPromise) return completionPrefixPromise;

  completionPrefixPromise = (async () => {
    try {
      completionPrefixTokenIds = getCompletionCachePrefix((fragment, context) =>
        getInputTokenIds(getCompletionInputs(tokenizer, fragment, context)),
      );
      const prefixLength = completionPrefixTokenIds.length;
      const inputIds = new Tensor(
        "int64",
        BigInt64Array.from(completionPrefixTokenIds),
        [1, prefixLength],
      );
      const attentionMask = new Tensor(
        "int64",
        new BigInt64Array(prefixLength).fill(BigInt(1)),
        [1, prefixLength],
      );
      const outputs = await model.forward({
        input_ids: inputIds,
        attention_mask: attentionMask,
        num_logits_to_keep: new Tensor("int64", [BigInt(1)], []),
      });

      const entries: Record<string, Tensor> = Object.create(null);
      for (const [name, tensor] of Object.entries(outputs) as Array<[string, Tensor]>) {
        if (name.startsWith("present")) {
          entries[presentToPastName(name)] = tensor;
        }
      }
      if (Object.keys(entries).length === 0) {
        throw new Error("Model did not return a reusable completion cache");
      }

      completionPrefixCache = new DynamicCache(entries);
      const cachedTensors = new Set(Object.values(entries));
      for (const tensor of Object.values(outputs) as Tensor[]) {
        if (tensor.location === "gpu-buffer" && !cachedTensors.has(tensor)) {
          tensor.dispose();
        }
      }
      return true;
    } catch (error) {
      completionPrefixTokenIds = [];
      completionPrefixUnavailable = true;
      console.warn("Completion prefix cache disabled:", error);
      return false;
    } finally {
      completionPrefixPromise = null;
    }
  })();

  return completionPrefixPromise;
}

function createBorrowedCompletionCache() {
  if (!completionPrefixCache) return null;

  const entries = Object.fromEntries(Object.entries(completionPrefixCache)) as Record<string, Tensor>;
  const borrowedTensors = new Set(Object.values(entries));
  const runCache = new DynamicCache(entries);

  // Generation updates its cache in place. Preserve the shared prefix tensors on
  // the first update, while still disposing every run-specific replacement.
  Object.defineProperty(runCache, "update", {
    enumerable: false,
    value(newEntries: Record<string, Tensor>) {
      const mutableCache = runCache as unknown as Record<string, Tensor>;
      for (const [key, newValue] of Object.entries(newEntries)) {
        const oldValue = mutableCache[key];
        if (
          oldValue &&
          oldValue !== newValue &&
          !borrowedTensors.has(oldValue) &&
          oldValue.location === "gpu-buffer"
        ) {
          oldValue.dispose();
        }
        mutableCache[key] = newValue;
      }
    },
  });

  return runCache;
}

function applyCompletionPrefixCache(inputs: Record<string, Tensor>) {
  if (!completionPrefixCache || completionPrefixTokenIds.length === 0) {
    return { inputs, runCache: null };
  }

  const fullIds = getInputTokenIds(inputs);
  const prefixMatches = completionPrefixTokenIds.every((token, index) => fullIds[index] === token);
  if (!prefixMatches || fullIds.length <= completionPrefixTokenIds.length) {
    return { inputs, runCache: null };
  }

  const suffixIds = fullIds.slice(completionPrefixTokenIds.length);
  const runCache = createBorrowedCompletionCache();
  return {
    inputs: {
      ...inputs,
      input_ids: new Tensor("int64", BigInt64Array.from(suffixIds), [1, suffixIds.length]),
      past_key_values: runCache,
    },
    runCache,
  };
}

async function disposeGenerationCache(result?: { past_key_values?: { dispose?: () => Promise<void> } }) {
  await result?.past_key_values?.dispose?.();
}

async function generate({ messages, reasonEnabled }: { messages: Array<{ role: string; content: string }>; reasonEnabled?: boolean }) {
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: reasonEnabled,
  } as Record<string, unknown>);

  const thinkTokens = tokenizer.encode("<think></think>", { add_special_tokens: false }) as number[];
  const START_THINKING_TOKEN_ID = thinkTokens[0];
  const END_THINKING_TOKEN_ID = thinkTokens[1];

  let state: "thinking" | "answering" = "answering";
  let startTime: number | null = null;
  let numTokens = 0;
  let tps = 0;

  const token_callback_function = (tokens: bigint[]) => {
    startTime ??= performance.now();

    if (numTokens++ > 0) {
      tps = (numTokens / (performance.now() - startTime)) * 1000;
    }
    const tokenId = Number(tokens[0]);
    switch (tokenId) {
      case START_THINKING_TOKEN_ID:
        state = "thinking";
        break;
      case END_THINKING_TOKEN_ID:
        state = "answering";
        break;
    }
  };

  const callback_function = (output: string) => {
    self.postMessage({
      status: "update",
      output,
      tps,
      numTokens,
      state,
    });
  };

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function,
    token_callback_function,
  });

  self.postMessage({ status: "start" });

  const generateOptions = {
    ...(inputs as Record<string, unknown>),
    past_key_values: past_key_values_cache,
    do_sample: true,
    top_k: 20,
    temperature: reasonEnabled ? 0.6 : 0.7,
    max_new_tokens: 160,
    streamer,
    stopping_criteria,
    return_dict_in_generate: true,
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let result: any;
  let decoded: string[];
  try {
    result = await model.generate(generateOptions);
    decoded = tokenizer.batch_decode(result.sequences, {
      skip_special_tokens: true,
    }) as string[];
  } finally {
    // A failed or interrupted run must not leave a full conversation cache resident.
    past_key_values_cache = null;
    await disposeGenerationCache(result);
  }

  self.postMessage({
    status: "complete",
    output: decoded[0],
  });
}

async function generateCompletion({
  partialText,
  context,
  usePrefixCache = true,
  requestId,
  postedAtEpoch,
  earlyStop = true,
  maxResponseTokens = TextGenerationPipeline.config.maxResponseTokens,
}: {
  partialText: string;
  context?: ContextTurn[];
  usePrefixCache?: boolean;
  requestId?: number;
  postedAtEpoch?: number;
  earlyStop?: boolean;
  maxResponseTokens?: number;
}) {
  const receivedAt = performance.now();
  const queuedMs = postedAtEpoch === undefined ? null : Math.max(0, nowEpoch() - postedAtEpoch);
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();

  await buildCompletionPrefixCache(tokenizer, model);
  const fullInputs = getCompletionInputs(tokenizer, partialText, context);
  const { inputs, runCache } = usePrefixCache
    ? applyCompletionPrefixCache(fullInputs)
    : { inputs: fullInputs, runCache: null };
  const prefixCacheUsed = runCache !== null;
  const inputsMs = performance.now() - receivedAt;

  let startTime: number | null = null;
  const generationStartedAt = performance.now();
  let firstTokenMs: number | null = null;
  const tokenMs: number[] = [];
  let numTokens = 0;
  let tps = 0;

  const token_callback_function = () => {
    if (startTime === null) {
      startTime = performance.now();
      firstTokenMs = startTime - generationStartedAt;
    } else {
      tokenMs.push(performance.now() - receivedAt);
    }
    if (numTokens++ > 0) {
      tps = (numTokens / (performance.now() - startTime)) * 1000;
    }
  };

  const callback_function = (output: string) => {
    self.postMessage({
      status: "update",
      output,
      tps,
      numTokens,
      state: "answering",
      requestId,
    });
  };

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function,
    token_callback_function,
  });

  self.postMessage({ status: "start", requestId });

  const MAX_NEW_TOKENS = normalizeResponseTokenLimit(maxResponseTokens);
  const MIN_NEW_TOKENS = Math.min(4, MAX_NEW_TOKENS);
  const finishedCriteria = new CompletionFinishedCriteria(
    tokenizer,
    partialText,
    (inputs.input_ids as Tensor).dims[1],
    MIN_NEW_TOKENS,
  );
  const criteria = new StoppingCriteriaList();
  criteria.extend(earlyStop ? [stopping_criteria, finishedCriteria] : [stopping_criteria]);

  const generateOptions = {
    ...(inputs as Record<string, unknown>),
    do_sample: true,
    top_k: 24,
    top_p: 0.9,
    temperature: 0.62,
    repetition_penalty: 1.08,
    min_new_tokens: MIN_NEW_TOKENS,
    max_new_tokens: MAX_NEW_TOKENS,
    streamer,
    stopping_criteria: criteria,
    return_dict_in_generate: true,
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let result: any;
  try {
    result = await model.generate(generateOptions);
    const generateMs = performance.now() - receivedAt;
    const decoded = tokenizer.batch_decode(result.sequences, {
      skip_special_tokens: true,
    }) as string[];
    const stopReason: CompletionTimings["stopReason"] = stopping_criteria.interrupted
      ? "interrupted"
      : finishedCriteria.triggered
        ? "finished"
        : numTokens >= MAX_NEW_TOKENS
          ? "max"
          : "eos";

    // The page can start speaking now; freeing this run's cache can wait.
    self.postMessage({
      status: "complete",
      output: decoded[0],
      firstTokenMs,
      numTokens,
      tps,
      prefixCacheUsed,
      requestId,
      timings: {
        queuedMs,
        inputsMs,
        firstTokenMs: firstTokenMs === null ? null : firstTokenMs + inputsMs,
        tokenMs,
        generateMs,
        completeMs: performance.now() - receivedAt,
        stopReason,
      } satisfies CompletionTimings,
    });
  } catch (error) {
    self.postMessage({
      status: "error",
      data: error instanceof Error ? error.message : String(error),
      requestId,
    });
  } finally {
    await disposeGenerationCache(result ?? { past_key_values: runCache });
  }
}

/**
 * One token against the cached prefix. The first completion after a minute
 * or two of idleness paid ~800 ms to its first token against ~250 ms when
 * runs were seconds apart; touching the weights regularly keeps them hot.
 */
async function keepWarm() {
  if (!completionPrefixCache) return;
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();
  const warm = applyCompletionPrefixCache(getCompletionInputs(tokenizer, "so"));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let result: any;
  try {
    result = await model.generate({
      ...warm.inputs,
      do_sample: false,
      max_new_tokens: 1,
      return_dict_in_generate: true,
    });
  } finally {
    await disposeGenerationCache(result ?? { past_key_values: warm.runCache });
  }
}

async function generateBranches({ partialText, numBranches = 3, context, maxResponseTokens = TextGenerationPipeline.config.maxResponseTokens }: { partialText: string; numBranches?: number; context?: ContextTurn[]; maxResponseTokens?: number }) {
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();
  await buildCompletionPrefixCache(tokenizer, model);
  const fullInputs = getCompletionInputs(tokenizer, partialText, context);

  self.postMessage({ status: "start" });

  for (let branchId = 0; branchId < numBranches; branchId++) {
    const { inputs, runCache } = applyCompletionPrefixCache(fullInputs);
    // Increase temperature for each branch for variety
    const baseTemperature = 0.55;
    const temperature = Math.min(0.95, baseTemperature + (branchId * 0.15));

    let startTime: number | null = null;
    let numTokens = 0;
    let tps = 0;

    const token_callback_function = () => {
      startTime ??= performance.now();
      if (numTokens++ > 0) {
        tps = (numTokens / (performance.now() - startTime)) * 1000;
      }
    };

    const callback_function = (output: string) => {
      self.postMessage({
        status: "update",
        output,
        tps,
        numTokens,
        state: "answering",
        branchId,
      });
    };

    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function,
      token_callback_function,
    });

    const generateOptions = {
      ...(inputs as Record<string, unknown>),
      do_sample: true,
      top_k: Math.min(64, 24 + (branchId * 8)),
      top_p: 0.92,
      temperature,
      repetition_penalty: 1.08,
      min_new_tokens: Math.min(4, normalizeResponseTokenLimit(maxResponseTokens)),
      max_new_tokens: normalizeResponseTokenLimit(maxResponseTokens),
      streamer,
      stopping_criteria,
      return_dict_in_generate: true,
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let result: any;
    let decoded: string[];
    try {
      result = await model.generate(generateOptions);
      decoded = tokenizer.batch_decode(result.sequences, {
        skip_special_tokens: true,
      }) as string[];
    } finally {
      await disposeGenerationCache(result ?? { past_key_values: runCache });
    }

    // Signal this branch is complete (triggers TTS immediately)
    self.postMessage({
      status: "branchComplete",
      output: decoded[0],
      branchId,
      branchComplete: true,
    });

    // Reset stopping criteria for next branch
    stopping_criteria.reset();
  }

  // Signal all branches are done
  self.postMessage({
    status: "complete",
  });
}

async function load() {
  const { dtype, externalData } = TextGenerationPipeline.config;
  const graph = `onnx/model_${dtype}.onnx`;
  const profile = createModelLoadProfile(
    externalData ? [graph, `${graph}_data`] : [graph],
    () => self.postMessage({ status: "weights-ready" }),
  );
  self.postMessage({
    status: "loading",
    data: "Loading model...",
  });

  const [tokenizer, model] = await TextGenerationPipeline.getInstance((x) => {
    profile.progress(x);
    self.postMessage(x);
  });
  profile.initialized();

  self.postMessage({
    status: "loading",
    data: "Warming up model...",
  });

  const hasPrefixCache = await buildCompletionPrefixCache(tokenizer, model);
  if (hasPrefixCache) {
    // The first completion after loading paid ~800 ms to its first token
    // against ~250 ms afterwards, even after a short warmup. Live transcripts
    // are 20-40 tokens of unpunctuated speech, so warm with one of those as
    // well as a short fragment.
    const warmupFragments = [
      "so my name is max and i like the sea and going out with my friends and doing things and what i keep thinking about is",
      "I think the reason this feels delayed is",
    ];
    for (const fragment of warmupFragments) {
      const warmup = applyCompletionPrefixCache(getCompletionInputs(tokenizer, fragment));
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let result: any;
      try {
        result = await model.generate({
          ...warmup.inputs,
          do_sample: true,
          top_k: 20,
          top_p: 0.88,
          temperature: 0.4,
          repetition_penalty: 1.08,
          min_new_tokens: 3,
          max_new_tokens: 4,
          return_dict_in_generate: true,
        });
      } finally {
        await disposeGenerationCache(result ?? { past_key_values: warmup.runCache });
      }
    }
  } else {
    const inputs = tokenizer("a");
    await model.generate({ ...inputs, max_new_tokens: 1 });
  }
  self.postMessage({ status: "ready", loadProfile: profile.finish() });
}

async function reloadCompletionModel(config?: { modelId?: string; dtype?: string; externalData?: boolean }) {
  stopping_criteria.interrupt();
  // The prefix cache belongs to the previous model. Clearing it first also
  // makes a keep-warm tick that lands during the swap a no-op instead of a
  // load of the outgoing model.
  past_key_values_cache = null;
  completionPrefixCache = null;
  completionPrefixTokenIds = [];
  completionPrefixPromise = null;
  completionPrefixUnavailable = false;
  await TextGenerationPipeline.reset();
  TextGenerationPipeline.config = { ...DEFAULT_COMPLETION_MODEL };
  TextGenerationPipeline.configure(config);
  stopping_criteria.reset();
  await load();
}

self.addEventListener("message", async (e: MessageEvent<WorkerMessage>) => {
  const { type, data } = e.data;

  switch (type) {
    case "check":
      check();
      break;

    case "load":
      TextGenerationPipeline.configure(data?.completionModel);
      load().catch((err) => {
        // Surfaces through the loading message; the page has no error case.
        self.postMessage({
          status: "loading",
          data: `LLM load failed: ${err instanceof Error ? err.message : String(err)}`,
        });
      });
      break;

    case "setModel":
      reloadCompletionModel(data?.completionModel).catch((err) => {
        self.postMessage({
          status: "loading",
          data: `LLM load failed: ${err instanceof Error ? err.message : String(err)}`,
        });
      });
      break;

    case "generate":
      stopping_criteria.reset();
      if (data) {
        generate(data);
      }
      break;

    case "generateCompletion":
      if (data?.partialText) {
        const request = data;
        void completionQueue.enqueue(async () => {
          stopping_criteria.reset();
          await generateCompletion({
            partialText: request.partialText!,
            context: request.context,
            usePrefixCache: request.usePrefixCache,
            requestId: request.requestId,
            postedAtEpoch: request.postedAtEpoch,
            earlyStop: request.earlyStop,
            maxResponseTokens: request.maxResponseTokens,
          });
        }, () => {
          // The coordinator needs the terminal message to release the stale
          // candidate and dispatch its replacement, even without inference.
          self.postMessage({ status: "complete", output: "", numTokens: 0, requestId: request.requestId });
        }).catch((error) => {
          self.postMessage({ status: "error", data: String(error), requestId: request.requestId });
        });
      }
      break;

    case "generateBranches":
      stopping_criteria.reset();
      if (data?.partialText) {
        generateBranches({ partialText: data.partialText, numBranches: data.numBranches, context: data.context, maxResponseTokens: data.maxResponseTokens });
      }
      break;

    case "interrupt":
      completionQueue.interrupt();
      stopping_criteria.interrupt();
      break;

    case "keepWarm":
      void completionQueue.enqueueWarmup(keepWarm).catch(() => {});
      break;

    case "reset":
      past_key_values_cache = null;
      stopping_criteria.reset();
      break;
  }
});
