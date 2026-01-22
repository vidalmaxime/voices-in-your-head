import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";

export interface WorkerMessage {
  type: "check" | "load" | "generate" | "interrupt" | "reset";
  data?: {
    messages: Array<{ role: string; content: string }>;
    reasonEnabled?: boolean;
  };
}

export interface WorkerResponse {
  status: "error" | "loading" | "initiate" | "progress" | "done" | "ready" | "start" | "update" | "complete";
  data?: string;
  output?: string;
  tps?: number;
  numTokens?: number;
  state?: "thinking" | "answering";
  file?: string;
  progress?: number;
  total?: number;
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

class TextGenerationPipeline {
  static model_id = "onnx-community/Qwen3-0.6B-ONNX";
  static tokenizer: Promise<AnyTokenizer> | null = null;
  static model: Promise<AnyModel> | null = null;

  static async getInstance(progress_callback?: (x: unknown) => void) {
    this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
      progress_callback,
    });

    this.model ??= AutoModelForCausalLM.from_pretrained(this.model_id, {
      dtype: "q4f16",
      device: "webgpu",
      progress_callback,
    } as Record<string, unknown>);

    return Promise.all([this.tokenizer, this.model]);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();

let past_key_values_cache: unknown = null;

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
    max_new_tokens: 16384,
    streamer,
    stopping_criteria,
    return_dict_in_generate: true,
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = await model.generate(generateOptions) as any;

  past_key_values_cache = result.past_key_values;

  const decoded = tokenizer.batch_decode(result.sequences, {
    skip_special_tokens: true,
  }) as string[];

  self.postMessage({
    status: "complete",
    output: decoded[0],
  });
}

async function load() {
  self.postMessage({
    status: "loading",
    data: "Loading model...",
  });

  const [tokenizer, model] = await TextGenerationPipeline.getInstance((x) => {
    self.postMessage(x);
  });

  self.postMessage({
    status: "loading",
    data: "Compiling shaders and warming up model...",
  });

  const inputs = tokenizer("a");
  await model.generate({ ...inputs, max_new_tokens: 1 });
  self.postMessage({ status: "ready" });
}

self.addEventListener("message", async (e: MessageEvent<WorkerMessage>) => {
  const { type, data } = e.data;

  switch (type) {
    case "check":
      check();
      break;

    case "load":
      load();
      break;

    case "generate":
      stopping_criteria.reset();
      if (data) {
        generate(data);
      }
      break;

    case "interrupt":
      stopping_criteria.interrupt();
      break;

    case "reset":
      past_key_values_cache = null;
      stopping_criteria.reset();
      break;
  }
});
