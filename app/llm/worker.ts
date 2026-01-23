import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";

export interface WorkerMessage {
  type: "check" | "load" | "generate" | "generateCompletion" | "generateBranches" | "interrupt" | "reset";
  data?: {
    messages: Array<{ role: string; content: string }>;
    reasonEnabled?: boolean;
    partialText?: string;
    numBranches?: number;
  };
}

export interface WorkerResponse {
  status: "error" | "loading" | "initiate" | "progress" | "done" | "ready" | "start" | "update" | "complete" | "branchComplete";
  data?: string;
  output?: string;
  tps?: number;
  numTokens?: number;
  state?: "thinking" | "answering";
  file?: string;
  progress?: number;
  total?: number;
  branchId?: number;
  branchComplete?: boolean;
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
  static model_id =  "onnx-community/granite-4.0-1b-ONNX";//"onnx-community/granite-4.0-350m-ONNX"; try fp16 or "onnx-community/granite-4.0-1b-ONNX" with q4; or onnx-community/granite-4.0-micro-ONNX-web  q4f16? (too much ram)
  static tokenizer: Promise<AnyTokenizer> | null = null;
  static model: Promise<AnyModel> | null = null;

  static async getInstance(progress_callback?: (x: unknown) => void) {
    this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
      progress_callback,
    });

    this.model ??= AutoModelForCausalLM.from_pretrained(this.model_id, {
      dtype: "q4",
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
    max_new_tokens: 500,
    streamer,
    stopping_criteria,
    return_dict_in_generate: true,
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = await model.generate(generateOptions) as any;

  // Clear KV cache after generation to prevent unbounded memory growth
  // The cache is only useful for multi-turn within a single generate call
  past_key_values_cache = null;

  const decoded = tokenizer.batch_decode(result.sequences, {
    skip_special_tokens: true,
  }) as string[];

  self.postMessage({
    status: "complete",
    output: decoded[0],
  });
}

async function generateCompletion({ partialText }: { partialText: string }) {
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();

  const systemPrompt = `Finish the sentence. Rules:
- Output ONLY the rest of the sentence, never repeat the input
- Be specific to context, not generic, be creative and interesting
- Minimum 10 words
- No punctuation at start`;

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: partialText },
  ];

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: false,
  } as Record<string, unknown>);

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
    do_sample: true,
    top_k: 10,
    temperature: 0.5,
    max_new_tokens: 50,
    streamer,
    stopping_criteria,
    return_dict_in_generate: true,
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = await model.generate(generateOptions) as any;

  const decoded = tokenizer.batch_decode(result.sequences, {
    skip_special_tokens: true,
  }) as string[];

  self.postMessage({
    status: "complete",
    output: decoded[0],
  });
}

async function generateBranches({ partialText, numBranches = 3 }: { partialText: string; numBranches?: number }) {
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();

  const systemPrompt = `Finish the sentence. Rules:
- Output ONLY the rest of the sentence, never repeat the input
- Be specific to context, not generic, be creative and interesting
- Minimum 10 words
- No punctuation at start`;

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: partialText },
  ];

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
    enable_thinking: false,
  } as Record<string, unknown>);

  self.postMessage({ status: "start" });

  for (let branchId = 0; branchId < numBranches; branchId++) {
    // Increase temperature for each branch for variety
    const baseTemperature = 0.5;
    const temperature = baseTemperature + (branchId * 0.15);

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
      top_k: 10 + (branchId * 5), // Also vary top_k for diversity
      temperature,
      max_new_tokens: 50,
      streamer,
      stopping_criteria,
      return_dict_in_generate: true,
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result = await model.generate(generateOptions) as any;

    const decoded = tokenizer.batch_decode(result.sequences, {
      skip_special_tokens: true,
    }) as string[];

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

    case "generateCompletion":
      stopping_criteria.reset();
      if (data?.partialText) {
        generateCompletion({ partialText: data.partialText });
      }
      break;

    case "generateBranches":
      stopping_criteria.reset();
      if (data?.partialText) {
        generateBranches({ partialText: data.partialText, numBranches: data.numBranches });
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
