import {
  AutoTokenizer,
  AutoProcessor,
  WhisperForConditionalGeneration,
  TextStreamer,
  full,
} from "@huggingface/transformers";

const MAX_NEW_TOKENS = 128;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyModel = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyTokenizer = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyProcessor = any;

class AutomaticSpeechRecognitionPipeline {
  static model_id = "onnx-community/whisper-base";
  static tokenizer: Promise<AnyTokenizer> | null = null;
  static processor: Promise<AnyProcessor> | null = null;
  static model: Promise<AnyModel> | null = null;

  static async getInstance(progress_callback?: (x: unknown) => void) {
    this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
      progress_callback,
    });
    this.processor ??= AutoProcessor.from_pretrained(this.model_id, {
      progress_callback,
    });

    this.model ??= WhisperForConditionalGeneration.from_pretrained(
      this.model_id,
      {
        dtype: {
          encoder_model: "fp32",
          decoder_model_merged: "q4",
        },
        device: "webgpu",
        progress_callback,
      } as Record<string, unknown>,
    );

    return Promise.all([this.tokenizer, this.processor, this.model]);
  }
}

let processing = false;

async function generate({ audio, language }: { audio: Float32Array; language: string }) {
  if (processing) return;
  processing = true;

  self.postMessage({ status: "start" });

  const [tokenizer, processor, model] =
    await AutomaticSpeechRecognitionPipeline.getInstance();

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
    });
  };

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function,
    token_callback_function,
  });

  const inputs = await processor(audio);

  const outputs = await model.generate({
    ...inputs,
    max_new_tokens: MAX_NEW_TOKENS,
    language,
    streamer,
  });

  const decoded = tokenizer.batch_decode(outputs, {
    skip_special_tokens: true,
  }) as string[];

  self.postMessage({
    status: "complete",
    output: decoded[0],
  });
  processing = false;
}

async function generatePartial({ audio, language }: { audio: Float32Array; language: string }) {
  // Quick transcription for thought completion - no streaming needed
  if (processing) return;
  processing = true;

  const [tokenizer, processor, model] =
    await AutomaticSpeechRecognitionPipeline.getInstance();

  const inputs = await processor(audio);

  const outputs = await model.generate({
    ...inputs,
    max_new_tokens: MAX_NEW_TOKENS,
    language,
  });

  const decoded = tokenizer.batch_decode(outputs, {
    skip_special_tokens: true,
  }) as string[];

  self.postMessage({
    status: "partialComplete",
    output: decoded[0],
  });
  processing = false;
}

async function load() {
  self.postMessage({
    status: "loading",
    data: "Loading Whisper model...",
  });

  const [, , model] =
    await AutomaticSpeechRecognitionPipeline.getInstance((x) => {
      self.postMessage(x);
    });

  self.postMessage({
    status: "loading",
    data: "Compiling shaders and warming up model...",
  });

  await model.generate({
    input_features: full([1, 80, 3000], 0.0),
    max_new_tokens: 1,
  });
  self.postMessage({ status: "ready" });
}

export interface WhisperWorkerMessage {
  type: "load" | "generate" | "generatePartial";
  data?: {
    audio: Float32Array;
    language: string;
  };
}

self.addEventListener("message", async (e: MessageEvent<WhisperWorkerMessage>) => {
  const { type, data } = e.data;

  switch (type) {
    case "load":
      load();
      break;

    case "generate":
      if (data) {
        generate(data);
      }
      break;

    case "generatePartial":
      if (data) {
        generatePartial(data);
      }
      break;
  }
});
