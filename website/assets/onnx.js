// In-browser model runtime for models the router places at location "browser".
// Loads ONNX weights from the Hugging Face Hub with transformers.js and runs them
// on WebGPU when the browser has it, WASM otherwise. Weights are cached by the
// browser after the first download.

let lib;
const load = async () => (lib ??= await import("@huggingface/transformers"));

export async function detectRuntimes() {
  const runtimes = ["wasm"];
  try {
    if (navigator.gpu && (await navigator.gpu.requestAdapter())) runtimes.unshift("webgpu");
  } catch { /* no WebGPU */ }
  return runtimes;
}

export class BrowserModels {
  constructor(device, onProgress = () => {}) {
    this.device = device; // "webgpu" | "wasm"
    this.onProgress = onProgress;
    this.cache = new Map();
    this.files = new Map();
  }

  _progress = (p) => {
    if (p.status !== "progress" && p.status !== "done") return;
    const f = this.files.get(p.file) || { loaded: 0, total: 0 };
    if (p.total) f.total = p.total;
    f.loaded = p.status === "done" ? f.total : p.loaded ?? f.loaded;
    this.files.set(p.file, f);
    let loaded = 0, total = 0;
    for (const v of this.files.values()) { loaded += v.loaded; total += v.total; }
    this.onProgress({ loaded, total, file: p.file });
  };

  _dtype(spec) {
    const d = spec.onnx.dtype;
    return d && typeof d === "object" && (d.webgpu || d.wasm) ? d[this.device] : d;
  }

  async get(spec) {
    if (this.cache.has(spec.name)) return this.cache.get(spec.name);
    const tf = await load();
    const opts = { device: this.device, dtype: this._dtype(spec), progress_callback: this._progress };
    let entry;
    if (spec.onnx.task === "text-generation") {
      entry = { generator: await tf.pipeline("text-generation", spec.onnx.hf_id, opts) };
    } else {
      // Florence-2 style grounding model.
      const [model, processor] = await Promise.all([
        tf.Florence2ForConditionalGeneration.from_pretrained(spec.onnx.hf_id, opts),
        tf.AutoProcessor.from_pretrained(spec.onnx.hf_id, { progress_callback: this._progress }),
      ]);
      entry = { model, processor };
    }
    this.cache.set(spec.name, entry);
    return entry;
  }

  /**
   * Ground phrases in the image with one Florence-2 call ("red cube. bowl").
   * Returns [{label, box: [x1, y1, x2, y2]}] in image pixels, at most one box per
   * phrase. A box already claimed by an earlier phrase is not reused.
   */
  async ground(spec, dataUrl, phrases) {
    const tf = await load();
    const { model, processor } = await this.get(spec);
    const image = await tf.RawImage.fromURL(dataUrl);
    const task = spec.onnx.task;
    const inputs = await processor(image, processor.construct_prompts(task + phrases.join(". ")));
    const ids = await model.generate({ ...inputs, max_new_tokens: 128 });
    const text = processor.batch_decode(ids, { skip_special_tokens: false })[0];
    const { labels = [], bboxes = [] } = processor.post_process_generation(text, task, image.size)[task] || {};
    const out = [];
    labels.forEach((label, i) => {
      const box = bboxes[i];
      if (!phrases.includes(label) || out.some((o) => o.label === label || iou(o.box, box) > 0.5)) return;
      out.push({ label, box });
    });
    return out;
  }

  /** Chat completion; streams text through onToken and returns the full reply. */
  async chat(spec, messages, onToken = () => {}, maxNewTokens = 96) {
    const tf = await load();
    const { generator } = await this.get(spec);
    const streamer = new tf.TextStreamer(generator.tokenizer, { skip_prompt: true, skip_special_tokens: true, callback_function: onToken });
    const out = await generator(messages, { max_new_tokens: maxNewTokens, do_sample: false, streamer });
    return out[0].generated_text.at(-1).content;
  }
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}
