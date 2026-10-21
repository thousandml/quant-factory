# Training Factory

**Fine-tune a quantized LLM. Spend less memory on the base model, and more of your budget on adaptation.**

Training Factory by **ThousandML** is built around a simple idea: **start with a high-quality 4-bit CQ model and train lightweight LoRA adapters directly on it**, without first restoring the stored base weights to a full-precision checkpoint. Quantization, fine-tuning, and inference live in one repository.

> **Scope:** Training requires a CQ-quantized architecture supported by the installed training loader and LoRA target mapping. The Qwen3.8-27B-CQ-4bit commands below assume compatibility with your release; verify Qwen support before publishing these as tested instructions. The CQ base is frozen; LoRA adapters are trained. Lower base-weight memory use can improve hardware accessibility and economics, but end-to-end cost savings and model quality depend on the model, task, sequence length, hardware, and baseline. No universal savings or accuracy guarantee is implied.

## Quick start — fine-tune a 4-bit model

**Requirements:** Linux x86-64 or WSL2, NVIDIA GPU with CUDA access, Conda, Python 3.12-compatible CQ native binaries, and a ThousandML license if required by the distribution.

```bash
# 1. Install all three workflows (training, quantization, inference)
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
git clone https://github.com/thousandml/quant-factory.git
cd quant-factory
chmod +x install.sh
./install.sh

# 2. Get a compatible CQ 4-bit checkpoint
hf download thousandml/Qwen3.8-27B-CQ-4bit \
  --local-dir ./models/Qwen3.8-27B-CQ-4bit

# 3. Create a small training dataset
cat > train.jsonl <<'JSONL'
{"text":"Question: What does LoRA train? Answer: Small task-specific adapter weights."}
{"text":"Question: What is a quantized base model? Answer: A model stored using reduced-precision weight representations."}
JSONL

# 4. Run a two-step smoke test
python training/train.py \
  --model ./models/Qwen3.8-27B-CQ-4bit \
  --data train.jsonl \
  --output ./outputs/smoke \
   --text-only \
  --targets q_proj,v_proj \
  --kernel-module cq.cq_kernels \
  --max-steps 2 --max-length 128 --batch-size 1 --grad-accum 1
```

`./install.sh` is the unified installer for the repository. Native extensions must match your Python/CUDA/PyTorch environment. A two-step smoke test validates the basic path, **not** task accuracy or a full production run. Use a representative dataset for real fine-tuning.

## Why train on a quantized model?

Most model customization workflows treat quantization as the *last* step: fine-tune first, compress later. Training Factory makes **the quantized checkpoint the starting point**.

- **Smaller frozen base:** CQ 4-bit storage reduces the base model's weight footprint compared with conventional 16-bit storage. This can lower base-weight memory pressure; activations, adapters, optimizer state, and temporary buffers still consume memory.
- **Less duplicated work:** Reuse one CQ checkpoint across tasks and train separate LoRA adapters instead of maintaining a full fine-tuned copy for every task.
- **Quality-conscious compression:** ThousandML's **Coded Quantization (CQ)** is designed to preserve model usefulness at low bit widths. Judge quality on your own evaluation sets; the repository does not claim that quantization is lossless.
- **One workflow:** Quantize a supported model, fine-tune its CQ checkpoint, and use CQ-enabled inference tools without switching model formats for the base.

**What is trained?** Only the LoRA adapters. The compressed CQ base stays frozen. This is **parameter-efficient fine-tuning**, not full-parameter training and not a claim that all intermediate GPU calculations are 4-bit.

## Train on your data

Supply `train.jsonl` as either plain-text records or conversations:

```jsonl
{"text":"Question: When does the warranty expire? Answer: Check the purchase date and warranty terms."}
{"messages":[{"role":"user","content":"Summarize LoRA in one sentence."},{"role":"assistant","content":"LoRA fine-tunes small adapter matrices while the base stays frozen."}]}
```

The trainer uses the tokenizer's chat template for `messages`. In the current implementation, loss is calculated on all non-padding tokens, **including user/prompt tokens**, rather than assistant-only tokens.

Start a longer LoRA run:

```bash
python training/train.py \
  --model ./models/Qwen3.8-27B-CQ-4bit \
  --data train.jsonl \
  --output ./outputs/my-lora \
   --text-only \
  --kernel-module cq.cq_kernels \
  --targets q_proj,v_proj \
  --rank 16 --alpha 32 \
  --epochs 1 \
  --max-length 1024 \
  --batch-size 1 --grad-accum 16 \
  --learning-rate 2e-4
```

The output is a PEFT-style LoRA adapter (for example, `adapter_model.safetensors` and `adapter_config.json`) stored **separately** from the CQ base model. Do not merge it back into the frozen CQ base with `merge_and_unload()`.

| Option | What it controls |
| --- | --- |
| `--model` | Local directory containing the CQ checkpoint, manifest, shards and tokenizer |
| `--data` | JSONL training file |
| `--output` | Adapter and checkpoint destination |
| `--targets` | LoRA projection/module names (must match the supported model mapping) |
| `--rank`, `--alpha` | LoRA rank and scaling |
| `--epochs` or `--max-steps` | Training duration |
| `--max-length` | Token sequence limit |
| `--batch-size`, `--grad-accum` | Batch and accumulation settings |
| `--learning-rate` | Optimizer learning rate |
| `--resume` | Resume from an existing trainer checkpoint |
| `--kernel-module` | Use `cq.cq_kernels` with the packaged inference extension |

For full options, use `python training/train.py --help` after configuring the required license.

## Quantize a model (optional)

Already have a ThousandML CQ checkpoint? **Skip this section.** Otherwise, from the repository root:

```bash
./quantize-model YOUR_ORG/Qwen3.8-27B \
  --output-dir ./quantized_models

python build_convert_upload.py \
  --model YOUR_ORG/Qwen3.8-27B \
  --repo-id YOUR_USERNAME/Qwen3.8-27B-CQ-4bit \
  --no-upload
```

CQ quantization supports a calibration-free workflow for compatible models. Follow the tool's output and `--help` to locate the complete checkpoint and verify its manifest and shards before training. Publishing to Hugging Face requires separate authentication and a deliberate upload choice.

## Serve the CQ base model

```bash
vllm serve thousandml/Qwen3.8-27B-CQ-4bit \
  --quantization cq
```

**Serving a trained LoRA adapter** requires a vLLM/CQ runtime that actually supports adapter injection and target mapping. Once validated for your release, its configuration can look like:

```bash
vllm serve ./models/Qwen3.8-27B-CQ-4bit \
  --quantization cq \
  --enable-lora --max-lora-rank 16 \
  --lora-modules my-lora=./outputs/my-lora
```

Do not assume adapter serving is compatible solely because training succeeds; validate end-to-end predictions on your deployment build.

**Call the fine-tuned adapter** (in another terminal, after the server is ready):

```bash
# Check that the adapter is listed as "my-lora"
curl http://localhost:8000/v1/models

# Request a response from the fine-tuned model, not just the base checkpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-lora",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain how LoRA fine-tuning works in one sentence."}
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

The `model` field must match the adapter name given in `--lora-modules my-lora=...`. Use `thousandml/Qwen3.8-27B-CQ-4bit` instead to query the *base* model, if that is the ID exposed by `/v1/models`. This request only works after the CQ/vLLM runtime has been validated to load and apply the trained adapter.


## Installation checks and compatibility

```bash
python -c "import torch, peft, transformers, cq.cq_kernels; print('CUDA:', torch.cuda.is_available())"
python -m pip check
```

- Training's kernel module path is **`cq.cq_kernels`**, not the top-level `cq_kernels`.
- Use **Python 3.12** for the current repository package set; compiled `.so` files are Python-ABI-specific.
- A valid license server and key may be required: `CQ_LICENSE_KEY` and, for custom deployments, `CQ_LICENSE_URL`.
- CUDA drivers, Toolkit, PyTorch, and precompiled extensions must be mutually compatible.
- If a shell script checked out from Windows fails with `bash\r`, normalize line endings with `sed -i 's/\r$//' install.sh`.
- Verify memory, throughput, model accuracy, and training costs on **your actual workload** before treating them as production claims.

## How the pieces fit

```text
Existing CQ 4-bit model ──────────────────┐
                                         v
Hugging Face model -> CQ quantization -> Frozen CQ 4-bit base
                                         |              |
                                         v              v
                                    LoRA training   vLLM serving
                                         |
                                         v
                                    LoRA adapter
                                         |
                                         v
                            CQ + LoRA serving (after validation)
```

**ThousandML Training Factory:** build task-specific adapters on top of a reusable, compact CQ model—rather than paying to fine-tune a full-precision copy for every adaptation.

[Source and issues](https://github.com/thousandml/quant-factory)
