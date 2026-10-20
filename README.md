# CQ Quant Factory

**CQ stands for Coded Quantization.**

**Calibration-free, information-theoretic 4-bit quantization for Large Language Models.**

CQ Quant Factory transforms pretrained Hugging Face models into **CQ 4-bit models** that can be deployed with vLLM.

**No calibration dataset. No calibration forward passes.**

```text
Hugging Face Model
        │
        ▼
   Quant Factory
        │
        ▼
   CQ 4-bit Model
        │
        ▼
       vLLM
```

---

# Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/thousandml-api/quant-factory.git
cd quant-factory
```

---

## 2. Quantize a Model

From the CQ quantizer directory, run:

```bash
./quantize-model meta-llama/Llama-3.1-8B-Instruct
```

That's it.

**No calibration dataset needs to be downloaded or supplied.**

**No calibration forward passes are required.**

CQ reads the model weights directly, processes supported targets incrementally, and persists the quantized results as it runs.

### Custom Output Directory

```bash
./quantize-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir /path/to/output
```

Or configure a default output root:

```bash
export CQ_OUTPUT_ROOT=/path/to/quantized_models
```

The quantization pipeline is incremental, so completed records are persisted as the model is processed.

---

## 3. Build the Final CQ Model

After quantization finishes:

```bash
python build_convert_upload.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-CQ-4bit \
  --no-upload
```

This converts the incremental quantization records into the final CQ model layout expected by the runtime.

For example:

```bash
python build_convert_upload.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id my-user/Llama-3.1-8B-Instruct-CQ-4bit \
  --no-upload
```

---

## 4. Optional: Upload to Hugging Face

To build and publish the finished model to Hugging Face, omit `--no-upload`:

```bash
python build_convert_upload.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-CQ-4bit
```

You must be authenticated with Hugging Face and have permission to create or update the specified repository.

---

## 5. Install CQ Inference Support

CQ models can be served through the ThousandML CQ integration for vLLM.

From the repository:

```bash
cd inference
pip install .
```

---

## 6. Serve the Model with vLLM

```bash
vllm serve YOUR_USERNAME/Llama-3.1-8B-Instruct-CQ-4bit \
  --quantization cq
```

For example:

```bash
vllm serve thousandml/Llama-3.1-8B-Instruct-CQ-4bit \
  --quantization cq
```

vLLM loads the CQ metadata and packed weights and dispatches supported operations through the CQ native runtime.

---

# Complete Workflow

```text
Pretrained Hugging Face Model
            │
            ▼
      ./quantize-model
            │
            │
            │  No calibration data
            │  No calibration passes
            ▼
   Incremental CQ Records
            │
            ▼
       Build / Convert
            │
            ▼
       CQ 4-bit Model
            │
       ┌────┴────┐
       │         │
       ▼         ▼
 Hugging Face   vLLM
                  │
                  ▼
           CQ Native Runtime
```

In short:

```bash
# 1. Quantize
./quantize-model meta-llama/Llama-3.1-8B-Instruct

# 2. Build
python build_convert_upload.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-CQ-4bit \
  --no-upload

# 3. Install CQ support
cd inference
pip install .

# 4. Serve
vllm serve YOUR_USERNAME/Llama-3.1-8B-Instruct-CQ-4bit \
  --quantization cq
```

---

# What is CQ?

**CQ (Coded Quantization)** is ThousandML's proprietary 4-bit LLM quantization and inference technology.

Unlike many post-training quantization workflows, CQ requires:

```text
Calibration datasets       0
Calibration samples        0
Calibration forward passes 0
```

CQ operates directly from the model weights.

```text
Model Weights
     │
     ▼
     CQ
     │
     ▼
CQ 4-bit Model
```

This makes quantization easier to automate and removes the need to choose a dataset intended to represent future inference workloads.

CQ approaches quantization from an **information-theoretic, lossy source-coding perspective**, using a proprietary coding-theoretic quantization core designed around the fundamental relationship between **rate and distortion**.

---

# Quantization Quality — Qwen3.8 27B

The benchmark below is specifically for **Qwen3.8 27B**.

It should not be interpreted as a claim that the same fidelity characteristics or relative performance apply to every model architecture or model size.

For this evaluation, we compare the **Qwen3.8 27B CQ 4-bit model** against **Unsloth UD-Q4_K_XL** at a nearly identical model footprint.

Both quantized models are evaluated against the unquantized Qwen3.8 27B reference model by comparing output probability distributions and top-token predictions.

| Qwen3.8 27B Quantization | Model Size\* |  Mean KLD ↓ | Top-1 Agreement ↑ |
| ------------------------ | -----------: | ----------: | ----------------: |
| **CQ 4-bit**             | **17.76 GB** | **0.02824** |       **92.419%** |
| Unsloth UD-Q4_K_XL       |     17.59 GB |     0.00772 |           95.779% |

\* Model size excluding MTP weights.

### Metrics

**Mean Kullback–Leibler Divergence (KLD)** measures how closely the quantized Qwen3.8 27B model's output probability distribution matches the unquantized reference model.

**Lower is better.**

**Top-1 Agreement** measures how often the quantized model and unquantized reference model select the same highest-probability token.

**Higher is better.**

---

## Qwen3.8 27B: The Tradeoff

On **Qwen3.8 27B**, the results show the tradeoff clearly.

At a nearly identical model size, **Unsloth UD-Q4_K_XL preserves the behavior of the unquantized Qwen3.8 27B reference more closely in this evaluation**.

Compared with Unsloth UD-Q4_K_XL, the CQ version shows:

- higher mean KLD (`0.02824` vs. `0.00772`);
- approximately **3.36 percentage points lower Top-1 Agreement** (`92.419%` vs. `95.779%`); and
- a nearly identical model footprint (`17.76 GB` vs. `17.59 GB`).

CQ, however, is designed around an additional constraint:

> **Zero calibration data and zero calibration forward passes.**

CQ quantizes Qwen3.8 27B directly from the model weights.

There is no calibration corpus to select, download, preprocess, version, or maintain as part of the CQ quantization process.

In this **Qwen3.8 27B evaluation**, that calibration-free constraint comes with a modest reduction in output fidelity compared with Unsloth UD-Q4_K_XL.

That is the engineering tradeoff CQ is designed to explore:

> **Preserve strong model fidelity at 4-bit while making quantization calibration-free, reproducible, and suitable for automated model-processing infrastructure.**

These results should be interpreted as a benchmark for **Qwen3.8 27B**, not as a universal ranking between CQ and Unsloth UD across all models.

---

# Why Calibration-Free?

A conventional calibration-based quantization workflow may look like this:

```text
Model
  +
Representative Dataset
        │
        ▼
Dataset Selection
        │
        ▼
Preprocessing
        │
        ▼
Calibration Forward Passes
        │
        ▼
Statistics / Optimization
        │
        ▼
Quantization
```

CQ reduces the workflow to:

```text
Model Weights
     │
     ▼
     CQ
     │
     ▼
Quantized Model
```

This eliminates several external dependencies from the quantization process.

There is no need to:

- select a representative calibration dataset;
- download or store calibration data;
- preprocess calibration samples;
- choose calibration sequences;
- run calibration forward passes;
- determine whether the calibration distribution represents future workloads; or
- maintain calibration datasets as part of the model pipeline.

This is particularly useful when:

- representative data is unavailable;
- deployment workloads are unknown;
- application data is private or restricted;
- many models need to be processed automatically;
- reproducibility is important; or
- calibration compute and infrastructure are undesirable.

---

# Why This Matters for a Quant Factory

Quantizing one model manually and building infrastructure capable of quantizing many models are different engineering problems.

**Quant Factory is designed for the second.**

If quantization depends on representative data, every model potentially introduces another data pipeline:

```text
Model A + Dataset A ──► Calibration ──► Quantization
Model B + Dataset B ──► Calibration ──► Quantization
Model C + Dataset C ──► Calibration ──► Quantization
```

With CQ:

```text
Model A ──► CQ ──► Quantized Model A
Model B ──► CQ ──► Quantized Model B
Model C ──► CQ ──► Quantized Model C
Model D ──► CQ ──► Quantized Model D
               .
               .
               .
```

That makes quantization easier to treat as **infrastructure rather than a model-specific research project**.

Quant Factory can:

- quantize Hugging Face models to CQ;
- quantize without calibration datasets;
- quantize without calibration forward passes;
- stream model weights incrementally;
- process large models layer by layer;
- persist results incrementally;
- resume interrupted jobs;
- build final CQ model artifacts;
- optionally publish models to Hugging Face; and
- serve CQ models through vLLM.

---

# Quantization Through Information Theory

CQ approaches model quantization as a **lossy source-coding problem**.

Any lossy compression system involves a fundamental tradeoff between two quantities:

**Rate** — how many bits are used to represent information.

**Distortion** — how much information is lost when that information is compressed.

Information theory describes the fundamental relationship between these quantities through the **rate–distortion function**:

```text
R(D)
```

where `R(D)` describes the theoretical minimum rate required to represent a source while maintaining a specified distortion level `D`.

CQ's proprietary quantization core is grounded in **coding-theoretic techniques designed around these fundamental rate–distortion limits**.

Rather than relying on representative activation data to determine how model weights should be represented, CQ treats the model weights themselves as the source being compressed.

Conceptually:

```text
                Model Weights
                     │
                     ▼
            ┌─────────────────┐
            │   CQ Quantizer  │
            │                 │
            │  Source Coding  │
            │  Rate ↔ Dist.   │
            └────────┬────────┘
                     │
                     ▼
            Low-Bit Representation
```

The objective is simple:

> **Represent model weights using fewer bits while minimizing the distortion introduced by that representation.**

The specific coding construction, production algorithm, internal representation, and inference implementation remain proprietary.

---

# Architecture

Quant Factory intentionally separates the **open model-processing pipeline** from the **proprietary quantization and inference technology**.

```text
┌─────────────────────────────────────────────┐
│                Quant Factory                │
│                                             │
│  Hugging Face discovery                    │
│  Model traversal                           │
│  Streaming                                 │
│  Quantization orchestration                │
│  Incremental persistence                   │
│  Resume support                            │
│  Model conversion                          │
│  Hugging Face publishing                   │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Native CQ Core   │
              │                   │
              │    Proprietary    │
              │    Quantization   │
              └─────────┬─────────┘
                        │
                        ▼
                  CQ 4-bit Model
                        │
                        ▼
                      vLLM
                        │
                        ▼
              ┌───────────────────┐
              │ CQ Native Runtime │
              │                   │
              │    Proprietary    │
              │     Inference     │
              └───────────────────┘
```

---

## Open Pipeline

The visible code in this repository is responsible for:

- Hugging Face checkpoint discovery
- Model-specific target discovery
- Streaming model weights
- Quantization orchestration
- Incremental persistence
- Resume support
- Model conversion
- Hugging Face integration
- vLLM integration

---

## Proprietary CQ Core

The production CQ quantization implementation is distributed as a compiled Linux extension:

```text
_cq_core.so
```

The native core contains the proprietary **coding-theoretic CQ quantization algorithm** and executes tensor operations through ATen CUDA.

The Python pipeline passes tensors into the native core but does not contain the proprietary quantization implementation.

This separation keeps the surrounding infrastructure inspectable while protecting the underlying quantization technology.

---

# Open Pipeline, Proprietary Core

Quant Factory uses a hybrid open/proprietary architecture.

### Available in this repository

- Quantization orchestration
- Model discovery
- Streaming pipeline
- Incremental persistence
- Resume logic
- Model conversion
- Hugging Face integration
- vLLM integration

### Proprietary CQ Technology

The following components are intentionally not open sourced:

- Core quantization algorithm
- Coding-theoretic construction
- Internal weight representation
- Native quantization implementation
- Optimized inference operators
- Native CUDA runtime

These components are distributed as compiled extensions rather than source code.

---

# Design Principles

## Calibration-Free

CQ does not require a representative calibration dataset.

```text
0 calibration samples
```

## No Calibration Forward Passes

CQ operates directly on model weights without running a calibration corpus through the model.

## Information-Theoretic

CQ approaches quantization as a lossy source-coding problem and is grounded in coding-theoretic techniques designed around fundamental rate–distortion limits.

## Streaming

Models are processed incrementally instead of requiring the complete model to reside in GPU memory.

## Reproducible

Removing the calibration dataset eliminates an external, workload-dependent input from the quantization pipeline.

## Deployment-Oriented

CQ is designed as both a quantization technology and an inference system, with direct integration into vLLM.

---

# CQ vs. Calibration-Based Quantization

|                                | CQ                  | Calibration-Based PCQ |
| ------------------------------ | ------------------- | --------------------- |
| Calibration dataset            | **Not required**    | Typically required    |
| Calibration forward passes     | **Not required**    | Typically required    |
| Representative workload needed | **No**              | Often                 |
| Dataset preprocessing          | **None**            | Typically required    |
| Dataset versioning             | **None**            | May be required       |
| Weight streaming               | **Yes**             | Method-dependent      |
| Automated model processing     | **Designed for it** | Method-dependent      |
| vLLM inference                 | **Yes**             | Method-dependent      |

Calibration-free does **not** mean calibration-based approaches are inherently inferior.

Calibration provides a quantizer with additional information about model behavior and can be valuable for aggressive low-bit compression.

CQ explores a different engineering and theoretical tradeoff:

> **Can high-quality low-bit LLM quantization be achieved without making representative data part of the quantization process?**

---

# CQ Philosophy

We believe quantization should eventually behave like infrastructure.

You should not need to build a new calibration pipeline every time you want to optimize another model.

The desired workflow is:

```text
Pretrained Model
       │
       ▼
    Quantize
       │
       ▼
     Deploy
       │
       ▼
      Infer
```

That's what we're building with CQ and Quant Factory.

**Quantize. Deploy. Infer.**

---

# Learn More

For more information about CQ, calibration-free quantization, benchmarks, supported models, and ThousandML:

**[https://www.thousandml.com](https://www.thousandml.com)**

Technical background:

**[https://www.thousandml.com/blog/calibration-free-quantization](https://www.thousandml.com/blog/calibration-free-quantization)**

GitHub:

**[https://github.com/thousandml-api/quant-factory](https://github.com/thousandml-api/quant-factory)**

---

# About ThousandML

ThousandML is building infrastructure for efficient Large Language Models.

CQ combines **calibration-free quantization**, an **information-theoretic approach to model compression**, and a **purpose-built inference runtime** to make large models smaller and more efficient to deploy.

Quant Factory provides the open pipeline around that technology for transforming pretrained models into CQ-quantized, inference-ready artifacts.

**Quantize. Deploy. Infer.**
