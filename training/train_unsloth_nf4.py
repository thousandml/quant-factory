#!/usr/bin/env python3
"""Matched text-only QLoRA baseline using Unsloth's PREQUANTIZED Qwen3.8-27B NF4 checkpoint."""
import argparse
import json
import os
import time
import statistics

import torch
from datasets import Dataset
from unsloth import FastModel  # Import Unsloth before transformers/trl.
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = 'unsloth/Qwen3.8-27B-unsloth-bnb-4bit'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj.get('text'), str):
                raise ValueError(f'{path}:{n}: expected a string text field')
            rows.append({'text': obj['text']})
    if not rows:
        raise ValueError(f'No examples in {path}')
    return Dataset.from_list(rows)


class JsonlLogger(TrainerCallback):
    def __init__(self, output):
        self.path = os.path.join(output, 'nf4_logs.jsonl')
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record = {'step': state.global_step,
                  'elapsed_seconds': time.perf_counter() - self.started if self.started else None,
                  **logs}
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')



def append_jsonl(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=str) + '\n')


class CountedSFTTrainer(SFTTrainer):
    """Count inputs actually passed to training loss, not prefetched batches."""
    def compute_loss(self, model, inputs, *args, **kwargs):
        if model.training and hasattr(self, 'comparison_counter'):
            ids = inputs.get('input_ids')
            mask = inputs.get('attention_mask')
            labels = inputs.get('labels')
            if ids is not None:
                if mask is not None:
                    nonpad = int(mask.detach().sum().item())
                else:
                    nonpad = int(ids.numel())
                self.comparison_counter['input_tokens'] += int(ids.numel())
                self.comparison_counter['nonpadding_tokens'] += nonpad
                self.comparison_counter['microbatches'] += 1
                if labels is not None:
                    self.comparison_counter['supervised_tokens'] += int(
                        (labels.detach() != -100).sum().item()
                    )
        return super().compute_loss(model, inputs, *args, **kwargs)


class ComparisonLogger(TrainerCallback):
    def __init__(self, output, warmup_steps=10):
        self.output = output
        self.warmup_steps = warmup_steps
        self.counter = dict(input_tokens=0, nonpadding_tokens=0,
                            supervised_tokens=0, microbatches=0)
        self.records = []
        self.started = None
        self.step_started = None
        self.before = None
        self.step_path = os.path.join(output, 'comparison_steps.jsonl')
        self.log_path = os.path.join(output, 'comparison_logs.jsonl')

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_begin(self, args, state, control, **kwargs):
        self.sync()
        self.started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kwargs):
        self.sync()
        self.step_started = time.perf_counter()
        self.before = self.counter.copy()

    def on_step_end(self, args, state, control, **kwargs):
        self.sync()
        elapsed = time.perf_counter() - self.step_started
        delta = {k: self.counter[k] - self.before[k] for k in self.counter}
        record = {'step': state.global_step, 'optimizer_step_seconds': elapsed,
                  **delta, 'tokens_per_second': delta['input_tokens'] / elapsed if elapsed else None,
                  'nonpadding_tokens_per_second': delta['nonpadding_tokens'] / elapsed if elapsed else None,
                  'sequences_per_second': (delta['microbatches'] * args.per_device_train_batch_size) / elapsed if elapsed else None}
        if torch.cuda.is_available():
            record['pytorch_allocated_gib'] = torch.cuda.memory_allocated() / 2**30
            record['pytorch_reserved_gib'] = torch.cuda.memory_reserved() / 2**30
        self.records.append(record)
        append_jsonl(self.step_path, record)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.sync()
            append_jsonl(self.log_path, {'step': state.global_step,
                        'elapsed_seconds': time.perf_counter() - self.started if self.started else None,
                        **logs})

    def on_train_end(self, args, state, control, **kwargs):
        self.sync()
        wall = time.perf_counter() - self.started
        steady = [r for r in self.records if r['step'] > self.warmup_steps]
        durations = [r['optimizer_step_seconds'] for r in steady]
        steady_seconds = sum(durations)
        totals = {k: sum(r[k] for r in steady) for k in self.counter}
        peak = {}
        if torch.cuda.is_available():
            peak = {'peak_pytorch_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'peak_pytorch_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
        benchmark = {'optimizer_steps': state.global_step,
                     'training_wall_seconds': wall,
                     'warmup_steps_excluded': min(self.warmup_steps, len(self.records)),
                     'median_steady_step_seconds': statistics.median(durations) if durations else None,
                     'steady_optimizer_steps_per_second': len(steady) / steady_seconds if steady_seconds else None,
                     **peak,
                     'notes': ['Step timing includes gradient accumulation and Trainer overhead, but excludes evaluation and checkpoint saving performed after on_step_end.',
                               'CUDA synchronized before and after each optimizer step.',
                               'PyTorch memory metrics exclude external CUDA allocations.',
                               'Token counts are measured in training compute_loss, not in a prefetched collator.']}
        summary = {'optimizer_steps': state.global_step, 'training_wall_seconds': wall,
                   'warmup_steps_excluded': min(self.warmup_steps, len(self.records)),
                   'steady_optimizer_step_seconds_total': steady_seconds,
                   'steady_input_tokens': totals['input_tokens'],
                   'steady_nonpadding_tokens': totals['nonpadding_tokens'],
                   'steady_supervised_tokens': totals['supervised_tokens'],
                   'steady_microbatches': totals['microbatches'],
                   'steady_input_tokens_per_second': totals['input_tokens'] / steady_seconds if steady_seconds else None,
                   'steady_nonpadding_tokens_per_second': totals['nonpadding_tokens'] / steady_seconds if steady_seconds else None,
                   'steady_supervised_tokens_per_second': totals['supervised_tokens'] / steady_seconds if steady_seconds else None,
                   'note': 'Token counts use actual compute_loss calls; input_tokens includes padding, supervised_tokens counts labels != -100. Timing excludes eval/checkpoint work.'}
        for name, obj in [('benchmark.json', benchmark), ('comparison_summary.json', summary)]:
            with open(os.path.join(self.output, name), 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=DEFAULT_MODEL,
                   help='Prequantized Unsloth bnb NF4 checkpoint; NOT original BF16 or CQ')
    p.add_argument('--train-data', required=True)
    p.add_argument('--eval-data')
    p.add_argument('--output', default='./outputs/unsloth-nf4-r32')
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-steps', type=int, default=100)
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--benchmark-warmup-steps', type=int, default=10)
    a = p.parse_args()

    os.makedirs(a.output, exist_ok=True)
    train_ds = read_jsonl(a.train_data)
    eval_ds = read_jsonl(a.eval_data) if a.eval_data else None

    # Qwen3.8-27B is a vision-language model, even when training on text only.
    # FastModel, rather than FastLanguageModel, supports Unsloth's VLM path.
    model, tokenizer = FastModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.max_length,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
    )

    # Check the loaded config and actual bnb linear modules, not the repo suffix.
    cfg = getattr(model.config, 'quantization_config', None)
    if cfg is not None:
        cfg = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
        if cfg.get('bnb_4bit_quant_type', '').lower() != 'nf4':
            raise RuntimeError(f'Expected NF4 checkpoint, config says: {cfg}')
    qtypes = set()
    n_bnb = 0
    for module in model.modules():
        if module.__class__.__name__ == 'Linear4bit':
            n_bnb += 1
            qt = getattr(module, 'quant_type', None)
            if qt is None:
                qt = getattr(getattr(module, 'weight', None), 'quant_type', None)
            qtypes.add(qt)
    if not n_bnb or qtypes != {'nf4'}:
        raise RuntimeError(f'Expected NF4 bitsandbytes Linear4bit modules; found {n_bnb}, types={qtypes}')
    print(f'Confirmed {n_bnb} bitsandbytes NF4 Linear4bit modules')

    model = FastModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=TARGETS,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=a.seed,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    config = SFTConfig(
        output_dir=a.output,
        dataset_text_field='text',
        max_length=a.max_length,
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        max_steps=a.max_steps,
        learning_rate=2e-5,
        lr_scheduler_type='linear',
        warmup_steps=0,
        weight_decay=0.0,
        optim='adamw_torch',
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy='steps',
        save_steps=a.save_steps,
        save_total_limit=None,
        eval_strategy='steps' if eval_ds is not None else 'no',
        eval_steps=a.eval_steps if eval_ds is not None else None,
        report_to='none',
        seed=a.seed,
        data_seed=a.seed,
        remove_unused_columns=False,
    )
    with open(os.path.join(a.output, 'comparison_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': a.model, 'quantization': 'bnb_nf4_prequantized',
                   'targets': TARGETS, 'rank': 32, 'alpha': 32, 'dropout': 0,
                   'train_data': a.train_data, 'eval_data': a.eval_data,
                   'max_length': a.max_length, 'seed': a.seed,
                   'max_steps': a.max_steps, 'batch_size': 2, 'grad_accum': 4,
                   'learning_rate': 2e-5, 'optimizer': 'adamw_torch',
                   'scheduler': 'linear', 'warmup_steps': 0, 'weight_decay': 0.0,
                   'eval_steps': a.eval_steps, 'save_steps': a.save_steps,
                   'benchmark_warmup_steps': a.benchmark_warmup_steps}, f, indent=2)

    logger = ComparisonLogger(a.output, a.benchmark_warmup_steps)
    trainer = CountedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[logger],
    )
    trainer.comparison_counter = logger.counter
    trainer.train()
    trainer.save_model(os.path.join(a.output, 'final_adapter'))
    tokenizer.save_pretrained(os.path.join(a.output, 'final_adapter'))


if __name__ == '__main__':
    main()
