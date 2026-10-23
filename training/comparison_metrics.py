"""Actual-consumed-token and loss/evaluation logs for controlled CQ vs NF4 runs."""
import json
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback


class CountedTrainer(Trainer):
    """Count actual microbatches passed into compute_loss, not prefetched batches."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consumed_nonpadding_tokens = 0
        self.consumed_input_tokens = 0
        self.consumed_supervised_tokens = 0
        self.consumed_sequences = 0

    def compute_loss(self, model, inputs, *args, **kwargs):
        # compute_loss is called once per training microbatch, not during
        # checkpoint recomputation. Exclude evaluation calls.
        if model.training:
            ids = inputs.get('input_ids')
            mask = inputs.get('attention_mask')
            labels = inputs.get('labels')
            if ids is not None:
                self.consumed_input_tokens += ids.numel()
                self.consumed_sequences += ids.shape[0]
                self.consumed_nonpadding_tokens += int(mask.sum().item()) if mask is not None else ids.numel()
            if labels is not None:
                # Shifted causal-LM targets: first label is not predicted.
                valid = labels[..., 1:] != -100
                if mask is not None:
                    valid = valid & mask[..., 1:].bool()
                self.consumed_supervised_tokens += int(valid.sum().item())
        return super().compute_loss(model, inputs, *args, **kwargs)


class ComparisonMetricsCallback(TrainerCallback):
    """Write step-level metrics incrementally so partial runs are useful."""
    def __init__(self, output_dir, metadata, warmup_steps=10):
        self.root = Path(output_dir)
        self.metadata = metadata
        self.warmup_steps = warmup_steps
        self.trainer_ref = None
        self.last_step_time = None
        self.last_tokens = 0
        self.last_sequences = 0
        self.steady_seconds = 0.0
        self.steady_tokens = 0
        self.steady_sequences = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'comparison_config.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.last_step_time = time.perf_counter()
        self.last_tokens = 0
        self.last_sequences = 0
        (self.root / 'comparison_steps.jsonl').write_text('')
        (self.root / 'comparison_logs.jsonl').write_text('')

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        trainer = self.trainer_ref
        tokens = trainer.consumed_nonpadding_tokens
        seqs = trainer.consumed_sequences
        seconds = now - self.last_step_time
        delta_tokens = tokens - self.last_tokens
        delta_seqs = seqs - self.last_sequences
        step = int(state.global_step)
        record = {'step': step, 'seconds': seconds,
                  'consumed_nonpadding_tokens': delta_tokens,
                  'consumed_sequences': delta_seqs,
                  'nonpadding_tokens_per_second': delta_tokens / seconds if seconds else None,
                  'sequences_per_second': delta_seqs / seconds if seconds else None,
                  'pytorch_allocated_gib': torch.cuda.memory_allocated() / 1024**3,
                  'pytorch_reserved_gib': torch.cuda.memory_reserved() / 1024**3}
        with (self.root / 'comparison_steps.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if step > self.warmup_steps:
            self.steady_seconds += seconds
            self.steady_tokens += delta_tokens
            self.steady_sequences += delta_seqs
        self.last_step_time = now
        self.last_tokens = tokens
        self.last_sequences = seqs

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with (self.root / 'comparison_logs.jsonl').open('a') as f:
                f.write(json.dumps({'step': int(state.global_step), **logs}, default=str) + '\n')

    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref
        summary = {'actual_consumed_input_tokens': trainer.consumed_input_tokens,
                   'actual_consumed_nonpadding_tokens': trainer.consumed_nonpadding_tokens,
                   'actual_consumed_supervised_tokens': trainer.consumed_supervised_tokens,
                   'actual_consumed_sequences': trainer.consumed_sequences,
                   'steady_nonpadding_tokens_per_second': self.steady_tokens / self.steady_seconds if self.steady_seconds else None,
                   'steady_sequences_per_second': self.steady_sequences / self.steady_seconds if self.steady_seconds else None,
                   'warmup_optimizer_steps_excluded': self.warmup_steps,
                   'notes': ['Counts actual compute_loss training microbatches, excluding dataloader prefetch.',
                             'Nonpadding tokens/sec is not supervised target tokens/sec.',
                             'Step times include Trainer overhead and any intervening evaluation or checkpoint saving.',
                             'Compare evaluation loss only on an identical held-out dataset and preprocessing.']}
        (self.root / 'comparison_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('\n===== COMPARISON METRICS =====\n' + json.dumps(summary, indent=2))
