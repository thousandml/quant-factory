"""Opt-in training benchmarks for the existing Hugging Face Trainer loop."""
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


class TokenCountingCollator:
    """Counts batch tokens as delivered by the DataLoader (may include prefetch)."""
    def __init__(self, collate):
        self.collate = collate
        self.total_input_tokens = 0
        self.total_nonpadding_tokens = 0

    def __call__(self, rows):
        batch = self.collate(rows)
        self.total_input_tokens += int(batch['input_ids'].numel())
        self.total_nonpadding_tokens += int(batch['attention_mask'].sum().item())
        return batch


class BenchmarkCallback(TrainerCallback):
    """Times optimizer steps, with CUDA synchronization at step boundaries."""
    def __init__(self, output_file, warmup_steps=10, token_counter=None):
        self.output_file = Path(output_file)
        self.warmup_steps = warmup_steps
        self.token_counter = token_counter
        self.step_times = []
        self.started = None
        self.last_step = None
        self.step_records = []

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.last_step = self.started
        self.step_times = []
        self.step_records = []

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        duration = now - self.last_step
        self.step_times.append(duration)
        self.step_records.append({'step': int(state.global_step), 'seconds': duration})
        self.last_step = now

    def on_train_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        steady = self.step_times[self.warmup_steps:]
        steady_seconds = sum(steady)
        output = {
            'optimizer_steps': len(self.step_times),
            'training_wall_seconds': elapsed,
            'warmup_steps_excluded': min(self.warmup_steps, len(self.step_times)),
            'median_steady_step_seconds': statistics.median(steady) if steady else None,
            'steady_optimizer_steps_per_second': len(steady) / steady_seconds if steady_seconds else None,
            'peak_pytorch_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_pytorch_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'step_timings': self.step_records,
            'notes': [
                'Step timings include gradient accumulation and Trainer overhead.',
                'PyTorch peak excludes allocations outside its CUDA caching allocator.',
                'Token counts from the collator may include prefetched batches; do not interpret as exact consumed tokens.',
                'Wall time includes any initialization inside Trainer.train and callback overhead.',
            ],
        }
        if self.token_counter is not None:
            output['collated_input_tokens_including_prefetch'] = self.token_counter.total_input_tokens
            output['collated_nonpadding_tokens_including_prefetch'] = self.token_counter.total_nonpadding_tokens
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(json.dumps(output, indent=2) + '\n')
        print('\n===== CQ TRAINING BENCHMARK =====')
        print(json.dumps({k: v for k, v in output.items() if k != 'step_timings'}, indent=2))
