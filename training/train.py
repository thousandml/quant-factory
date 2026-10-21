import argparse
import json
import os
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments
from load_model import load_model


def main():
    from _cq_training_license import require_valid
    require_valid()  # fail closed before loading checkpoint or allocating CUDA memory
    p = argparse.ArgumentParser()
    p.add_argument('--kernel-module', default='cq.cq_kernels')
    p.add_argument('--kernel-backend', choices=['auto','raw','registered'], default='auto')
    p.add_argument('--model', required=True, help='Local complete CQ checkpoint')
    p.add_argument('--data', required=True, help='JSONL: text or messages')
    p.add_argument('--output', default='outputs/cq-lora')
    p.add_argument('--layer-map')
    p.add_argument('--targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=16)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--resume')
    p.add_argument('--trust-remote-code', action='store_true')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('This initial bridge supports one GPU only')
    if a.max_length < 2 or a.rank < 1:
        raise ValueError('max-length >= 2 and rank >= 1 required')
    from inspect_checkpoint import inspect
    inspect(a.model, a.layer_map)  # fail before CUDA weight allocation on layout/header errors
    from kernel_backend import configure_backend
    configure_backend(a.kernel_module, a.kernel_backend)
    model = load_model(a.model, a.layer_map, a.trust_remote_code)
    targets = a.targets.split(',')
    matched = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in targets]
    if not matched:
        raise ValueError('No target linears matched')
    config = LoraConfig(task_type='CAUSAL_LM', r=a.rank, lora_alpha=a.alpha,
                        lora_dropout=0.05, target_modules=matched, bias='none',
                        base_model_name_or_path=str(Path(a.model).resolve()))
    load_report = model._cq_load_report
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer needs a pad or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    dataset = load_dataset('json', data_files=a.data, split='train')
    def tokenize(row):
        if row.get('messages'):
            ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False)
            ids = ids[:a.max_length]
        elif isinstance(row.get('text'), str):
            ids = tokenizer(row['text'], truncation=True, max_length=a.max_length)['input_ids']
        else:
            raise ValueError('Each row needs text or messages')
        if len(ids) < a.max_length and tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
            ids.append(tokenizer.eos_token_id)
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda row: len(row['input_ids']) >= 2)
    if not len(dataset):
        raise ValueError('No usable training examples')
    def collate(rows):
        batch = tokenizer.pad(rows, return_tensors='pt', pad_to_multiple_of=8)
        labels = batch['input_ids'].clone()
        labels[batch['attention_mask'] == 0] = -100
        batch['labels'] = labels
        return batch
    args = TrainingArguments(output_dir=a.output, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
        max_steps=a.max_steps, learning_rate=a.learning_rate, fp16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch', logging_steps=1, save_steps=100, save_total_limit=2,
        report_to='none', remove_unused_columns=False, seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=collate, processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=a.resume)
    model.save_pretrained(a.output, safe_serialization=True)
    tokenizer.save_pretrained(a.output)
    trainer.save_metrics('train', result.metrics)
    Path(a.output, 'cq_load_report.json').write_text(json.dumps(load_report, indent=2))
    Path(a.output, 'cq_training.json').write_text(json.dumps(vars(a), indent=2))

if __name__ == '__main__':
    main()
