"""Metadata-only CQ training coverage audit. No proprietary operator imports.

This reports implementation gaps, not an assertion of numerical parity.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from checkpoint_layout import namespace


def classify_record(name, location, *, text_only=False):
    meta = location.get('meta') or {}
    scope = namespace(name)
    if text_only and scope != 'backbone':
        return 'auxiliary_skippable', f'{scope} tower (only if omitted from the constructed CausalLM)'
    kind = str(meta.get('target_kind') or '')
    if 'ngram' in kind.lower() or 'ngram' in name.lower():
        return 'needs_training_bridge', 'N-gram embedding requires specialized autograd and load mapping'
    if meta.get('slice_index') is not None or '.experts.' in name or '.expert_' in name:
        return 'needs_training_bridge', 'routed expert requires expert-aware HF mapping and gradient validation'
    if meta.get('transpose_source'):
        return 'needs_training_bridge', 'transposed source needs verified orientation and gradients'
    shape = meta.get('matrix_shape') or meta.get('original_shape')
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return 'invalid_metadata', 'missing two-dimensional CQ matrix shape'
    try:
        m, n = [int(v) for v in shape]
    except (ValueError, TypeError):
        return 'invalid_metadata', 'noninteger matrix dimensions'
    if m <= 0 or n <= 0:
        return 'invalid_metadata', 'nonpositive matrix dimensions'
    if n % 1024 or m % 2:
        return 'needs_training_bridge', f'[{m}, {n}] outside verified dense training kernel geometry'
    return 'dense_candidate', 'geometry supported; HF mapping and numerical backward still require validation'


def audit(root, *, text_only=False):
    root = Path(root)
    path = root / 'quantization_data' / 'cq_shards.json'
    index = json.loads(path.read_text())
    if index.get('format') != 'cq_safetensors_shards_v1':
        raise ValueError('Unsupported CQ checkpoint index format')
    records = []
    for name, entry in sorted(index.get('layers', {}).items()):
        state, reason = classify_record(name, entry, text_only=text_only)
        records.append({'name': name, 'status': state, 'reason': reason,
                        'target_kind': (entry.get('meta') or {}).get('target_kind'),
                        'shape': (entry.get('meta') or {}).get('matrix_shape') or (entry.get('meta') or {}).get('original_shape')})
    totals = dict(Counter(r['status'] for r in records))
    return {'checkpoint': str(root), 'mode': 'text_only' if text_only else 'full',
            'summary': totals, 'records': records,
            'all_records_dense_candidates': bool(records) and all(r['status'] == 'dense_candidate' for r in records),
            'notice': 'Dense candidates are not certified trainable; HF names, residuals, kernel execution and gradients must pass separate tests.'}


def main():
    p = argparse.ArgumentParser(description='Audit inference CQ records against current training backend')
    p.add_argument('--model', required=True)
    p.add_argument('--text-only', action='store_true')
    p.add_argument('--output')
    a = p.parse_args()
    report = audit(a.model, text_only=a.text_only)
    body = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).write_text(body + '\n')
    print(body)

if __name__ == '__main__':
    main()
