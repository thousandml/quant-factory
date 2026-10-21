"""Strict HF checkpoint-to-model tensor name resolution.

Match exact names first; otherwise only remove nonsemantic model/language wrappers.
Never resolve by suffix alone: that could silently put weights into the wrong tower.
"""
from checkpoint_layout import normalize, namespace


def resolve_tensor_name(name, available):
    if name in available:
        return name
    scope = namespace(name)
    canonical = normalize(name)
    matches = [candidate for candidate in available
               if namespace(candidate) == scope and normalize(candidate) == canonical]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f'Unknown residual tensor: {name}')
    raise ValueError(f'Ambiguous residual tensor {name}: {matches[:12]}')


def is_unmodeled_auxiliary(name, available_names, text_only=False):
    """Ignore tensors from towers that a text-only skeleton deliberately omits.

    A partially instantiated tower is NOT silently ignored.
    """
    scope = namespace(name)
    return bool(text_only and scope in ('vision', 'audio', 'mtp') and
                not any(namespace(candidate) == scope for candidate in available_names))


def covered_quantized_weight(name, destinations):
    """Return True only when a checkpoint weight is covered by a planned CQ destination.

    Canonicalize *both* names to allow text-only language_model/model wrappers.
    Never skip biases or other tensors, even under a CQ-covered module.
    """
    if not name.endswith('.weight'):
        return False
    module = name[:-len('.weight')]
    scope = namespace(module)
    canonical = normalize(module)
    return any(namespace(dest) == scope and normalize(dest) == canonical
               for dest in destinations)
