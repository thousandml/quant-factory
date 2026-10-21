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
