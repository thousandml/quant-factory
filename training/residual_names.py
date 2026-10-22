"""Resolve CQ residual names against the instantiated HF model without guessing."""
from checkpoint_layout import _strip_wrappers, namespace


def resolve_residual(name, targets, *, text_only=False):
    if name in targets:
        return name
    # MTP is an optional speculative-decoding head in inference. A standard
    # multimodal HF training skeleton may not instantiate it. Only ignore MTP
    # residuals if that entire namespace is absent; a partially instantiated
    # MTP head must be mapped strictly, not silently skipped.
    if namespace(name) == 'mtp' and not any(namespace(key) == 'mtp' for key in targets):
        return None
    # Text-only skeletons may remove model.language_model wrappers.
    if text_only and namespace(name) != 'backbone':
        return None
    if namespace(name) != 'backbone':
        raise ValueError(f'Unknown residual tensor: {name}')
    normalized = _strip_wrappers(name)
    candidates = [candidate for candidate in targets
                  if namespace(candidate) == 'backbone' and _strip_wrappers(candidate) == normalized]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f'Unknown or ambiguous residual tensor: {name}; candidates={candidates}')
