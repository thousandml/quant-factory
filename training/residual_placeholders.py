"""Validate zero-sized dense placeholders for CQ-backed projections."""

def is_cq_placeholder(name, shape, destination, cq_destinations):
    """Return True only for an empty .weight whose destination has CQ coverage.

    Empty tensors elsewhere are errors, not missing weights to silently ignore.
    """
    if not shape or all(dim != 0 for dim in shape):
        return False
    if name.endswith('.weight') and destination.endswith('.weight') and destination[:-7] in cq_destinations:
        return True
    raise ValueError(
        f'Unexpected empty residual tensor: {name} -> {destination}; '
        f'checkpoint_shape={list(shape)}; destination has no matching CQ weight record'
    )
