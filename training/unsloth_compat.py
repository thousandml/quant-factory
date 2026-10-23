"""Opt-in Unsloth-adjacent training acceleration; never replace CQ quantized layers.

These paths are intentionally fail-closed: an unsupported installed Unsloth API
must not silently change numerical behavior or silently fall back to native.
"""
import importlib


def configure_checkpointing(mode):
    if mode == "native":
        return
    if mode != "unsloth":
        raise ValueError(f"Unknown checkpointing mode: {mode}")
    try:
        import unsloth
        module = importlib.import_module("unsloth_zoo.gradient_checkpointing")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "--checkpointing unsloth requires a compatible unsloth-zoo installation; "
            "use --checkpointing native to retain the tested CQ path"
        ) from exc
    patch = getattr(module, "patch_unsloth_smart_gradient_checkpointing", None)
    if not callable(patch):
        raise RuntimeError(
            "Installed unsloth-zoo lacks patch_unsloth_smart_gradient_checkpointing; "
            "use --checkpointing native or install a compatible version"
        )
    patch()


def optimizer_name(name):
    """Select a CQ-compatible optimizer; never require bitsandbytes."""
    if name not in ("adamw_torch", "adamw_torch_fused"):
        raise ValueError(f"Unsupported CQ optimizer: {name}")
    if name == "adamw_torch_fused":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Fused AdamW requires CUDA")
        try:
            from torch.optim import AdamW
            import inspect
            if "fused" not in inspect.signature(AdamW).parameters:
                raise RuntimeError("Installed PyTorch does not expose fused AdamW")
        except ImportError as exc:
            raise RuntimeError("Fused AdamW unavailable") from exc
    return name
