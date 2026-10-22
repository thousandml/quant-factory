"""Numerical safeguards for frozen-CQ LoRA fine-tuning."""
import torch


def prepare_trainable_lora(model):
    """Keep adapter weights and their optimizer state in FP32."""
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not parameter.is_floating_point():
                raise TypeError(f'Trainable parameter {name} is not floating point')
            parameter.data = parameter.data.float()
            count += 1
    if not count:
        raise ValueError('No trainable adapter parameters')
    return count


def install_nonfinite_gradient_guards(model):
    """Raise at the first nonfinite adapter gradient instead of silently saving it."""
    handles = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        def check(gradient, *, parameter_name=name):
            if not torch.isfinite(gradient).all().item():
                raise FloatingPointError(
                    f'Nonfinite LoRA gradient in {parameter_name}; aborting before optimizer step. '
                    'Inspect CQ backward and activation magnitudes; do not disable this check.')
            return gradient
        handles.append(parameter.register_hook(check))
    return handles
