"""GPU auto-detection and optimal training settings.

Auto-detects NVIDIA vs AMD GPUs and configures torch.compile and SDPA
backend for best performance. ROCm's flash attention backward has stride
mismatches with torch.compile, so we fall back to the MATH SDPA backend
on AMD GPUs (still ~30% faster than no compile at all).

Raises RuntimeError if no GPU is detected at runtime, unless the
environment variable PAWN_ALLOW_CPU=1 is set.
"""

import os


def is_rocm() -> bool:
    """Check if the current CUDA device is AMD/ROCm."""
    import torch
    if not torch.cuda.is_available():
        return False
    return getattr(torch.version, "hip", None) is not None


def configure_gpu(
    device: str = "cuda",
    *,
    no_compile: bool = False,
    no_amp: bool = False,
    sdpa_math: bool = False,
) -> dict:
    """Auto-detect GPU and return optimal training settings.

    Returns a dict with:
        use_compile: bool — whether to torch.compile the forward pass
        use_amp: bool — whether to use automatic mixed precision
        sdpa_backend: SDPBackend | None — SDPA backend override (None = default)

    CLI flags (no_compile, no_amp, sdpa_math) act as overrides. When not
    set, the function picks the fastest settings for the detected GPU:

        NVIDIA: compile + AMP + flash attention (default SDPA)
        AMD:    compile + AMP + MATH SDPA (avoids flash attn backward bug)
        CPU:    no compile, no AMP (requires PAWN_ALLOW_CPU=1)
    """
    import torch
    from torch.nn.attention import SDPBackend

    is_cuda = torch.cuda.is_available()
    rocm = is_cuda and is_rocm()

    # Defaults: compile and AMP on for CUDA
    use_compile = is_cuda and not no_compile
    use_amp = is_cuda and not no_amp

    # SDPA backend: use MATH on ROCm (flash attn backward is broken with compile)
    if sdpa_math:
        sdpa_backend = SDPBackend.MATH
    elif rocm and use_compile:
        sdpa_backend = SDPBackend.MATH
    else:
        sdpa_backend = None

    # Log what we're doing
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        platform = "ROCm" if rocm else "CUDA"
        print(f"GPU: {gpu_name} ({platform})")
    elif os.environ.get("PAWN_ALLOW_CPU") == "1":
        print("GPU: none (CPU mode — PAWN_ALLOW_CPU=1)")
    else:
        raise RuntimeError(
            "No GPU available. Training and evaluation require a CUDA or ROCm GPU.\n"
            "Set PAWN_ALLOW_CPU=1 to override."
        )

    if use_compile:
        print(f"  torch.compile: enabled (inductor)")
    else:
        print(f"  torch.compile: disabled")

    if use_amp:
        print(f"  AMP: enabled (fp16)")
    else:
        print(f"  AMP: disabled")

    if sdpa_backend is not None:
        print(f"  SDPA backend: {sdpa_backend.name}")
    else:
        print(f"  SDPA backend: default")

    return {
        "use_compile": use_compile,
        "use_amp": use_amp,
        "sdpa_backend": sdpa_backend,
    }


def apply_gpu_config(config: dict, model_module, forward_fn):
    """Apply GPU config: set SDPA backend and optionally compile forward_fn.

    Args:
        config: dict from configure_gpu()
        model_module: the pawn.model module (to set SDPA_BACKEND)
        forward_fn: the forward function to compile (e.g. model.forward_hidden)

    Returns:
        The (possibly compiled) forward function.
    """
    # IMPORTANT: Set SDPA_BACKEND before torch.compile — compiled code
    # captures the backend at trace time.
    if config["sdpa_backend"] is not None:
        model_module.SDPA_BACKEND = config["sdpa_backend"]

    if config["use_compile"]:
        import torch
        forward_fn = torch.compile(forward_fn)

    return forward_fn
