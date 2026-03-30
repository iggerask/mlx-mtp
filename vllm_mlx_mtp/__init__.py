"""
MTP Speculative Decoding for vllm-mlx.

Adds Multi-Token Prediction (MTP) speculative decoding support
for Qwen3.5 and GLM-4.7-Flash models on Apple Silicon via MLX.

Quick start:
    from mlx_lm import load
    from vllm_mlx_mtp import MTPModelWrapper

    model, tokenizer = load("mlx-community/Qwen3.5-35B-A3B-4bit")
    wrapper = MTPModelWrapper(model, "mlx-community/Qwen3.5-35B-A3B-4bit")
    text, stats = wrapper.generate_mtp("Hello, world!", tokenizer)
"""

from .mtp_head import MTPHead, MTPMoEMLP, build_mtp_head, load_mtp_weights, detect_mtp_support
from .mtp_decoder import MTPDecoder, MTPConfig, MTPStats
from .integration import MTPModelWrapper, parse_speculative_config
from .cache_utils import trim_hybrid_cache, can_trim_hybrid_cache
from .hidden_capture import HiddenStateCapture
from .gdn_capture import GDNStateCapture

__all__ = [
    # Primary API
    "MTPModelWrapper",
    "parse_speculative_config",
    # Core components
    "MTPHead",
    "MTPMoEMLP",
    "MTPDecoder",
    "MTPConfig",
    "MTPStats",
    "GDNStateCapture",
    # Weight loading
    "build_mtp_head",
    "load_mtp_weights",
    "detect_mtp_support",
    # Cache utilities
    "trim_hybrid_cache",
    "can_trim_hybrid_cache",
    "HiddenStateCapture",
]
