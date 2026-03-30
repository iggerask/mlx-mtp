"""
vllm-mlx Integration Layer.

Provides the glue between vllm_mlx_mtp and vllm-mlx's server/engine.
This module can be used to patch vllm-mlx for MTP support without
modifying vllm-mlx source code directly.

The default configuration enables all optimizations for maximum throughput:
- MTP K=1 speculative decoding with batch verify + lazy draft
- Fused MoE SIMD kernels (gate+up+SwiGLU and down+reduce in single dispatches)
- Zero-replay rejection (split Metal kernel GDN state capture)
- Q4 quantized MTP head

Usage in vllm-mlx server:
    from vllm_mlx_mtp.integration import MTPModelWrapper

    model, tokenizer = load(model_name)
    wrapper = MTPModelWrapper(model, model_name)
    if wrapper.mtp_available:
        # Use wrapper.stream_generate_mtp() instead of stream_generate()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.cache import make_prompt_cache

from .cache_utils import get_cache_info, trim_hybrid_cache
from .hidden_capture import HiddenStateCapture
from .mtp_decoder import MTPConfig, MTPDecoder, MTPStats
from .mtp_head import MTPHead, build_mtp_head, detect_mtp_support, load_mtp_weights

logger = logging.getLogger(__name__)


def _default_fast_config() -> MTPConfig:
    """Default configuration optimized for maximum throughput on Apple Silicon."""
    return MTPConfig(
        num_speculative_tokens=1,
        batch_verify=True,
        lazy_draft=True,
        zero_replay=True,
        quantize_head=True,
        quantize_head_bits=4,
        quantize_head_group_size=64,
    )


def _try_patch_fused_moe(model, verbose: bool = True) -> bool:
    """Attempt to patch MoE layers with fused SIMD kernels.

    Returns True if patching succeeded, False if the extension isn't available
    or the model doesn't use the expected MoE architecture.
    """
    try:
        from mlx_fused_moe.patch_moe_full import patch_moe_full
        n_patched = patch_moe_full(model, verbose=verbose)
        return n_patched > 0
    except ImportError:
        if verbose:
            logger.info(
                "Fused MoE extension not available. "
                "Build with: pip install -e ./mlx_fused_moe"
            )
        return False
    except Exception as e:
        logger.debug(f"Fused MoE patching skipped: {e}")
        return False


def _try_unpatch_fused_moe():
    """Unpatch fused MoE if it was applied."""
    try:
        from mlx_fused_moe.patch_moe_full import unpatch_moe_full
        unpatch_moe_full()
    except ImportError:
        pass


def _find_local_mtp_weights(model_name: str, config: dict) -> Optional[dict]:
    """Search local mtp_weights/ directory for extracted MTP weight files.

    Tries multiple naming patterns to match quantized model names
    (e.g., mlx-community/Qwen3.5-35B-A3B-4bit) back to their BF16 source
    (e.g., Qwen/Qwen3.5-35B-A3B).
    """
    from .mtp_head import load_mtp_weights_from_file
    import glob

    mtp_dir = Path("mtp_weights")
    if not mtp_dir.exists():
        return None

    # Build candidate names from the model name
    candidates = []
    safe = model_name.replace("/", "_")
    candidates.append(safe)

    # Strip common quantization suffixes (4bit, 8bit, 3bit, bf16, fp16, etc.)
    import re
    base = re.sub(r'[-_](?:\d+bit|bf16|fp16|MLX|mlx|GGUF|gguf)$', '', safe)
    if base != safe:
        candidates.append(base)

    # Strip org prefix for mlx-community repos that mirror other orgs
    for prefix in ("mlx-community_", "mlx_community_"):
        if safe.startswith(prefix):
            stripped = safe[len(prefix):]
            candidates.append(stripped)
            base_stripped = re.sub(r'[-_](?:\d+bit|bf16|fp16|MLX|mlx)$', '', stripped)
            if base_stripped != stripped:
                candidates.append(base_stripped)
                # Also try with original org prefix (Qwen_Qwen3.5-35B-A3B)
                # by looking at config for clues
                pass

    # Try each candidate
    for name in candidates:
        path = mtp_dir / f"{name}.safetensors"
        if path.exists():
            logger.info(f"Found MTP weights: {path}")
            return load_mtp_weights_from_file(path)

    # Fallback: glob for any file containing the model's short name
    # e.g., "35B-A3B" matches "Qwen_Qwen3.5-35B-A3B.safetensors"
    short_parts = model_name.split("/")[-1].split("-")
    # Find the most distinctive part (largest, non-suffix)
    for part in reversed(short_parts):
        if re.match(r'^\d+bit$', part):
            continue
        matches = list(mtp_dir.glob(f"*{part}*.safetensors"))
        if len(matches) == 1:
            logger.info(f"Found MTP weights via fuzzy match: {matches[0]}")
            return load_mtp_weights_from_file(matches[0])

    return None


class MTPModelWrapper:
    """
    Wraps an mlx-lm model with MTP speculative decoding capability.

    Drop-in enhancement for vllm-mlx's model usage. When MTP is available,
    provides an alternative generation path that uses speculative decoding.
    Falls back to standard generation when MTP is not available.

    By default, enables all optimizations (fused MoE, zero-replay, Q4 head)
    for maximum throughput. Pass a custom MTPConfig to override.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        mtp_config: Optional[MTPConfig] = None,
        fused_moe: bool = True,
    ):
        self.model = model
        self.model_name = model_name
        self.mtp_config = mtp_config or _default_fast_config()
        self.mtp_head: Optional[MTPHead] = None
        self.mtp_decoder: Optional[MTPDecoder] = None
        self._config: Optional[dict] = None
        self._stats = MTPStats()
        self._fused_moe_active = False

        # Apply fused MoE kernel patching
        if fused_moe:
            self._fused_moe_active = _try_patch_fused_moe(model)

        self._try_init_mtp()

    def _try_init_mtp(self):
        """Attempt to initialize MTP head from model weights."""
        try:
            # Download/locate model files
            model_path = Path(
                snapshot_download(
                    self.model_name,
                    allow_patterns=["*.json", "model*.safetensors", "mtp_weights.safetensors"],
                )
            )

            # Load config
            with open(model_path / "config.json") as f:
                self._config = json.load(f)

            if not detect_mtp_support(self._config):
                logger.info(f"Model {self.model_name} does not support MTP")
                return

            # Load MTP weights (checks for extracted file first, then model files)
            mtp_weights = load_mtp_weights(model_path)

            # Also check local mtp_weights/ directory with various name patterns
            if not mtp_weights:
                from .mtp_head import load_mtp_weights_from_file
                mtp_weights = _find_local_mtp_weights(self.model_name, self._config)

            if not mtp_weights:
                logger.warning(
                    f"Model config indicates MTP support but no MTP weights found. "
                    f"Run: python scripts/extract_mtp_weights.py --source <original-model>"
                )
                return

            # Build MTP head (norm_shift=True for Qwen3.5 weight format)
            self.mtp_head = build_mtp_head(mtp_weights, self._config, norm_shift=True)
            if self.mtp_head is None:
                logger.warning("Failed to build MTP head from weights")
                return

            # Create decoder (handles Q4 quantization, zero-replay patching, etc.)
            self.mtp_decoder = MTPDecoder(self.model, self.mtp_head, self.mtp_config)
            logger.info(
                f"MTP speculative decoding initialized for {self.model_name}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize MTP: {e}")
            self.mtp_head = None
            self.mtp_decoder = None

    @property
    def mtp_available(self) -> bool:
        """Whether MTP speculative decoding is available."""
        return self.mtp_decoder is not None

    @property
    def stats(self) -> MTPStats:
        """Current MTP statistics."""
        if self.mtp_decoder:
            return self.mtp_decoder.stats
        return self._stats

    def generate_mtp(
        self,
        prompt: str,
        tokenizer,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Tuple[str, MTPStats]:
        """
        Generate text using MTP speculative decoding.

        Args:
            prompt: Input text
            tokenizer: The tokenizer
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_text, statistics)
        """
        if not self.mtp_available:
            raise RuntimeError("MTP not available for this model")

        tokens = tokenizer.encode(prompt)
        prompt_arr = mx.array(tokens)
        cache = make_prompt_cache(self.model)

        # Resolve EOS tokens
        eos_set = set()
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, list):
                eos_set = set(eid)
            elif eid is not None:
                eos_set = {eid}

        generated = []
        for token_id in self.mtp_decoder.generate(
            prompt_arr,
            cache,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_tokens=eos_set,
        ):
            generated.append(token_id)
            if token_id in eos_set:
                break

        output_text = tokenizer.decode(generated)
        return output_text, self.mtp_decoder.stats

    def stream_generate_mtp(
        self,
        prompt: str,
        tokenizer,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Generator[dict, None, None]:
        """
        Stream tokens using MTP speculative decoding.

        Yields dicts compatible with vllm-mlx's streaming format:
            {"text": str, "token": int, "finished": bool, "finish_reason": str|None}
        """
        if not self.mtp_available:
            raise RuntimeError("MTP not available for this model")

        tokens = tokenizer.encode(prompt)
        prompt_arr = mx.array(tokens)
        cache = make_prompt_cache(self.model)

        eos_set = set()
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, list):
                eos_set = set(eid)
            elif eid is not None:
                eos_set = {eid}

        count = 0
        for token_id in self.mtp_decoder.generate(
            prompt_arr,
            cache,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_tokens=eos_set,
        ):
            count += 1
            text = tokenizer.decode([token_id])
            is_eos = token_id in eos_set
            at_limit = count >= max_tokens

            yield {
                "text": text,
                "token": token_id,
                "finished": is_eos or at_limit,
                "finish_reason": "stop" if is_eos else ("length" if at_limit else None),
            }

            if is_eos or at_limit:
                break

    def get_health_info(self) -> dict:
        """Return MTP stats for the /health endpoint."""
        if not self.mtp_available:
            return {"mtp": {"enabled": False}}
        return {"mtp": self.stats.to_dict()}

    def cleanup(self):
        """Release resources and restore model to original state."""
        if self.mtp_decoder:
            self.mtp_decoder.cleanup()
        if self._fused_moe_active:
            _try_unpatch_fused_moe()
            self._fused_moe_active = False


def parse_speculative_config(config_str: str) -> Optional[MTPConfig]:
    """
    Parse --speculative-config CLI argument.

    Expected format: '{"method":"mtp"}' (uses optimized defaults)
    Or with overrides: '{"method":"mtp","zero_replay":false,"quantize_head":false}'
    """
    if not config_str:
        return None

    try:
        data = json.loads(config_str)
    except json.JSONDecodeError:
        logger.error(f"Invalid speculative config JSON: {config_str}")
        return None

    method = data.get("method", "")
    if method not in ("mtp", "qwen3_next_mtp"):
        logger.error(f"Unknown speculative method: {method}")
        return None

    # Start from fast defaults, override with provided values
    cfg = _default_fast_config()
    cfg.method = method
    for key in ("num_speculative_tokens", "batch_verify", "lazy_draft",
                "zero_replay", "cascade_verify", "adaptive_k",
                "adaptive_k_threshold", "quantize_head"):
        if key in data:
            setattr(cfg, key, data[key])
    return cfg
