"""
vllm-mlx Server Patch for MTP Support.

This module provides functions to patch vllm-mlx's server and CLI
to support MTP speculative decoding via --speculative-config flag.

Phase 3: Continuous batching compatibility.
Strategy: MTP is used for single-request mode only. When continuous
batching is active, MTP is disabled with a warning. This is the
correct approach because MTP's value is reducing single-user latency,
while continuous batching is for throughput with concurrent users.

Usage:
    # In vllm-mlx's server.py or cli.py:
    from vllm_mlx_mtp.server_patch import (
        patch_cli_args,
        create_mtp_wrapper,
        patch_server_routes,
    )
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def patch_cli_args(parser):
    """
    Add --speculative-config argument to an argparse parser.

    Call this on the serve_parser in vllm-mlx's cli.py.
    """
    parser.add_argument(
        "--speculative-config",
        type=str,
        default=None,
        help=(
            'MTP speculative decoding config JSON. '
            'Example: \'{"method":"mtp","num_speculative_tokens":1}\''
        ),
    )
    return parser


def create_mtp_wrapper(model, model_name: str, speculative_config: Optional[str] = None):
    """
    Create an MTP wrapper if speculative config is provided.

    Returns the wrapper (or None) for use in the server's generation path.
    """
    if not speculative_config:
        return None

    from .integration import MTPModelWrapper, parse_speculative_config

    mtp_config = parse_speculative_config(speculative_config)
    if mtp_config is None:
        return None

    wrapper = MTPModelWrapper(model, model_name, mtp_config)
    if wrapper.mtp_available:
        logger.info("MTP speculative decoding enabled")
        return wrapper
    else:
        logger.warning("MTP requested but not available for this model")
        return None


def should_use_mtp(wrapper, continuous_batching: bool = False) -> bool:
    """
    Determine if MTP should be used for the current request.

    MTP is disabled in continuous batching mode (Phase 3, Option A).
    """
    if wrapper is None or not wrapper.mtp_available:
        return False

    if continuous_batching:
        logger.info(
            "MTP speculative decoding disabled in continuous batching mode. "
            "MTP is optimized for single-user latency; continuous batching "
            "is for multi-user throughput."
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Example: How to integrate into vllm-mlx's server.py
# ---------------------------------------------------------------------------
INTEGRATION_EXAMPLE = """
# In vllm-mlx/vllm_mlx/server.py, add these changes:

# 1. At the top, add import:
from vllm_mlx_mtp.server_patch import create_mtp_wrapper, should_use_mtp

# 2. In load_model(), after loading:
_mtp_wrapper = create_mtp_wrapper(model, model_name, speculative_config)

# 3. In the /v1/completions endpoint, before generation:
if should_use_mtp(_mtp_wrapper):
    # Use MTP streaming path
    async def mtp_stream():
        for chunk in _mtp_wrapper.stream_generate_mtp(
            prompt, tokenizer, max_tokens=max_tokens, temperature=temperature
        ):
            yield chunk
    return StreamingResponse(mtp_stream())
else:
    # Use standard generation path
    ...

# 4. In the /health endpoint:
if _mtp_wrapper:
    health_info.update(_mtp_wrapper.get_health_info())

# 5. In cli.py, add --speculative-config to serve_parser:
from vllm_mlx_mtp.server_patch import patch_cli_args
patch_cli_args(serve_parser)
"""
