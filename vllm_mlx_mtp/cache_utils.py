"""
Cache utilities for hybrid architectures (GatedDeltaNet + Attention).

Qwen3.5 uses a hybrid architecture:
- GatedDeltaNet layers: fixed-size recurrent state (ArraysCache) - needs save/restore
- Attention layers: standard KV cache (KVCache) - can be trimmed

For speculative decoding, we must:
1. Save ALL cache state before the verify step
2. On accept: keep the new state
3. On reject: restore saved state + trim KV cache
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import KVCache, ArraysCache


def trim_hybrid_cache(cache: List[Any], n: int) -> int:
    """
    Trim n tokens from a hybrid cache.

    Only KVCache entries (attention layers) are trimmed.
    ArraysCache entries (GatedDeltaNet recurrent state) are left unchanged.
    """
    trimmed = 0
    for c in cache:
        if isinstance(c, KVCache) and c.is_trimmable():
            trimmed = c.trim(n)
    return trimmed


def save_cache_state(cache: List[Any]) -> List[Any]:
    """
    Save a snapshot of the entire hybrid cache state.

    For KVCache: saves offset (keys/values are not copied — trim handles rollback)
    For ArraysCache: deep copies the recurrent state arrays and evaluates them
                     immediately to ensure they are materialized before any
                     subsequent operations modify the computation graph.

    Returns a list of saved states, one per cache entry.
    """
    saved = []
    all_copies = []
    for c in cache:
        if isinstance(c, KVCache):
            saved.append(("kv", c.offset))
        elif isinstance(c, ArraysCache):
            state_copy = []
            for item in c.cache:
                if item is not None:
                    copy = mx.array(item)
                    state_copy.append(copy)
                    all_copies.append(copy)
                else:
                    state_copy.append(None)
            saved.append(("arrays", state_copy))
        else:
            saved.append(("unknown", None))
    # Materialize all copies immediately so they're independent of future ops
    if all_copies:
        mx.eval(*all_copies)
    return saved


def restore_cache_state(cache: List[Any], saved: List[Any]):
    """
    Restore cache state from a snapshot.

    For KVCache: restores offset (effectively trimming)
    For ArraysCache: restores the recurrent state arrays
    """
    for c, (cache_type, state) in zip(cache, saved):
        if cache_type == "kv" and isinstance(c, KVCache):
            c.offset = state
        elif cache_type == "arrays" and isinstance(c, ArraysCache):
            for i, item in enumerate(state):
                c.cache[i] = item


def lossy_rollback(cache: List[Any], saved: List[Any], n_kv_trim: int):
    """
    Cheap cache rollback: restore GDN state from snapshot, trim KV by n_kv_trim.

    This avoids the expensive replay forward on rejection. The GDN state is
    restored to pre-verify (before token_0), so it MISSES token_0's contribution.
    This is less corrupting than keeping the wrong draft token's state, because
    the delta-rule decay naturally ages out the missing update.

    Args:
        cache: Active cache list
        saved: Snapshot from save_cache_state
        n_kv_trim: Number of tokens to remove from KV cache (1 for K=1, 2 for K=2 batch)
    """
    for c, (cache_type, state) in zip(cache, saved):
        if cache_type == "kv" and isinstance(c, KVCache):
            # Trim KV by n_kv_trim instead of full restore (keep token_0's KV)
            c.offset = max(0, c.offset - n_kv_trim)
        elif cache_type == "arrays" and isinstance(c, ArraysCache):
            # Restore GDN state to pre-verify snapshot
            for i, item in enumerate(state):
                c.cache[i] = item


def can_trim_hybrid_cache(cache: List[Any]) -> bool:
    """Check if the hybrid cache has any trimmable (KVCache) entries."""
    return any(isinstance(c, KVCache) for c in cache)


def get_cache_info(cache: List[Any]) -> dict:
    """Get diagnostic info about a hybrid cache."""
    kv_count = sum(1 for c in cache if isinstance(c, KVCache))
    arrays_count = sum(1 for c in cache if isinstance(c, ArraysCache))
    kv_offset = None
    for c in cache:
        if isinstance(c, KVCache) and c.offset > 0:
            kv_offset = c.offset
            break
    return {
        "total_layers": len(cache),
        "kv_cache_layers": kv_count,
        "arrays_cache_layers": arrays_count,
        "kv_offset": kv_offset,
    }
