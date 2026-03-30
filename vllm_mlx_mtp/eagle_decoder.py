"""
EAGLE-Style Speculative Decoder using MTP Head.

Enhances basic MTP speculation with EAGLE-inspired features:

1. Configurable chain depth (tree_depth): draft more tokens per step
2. Multi-chain starting from top-K candidates (tree_width): generate
   multiple candidate continuations and verify the most promising one
3. Adaptive depth cutoff (min_confidence): stop drafting when the MTP
   head's confidence drops below a threshold

On Qwen3.5's hybrid architecture (GatedDeltaNet + Attention), true tree
attention is not feasible because GatedDeltaNet layers are sequential.
Instead, we:
- Use sequential chain drafting (MTP head chained D times)
- Optionally generate K parallel chains from top-K first candidates
- Verify the best chain in a single batch forward pass

This captures most of EAGLE's benefit (deeper speculation with higher
acceptance) while working within the hybrid architecture constraints.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

from .hidden_capture import HiddenStateCapture
from .mtp_head import MTPHead
from .cache_utils import save_cache_state, restore_cache_state

logger = logging.getLogger(__name__)


@dataclass
class EAGLEConfig:
    """Configuration for EAGLE-style speculative decoding."""

    # Maximum draft chain depth (number of tokens to draft per step)
    tree_depth: int = 3
    # Number of parallel chains from top-K first candidates
    # 1 = single greedy chain (like standard MTP with higher K)
    # >1 = generate tree_width chains, verify best one
    tree_width: int = 1
    # Minimum MTP softmax probability to continue chaining
    # 0.0 = always chain to full depth
    # 0.3-0.5 = stop when uncertain (adaptive depth)
    min_confidence: float = 0.0
    # Use greedy (argmax) drafting. Non-greedy uses sampling.
    greedy_draft: bool = True


@dataclass
class EAGLEStats:
    """Runtime statistics for EAGLE decoding."""

    total_tokens: int = 0
    draft_attempts: int = 0
    draft_accepted: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    prefill_time: float = 0.0
    total_draft_depth: int = 0  # Sum of draft depths across steps
    chains_generated: int = 0   # Total chains generated

    @property
    def acceptance_rate(self) -> float:
        return self.draft_accepted / self.draft_attempts if self.draft_attempts else 0.0

    @property
    def tokens_per_step(self) -> float:
        return self.total_tokens / self.total_steps if self.total_steps else 0.0

    @property
    def avg_draft_depth(self) -> float:
        return self.total_draft_depth / self.total_steps if self.total_steps else 0.0

    @property
    def tokens_per_second(self) -> float:
        decode_time = self.total_time - self.prefill_time
        return self.total_tokens / decode_time if decode_time > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "enabled": True,
            "total_tokens": self.total_tokens,
            "draft_attempts": self.draft_attempts,
            "draft_accepted": self.draft_accepted,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "avg_tokens_per_step": round(self.tokens_per_step, 2),
            "avg_draft_depth": round(self.avg_draft_depth, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
        }

    def __repr__(self):
        return (
            f"EAGLEStats(tokens={self.total_tokens}, "
            f"acceptance={self.acceptance_rate:.1%}, "
            f"tok/step={self.tokens_per_step:.2f}, "
            f"avg_depth={self.avg_draft_depth:.1f}, "
            f"tok/s={self.tokens_per_second:.1f})"
        )


class EAGLEDecoder:
    """
    EAGLE-style speculative decoder using MTP head.

    Uses the MTP head (structurally identical to EAGLE-1's draft model)
    to generate candidate token chains, then verifies them in a single
    batch forward pass through the main model.
    """

    def __init__(
        self,
        model: nn.Module,
        mtp_head: MTPHead,
        config: EAGLEConfig = None,
    ):
        self.model = model
        self.mtp_head = mtp_head
        self.config = config or EAGLEConfig()
        self.stats = EAGLEStats()

        # Resolve model components
        if hasattr(model, "language_model"):
            self._text_model = model.language_model
        else:
            self._text_model = model

        self._embed_tokens = self._text_model.model.embed_tokens

        if self._text_model.args.tie_word_embeddings:
            self._lm_head = self._text_model.model.embed_tokens.as_linear
        else:
            self._lm_head = self._text_model.lm_head

        self._capture = HiddenStateCapture(model)

    def _sample(self, logits: mx.array, temperature: float = 0.0) -> mx.array:
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits / temperature)

    def _draft_chain(
        self,
        last_hidden: mx.array,
        start_token: mx.array,
        max_depth: int,
        min_confidence: float,
        temperature: float,
        eos_tokens: Set[int],
    ) -> Tuple[List[mx.array], float]:
        """
        Draft a single chain of tokens by chaining the MTP head.

        Returns (draft_tokens, cumulative_probability).
        """
        drafts = []
        cum_prob = 1.0
        h = last_hidden
        tok = start_token

        for _ in range(max_depth):
            tok_embed = self._embed_tokens(tok[None])
            if tok_embed.ndim == 2:
                tok_embed = tok_embed[:, None, :]

            h = self.mtp_head(h, tok_embed)
            logits = self._lm_head(h)
            probs = mx.softmax(logits[:, -1, :], axis=-1)

            if self.config.greedy_draft:
                tok = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                tok = self._sample(logits[:, -1, :], temperature)

            mx.eval(tok, probs)
            confidence = probs[0, tok.item()].item()
            cum_prob *= confidence

            drafts.append(tok)
            self.stats.draft_attempts += 1

            # Adaptive depth cutoff
            if min_confidence > 0 and confidence < min_confidence:
                break

            if tok.item() in eos_tokens:
                break

        return drafts, cum_prob

    def _draft_tree(
        self,
        last_hidden: mx.array,
        current_token: mx.array,
        temperature: float,
        eos_tokens: Set[int],
    ) -> List[Tuple[List[mx.array], float]]:
        """
        Generate tree of candidate chains.

        For tree_width=1: single greedy chain (equivalent to MTPDecoder).
        For tree_width>1: generate multiple chains from top-K first tokens.

        Returns list of (chain_tokens, cumulative_prob) sorted by prob desc.
        """
        cfg = self.config
        chains = []

        if cfg.tree_width <= 1:
            # Single chain: standard greedy drafting
            chain, prob = self._draft_chain(
                last_hidden, current_token, cfg.tree_depth,
                cfg.min_confidence, temperature, eos_tokens,
            )
            chains.append((chain, prob))
        else:
            # Multi-chain: get top-K candidates for first token
            tok_embed = self._embed_tokens(current_token[None])
            if tok_embed.ndim == 2:
                tok_embed = tok_embed[:, None, :]

            h0 = self.mtp_head(last_hidden, tok_embed)
            logits0 = self._lm_head(h0)
            probs0 = mx.softmax(logits0[:, -1, :], axis=-1)

            # Get top-K candidates
            top_k_indices = mx.argpartition(
                -probs0, kth=cfg.tree_width, axis=-1
            )[..., : cfg.tree_width]
            top_k_probs = mx.take_along_axis(probs0, top_k_indices, axis=-1)
            mx.eval(top_k_indices, top_k_probs)

            for i in range(cfg.tree_width):
                first_tok = top_k_indices[0, i].reshape(1)
                first_prob = top_k_probs[0, i].item()

                self.stats.draft_attempts += 1

                if first_tok.item() in eos_tokens:
                    continue

                # Chain remaining depth from this first candidate
                if cfg.tree_depth > 1:
                    rest_chain, rest_prob = self._draft_chain(
                        h0, first_tok, cfg.tree_depth - 1,
                        cfg.min_confidence, temperature, eos_tokens,
                    )
                    chain = [first_tok] + rest_chain
                    cum_prob = first_prob * rest_prob
                else:
                    chain = [first_tok]
                    cum_prob = first_prob

                chains.append((chain, cum_prob))

            self.stats.chains_generated += len(chains)

        # Sort by cumulative probability (best first)
        chains.sort(key=lambda x: x[1], reverse=True)
        return chains

    def step(
        self,
        cache: list,
        current_token: mx.array,
        last_hidden: mx.array,
        temperature: float = 0.0,
        eos_tokens: Optional[Set[int]] = None,
    ) -> Tuple[List[int], mx.array, mx.array]:
        """
        Run one EAGLE-style speculative decode step.

        Returns (accepted_tokens, next_token, next_hidden).
        """
        eos_tokens = eos_tokens or set()
        self.stats.total_steps += 1

        if current_token.ndim == 0:
            current_token = current_token.reshape(1)
        token_0 = current_token

        # Generate candidate chains
        chains = self._draft_tree(last_hidden, token_0, temperature, eos_tokens)

        if not chains or not chains[0][0]:
            # No valid drafts — just process token_0
            logits = self.model(token_0.reshape(1, 1), cache=cache)
            hidden = self._capture.get_hidden_state()
            verified = self._sample(logits[:, -1, :], temperature)
            mx.eval(verified, hidden)
            self.stats.total_tokens += 1
            return [verified.item()], verified.reshape(1), hidden[:, -1:, :]

        # Take best chain and verify
        best_chain, _ = chains[0]
        draft_tokens = best_chain
        self.stats.total_draft_depth += len(draft_tokens)

        # Verify via batch forward pass (same as MTPDecoder._step_multi)
        draft_ids = [d.item() for d in draft_tokens]

        # Truncate at first EOS
        for i, did in enumerate(draft_ids):
            if did in eos_tokens:
                draft_tokens = draft_tokens[:i]
                draft_ids = draft_ids[:i]
                break

        if not draft_tokens:
            logits = self.model(token_0.reshape(1, 1), cache=cache)
            hidden = self._capture.get_hidden_state()
            mx.eval(logits, hidden)
            verified = self._sample(logits[:, -1, :], temperature)
            mx.eval(verified)
            self.stats.total_tokens += 1
            return [verified.item()], verified.reshape(1), hidden[:, -1:, :]

        saved_state = save_cache_state(cache)

        # Build verify input: [token_0, draft_0, ..., draft_{K-1}]
        all_tokens = [token_0.reshape(1, 1)] + [d.reshape(1, 1) for d in draft_tokens]
        verify_input = mx.concatenate(all_tokens, axis=1)

        verify_logits = self.model(verify_input, cache=cache)
        verify_hidden = self._capture.get_hidden_state()
        mx.eval(verify_logits, verify_hidden)

        # Check each draft position
        accepted = []
        last_accept_pos = -1

        for i, draft_id in enumerate(draft_ids):
            verified = self._sample(verify_logits[:, i, :], temperature)
            mx.eval(verified)
            verified_id = verified.item()

            if verified_id == draft_id:
                self.stats.draft_accepted += 1
                accepted.append(draft_id)
                last_accept_pos = i
            else:
                accepted.append(verified_id)
                break
        else:
            # All drafts accepted — sample bonus token
            bonus = self._sample(verify_logits[:, len(draft_ids), :], temperature)
            mx.eval(bonus)
            accepted.append(bonus.item())

        self.stats.total_tokens += len(accepted)

        if last_accept_pos == len(draft_ids) - 1:
            # All accepted — cache is valid
            next_token_arr = mx.array([accepted[-1]]).reshape(1)
            next_hidden = verify_hidden[:, len(draft_ids) : len(draft_ids) + 1, :]
            return accepted, next_token_arr, next_hidden
        else:
            # Partial accept — restore cache, replay accepted tokens
            restore_cache_state(cache, saved_state)
            n_accepted = len(accepted)
            replay_input = mx.concatenate(
                [token_0.reshape(1, 1)]
                + [mx.array([[tid]]) for tid in accepted[:-1]],
                axis=1,
            )
            replay_logits = self.model(replay_input, cache=cache)
            replay_hidden = self._capture.get_hidden_state()
            mx.eval(replay_logits, replay_hidden)

            next_token_arr = mx.array([accepted[-1]]).reshape(1)
            next_hidden = replay_hidden[:, -1:, :]
            return accepted, next_token_arr, next_hidden

    def generate(
        self,
        prompt_tokens: mx.array,
        cache: list,
        max_tokens: int = 256,
        temperature: float = 0.0,
        eos_tokens: Optional[Set[int]] = None,
    ) -> Generator[int, None, None]:
        """Generate tokens using EAGLE-style speculative decoding."""
        eos_tokens = eos_tokens or set()
        self.stats = EAGLEStats()
        start_time = time.perf_counter()

        # Prefill
        logits = self.model(prompt_tokens[None], cache=cache)
        hidden = self._capture.get_hidden_state()
        mx.eval(logits, hidden, *[c.state for c in cache if hasattr(c, "state")])

        self.stats.prefill_time = time.perf_counter() - start_time

        # Sample first token
        token_0 = self._sample(logits[:, -1, :], temperature)
        mx.eval(token_0)
        yield token_0.item()
        self.stats.total_tokens += 1

        last_hidden = hidden[:, -1:, :]
        current_token = token_0.reshape(1)
        generated = 1

        while generated < max_tokens:
            if current_token.item() in eos_tokens:
                break

            accepted, next_token, next_hidden = self.step(
                cache=cache,
                current_token=current_token,
                last_hidden=last_hidden,
                temperature=temperature,
                eos_tokens=eos_tokens,
            )

            for tid in accepted:
                if generated >= max_tokens:
                    break
                yield tid
                generated += 1
                if tid in eos_tokens:
                    self.stats.total_time = time.perf_counter() - start_time
                    return

            current_token = next_token
            last_hidden = next_hidden

        self.stats.total_time = time.perf_counter() - start_time

    def cleanup(self):
        """Restore model to original state."""
        self._capture.restore()
