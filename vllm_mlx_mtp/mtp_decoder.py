"""
MTP Speculative Decoder.

Wraps the MTP head and main model into a speculative decode loop that:
1. Runs the MTP head to draft the next token (lazy — no sync)
2. Batch-verifies [current_token, draft] in a single forward pass
3. If draft matches: accept both + bonus token (2 tokens/step)
4. If draft misses: fall back to verified token (1 token/step)

Optimized for maximum throughput on Apple Silicon:
- Lazy draft evaluation: MTP head + model forward in single MLX graph
- Q4 MTP head: 4-bit quantized head reduces draft overhead by ~30%
- Batch verify: always processes 2 tokens in one forward pass
- Sequential fallback: available for bit-exact output when needed

Designed for integration into vllm-mlx's generation path.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from .hidden_capture import HiddenStateCapture
from .mtp_head import MTPHead

logger = logging.getLogger(__name__)


@dataclass
class MTPConfig:
    """Configuration for MTP speculative decoding."""

    method: str = "mtp"
    # Number of tokens to draft per step. 1 = standard MTP-1.
    # 2-3 = chain the MTP head multiple times (accuracy degrades per depth).
    num_speculative_tokens: int = 1
    # Whether to use greedy drafting (recommended for correctness)
    greedy_draft: bool = True
    # Use batch verification (N+1 tokens at once). Faster but may have
    # tiny numerical differences on hybrid GDN+Attention architectures.
    batch_verify: bool = True
    # Use lazy draft evaluation — don't mx.eval(draft) before building the
    # verify graph. Lets MLX fuse the MTP head + model forward into one graph.
    # ~2% faster. Only applies to K=1 batch verify path.
    lazy_draft: bool = True
    # Use cascade verification for K=2: verify draft1 with 2-token batch
    # (same as K=1), then conditionally verify draft2 with 1-token forward.
    # Avoids expensive 3-token batch verify that loads 3x MoE expert weights.
    cascade_verify: bool = False
    # Adaptive K: dynamically switch between K=1 and K=2 based on rolling
    # acceptance rate. When enabled, num_speculative_tokens is the max K.
    adaptive_k: bool = False
    # Threshold for adaptive K to upgrade from K=1 to K=2 (0.0-1.0)
    adaptive_k_threshold: float = 0.90
    # Rolling window size for adaptive K acceptance tracking
    adaptive_k_window: int = 20
    # Zero-replay rejection: splits GDN Metal kernel calls during batch verify
    # to capture intermediate recurrent state, enabling exact cache restoration
    # on rejection without replaying accepted tokens. Eliminates the ~14ms
    # replay forward cost with minimal overhead (~1.5ms from extra dispatches).
    zero_replay: bool = False
    # Quantize MTP head to 4-bit on init. Saves ~30% draft overhead.
    quantize_head: bool = False
    quantize_head_bits: int = 4
    quantize_head_group_size: int = 64


@dataclass
class MTPStats:
    """Runtime statistics for MTP decoding."""

    total_tokens: int = 0
    draft_attempts: int = 0
    draft_accepted: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    prefill_time: float = 0.0
    # Rolling acceptance window for adaptive K
    _rolling_results: list = field(default_factory=list)
    _rolling_window: int = 20
    # Track which K was used (for reporting)
    k1_steps: int = 0
    k2_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_attempts == 0:
            return 0.0
        return self.draft_accepted / self.draft_attempts

    @property
    def tokens_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_tokens / self.total_steps

    @property
    def tokens_per_second(self) -> float:
        decode_time = self.total_time - self.prefill_time
        if decode_time <= 0:
            return 0.0
        return self.total_tokens / decode_time

    def to_dict(self) -> dict:
        return {
            "enabled": True,
            "total_tokens": self.total_tokens,
            "draft_attempts": self.draft_attempts,
            "draft_accepted": self.draft_accepted,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "avg_tokens_per_step": round(self.tokens_per_step, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
        }

    @property
    def rolling_acceptance(self) -> float:
        """Recent acceptance rate over the rolling window."""
        if not self._rolling_results:
            return 1.0  # Optimistic start
        return sum(self._rolling_results) / len(self._rolling_results)

    def record_draft_result(self, accepted: bool):
        """Record a draft accept/reject for rolling window tracking."""
        self._rolling_results.append(1.0 if accepted else 0.0)
        if len(self._rolling_results) > self._rolling_window:
            self._rolling_results.pop(0)

    def __repr__(self):
        return (
            f"MTPStats(tokens={self.total_tokens}, "
            f"acceptance={self.acceptance_rate:.1%}, "
            f"tok/step={self.tokens_per_step:.2f}, "
            f"tok/s={self.tokens_per_second:.1f})"
        )


class MTPDecoder:
    """
    MTP-1 speculative decoder for Qwen3.5 models.

    Manages the draft-verify-accept/reject loop with proper
    KV cache handling for hybrid (GatedDeltaNet + Attention) architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        mtp_head: MTPHead,
        config: MTPConfig = None,
    ):
        self.model = model
        self.mtp_head = mtp_head
        self.config = config or MTPConfig()
        self.stats = MTPStats()

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

        # Hidden state capture
        self._capture = HiddenStateCapture(model)

        # GDN intermediate state capture for zero-replay rejection
        self._gdn_capture = None
        if self.config.zero_replay:
            from .gdn_capture import GDNStateCapture
            self._gdn_capture = GDNStateCapture(model)
            self._gdn_capture.patch()

        # Optionally quantize MTP head for faster drafting (~30% overhead reduction)
        if self.config.quantize_head:
            from .optimizations import quantize_mtp_head
            quantize_mtp_head(
                self.mtp_head,
                bits=self.config.quantize_head_bits,
                group_size=self.config.quantize_head_group_size,
            )
            logger.info("MTP head quantized to %d-bit", self.config.quantize_head_bits)

    def _sample(self, logits: mx.array, temperature: float = 0.0) -> mx.array:
        """Sample a token from logits."""
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits / temperature)

    def _draft_tokens(
        self,
        last_hidden: mx.array,
        current_token: mx.array,
        num_tokens: int,
        temperature: float,
        eos_tokens: Set[int],
    ) -> Tuple[List[mx.array], mx.array]:
        """
        Draft N tokens by chaining the MTP head.

        Returns:
            (draft_tokens, last_draft_hidden)
        """
        drafts = []
        h = last_hidden
        tok = current_token

        for _ in range(num_tokens):
            tok_embed = self._embed_tokens(tok[None])
            if tok_embed.ndim == 2:
                tok_embed = tok_embed[:, None, :]

            h = self.mtp_head(h, tok_embed)
            logits = self._lm_head(h)

            if self.config.greedy_draft:
                tok = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                tok = self._sample(logits[:, -1, :], temperature)

            drafts.append(tok)
            self.stats.draft_attempts += 1

        return drafts, h

    def step(
        self,
        cache: list,
        current_token: mx.array,
        last_hidden: mx.array,
        temperature: float = 0.0,
        eos_tokens: Optional[Set[int]] = None,
    ) -> Tuple[List[int], mx.array, mx.array]:
        """
        Run one MTP speculative decode step.

        Args:
            cache: Model KV cache (hybrid)
            current_token: (1,) or (1,1) current token to process
            last_hidden: (1, 1, hidden_size) hidden state for MTP head
            temperature: Sampling temperature
            eos_tokens: Set of EOS token IDs

        Returns:
            Tuple of:
            - accepted_tokens: List of accepted token IDs
            - next_token: The token to use for the next step (as mx.array)
            - next_hidden: Hidden state for the next MTP draft
        """
        eos_tokens = eos_tokens or set()
        self.stats.total_steps += 1

        if current_token.ndim == 0:
            current_token = current_token.reshape(1)
        token_0 = current_token

        K = self.config.num_speculative_tokens

        # Adaptive K: choose K dynamically based on rolling acceptance
        if self.config.adaptive_k and K >= 2:
            if self.stats.rolling_acceptance >= self.config.adaptive_k_threshold:
                effective_k = 2
            else:
                effective_k = 1
        else:
            effective_k = K

        if self.config.batch_verify and self.config.lazy_draft:
            # --- Fast path: lazy draft + batch verify (best throughput) ---
            if effective_k == 1:
                self.stats.k1_steps += 1
                return self._step_lazy_batch(
                    cache, token_0, last_hidden, temperature, eos_tokens,
                )
            elif effective_k == 2 and self.config.cascade_verify:
                self.stats.k2_steps += 1
                return self._step_cascade_k2(
                    cache, token_0, last_hidden, temperature, eos_tokens,
                )
            else:
                self.stats.k2_steps += 1
                return self._step_lazy_batch_kn(
                    cache, token_0, last_hidden, temperature, eos_tokens,
                    effective_k,
                )

        # --- Standard path: draft first, then verify ---
        draft_tokens, _ = self._draft_tokens(
            last_hidden, token_0, K, temperature, eos_tokens,
        )

        if K == 1:
            draft_token = draft_tokens[0]
            if self.config.batch_verify:
                mx.eval(draft_token)
                return self._step_batch(cache, token_0, draft_token, draft_token.item(), temperature, eos_tokens)
            else:
                return self._step_sequential(cache, token_0, draft_token, temperature, eos_tokens)
        else:
            # Multi-token path: always uses batch verification
            mx.eval(*draft_tokens)
            return self._step_multi(cache, token_0, draft_tokens, temperature, eos_tokens)

    def _step_sequential(self, cache, token_0, draft_token, temperature, eos_tokens):
        """Sequential verify: bit-exact output."""
        # Process token_0 through main model.
        # Build model graph BEFORE eval'ing draft so MLX can overlap computation.
        step1_logits = self.model(token_0.reshape(1, 1), cache=cache)
        step1_hidden = self._capture.get_hidden_state()
        verified_token = self._sample(step1_logits[:, -1, :], temperature)
        # Eval draft and verify together - MLX can parallelize
        mx.eval(draft_token, verified_token, step1_hidden)
        draft_id = draft_token.item()
        verified_id = verified_token.item()

        if draft_id in eos_tokens:
            return [], verified_token.reshape(1), step1_hidden[:, -1:, :]

        if verified_id == draft_id:
            # ACCEPT: also process draft_token
            self.stats.draft_accepted += 1
            step2_logits = self.model(draft_token.reshape(1, 1), cache=cache)
            step2_hidden = self._capture.get_hidden_state()
            mx.eval(step2_logits, step2_hidden)

            bonus_token = self._sample(step2_logits[:, -1, :], temperature)
            mx.eval(bonus_token)

            accepted = [draft_id, bonus_token.item()]
            self.stats.total_tokens += 2
            return accepted, bonus_token.reshape(1), step2_hidden[:, -1:, :]
        else:
            # REJECT: cache already has token_0 processed
            accepted = [verified_id]
            self.stats.total_tokens += 1
            return accepted, verified_token.reshape(1), step1_hidden[:, -1:, :]

    def _step_lazy_batch(self, cache, token_0, last_hidden, temperature, eos_tokens):
        """Lazy draft + batch verify: fastest K=1 path.

        Key optimization: the MTP head draft and model batch-verify are built
        as a single MLX computation graph before any mx.eval(). This lets MLX
        schedule them optimally, saving ~0.5ms per step vs eager eval.

        With zero_replay: captures GDN intermediate state during verify forward,
        enabling exact restoration on rejection without replay forward (~14ms saved).
        """
        from .cache_utils import save_cache_state, restore_cache_state

        zr = self._gdn_capture is not None

        # Build draft graph (lazy — no eval yet)
        tok_embed = self._embed_tokens(token_0[None])
        if tok_embed.ndim == 2:
            tok_embed = tok_embed[:, None, :]
        mtp_h = self.mtp_head(last_hidden, tok_embed)
        mtp_logits = self._lm_head(mtp_h)
        draft_token = self._sample(mtp_logits[:, -1, :], temperature)

        self.stats.draft_attempts += 1

        # Save cache state (needed for non-zero-replay fallback)
        if not zr:
            saved_state = save_cache_state(cache)

        # Enable GDN capture before verify forward
        if zr:
            self._gdn_capture.prepare(cache)

        # Build batch verify graph with lazy draft token
        verify_input = mx.concatenate(
            [token_0.reshape(1, 1), draft_token.reshape(1, 1)], axis=1
        )
        verify_logits = self.model(verify_input, cache=cache)
        verify_hidden = self._capture.get_hidden_state()

        # Sample verified token from position 0 (what model predicts after token_0)
        verified_token = self._sample(verify_logits[:, 0, :], temperature)

        # Evaluate entire graph at once, including GDN intermediates
        eval_args = [draft_token, verified_token, verify_logits, verify_hidden]
        if zr:
            eval_args.extend(self._gdn_capture.get_intermediates())
            self._gdn_capture.disable()
        mx.eval(*eval_args)

        draft_id = draft_token.item()
        verified_id = verified_token.item()

        if draft_id in eos_tokens:
            self.stats.total_tokens += 1
            self.stats.record_draft_result(False)
            if zr:
                # Restore to post-token_0 state, trim KV by 1 (drop draft)
                self._gdn_capture.restore(cache, position=0, n_kv_trim=1)
                return [verified_id], verified_token.reshape(1), verify_hidden[:, 0:1, :]
            else:
                restore_cache_state(cache, saved_state)
                fallback_logits = self.model(token_0.reshape(1, 1), cache=cache)
                fallback_hidden = self._capture.get_hidden_state()
                mx.eval(fallback_logits, fallback_hidden)
                return [verified_id], verified_token.reshape(1), fallback_hidden[:, -1:, :]

        if verified_id == draft_id:
            # ACCEPT: cache contains both tokens, get bonus from position 1
            self.stats.draft_accepted += 1
            self.stats.record_draft_result(True)
            bonus_token = self._sample(verify_logits[:, 1, :], temperature)
            mx.eval(bonus_token)

            accepted = [draft_id, bonus_token.item()]
            self.stats.total_tokens += 2
            return accepted, bonus_token.reshape(1), verify_hidden[:, 1:2, :]
        else:
            # REJECT
            self.stats.record_draft_result(False)
            self.stats.total_tokens += 1

            if zr:
                # Restore GDN to post-token_0 (exact), trim KV by 1
                self._gdn_capture.restore(cache, position=0, n_kv_trim=1)
                return [verified_id], verified_token.reshape(1), verify_hidden[:, 0:1, :]
            else:
                restore_cache_state(cache, saved_state)
                rerun_logits = self.model(token_0.reshape(1, 1), cache=cache)
                rerun_hidden = self._capture.get_hidden_state()
                mx.eval(rerun_logits, rerun_hidden)
                return [verified_id], verified_token.reshape(1), rerun_hidden[:, -1:, :]

    def _step_cascade_k2(self, cache, token_0, last_hidden, temperature, eos_tokens):
        """Cascade K=2: uses 2-token batch verify, checks draft2 from position 1.

        With zero_replay: on draft1 rejection, restore GDN intermediate + trim KV.
        """
        from .cache_utils import save_cache_state, restore_cache_state

        zr = self._gdn_capture is not None

        # Build lazy draft chain (no eval) — 2 MTP head passes
        tok_embed_0 = self._embed_tokens(token_0[None])
        if tok_embed_0.ndim == 2:
            tok_embed_0 = tok_embed_0[:, None, :]
        mtp_h1 = self.mtp_head(last_hidden, tok_embed_0)
        mtp_logits1 = self._lm_head(mtp_h1)
        draft1 = self._sample(mtp_logits1[:, -1, :], temperature)

        tok_embed_1 = self._embed_tokens(draft1[None])
        if tok_embed_1.ndim == 2:
            tok_embed_1 = tok_embed_1[:, None, :]
        mtp_h2 = self.mtp_head(mtp_h1, tok_embed_1)
        mtp_logits2 = self._lm_head(mtp_h2)
        draft2 = self._sample(mtp_logits2[:, -1, :], temperature)

        self.stats.draft_attempts += 2

        if not zr:
            saved_state = save_cache_state(cache)

        if zr:
            self._gdn_capture.prepare(cache)

        # Stage 1: 2-token batch verify [token_0, draft1] — same as K=1
        verify_input = mx.concatenate(
            [token_0.reshape(1, 1), draft1.reshape(1, 1)], axis=1
        )
        verify_logits = self.model(verify_input, cache=cache)
        verify_hidden = self._capture.get_hidden_state()

        verified1 = self._sample(verify_logits[:, 0, :], temperature)

        eval_args = [draft1, draft2, verified1, verify_logits, verify_hidden]
        if zr:
            eval_args.extend(self._gdn_capture.get_intermediates())
            self._gdn_capture.disable()
        mx.eval(*eval_args)

        d1 = draft1.item()
        d2 = draft2.item()
        v1 = verified1.item()

        # Check draft1
        if d1 in eos_tokens or v1 != d1:
            # Draft1 rejected (or EOS)
            self.stats.total_tokens += 1
            self.stats.record_draft_result(False)
            self.stats.record_draft_result(False)
            if zr:
                self._gdn_capture.restore(cache, position=0, n_kv_trim=1)
                return [v1], verified1.reshape(1), verify_hidden[:, 0:1, :]
            else:
                restore_cache_state(cache, saved_state)
                fb = self.model(token_0.reshape(1, 1), cache=cache)
                fbh = self._capture.get_hidden_state()
                mx.eval(fb, fbh)
                return [v1], verified1.reshape(1), fbh[:, -1:, :]

        # Draft1 ACCEPTED — check draft2 from position 1 (FREE!)
        self.stats.draft_accepted += 1
        self.stats.record_draft_result(True)

        verified2 = self._sample(verify_logits[:, 1, :], temperature)
        mx.eval(verified2)
        v2 = verified2.item()

        if d2 not in eos_tokens and v2 == d2:
            # BOTH ACCEPTED — process draft2 with 1-token forward
            self.stats.draft_accepted += 1
            self.stats.record_draft_result(True)

            stage2_logits = self.model(draft2.reshape(1, 1), cache=cache)
            stage2_hidden = self._capture.get_hidden_state()
            mx.eval(stage2_logits, stage2_hidden)

            bonus = self._sample(stage2_logits[:, -1, :], temperature)
            mx.eval(bonus)

            accepted = [d1, d2, bonus.item()]
            self.stats.total_tokens += 3
            return accepted, bonus.reshape(1), stage2_hidden[:, -1:, :]
        else:
            # Draft2 rejected — return [draft1, corrected]. No extra forward.
            self.stats.record_draft_result(False)
            accepted = [d1, v2]
            self.stats.total_tokens += 2
            return accepted, verified2.reshape(1), verify_hidden[:, 1:2, :]

    def _step_lazy_batch_kn(self, cache, token_0, last_hidden, temperature, eos_tokens, K):
        """Lazy draft K tokens + batch verify K+1 positions.

        With zero_replay: captures GDN intermediates at K positions during the
        (K+1)-token verify. On rejection at position i: restore to post-token[0..i]
        state, trim KV by (K - i).
        """
        from .cache_utils import save_cache_state, restore_cache_state

        zr = self._gdn_capture is not None

        # Build lazy draft chain (no eval) — K MTP head passes
        drafts = []
        h = last_hidden
        tok = token_0
        for _ in range(K):
            tok_embed = self._embed_tokens(tok[None])
            if tok_embed.ndim == 2:
                tok_embed = tok_embed[:, None, :]
            h = self.mtp_head(h, tok_embed)
            draft = self._sample(self._lm_head(h)[:, -1, :], temperature)
            drafts.append(draft)
            tok = draft

        self.stats.draft_attempts += K

        if not zr:
            saved_state = save_cache_state(cache)

        if zr:
            self._gdn_capture.prepare(cache)

        # Batch verify [token_0, draft1, ..., draftK] in one forward pass
        verify_input = mx.concatenate(
            [token_0.reshape(1, 1)] + [d.reshape(1, 1) for d in drafts],
            axis=1,
        )
        verify_logits = self.model(verify_input, cache=cache)
        verify_hidden = self._capture.get_hidden_state()

        verified0 = self._sample(verify_logits[:, 0, :], temperature)

        eval_args = [verified0, verify_logits, verify_hidden] + list(drafts)
        if zr:
            eval_args.extend(self._gdn_capture.get_intermediates())
            self._gdn_capture.disable()
        mx.eval(*eval_args)

        draft_ids = [d.item() for d in drafts]
        v0 = verified0.item()

        # Verify each draft position sequentially
        accepted = []
        for i in range(K):
            did = draft_ids[i]

            if did in eos_tokens:
                # EOS in draft — stop here
                for _ in range(i, K):
                    self.stats.record_draft_result(False)
                break

            if i == 0:
                verified_id = v0
            else:
                vi = self._sample(verify_logits[:, i, :], temperature)
                mx.eval(vi)
                verified_id = vi.item()

            if verified_id == did:
                # Draft i accepted
                self.stats.draft_accepted += 1
                self.stats.record_draft_result(True)
                accepted.append(did)
            else:
                # Draft i rejected — accept corrected token, stop
                for _ in range(i, K):
                    self.stats.record_draft_result(False)
                accepted.append(verified_id)
                break
        else:
            # All K drafts accepted — get bonus from position K
            bonus = self._sample(verify_logits[:, K, :], temperature)
            mx.eval(bonus)
            accepted.append(bonus.item())

        n_accepted = len(accepted)
        self.stats.total_tokens += n_accepted

        if n_accepted == K + 1:
            # All accepted + bonus — cache is valid
            next_tok = mx.array([accepted[-1]]).reshape(1)
            return accepted, next_tok, verify_hidden[:, K:K+1, :]

        # Some rejection occurred
        # n_accepted includes the corrected token at the rejection point
        # Number of drafts that were truly accepted = n_accepted - 1
        n_drafts_ok = n_accepted - 1
        n_kv_trim = K - n_drafts_ok

        if zr:
            self._gdn_capture.restore(cache, position=n_drafts_ok, n_kv_trim=n_kv_trim)
            next_tok = mx.array([accepted[-1]]).reshape(1)
            return accepted, next_tok, verify_hidden[:, n_drafts_ok:n_drafts_ok+1, :]
        else:
            restore_cache_state(cache, saved_state)
            if n_drafts_ok > 0:
                replay_tokens = [token_0.reshape(1, 1)] + [mx.array([[did]]) for did in accepted[:n_drafts_ok]]
                replay_input = mx.concatenate(replay_tokens, axis=1)
            else:
                replay_input = token_0.reshape(1, 1)
            replay_logits = self.model(replay_input, cache=cache)
            replay_hidden = self._capture.get_hidden_state()
            mx.eval(replay_logits, replay_hidden)
            next_tok = mx.array([accepted[-1]]).reshape(1)
            return accepted, next_tok, replay_hidden[:, -1:, :]

    def _step_multi(self, cache, token_0, draft_tokens, temperature, eos_tokens):
        """Verify K draft tokens in a single batch forward pass."""
        from .cache_utils import save_cache_state, restore_cache_state

        draft_ids = [d.item() for d in draft_tokens]

        # Check for EOS in drafts — truncate at first EOS
        for i, did in enumerate(draft_ids):
            if did in eos_tokens:
                draft_tokens = draft_tokens[:i]
                draft_ids = draft_ids[:i]
                break

        if not draft_tokens:
            # All drafts were EOS — just process token_0
            logits = self.model(token_0.reshape(1, 1), cache=cache)
            hidden = self._capture.get_hidden_state()
            mx.eval(logits, hidden)
            return [], token_0, hidden[:, -1:, :]

        saved_state = save_cache_state(cache)

        # Build verify input: [token_0, draft_0, draft_1, ..., draft_{K-1}]
        all_tokens = [token_0.reshape(1, 1)] + [d.reshape(1, 1) for d in draft_tokens]
        verify_input = mx.concatenate(all_tokens, axis=1)  # (1, K+1)

        verify_logits = self.model(verify_input, cache=cache)
        verify_hidden = self._capture.get_hidden_state()
        mx.eval(verify_logits, verify_hidden)

        # Verify each draft position:
        # verify_logits[:, 0, :] = logits after processing token_0 → should predict draft_0
        # verify_logits[:, 1, :] = logits after processing draft_0 → should predict draft_1
        # verify_logits[:, K, :] = logits after processing draft_{K-1} → bonus token
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
                # First rejection — accept verified token instead, stop
                accepted.append(verified_id)
                break
        else:
            # All drafts accepted — sample bonus token from position K
            bonus = self._sample(verify_logits[:, len(draft_ids), :], temperature)
            mx.eval(bonus)
            accepted.append(bonus.item())

        self.stats.total_tokens += len(accepted)

        if last_accept_pos == len(draft_ids) - 1:
            # All K drafts accepted — cache is valid, use hidden at last position
            next_token_arr = mx.array([accepted[-1]]).reshape(1)
            next_hidden = verify_hidden[:, len(draft_ids):len(draft_ids)+1, :]
            return accepted, next_token_arr, next_hidden
        else:
            # Partial accept — restore cache, replay accepted tokens
            restore_cache_state(cache, saved_state)
            n_accepted = len(accepted)
            replay_input = mx.concatenate(
                [token_0.reshape(1, 1)] + [mx.array([[tid]]) for tid in accepted[:-1]],
                axis=1,
            )  # (1, n_accepted)
            replay_logits = self.model(replay_input, cache=cache)
            replay_hidden = self._capture.get_hidden_state()
            mx.eval(replay_logits, replay_hidden)

            next_token_arr = mx.array([accepted[-1]]).reshape(1)
            next_hidden = replay_hidden[:, -1:, :]
            return accepted, next_token_arr, next_hidden

    def _step_batch(self, cache, token_0, draft_token, draft_id, temperature, eos_tokens):
        """Batch verify: faster but may have tiny numerical drift on GDN models."""
        from .cache_utils import save_cache_state, restore_cache_state

        if draft_id in eos_tokens:
            fallback_logits = self.model(token_0.reshape(1, 1), cache=cache)
            fallback_hidden = self._capture.get_hidden_state()
            mx.eval(fallback_logits, fallback_hidden)
            return [], token_0, fallback_hidden[:, -1:, :]

        saved_state = save_cache_state(cache)

        # Verify [token_0, draft_token] in one batch
        verify_input = mx.concatenate(
            [token_0.reshape(1, 1), draft_token.reshape(1, 1)], axis=1
        )
        verify_logits = self.model(verify_input, cache=cache)
        verify_hidden = self._capture.get_hidden_state()
        mx.eval(verify_logits, verify_hidden)

        verified_token = self._sample(verify_logits[:, 0, :], temperature)
        mx.eval(verified_token)
        verified_id = verified_token.item()

        if verified_id == draft_id:
            # ACCEPT
            self.stats.draft_accepted += 1
            bonus_token = self._sample(verify_logits[:, 1, :], temperature)
            mx.eval(bonus_token)

            accepted = [draft_id, bonus_token.item()]
            self.stats.total_tokens += 2
            return accepted, bonus_token.reshape(1), verify_hidden[:, 1:2, :]
        else:
            # REJECT: restore cache, process token_0 alone
            restore_cache_state(cache, saved_state)
            rerun_logits = self.model(token_0.reshape(1, 1), cache=cache)
            rerun_hidden = self._capture.get_hidden_state()
            mx.eval(rerun_logits, rerun_hidden)

            accepted = [verified_id]
            self.stats.total_tokens += 1
            return accepted, verified_token.reshape(1), rerun_hidden[:, -1:, :]

    def generate(
        self,
        prompt_tokens: mx.array,
        cache: list,
        max_tokens: int = 256,
        temperature: float = 0.0,
        eos_tokens: Optional[Set[int]] = None,
    ) -> Generator[int, None, None]:
        """
        Generate tokens using MTP speculative decoding.

        Yields individual token IDs as they are accepted.

        Args:
            prompt_tokens: Tokenized prompt as mx.array
            cache: Pre-created model cache
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            eos_tokens: Set of EOS token IDs
        """
        eos_tokens = eos_tokens or set()
        self.stats = MTPStats(
            _rolling_window=self.config.adaptive_k_window,
        )
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
        if self._gdn_capture is not None:
            self._gdn_capture.unpatch()
