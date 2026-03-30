# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-optimized MTP K=1 decode loop with zero-replay.

Supports optional top-K cached f16 draft to eliminate the 1.3ms LM head cost.
"""

import time as _time
import mlx.core as mx


def generate_k1_zr(
    object model,
    object mtp_head,
    object capture,        # HiddenStateCapture
    object gdn_cap,        # GDNStateCapture
    object embed_fn,       # model.language_model.model.embed_tokens
    object lm_head_fn,     # model.language_model.lm_head (or as_linear)
    object prompt_tokens,  # mx.array
    list cache,
    int max_tokens,
    float temperature,
    set eos_tokens,
    object stats,          # MTPStats
    # Adaptive K=2 support
    bint use_adaptive,
    float adaptive_threshold,
    object step_kn_fn,     # MTPDecoder._step_lazy_batch_kn (bound method)
    # Top-K cached f16 draft support
    int draft_top_k = 0,
    object lm_head_f16 = None,       # (vocab, hidden) f16 full LM head
    object update_topk_fn = None,    # MTPDecoder._update_top_k_candidates
    object draft_from_fn = None,     # MTPDecoder._draft_from_candidates
    object get_topk_indices_fn = None,  # lambda: decoder._cached_top_k_indices
):
    """Cython K=1 ZR decode loop with optional top-K f16 draft."""

    # ── Local references ──
    cdef object _eval = mx.eval
    cdef object _argmax = mx.argmax
    cdef object _concat = mx.concatenate
    cdef object _argpartition = mx.argpartition
    cdef object _prepare = gdn_cap.prepare
    cdef object _disable = gdn_cap.disable
    cdef object _restore = gdn_cap.restore
    cdef object _get_ims = gdn_cap.get_intermediates
    cdef object _get_hidden = capture.get_hidden_state
    cdef object _record = stats.record_draft_result

    # ── C-typed counters ──
    cdef int c_total_tokens = 0
    cdef int c_draft_attempts = 0
    cdef int c_draft_accepted = 0
    cdef int c_total_steps = 0
    cdef int c_k1_steps = 0
    cdef int c_k2_steps = 0

    cdef int generated = 0
    cdef int draft_id, verified_id, bonus_id, tid
    cdef bint hit_eos = False
    cdef bint use_topk = draft_top_k > 0 and lm_head_f16 is not None

    cdef double start_time = _time.perf_counter()

    # ── Prefill ──
    logits = model(prompt_tokens[None], cache=cache)
    hidden = _get_hidden()
    _eval(logits, hidden, *[c.state for c in cache if hasattr(c, 'state')])
    stats.prefill_time = _time.perf_counter() - start_time

    # ── First token ──
    tok = _argmax(logits[:, -1, :], axis=-1)
    _eval(tok)
    tid = tok.item()
    yield tid
    c_total_tokens += 1
    generated = 1

    if tid in eos_tokens:
        stats.total_tokens += c_total_tokens
        stats.total_time = _time.perf_counter() - start_time
        return

    # Initialize top-K candidates from prefill logits
    if use_topk and update_topk_fn is not None:
        update_topk_fn(logits[:, -1, :])

    last_hidden = hidden[:, -1:, :]
    cur_tok = tok.reshape(1)

    # ── Main decode loop ──
    while generated < max_tokens:
        c_total_steps += 1

        # Adaptive K=2 check
        if use_adaptive and stats.rolling_acceptance >= adaptive_threshold:
            c_k2_steps += 1
            stats.total_tokens += c_total_tokens
            stats.draft_attempts += c_draft_attempts
            stats.draft_accepted += c_draft_accepted
            stats.total_steps += c_total_steps
            stats.k1_steps += c_k1_steps
            stats.k2_steps += c_k2_steps
            c_total_tokens = 0
            c_draft_attempts = 0
            c_draft_accepted = 0
            c_total_steps = 0
            c_k1_steps = 0
            c_k2_steps = 0

            accepted, cur_tok, last_hidden = step_kn_fn(
                cache, cur_tok, last_hidden, temperature, eos_tokens, 2,
            )
        else:
            c_k1_steps += 1
            c_draft_attempts += 1

            # ── Build lazy draft graph ──
            tok_embed = embed_fn(cur_tok[None])
            if tok_embed.ndim == 2:
                tok_embed = tok_embed[:, None, :]
            mtp_h = mtp_head(last_hidden, tok_embed)

            if use_topk and get_topk_indices_fn is not None and get_topk_indices_fn() is not None:
                # Fast path: project against cached f16 sub-matrix
                draft = draft_from_fn(mtp_h, temperature)
            else:
                # Full LM head (first step or fallback)
                mtp_logits = lm_head_fn(mtp_h)
                draft = _argmax(mtp_logits[:, -1, :], axis=-1)

            # ── Build verify graph with GDN capture ──
            _prepare(cache)
            verify_input = _concat(
                [cur_tok.reshape(1, 1), draft.reshape(1, 1)], axis=1
            )
            verify_logits = model(verify_input, cache=cache)
            verify_hidden = _get_hidden()
            verified = _argmax(verify_logits[:, 0, :], axis=-1)
            bonus = _argmax(verify_logits[:, 1, :], axis=-1)

            # ── Single eval ──
            ims = _get_ims()
            _disable()
            if use_topk:
                _eval(draft, verified, bonus, verify_hidden, verify_logits, *ims)
            else:
                _eval(draft, verified, bonus, verify_hidden, *ims)

            draft_id = draft.item()
            verified_id = verified.item()

            if draft_id in eos_tokens:
                c_total_tokens += 1
                _record(False)
                if use_topk:
                    update_topk_fn(verify_logits[:, 0, :])
                _restore(cache, position=0, n_kv_trim=1)
                accepted = [verified_id]
                cur_tok = verified.reshape(1)
                last_hidden = verify_hidden[:, 0:1, :]
            elif verified_id == draft_id:
                # ACCEPT
                c_draft_accepted += 1
                _record(True)
                bonus_id = bonus.item()
                c_total_tokens += 2
                if use_topk:
                    update_topk_fn(verify_logits[:, 1, :])
                accepted = [draft_id, bonus_id]
                cur_tok = bonus.reshape(1)
                last_hidden = verify_hidden[:, 1:2, :]
            else:
                # REJECT
                _record(False)
                c_total_tokens += 1
                if use_topk:
                    update_topk_fn(verify_logits[:, 0, :])
                _restore(cache, position=0, n_kv_trim=1)
                accepted = [verified_id]
                cur_tok = verified.reshape(1)
                last_hidden = verify_hidden[:, 0:1, :]

        # ── Yield accepted tokens ──
        for tid in accepted:
            if generated >= max_tokens:
                break
            yield tid
            generated += 1
            if tid in eos_tokens:
                hit_eos = True
                break

        if hit_eos:
            break

    # ── Flush counters ──
    stats.total_tokens += c_total_tokens
    stats.draft_attempts += c_draft_attempts
    stats.draft_accepted += c_draft_accepted
    stats.total_steps += c_total_steps
    stats.k1_steps += c_k1_steps
    stats.k2_steps += c_k2_steps
    stats.total_time = _time.perf_counter() - start_time
