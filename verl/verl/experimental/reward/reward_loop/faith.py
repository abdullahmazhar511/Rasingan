# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
import logging
import os
import requests

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward.reward_loop import register as register_loop
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register as register_manager
import math

logger = logging.getLogger(__file__)

# Define the 6 CARE labels for therapy quality evaluation
CARE_LABELS = [
    "Non-Judgmental Language",
    "Warmth and Encouragement",
    "Respect for Autonomy",
    "Active Listening",
    "Reflecting Feelings",
    "Situational Appropriateness"
]

# CARE predictions/labels are discrete in [-2, 2].
CARE_LABEL_MIN = -2.0
CARE_LABEL_MAX = 2.0

# Hybrid reward: total = (1 - EMB_WEIGHT) * care_reward + EMB_WEIGHT * emb_reward.
# Embedding similarity gives a continuous, classifier-independent gradient
# that fills in regions where the quantized CARE reward is locally flat.
# Override via env var EMB_REWARD_WEIGHT (e.g. 0.4 for 60/40 CARE/embedding).
EMB_REWARD_WEIGHT = float(os.environ.get("EMB_REWARD_WEIGHT", "0.5"))


# ============================================================================
# Multi-turn reward = IR coverage + CTRS-P (Cognitive Therapy Rating Scale).
#
# When the agent loop produces a multi-turn rollout it puts the full
# conversation + checklist state into AgentLoopOutput.extra_fields, which the
# trainer surfaces as data_item.non_tensor_batch["tool_extra_fields"] and
# merges into extra_info. We detect that case here and switch to a
# session-level reward instead of single-utterance CARE distance.
#
# Mapping (matches scores/score_ctrs.py exactly, applied to argmax CARE scores
# per therapist turn):
#   Understanding (U)              = mean(AL, RF)  per turn
#   Interpersonal Effectiveness    = mean(WE, NJ)  per turn
#   Collaboration (C)              = RA            per turn
#   Technical Appropriateness (TA) = SA            per turn
# Per-construct score_k = Q_k - F_k + 0.5 * SP_k, where
#   Q_k  = mean(values over therapist turns)
#   F_k  = rate(values <= 0)
#   SP_k = rate(values == 2)
# CTRS-P = 0.35*U + 0.25*IE + 0.20*C + 0.20*TA   (range ~ [-1, 2.5])
# IR     = fraction of 17-item PHQ-9+GAD-7 checklist covered, in [0, 1].
# Total reward = IR + CTRS_P  (user spec).
# ============================================================================
CARE_ABBREV = {
    "Non-Judgmental Language":     "NJ",
    "Warmth and Encouragement":    "WE",
    "Respect for Autonomy":        "RA",
    "Active Listening":            "AL",
    "Reflecting Feelings":         "RF",
    "Situational Appropriateness": "SA",
}


def _construct_score(values):
    """Compute Q - F + 0.5*SP for a vector of per-turn values."""
    import statistics as _st
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    Q  = sum(values) / n
    F  = sum(1 for v in values if v <= 0) / n
    SP = sum(1 for v in values if v == 2) / n
    return Q, F, SP, Q - F + 0.5 * SP


def _ctrs_p_from_per_turn(per_turn):
    """per_turn: list of dicts {NJ, WE, RA, AL, RF, SA}. Returns (CTRS_P, sub_scores)."""
    if not per_turn:
        return 0.0, {"U": 0.0, "IE": 0.0, "C": 0.0, "TA": 0.0}
    U_vals  = [(t["AL"] + t["RF"]) / 2.0 for t in per_turn]
    IE_vals = [(t["WE"] + t["NJ"]) / 2.0 for t in per_turn]
    C_vals  = [t["RA"] for t in per_turn]
    TA_vals = [t["SA"] for t in per_turn]
    _, _, _, U_s  = _construct_score(U_vals)
    _, _, _, IE_s = _construct_score(IE_vals)
    _, _, _, C_s  = _construct_score(C_vals)
    _, _, _, TA_s = _construct_score(TA_vals)
    ctrs_p = 0.35 * U_s + 0.25 * IE_s + 0.20 * C_s + 0.20 * TA_s
    return ctrs_p, {"U": U_s, "IE": IE_s, "C": C_s, "TA": TA_s}


def _extract_therapist_pairs(conversation):
    """From a list of {role, content} dicts, return [(context_str, therapist_utt), ...]
    for each assistant turn; context_str is the prior turns formatted as
    'Patient: ...\\nTherapist: ...\\n...' (consistent with score_ctrs.py)."""
    pairs = []
    history = []
    for msg in conversation:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role == "assistant":
            ctx = "\n".join(history) if history else ""
            pairs.append((ctx, content))
            history.append(f"Therapist: {content}")
        elif role == "user":
            history.append(f"Patient: {content}")
        # ignore system / other roles in the context window
    return pairs


class AsyncTokenBucket:
    """Async token bucket for rate limiting with variable token consumption.

    The token bucket algorithm is a classic rate limiting technique that allows
    for burst traffic while maintaining an average rate limit. This implementation
    is async-first and thread-safe, designed for use in concurrent environments.

    The bucket starts full and refills at a constant rate (rate_limit tokens/second).
    When tokens are acquired, they are consumed from the bucket. If insufficient
    tokens are available, the acquire() method will sleep until enough tokens
    have been refilled.

    This implementation supports variable token consumption, making it suitable
    for rate limiting based on request size (e.g., API token usage).

    Args:
        rate_limit (float): The rate at which tokens are added to the bucket,
            in tokens per second. For example, rate_limit=10.0 means 10 tokens
            are added per second (or 600 per minute).
        max_tokens (float, optional): The maximum capacity of the token bucket.
            Defaults to rate_limit if not specified. This value determines the
            maximum burst size allowed.

    Attributes:
        rate_limit (float): Tokens added per second.
        max_tokens (float): Maximum bucket capacity.
        tokens (float): Current number of available tokens.
        last_update (float | None): Timestamp of last token update (from event loop).
        lock (asyncio.Lock): Async lock for thread-safe token operations.

    Example:
        >>> # Limit to 60 requests per minute (1 request per second)
        >>> rpm_limiter = AsyncTokenBucket(rate_limit=1.0, max_tokens=1.0)
        >>> await rpm_limiter.acquire(1.0)  # Consumes 1 token
        >>>
        >>> # Limit to 10000 tokens per minute (~166.67 tokens per second)
        >>> tpm_limiter = AsyncTokenBucket(rate_limit=166.67, max_tokens=166.67)
        >>> await tpm_limiter.acquire(100.0)  # Consumes 100 tokens

    Thread Safety:
        All operations are protected by an asyncio.Lock, making this class safe
        for concurrent use across multiple coroutines.

    Algorithm Details:
        1. On each acquire(), calculate elapsed time since last update
        2. Refill tokens: tokens += elapsed * rate_limit (capped at max_tokens)
        3. If tokens >= num_tokens: consume tokens and return
        4. Otherwise: calculate wait_time = tokens_needed / rate_limit, then sleep
        5. Retry after sleep (loop back to step 1)
    """

    def __init__(self, rate_limit: float, max_tokens: float = None):
        self.rate_limit = rate_limit
        self.max_tokens = max_tokens or rate_limit
        self.tokens = self.max_tokens
        self.last_update = None
        self.lock = asyncio.Lock()

    async def acquire(self, num_tokens: float = 1.0) -> None:
        """Acquire tokens from the bucket, waiting if necessary.

        This method will block (using asyncio.sleep) until sufficient tokens
        are available. It automatically refills tokens based on elapsed time
        and the configured rate_limit.

        For requests exceeding max_tokens, the method will wait for enough time
        to accumulate the required tokens at the configured rate_limit, allowing
        tokens to temporarily go negative.

        Args:
            num_tokens (float): Number of tokens to consume. Defaults to 1.0.
                Can be fractional for fine-grained rate limiting.

        Returns:
            None: Returns when tokens have been successfully acquired.

        Raises:
            No exceptions are raised. This method will wait indefinitely until
            tokens become available.

        Example:
            >>> bucket = AsyncTokenBucket(rate_limit=10.0)
            >>> await bucket.acquire(5.0)  # Acquire 5 tokens
            >>> await bucket.acquire(1.0)  # Acquire 1 more token

        Implementation Notes:
            - Uses event loop's time() for high-precision timestamps
            - Lock is released during sleep to allow other coroutines to proceed
            - Tokens are refilled continuously based on elapsed time
            - For requests > max_tokens, allows temporary negative balance
        """
        # Handle requests larger than max_tokens separately
        if num_tokens > self.max_tokens:
            wait_time = 0.0
            async with self.lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                if self.last_update is None:
                    self.last_update = now

                elapsed = now - self.last_update
                new_tokens = elapsed * self.rate_limit
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)

                tokens_needed = num_tokens - self.tokens
                if tokens_needed > 0:
                    wait_time = tokens_needed / self.rate_limit

                self.tokens -= num_tokens
                self.last_update = now

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            return

        # Standard case: request <= max_tokens
        while True:
            wait_time = 0.0
            async with self.lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                if self.last_update is None:
                    self.last_update = now

                elapsed = now - self.last_update
                new_tokens = elapsed * self.rate_limit
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)
                self.last_update = now

                if self.tokens >= num_tokens:
                    self.tokens -= num_tokens
                    return

                tokens_needed = num_tokens - self.tokens
                wait_time = tokens_needed / self.rate_limit

            if wait_time > 0:
                await asyncio.sleep(wait_time)


@register_loop("care")
@register_manager("care")
class CareRewardLoopManager(RewardLoopManagerBase):
    """Reward loop manager with rate limiting for API-based reward functions.

    This manager implements a sophisticated three-layer rate limiting system
    designed for LLM-as-judge scenarios where reward computation involves
    external API calls (e.g., OpenAI, Anthropic, Claude) that have rate limits.

    The three layers of rate limiting are:
        1. **Concurrency limiting** (max_concurrent): Limits the number of
           simultaneous API requests using asyncio.Semaphore. This prevents
           overwhelming the API with too many parallel connections.

        2. **Request rate limiting** (max_rpm): Limits requests per minute
           using AsyncTokenBucket. Each request consumes 1 token. Useful for
           APIs with per-minute request quotas.

        3. **Token rate limiting** (max_tpm): Limits tokens per minute using
           AsyncTokenBucket. Each request consumes estimated_tokens_per_request
           tokens. Essential for APIs that bill or limit based on token usage
           (e.g., GPT-4 API).

    All rate limiters are **global class-level resources**, meaning they are
    shared across all instances of this manager. This ensures that rate limits
    are enforced consistently across multiple workers in distributed training.

    Rate Limiting Flow:
        When processing a reward request, the manager:
        1. Acquires RPM token (if rpm_limiter enabled)
        2. Acquires TPM tokens (if tpm_limiter enabled)
        3. Acquires concurrency semaphore
        4. Executes reward computation with timeout
        5. Releases concurrency semaphore
        6. Tokens are automatically refilled by the token buckets

    Args:
        config (DictConfig): Configuration object containing reward_model settings:
            - max_concurrent (int): Max parallel requests. Default: 1
            - max_rpm (int | None): Max requests per minute. Default: None (unlimited)
            - max_tpm (int | None): Max tokens per minute. Default: None (unlimited)
            - estimated_tokens_per_request (int): Estimated tokens per request for
              TPM limiting. Default: 2000
            - timeout (float): Timeout for reward computation in seconds. Default: 300
        tokenizer (AutoTokenizer): HuggingFace tokenizer for decoding responses.
        compute_score (callable, optional): Custom reward scoring function. Can be
            sync or async. Defaults to default_compute_score.
        reward_router_address (str | None): Address for reward router service.
        reward_model_tokenizer (AutoTokenizer | None): Optional tokenizer for reward model.

    Class Attributes (Global State):
        _semaphore (asyncio.Semaphore): Global concurrency limiter
        _max_concurrent (int): Max concurrent requests
        _rpm_limiter (AsyncTokenBucket | None): Request rate limiter
        _max_rpm (int | None): Max requests per minute
        _tpm_limiter (AsyncTokenBucket | None): Token rate limiter
        _max_tpm (int | None): Max tokens per minute
        _estimated_tokens_per_request (int): Estimated tokens per request
        _class_initialized (bool): Whether class has been initialized

    Example Configuration:
        >>> config = DictConfig({
        ...     "reward_model": {
        ...         "max_concurrent": 10,      # 10 parallel requests
        ...         "max_rpm": 500,            # 500 requests/minute
        ...         "max_tpm": 100000,         # 100k tokens/minute
        ...         "estimated_tokens_per_request": 2000,
        ...         "timeout": 60.0,
        ...     }
        ... })
        >>> manager = RateLimitedRewardLoopManager(config, tokenizer)

    Thread Safety:
        This class is designed for concurrent use. All rate limiting resources
        are protected by asyncio primitives (Lock, Semaphore).

    See Also:
        - AsyncTokenBucket: Token bucket implementation for rate limiting
        - RewardLoopManagerBase: Base class for reward loop managers
        - verl.utils.reward_score.default_compute_score: Default scoring function
    """

    # Class-level state for global rate limiting
    _semaphore = None
    _max_concurrent = None
    _rpm_limiter = None
    _max_rpm = None
    _tpm_limiter = None
    _max_tpm = None
    _estimated_tokens_per_request = None
    _class_initialized = False

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        # Check if already initialized before calling parent
        if cls._class_initialized:
            return

        super().init_class(config, tokenizer)

        # Concurrency limiter
        cls._max_concurrent = config.reward_model.get("max_concurrent", 1)
        cls._semaphore = asyncio.Semaphore(cls._max_concurrent)

        # Request rate limiter (RPM)
        cls._max_rpm = config.reward_model.get("max_rpm", None)
        if cls._max_rpm is not None:
            requests_per_second = cls._max_rpm / 60.0
            cls._rpm_limiter = AsyncTokenBucket(rate_limit=requests_per_second, max_tokens=requests_per_second)
        else:
            cls._rpm_limiter = None

        # Token rate limiter (TPM)
        cls._max_tpm = config.reward_model.get("max_tpm", None)
        cls._estimated_tokens_per_request = config.reward_model.get("estimated_tokens_per_request", 2000)
        if cls._max_tpm is not None:
            tokens_per_second = cls._max_tpm / 60.0
            cls._tpm_limiter = AsyncTokenBucket(rate_limit=tokens_per_second, max_tokens=tokens_per_second)
        else:
            cls._tpm_limiter = None

        log_msg = "Rate limiting configuration:\n"
        log_msg += f"  - Concurrency limit: {cls._max_concurrent}\n"
        if cls._max_rpm is not None:
            log_msg += f"  - Request rate limit: {cls._max_rpm} RPM ({cls._max_rpm / 60.0:.2f} RPS)\n"
        else:
            log_msg += "  - Request rate limit: unlimited\n"
        if cls._max_tpm is not None:
            log_msg += f"  - Token rate limit: {cls._max_tpm} TPM ({cls._max_tpm / 60.0:.2f} TPS)\n"
            log_msg += f"  - Estimated tokens per request: {cls._estimated_tokens_per_request}\n"
        else:
            log_msg += "  - Token rate limit: unlimited\n"
        log_msg += "All limiters are shared globally across all workers."
        logger.info(log_msg)

        cls._class_initialized = True

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.timeout = config.reward_model.get("timeout", 300.0)
        self.care_server_url = (
            config.reward_model.get("care_server_url", os.getenv("CARE_SERVER_URL", "http://127.0.0.1:8000"))
            .rstrip("/")
        )

    async def _care_batch_argmax(self, contexts, utterances):
        """One round-trip to /batch_predict, returning argmax integers per item.

        Returns a list of dicts keyed by CARE abbreviation
        ({NJ, WE, RA, AL, RF, SA}). Falls back to all-zeros on server error.
        """
        if not utterances:
            return []
        try:
            url = f"{self.care_server_url}/batch_predict"
            response = await self.loop.run_in_executor(
                None,
                lambda: requests.post(
                    url,
                    json={
                        "contexts": contexts,
                        "utterances": utterances,
                        "batch_size": 8,
                        "include_analysis": True,
                        # argmax (return_expected omitted/False) — CTRS-P is
                        # defined on integer labels.
                    },
                    timeout=120,
                )
            )
            response.raise_for_status()
            preds = response.json().get("predictions", [])
            # Convert full-name keys -> abbreviated for downstream code.
            out = []
            for p in preds:
                out.append({CARE_ABBREV[k]: float(v) for k, v in p.items() if k in CARE_ABBREV})
            return out
        except Exception as e:
            logger.error(f"Error calling /batch_predict at {self.care_server_url}: {e}")
            return [{"NJ": 0.0, "WE": 0.0, "RA": 0.0, "AL": 0.0, "RF": 0.0, "SA": 0.0}
                    for _ in utterances]

    # Phrases the agent loop's _should_end_session recognises as a natural
    # session close. Kept in sync with final_pipeline/shared_config.py.
    _CLOSING_PHRASES = (
        "see you next time", "take care", "good luck",
        "until next week", "same time next", "goodbye", "farewell",
    )

    @staticmethod
    def _ended_with_closing(conversation):
        """Return True iff the LAST therapist turn contains closing language.

        Used to gate the saved-turns part of the early-finish bonus: we only
        want to pay out for sessions that wrap up *gracefully*, not sessions
        that ended because they ran out of token budget (or otherwise
        terminated mid-thought).
        """
        if not conversation:
            return False
        last_asst = None
        for msg in reversed(conversation):
            if msg.get("role") == "assistant":
                last_asst = (msg.get("content") or "").lower()
                break
        if not last_asst:
            return False
        return any(p in last_asst for p in CareRewardLoopManager._CLOSING_PHRASES)

    async def _compute_multiturn_reward(self, conversation, extraction_summary,
                                        max_turns=None, supervisor_signaled_complete=False):
        """Multi-turn reward = IR coverage + CTRS-P + early-finish bonus.

        Args:
            conversation: list of {role, content} dicts from the agent loop.
            extraction_summary: dict with at least 'overall_pct' (0-100), and
                ideally 'covered_items' / 'total_items'.
            max_turns: turn budget the agent loop was configured with. Used to
                compute the early-finish bonus. If None, no bonus is awarded.
            supervisor_signaled_complete: True if the supervisor emitted the
                [SESSION_COMPLETE] token during the rollout.

        Returns:
            (reward, ir_frac, ctrs_p, sub_scores_dict, n_turns, early_bonus)
        """
        # IR coverage fraction in [0, 1].
        try:
            ir_frac = float(extraction_summary.get("overall_pct", 0.0)) / 100.0
        except Exception:
            ir_frac = 0.0
        ir_frac = max(0.0, min(1.0, ir_frac))

        # "Checklist complete" — prefer the exact integer comparison when
        # available (handles floating-point rounding from overall_pct).
        covered = extraction_summary.get("covered_items")
        total = extraction_summary.get("total_items")
        if isinstance(covered, (int, float)) and isinstance(total, (int, float)) and total > 0:
            checklist_complete = covered >= total
        else:
            checklist_complete = ir_frac >= 0.999

        # Score every therapist turn through CARE in a single batch call.
        pairs = _extract_therapist_pairs(conversation or [])
        if not pairs:
            return ir_frac + 0.0, ir_frac, 0.0, {"U": 0.0, "IE": 0.0, "C": 0.0, "TA": 0.0}, 0, 0.0

        contexts   = [c for c, _ in pairs]
        utterances = [u for _, u in pairs]
        per_turn   = await self._care_batch_argmax(contexts, utterances)
        ctrs_p, sub = _ctrs_p_from_per_turn(per_turn)

        n_turns = len(per_turn)

        # Early-finish bonus: reward fully covering the checklist with turns
        # to spare. The saved-turns scaling is GATED on the last therapist
        # turn ending the session with closing language — otherwise PPO will
        # exploit the bonus by writing longer per-turn responses that exhaust
        # the response_length budget at fewer turns and terminate
        # mid-thought (observed in long-run step 9). The small
        # SUPERVISOR_SIGNAL_BONUS still fires on coverage alone, so we
        # continue to reward reaching ir=1.0 even without a clean close.
        # 0.5 → 0.3: the 17-item checklist (vs the prior 27-item taxonomy) is
        # more reachable in a 20-turn session, so the early-finish bonus fires
        # on more rollouts. Halving prevents the bonus from dominating
        # advantages and pushing the model to rush through items at the
        # expense of Collaboration (which regressed from -0.45 → -0.60 in v3).
        EARLY_FINISH_MAX_BONUS = 0.3
        SUPERVISOR_SIGNAL_BONUS = 0.1
        early_bonus = 0.0
        ended_with_closing = self._ended_with_closing(conversation)
        if checklist_complete and isinstance(max_turns, (int, float)) and max_turns > 0:
            if ended_with_closing:
                saved_frac = max(0.0, (float(max_turns) - n_turns) / float(max_turns))
                early_bonus = EARLY_FINISH_MAX_BONUS * saved_frac
            if supervisor_signaled_complete:
                early_bonus += SUPERVISOR_SIGNAL_BONUS

        # === Shaping fix #1: boost IR weight ==============================
        # Old reward: 1.0 * ir_frac + ctrs_p. IR ∈ [0,1] vs CTRS_P ∈ ~[-1, 2.5]
        # → CTRS_P dominated and RL abandoned probing (qwen_mt_v1 regressed
        # IR 61% → 12%). Bumping IR's coefficient evens the magnitudes and
        # makes asking PHQ-9/GAD-7 questions worth more than maxing CARE's
        # Technical-Appropriateness via reflective boilerplate.
        # 2.0 → 2.2: with the swap to the stricter PHQ-9+GAD-7 17-item
        # checklist (was 27 items of fuzzier categories), ir_frac will run
        # lower per-rollout. A modest bump maintains roughly the same IR-vs-
        # CTRS contribution ratio (~1.5–2×) that worked in v3, without
        # over-weighting IR enough to crush Collaboration.
        IR_WEIGHT = 2.2

        # === Shaping fix #3: repetition penalty ===========================
        # qwen_mt_v1 collapsed to a fixed-point "Well, it seems like you're..."
        # boilerplate (T18 ≡ T19 verbatim in sampled sessions). Penalize high
        # token-Jaccard overlap between consecutive therapist turns so the
        # policy can't copy itself for free reward.
        # 0.25 → 0.35: the 17-item PHQ-9+GAD-7 checklist tempts the model to
        # rattle off items as a drill ("Have you had X? Have you had Y?").
        # Stronger Jaccard penalty discourages the boilerplate prober pattern
        # while still permitting natural follow-ups (threshold stays at 0.6).
        REP_PENALTY_MAX = 0.35
        REP_PENALTY_THRESHOLD = 0.6
        rep_penalty = 0.0
        therapist_msgs = [
            (m.get("content") or "").strip()
            for m in (conversation or [])
            if m.get("role") == "assistant"
        ]
        for prev, curr in zip(therapist_msgs, therapist_msgs[1:]):
            p_toks = set(prev.split())
            c_toks = set(curr.split())
            union = p_toks | c_toks
            if not union:
                continue
            jaccard = len(p_toks & c_toks) / len(union)
            if jaccard > REP_PENALTY_THRESHOLD:
                rep_penalty += REP_PENALTY_MAX * (jaccard - REP_PENALTY_THRESHOLD) / (1.0 - REP_PENALTY_THRESHOLD)

        reward = IR_WEIGHT * ir_frac + ctrs_p + early_bonus - rep_penalty
        return reward, ir_frac, ctrs_p, sub, n_turns, early_bonus

    async def _get_embedding_sim(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between MiniLM embeddings of two responses.

        Returns a scalar in [-1, 1]. Used as the continuous component of the
        hybrid reward; resilient to server errors (returns 0.0 on failure).
        """
        try:
            url = f"{self.care_server_url}/embedding_sim"
            response = await self.loop.run_in_executor(
                None,
                lambda: requests.post(
                    url,
                    json={"text_a": text_a, "text_b": text_b},
                    timeout=30,
                ),
            )
            response.raise_for_status()
            return float(response.json().get("cosine", 0.0))
        except Exception as e:
            logger.error(f"Error calling /embedding_sim at {self.care_server_url}: {e}")
            return 0.0

    async def _get_server_predictions(self, context: str, utterance: str) -> tuple[dict, dict]:
        """Get CARE predictions from the server.

        Returns (expected, argmax) — both come from the same forward pass:
            expected: E[label] = sum_c p_c * c   (continuous, used for reward)
            argmax:  argmax_c logits             (integer, used for logging/eval parity)
        """
        try:
            predict_url = f"{self.care_server_url}/predict"
            response = await self.loop.run_in_executor(
                None,
                lambda: requests.post(
                    predict_url,
                    json={
                        "context": context,
                        "utterance": utterance,
                        "include_analysis": True,
                        "return_expected": True,
                    },
                    timeout=60
                )
            )
            response.raise_for_status()
            result = response.json()
            expected = result.get("predictions", {})
            argmax   = result.get("predictions_argmax") or expected  # fallback
            return expected, argmax
        except Exception as e:
            logger.error(f"Error calling server at {self.care_server_url}: {e}")
            zeros = {label: 0.0 for label in CARE_LABELS}
            return zeros, zeros

    def _extract_reference_response(
        self, data_item, extra_info: dict, ground_truth_payload
    ) -> str | None:
        """Extract reference therapist response text for CARE-vs-CARE scoring."""
        candidate_keys = [
            "reference_response",
            "ground_truth_response",
            "target_response",
            "target",
            "answer",
            "utterance",
            "Utterance",
        ]

        for key in candidate_keys:
            val = extra_info.get(key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()

        non_tensor = getattr(data_item, "non_tensor_batch", {})
        if hasattr(non_tensor, "get"):
            for key in candidate_keys:
                val = non_tensor.get(key, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        # In some pipelines, ground_truth is the reference response string.
        if isinstance(ground_truth_payload, str) and ground_truth_payload.strip():
            return ground_truth_payload.strip()

        return None

    async def _compute_six_dim_reward(self, response_str: str, context: str, reference_response: str):
        """Compute hybrid reward = CARE-based + embedding-cosine-based.

        Returns (reward, l2_norm, raw_l2_norm, l2_norm_argmax, raw_l2_norm_argmax,
                 care_reward, emb_reward, emb_sim):
            reward = (1 - EMB_REWARD_WEIGHT) * care_reward + EMB_REWARD_WEIGHT * emb_reward
            care_reward uses expected-value CARE L2 (smooth).
            emb_reward = (cosine + 1)/2 in [0, 1] from MiniLM embedding sim.
            *_argmax mirror CARE metrics via argmax integers for eval parity.
        """
        try:
            # Issue CARE prediction calls and embedding-sim call concurrently.
            pred_task = asyncio.create_task(self._get_server_predictions(context, response_str))
            ref_task  = asyncio.create_task(self._get_server_predictions(context, reference_response))
            emb_task  = asyncio.create_task(self._get_embedding_sim(response_str, reference_response))

            pred_exp, pred_arg = await pred_task
            ref_exp,  ref_arg  = await ref_task
            emb_sim            = await emb_task

            def _l2(p, r):
                losses = [(float(p.get(l, 0.0)) - float(r.get(l, 0.0))) ** 2 for l in CARE_LABELS]
                raw = math.sqrt(sum(losses)) if losses else 0.0
                norm = math.sqrt(sum(losses) / len(losses)) if losses else 0.0
                return raw, norm

            raw_l2_norm,        normalized_l2_norm        = _l2(pred_exp, ref_exp)
            raw_l2_norm_argmax, normalized_l2_norm_argmax = _l2(pred_arg, ref_arg)

            # CARE component (smooth expected-value L2 → [0, 1] reward).
            reward_scale = (CARE_LABEL_MAX - CARE_LABEL_MIN) / 1.5
            care_reward = 1.0 - (normalized_l2_norm / reward_scale)
            care_reward = max(0.0, min(1.0, care_reward))

            # Embedding component: cosine in [-1, 1] → reward in [0, 1].
            emb_reward = max(0.0, min(1.0, (emb_sim + 1.0) / 2.0))

            reward = (1.0 - EMB_REWARD_WEIGHT) * care_reward + EMB_REWARD_WEIGHT * emb_reward

            return (reward, normalized_l2_norm, raw_l2_norm,
                    normalized_l2_norm_argmax, raw_l2_norm_argmax,
                    care_reward, emb_reward, emb_sim)

        except Exception as e:
            logger.error(f"Error computing 6D reward: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    async def _compute_reward(
        self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict
    ) -> dict | float:
        if self.is_async_reward_score:
            return await self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                reward_router_address=self.reward_router_address,
                reward_model_tokenizer=self.reward_model_tokenizer,
            )
        else:
            return await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    reward_router_address=self.reward_router_address,
                    reward_model_tokenizer=self.reward_model_tokenizer,
                ),
            )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        reward_extra_info = {}

        # Apply rate limiting layers
        if self._rpm_limiter is not None:
            await self._rpm_limiter.acquire(1.0)

        if self._tpm_limiter is not None:
            estimated_tokens = self._estimated_tokens_per_request
            await self._tpm_limiter.acquire(estimated_tokens)

        async with self._semaphore:
            try:
                # Multi-turn rollouts inject 'conversation' + 'extraction_summary'
                # via the agent loop's extra_fields. When both are present, use
                # the session-level IR + CTRS-P reward; otherwise fall back to
                # single-utterance CARE distance.
                conversation = extra_info.get("conversation", None)
                extraction_summary = extra_info.get("extraction_summary", None)

                if conversation and extraction_summary:
                    max_turns = extra_info.get("max_therapy_turns", None)
                    supervisor_done = bool(extra_info.get("supervisor_signaled_complete", False))
                    sup_failures = int(extra_info.get("supervisor_failures", 0))
                    sup_calls = int(extra_info.get("supervisor_calls", 0))
                    (reward_score, ir_frac, ctrs_p,
                     sub, n_turns, early_bonus) = await asyncio.wait_for(
                        self._compute_multiturn_reward(
                            conversation, extraction_summary,
                            max_turns=max_turns,
                            supervisor_signaled_complete=supervisor_done,
                        ),
                        timeout=self.timeout,
                    )
                    # Shaping fix #4: explicit supervisor-failure malus.
                    # Without this, sessions that overflowed the supervisor's
                    # context received no IR penalty → model learned to let
                    # sessions run long so the supervisor blanked.
                    SUP_FAILURE_MALUS = 0.05    # per failed supervisor call
                    sup_malus = SUP_FAILURE_MALUS * sup_failures
                    reward_score = reward_score - sup_malus

                    reward_extra_info["ir_frac"]       = ir_frac
                    reward_extra_info["ctrs_p"]        = ctrs_p
                    reward_extra_info["ctrs_U_score"]  = sub["U"]
                    reward_extra_info["ctrs_IE_score"] = sub["IE"]
                    reward_extra_info["ctrs_C_score"]  = sub["C"]
                    reward_extra_info["ctrs_TA_score"] = sub["TA"]
                    reward_extra_info["n_therapist_turns"] = float(n_turns)
                    reward_extra_info["early_finish_bonus"] = float(early_bonus)
                    reward_extra_info["supervisor_signaled_complete"] = float(supervisor_done)
                    reward_extra_info["supervisor_failures"] = float(sup_failures)
                    reward_extra_info["supervisor_calls"] = float(sup_calls)
                    reward_extra_info["supervisor_malus"] = float(sup_malus)
                    reward_extra_info["method"] = "ir_plus_ctrs_p"
                else:
                    # Single-turn CARE distance + embedding similarity hybrid.
                    context = str(extra_info.get("context", ""))
                    reference_response = self._extract_reference_response(data_item, extra_info, ground_truth)
                    if not reference_response:
                        raise ValueError("Missing reference response text for CARE ground truth scoring")

                    (reward_score, l2_norm, raw_l2_norm,
                     l2_norm_argmax, raw_l2_norm_argmax,
                     care_reward, emb_reward, emb_sim) = await asyncio.wait_for(
                        self._compute_six_dim_reward(response_str, context, reference_response),
                        timeout=self.timeout,
                    )
                    reward_extra_info["l2_norm"] = l2_norm
                    reward_extra_info["raw_l2_norm"] = raw_l2_norm
                    reward_extra_info["l2_norm_argmax"] = l2_norm_argmax
                    reward_extra_info["raw_l2_norm_argmax"] = raw_l2_norm_argmax
                    reward_extra_info["care_reward"] = care_reward
                    reward_extra_info["emb_reward"] = emb_reward
                    reward_extra_info["emb_sim"] = emb_sim
                    reward_extra_info["method"] = "care_plus_embedding"

            except asyncio.TimeoutError:
                logger.warning(
                    f"Reward computation timed out after {self.timeout}s for data_source={data_source}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward_score = 0.0
                reward_extra_info["timeout"] = True
                reward_extra_info["acc"] = 0.0

            except Exception as e:
                logger.error(
                    f"Reward computation failed for data_source={data_source}: {e}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward_score = 0.0
                reward_extra_info["error"] = str(e)
                reward_extra_info["acc"] = 0.0

        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Make the manager callable like traditional reward managers.

        This method provides compatibility with the existing reward manager interface
        by wrapping the async run_single method in a synchronous call.

        Args:
            data (DataProto): Input data containing prompts and responses.
            return_dict (bool): If True, return a dict with reward_tensor and reward_extra_info.
                               If False, return only the reward_tensor. Defaults to False.

        Returns:
            torch.Tensor | dict: If return_dict is False, returns a tensor of shape [batch_size, response_length]
                                with single scalar reward values. If return_dict is True, returns a dict with:
                                - reward_tensor: The reward tensor
                                - reward_extra_info: Dict containing extra information about rewards
        """
        from collections import defaultdict

        import torch

        # If there are pre-computed rm_scores, return them directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor with single dimension
        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]
        reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Process each data item through the async event loop
        async def process_batch():
            tasks = []
            for i in range(len(data)):
                data_item = data[i : i + 1]  # Get single item as DataProto slice
                tasks.append(self.run_single(data_item))

            results = await asyncio.gather(*tasks)
            return results

        # Run the async processing using self.loop property which lazily gets/creates event loop
        # This ensures rate limiters and semaphores work correctly by using the same loop
        results = self.loop.run_until_complete(process_batch())

        # Aggregate results into reward tensor and extra info
        for i, result in enumerate(results):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()

            reward_score = result["reward_score"]  # Single scalar value
            reward_tensor[i, valid_response_length - 1] = reward_score

            # Collect extra info
            if "reward_extra_info" in result:
                for key, value in result["reward_extra_info"].items():
                    reward_extra_info[key].append(value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
