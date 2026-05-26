"""
VERL-Compatible Multi-Turn Therapist Agent Loop
Wraps Agent1_PrimaryTherapist for training with VERL framework.

- Therapist (agent_1): Uses VERL's vLLM server (the model being trained)
- Patient: Uses external OpenAI-compatible API (separate model)
- Agent_2 (supervisor): Uses external OpenAI-compatible API (separate model, optional)
"""

import asyncio
import copy
import json
import logging
import os
import sys
from typing import Any, Optional
from uuid import uuid4

import aiohttp
import torch
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, 
    AgentLoopMetrics,
    AgentLoopOutput, 
    register
)
from verl.utils.profiler import simple_timer

# Import shared config from final_pipeline
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
_PIPELINE_CANDIDATES = [
    os.getenv("RASINGAN_FINAL_PIPELINE", ""),
    os.path.join(_ROOT, "final_pipeline"),
]
for _candidate in _PIPELINE_CANDIDATES:
    if _candidate and os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break

from shared_config import (
    get_therapist_system_prompt,
    get_patient_system_prompt,
    get_supervisor_system_prompt,
    get_session_phase,
    should_end_session,
)

# Checklist source: final_pipeline/checklists package. Same package the offline
# pipeline (agent_2.py) and the offline IR scorer use, so the training-time
# IR signal walks the SAME PHQ-9+GAD-7 items the eval measures. Previously
# training used shared_config.EXTRACTION_CHECKLIST (a custom 27-item
# taxonomy), which was the root cause of the train/eval IR divergence
# (val ir_frac climbing while offline IR regressed 62→45).
#
# Default = combined PHQ-9 + GAD-7 (17 items). Override at run-time with
# CHECKLIST_NAME=phq9|gad7|combined.
from checklists import (
    get_checklist as _get_checklist,
    init_status as _init_status,
    compute_summary as _compute_summary,
    format_status as _format_status,
    get_next_priority as _get_next_priority,
)

_CHECKLIST_NAME = os.environ.get("CHECKLIST_NAME", "combined")
_CHECKLIST_BUNDLE = _get_checklist(_CHECKLIST_NAME)
EXTRACTION_CHECKLIST = _CHECKLIST_BUNDLE["checklist"]
EXTRACTION_PRIORITY_ORDER = _CHECKLIST_BUNDLE["priority_order"]


def init_extraction_status():
    """Initialize tracking status for the configured checklist."""
    return _init_status(EXTRACTION_CHECKLIST)


def compute_extraction_summary(status):
    return _compute_summary(status)


def format_extraction_status(status):
    return _format_status(status)


def get_next_priority(status):
    return _get_next_priority(status, EXTRACTION_PRIORITY_ORDER)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Robust matching of supervisor-returned `covered_items` text against the
# extraction-status checklist keys.
#
# The pure substring match used previously failed almost always because the
# 4B supervisor paraphrases items (e.g. it returns "PHQ-9 mood" or
# "appetite changes" while the checklist key is
# "Persistent low mood (PHQ-9 item 1)" or "Changes in appetite or weight").
# That bug zeroed out IR coverage throughout multi-turn RL training (see
# run therapist_multiturn_23-23-14: 24/24 rollouts ir_frac=0).
#
# Strategy: try substring (cheap), then token-set Jaccard with stopword strip.
# Pick the best-scoring checklist item per supervisor entry across all
# categories. Threshold tuned to be permissive (small overlaps match) without
# letting unrelated items match — most short supervisor responses contain
# 2-4 informative tokens, so requiring ≥40% Jaccard is a reasonable balance.
# ---------------------------------------------------------------------------

import re as _re

_STOPWORDS = {
    "the", "of", "and", "a", "an", "to", "in", "or", "for", "on", "at",
    "is", "are", "be", "with", "as", "this", "that", "by", "from", "any",
    "their", "your", "you", "i", "it", "its", "has", "have", "had", "was",
    "were", "do", "does", "did", "not",
}


def _norm_text(s: str) -> str:
    return _re.sub(r"\s+", " ", str(s).lower().strip())


def _tokenize(s: str) -> set:
    return {t for t in _re.split(r"[^a-z0-9]+", _norm_text(s)) if t and t not in _STOPWORDS}


def _match_score(supervisor_text: str, item_key: str) -> int:
    """Higher = better match. 0 = no match.

    100  = exact normalized equality
     50  = either string is a substring of the other
    0-49 = Jaccard-percent token overlap
    """
    a, b = _norm_text(supervisor_text), _norm_text(item_key)
    if not a or not b:
        return 0
    if a == b:
        return 100
    if a in b or b in a:
        return 50
    ta, tb = _tokenize(a), _tokenize(b)
    union = ta | tb
    if not union:
        return 0
    return int(100 * len(ta & tb) / len(union))


def _best_checklist_match(supervisor_text: str, extraction_status: dict,
                          threshold: int = 40):
    """Find the highest-scoring checklist item across all categories.

    Returns (cat_data, item_key) for the best match above `threshold`, or
    None if no item clears the bar.
    """
    best = None
    best_score = threshold
    for cat_data in extraction_status.values():
        for item_key in cat_data["items"]:
            score = _match_score(supervisor_text, item_key)
            if score > best_score:
                best_score = score
                best = (cat_data, item_key)
    return best


class ExternalModelClient:
    """Async client for OpenAI-compatible API endpoints (vLLM, TGI, etc.).

    `instruct_mode=True` overrides per-call sampling with the Qwen3-Instruct-2507
    model-card values (non-thinking): temperature=0.7, top_p=0.8, top_k=20,
    min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, and disables
    thinking by sending chat_template_kwargs={"enable_thinking": False}.
    Default off → callers' temperature/top_p are used (legacy behavior).
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 instruct_mode: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.instruct_mode = instruct_mode
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=120),
            )
        return self._session

    async def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Call the /v1/chat/completions endpoint and return the text response.

        Retries with exponential backoff on connection / 5xx errors so a
        temporary server crash + restart doesn't drop the rollout's reward
        signal. Bash-side watchdog (in run_multiturn.sh) restarts the
        supervisor vLLM if it dies, so we just need to wait it out here.
        Does NOT retry on 4xx (e.g., 400 context-overflow) — those are real
        client errors and retrying won't help.
        """
        session = await self._get_session()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.instruct_mode:
            # Qwen3-Instruct-2507 model-card sampling (non-thinking). The first
            # five values match the card verbatim. presence_penalty is the only
            # knob that survives across separate chat_completion calls (because
            # prior patient turns enter this call's prompt as assistant
            # messages), so it carries the cross-turn anti-repetition load.
            payload.update({
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                # Disable Qwen3 thinking-mode prefix at template-render time.
                "chat_template_kwargs": {"enable_thinking": False},
            })
        else:
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        # Retry parameters: 30 attempts, sleeps 2,4,8,16,30,30,...,30 = ~13min max wait.
        # Caps at 30s per retry so we wake up regularly to check the server.
        MAX_RETRIES = 30
        last_err: Optional[Exception] = None
        sleep_s = 2.0
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                    if 400 <= resp.status < 500:
                        # Real client error — don't retry, surface immediately.
                        text = await resp.text()
                        raise RuntimeError(f"{resp.status} {resp.reason}: {text[:200]}")
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientConnectorError,
                    aiohttp.ServerDisconnectedError,
                    aiohttp.ClientOSError,
                    aiohttp.ServerTimeoutError,
                    asyncio.TimeoutError) as e:
                # Transient connection error — supervisor may be crashing /
                # restarting. Sleep and retry.
                last_err = e
                if attempt == 0:
                    logger.warning(
                        f"Server call failed (will retry up to {MAX_RETRIES}× with backoff): "
                        f"{type(e).__name__}: {str(e)[:120]}"
                    )
                await asyncio.sleep(min(sleep_s, 30.0))
                sleep_s *= 2
                # Reset session so we don't reuse a dead keep-alive connection.
                if self._session and not self._session.closed:
                    try:
                        await self._session.close()
                    except Exception:
                        pass
                session = await self._get_session()
                continue
            except aiohttp.ClientResponseError as e:
                # 5xx — also retry.
                if e.status >= 500:
                    last_err = e
                    if attempt == 0:
                        logger.warning(
                            f"Server returned {e.status} (will retry up to {MAX_RETRIES}×): {e.message}"
                        )
                    await asyncio.sleep(min(sleep_s, 30.0))
                    sleep_s *= 2
                    continue
                raise

        # Exhausted retries — propagate.
        raise RuntimeError(
            f"chat_completion failed after {MAX_RETRIES} retries; last error: "
            f"{type(last_err).__name__}: {last_err}"
        )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


@register("therapist_multiturn_agent")
class TherapistMultiTurnAgentLoop(AgentLoopBase):
    """Multi-turn agent loop for therapeutic conversations.
    
    - Therapist responses: generated via VERL's vLLM server (trained model)
    - Patient responses: generated via external API (separate model, not trained)
    - Supervisor feedback: generated via external API (separate model, optional)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Therapy session config
        self.max_turns = self.config.data.get("max_therapy_turns", 5)
        self.session_type = self.config.data.get("session_type", "counseling")
        self.enable_supervisor = self.config.data.get("enable_supervisor_feedback", False)

        # Instruct (non-thinking) sampling for the shared patient+supervisor
        # vLLM. Env-var toggled, default OFF — when ON, applies the Qwen3
        # recommended sampling and disables thinking-mode prefix rendering.
        instruct_mode = os.environ.get("INSTRUCT_MODE", "0").strip().lower() in ("1", "true", "yes", "on")

        # External model config for patient
        patient_cfg = self.config.data.get("patient_model", {})
        self.patient_client = ExternalModelClient(
            base_url=patient_cfg.get("base_url", "http://localhost:8001"),
            model=patient_cfg.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            api_key=patient_cfg.get("api_key", "EMPTY"),
            instruct_mode=instruct_mode,
        )
        self.patient_max_tokens = patient_cfg.get("max_tokens", 256)
        self.patient_temperature = patient_cfg.get("temperature", 0.7)

        # External model config for supervisor (agent_2)
        supervisor_cfg = self.config.data.get("supervisor_model", {})
        self.supervisor_client = ExternalModelClient(
            base_url=supervisor_cfg.get("base_url", "http://localhost:8002"),
            model=supervisor_cfg.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            api_key=supervisor_cfg.get("api_key", "EMPTY"),
            instruct_mode=instruct_mode,
        ) if self.enable_supervisor else None

        if instruct_mode:
            logger.info("[ExternalModelClient] INSTRUCT_MODE=ON — patient & supervisor "
                        "use Qwen3 instruct sampling (temp=1.0, top_p=1.0, top_k=40, "
                        "min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0) "
                        "with chat_template_kwargs={'enable_thinking': False}.")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run multi-turn therapeutic conversation.
        
        Args:
            sampling_params: LLM sampling parameters
            **kwargs: Dataset fields including:
                - raw_prompt: Initial message list
                - multi_modal_data: Optional image data
                - patient_context: Reddit post or patient scenario
                - patient_profile: Patient demographics
                - interaction_kwargs: Additional context
        
        Returns:
            AgentLoopOutput: Complete conversation with prompt, response, and metadata
        """
        metrics = {}
        request_id = uuid4().hex
        
        # Extract input data
        initial_messages = list(kwargs.get("raw_prompt", []))
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))
        patient_context = kwargs.get("patient_context", "No patient context provided")
        patient_profile = kwargs.get("patient_profile", {})
        interaction_kwargs = kwargs.get("interaction_kwargs", {})
        
        # Initialize conversation history
        conversation_messages = copy.deepcopy(initial_messages)
        all_response_ids = []
        all_response_masks = []
        all_logprobs = []
        num_turns = 0
        last_supervisor_feedback = None
        extraction_status = init_extraction_status()
        supervisor_signaled_complete = False
        # Count supervisor calls that returned no feedback (overflow / 400 / parse
        # failure). Without explicit penalty, the model learned to let sessions
        # run long so supervisor overflowed and IR penalty silently disappeared
        # (qwen_mt_v1 regression). Passed to faith.py for an explicit malus.
        supervisor_failures = 0
        supervisor_calls = 0
        
        with simple_timer("multi_turn_generation", metrics):
            # Run multi-turn conversation
            for turn_idx in range(self.max_turns):
                # Generate therapist response (with supervisor feedback from previous turn)
                therapist_response_ids, therapist_logprobs = await self._generate_therapist_response(
                    request_id=uuid4().hex,
                    messages=conversation_messages,
                    sampling_params=sampling_params,
                    turn=turn_idx,
                    image_data=image_data,
                    supervisor_feedback=last_supervisor_feedback,
                )
                
                if not therapist_response_ids:
                    break
                
                # Accumulate response tokens
                all_response_ids.extend(therapist_response_ids)
                all_response_masks.extend([1] * len(therapist_response_ids))  # 1 for LLM generated
                if therapist_logprobs:
                    all_logprobs.extend(therapist_logprobs)
                
                # Decode response to add to conversation
                therapist_text = self.tokenizer.decode(therapist_response_ids, skip_special_tokens=True)
                conversation_messages.append({"role": "assistant", "content": therapist_text})
                
                # Patient and supervisor calls in PARALLEL when both are needed.
                # Supervisor evaluates the therapist response (and the prior
                # patient turn) — it doesn't strictly need to see the patient's
                # reaction to this turn's therapist, so we can dispatch both
                # before the patient response is back. One-turn lag in
                # extraction (current patient revelations get marked next turn)
                # in exchange for ~halving per-turn wall clock.
                is_last_turn = (turn_idx == self.max_turns - 1)
                last_supervisor_feedback = None

                if turn_idx < self.max_turns - 1:
                    patient_task = asyncio.create_task(self._generate_patient_response(
                        messages=conversation_messages,
                        patient_context=patient_context,
                        patient_profile=patient_profile,
                    ))
                    supervisor_task = None
                    if self.enable_supervisor and self.supervisor_client and not is_last_turn:
                        supervisor_task = asyncio.create_task(self._supervise_turn(
                            conversation_messages=conversation_messages,
                            patient_context=patient_context,
                            extraction_status=extraction_status,
                            turn=turn_idx,
                        ))
                    patient_text = await patient_task
                    if supervisor_task is not None:
                        extraction_status, last_supervisor_feedback = await supervisor_task
                        supervisor_calls += 1
                        if last_supervisor_feedback is None:
                            # Either the LLM call raised (400/timeout/conn) or
                            # the response wasn't parseable JSON. Either way
                            # the IR reward signal for this turn is missing →
                            # count it so faith.py can apply a malus.
                            supervisor_failures += 1
                        post_summary = compute_extraction_summary(extraction_status)
                        metrics[f"extraction_status_turn_{turn_idx}"] = post_summary
                        # Programmatic checklist-complete signal: fire as soon as
                        # the checklist is fully covered, even if the supervisor (a
                        # ~4B base model) didn't emit [SESSION_COMPLETE] itself.
                        checklist_done = (
                            post_summary.get("covered_items", 0)
                            >= post_summary.get("total_items", 0) > 0
                        )
                        if checklist_done or (
                            last_supervisor_feedback
                            and "[SESSION_COMPLETE]" in last_supervisor_feedback
                        ):
                            supervisor_signaled_complete = True
                        # When the checklist is complete, override the supervisor
                        # feedback shown to the therapist next turn so it is told
                        # explicitly to wrap up. We do NOT break the loop — the
                        # therapist must learn to end the session naturally via
                        # its closing language (caught by _should_end_session), and
                        # the reward's early-finish bonus pays out the saved turns.
                        if checklist_done:
                            last_supervisor_feedback = (
                                "[CHECKLIST COMPLETE] Every required information-extraction "
                                "item has been gathered. Do NOT introduce new probes. "
                                "Begin wrapping up the session: briefly summarize what the "
                                "client shared, validate their effort, and close the session "
                                "naturally (e.g., 'Thank you for sharing this with me today. "
                                "Take care, and I'll see you next time.'). End the session "
                                "in this turn."
                            )
                        if last_supervisor_feedback:
                            metrics[f"supervisor_feedback_turn_{turn_idx}"] = last_supervisor_feedback

                    if patient_text:
                        conversation_messages.append({"role": "user", "content": patient_text})

                        # Tokenize patient text to track in response sequence
                        patient_token_ids = self.tokenizer.encode(patient_text, add_special_tokens=False)
                        # Mark patient tokens as mask=0 (NOT trained on)
                        all_response_ids.extend(patient_token_ids)
                        all_response_masks.extend([0] * len(patient_token_ids))

                num_turns += 1

                # Only natural endings stop the loop early. The model has to
                # learn to end the session itself — the supervisor's
                # "checklist complete" advice (injected into next turn's
                # therapist prompt) tells it to wrap up, and the reward's
                # early-finish bonus pays out the saved turns once it does.
                if self._should_end_session(therapist_text):
                    break
        
        # Prepare prompt tokens from initial messages
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    initial_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    initial_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs
                ),
            )
        
        # Truncate/pad response to response_length
        response_ids = all_response_ids[:self.response_length]
        response_mask = all_response_masks[:self.response_length]
        response_logprobs = all_logprobs[:self.response_length] if all_logprobs else None
        
        # Pad if necessary
        if len(response_ids) < self.response_length:
            pad_len = self.response_length - len(response_ids)
            response_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            response_mask.extend([0] * pad_len)
            if response_logprobs:
                response_logprobs.extend([0.0] * pad_len)
        
        metrics["num_therapy_turns"] = num_turns
        metrics["conversation_length"] = len(conversation_messages)
        metrics["total_tokens_generated"] = len(all_response_ids)
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=num_turns,
            metrics=metrics,
            extra_fields={
                "conversation": conversation_messages,
                "patient_context": patient_context,
                "patient_profile": patient_profile,
                "extraction_status": extraction_status,
                "extraction_summary": compute_extraction_summary(extraction_status),
                "max_therapy_turns": int(self.max_turns),
                "supervisor_signaled_complete": bool(supervisor_signaled_complete),
                # Reward shaping #4: explicit malus for supervisor failures
                # (overflow / 400 / parse fail). Without this, sessions that
                # blanked the supervisor got a free pass on the IR signal.
                "supervisor_failures": int(supervisor_failures),
                "supervisor_calls": int(supervisor_calls),
            }
        )

        # Close per-rollout aiohttp sessions so they don't leak TCP connections
        # or emit "Unclosed client session" warnings on GC. ExternalModelClient
        # lazily reopens via _get_session() if this loop instance is reused.
        for _client in (self.patient_client, self.supervisor_client):
            if _client is None:
                continue
            try:
                await _client.close()
            except Exception:
                pass

        return output
    
    async def _generate_therapist_response(
        self,
        request_id: str,
        messages: list,
        sampling_params: dict,
        turn: int,
        image_data: Any = None,
        supervisor_feedback: Optional[str] = None,
    ) -> tuple:
        """Generate therapist response using LLM.
        
        Args:
            request_id: Unique request identifier
            messages: Conversation history
            sampling_params: LLM sampling parameters
            turn: Current turn number
            image_data: Optional multimodal data
            supervisor_feedback: Checklist-based feedback from supervisor on previous turn
            
        Returns:
            Tuple of (response_ids, logprobs)
        """
        # Prepare therapist system prompt
        session_phase = self._get_session_phase(turn)
        therapist_messages = copy.deepcopy(messages)
        
        # Build system prompt with optional supervisor guidance
        system_prompt = get_therapist_system_prompt(session_phase, self.session_type)
        if supervisor_feedback:
            system_prompt += f"""\n\nSUPERVISOR GUIDANCE (use this to improve your next response):
{supervisor_feedback}

Apply the supervisor's suggestions while responding naturally to the patient. Do not mention the supervisor."""
        
        # Add therapist system prompt if not already present
        if not therapist_messages or therapist_messages[0]["role"] != "system":
            therapist_messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            therapist_messages[0]["content"] = system_prompt
        
        # Tokenize
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    therapist_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    therapist_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        # Headroom check: vLLM's max_model_len = prompt_length + response_length.
        # If len(prompt_ids) >= max_model_len, `vllm_async_server.py:400` would
        # compute max_tokens ≤ 0 and crash the whole training run with
        # `ValueError: max_tokens must be at least 1, got -N`. We do NOT
        # truncate the prompt here — doing so would corrupt the PPO trajectory
        # (training-time forward would not match the context the policy
        # actually conditioned on at rollout time). Instead we signal
        # "session over" by returning empty tensors; the run() loop treats
        # that as a graceful early-stop and the partial session continues
        # through to reward computation as-is.
        MIN_RESPONSE_BUDGET = 256
        max_model_len = self.prompt_length + self.response_length
        if len(prompt_ids) > max_model_len - MIN_RESPONSE_BUDGET:
            logger.warning(
                f"[therapist] prompt_ids={len(prompt_ids)} would overflow "
                f"max_model_len={max_model_len}; early-stopping session "
                f"(turn={turn}) to preserve trajectory faithfulness."
            )
            return [], []

        # Per-turn max_tokens cap. The verl rollout passes sampling_params with
        # max_tokens = response_length (12288 by default for this run), which is
        # the WHOLE-ROLLOUT budget. Without a per-turn cap, a single turn could
        # consume most of the budget; downstream CTRS scoring also tends to
        # over-credit "Technical Appropriateness" on long monologues, biasing
        # the reward. Cap per-turn at THERAPIST_PER_TURN_MAX_TOKENS so the
        # model has to be more concise (matches realistic therapist replies
        # and forces the trajectory budget to be spread across ≥ N/256 turns).
        # Override at run time via env: THERAPIST_PER_TURN_MAX_TOKENS=512.
        import os as _os
        THERAPIST_PER_TURN_MAX_TOKENS = int(
            _os.environ.get("THERAPIST_PER_TURN_MAX_TOKENS", "256")
        )
        # Shallow-copy and override so we don't mutate the shared dict across turns.
        per_turn_sampling = dict(sampling_params)
        existing_cap = per_turn_sampling.get("max_tokens", THERAPIST_PER_TURN_MAX_TOKENS)
        per_turn_sampling["max_tokens"] = min(existing_cap, THERAPIST_PER_TURN_MAX_TOKENS)

        # Generate
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=per_turn_sampling,
            image_data=image_data
        )
        
        return output.token_ids, output.log_probs
    
    async def _generate_patient_response(
        self,
        messages: list,
        patient_context: str,
        patient_profile: dict,
    ) -> Optional[str]:
        """Generate patient response via external API (separate model).
        
        Args:
            messages: Conversation history
            patient_context: Reddit post or patient scenario
            patient_profile: Patient demographics
            
        Returns:
            Patient response text, or None on failure
        """
        system_prompt = get_patient_system_prompt(patient_context, patient_profile)
        
        # Build messages for patient model: swap roles so patient sees therapist as "user"
        patient_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "assistant":
                # Therapist message -> patient sees as "user"
                patient_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "user":
                # Previous patient messages -> patient sees as "assistant"
                patient_messages.append({"role": "assistant", "content": msg["content"]})

        try:
            response = await self.patient_client.chat_completion(
                messages=patient_messages,
                max_tokens=self.patient_max_tokens,
                temperature=self.patient_temperature,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Patient model API call failed: {e}")
            return None

    async def _generate_supervisor_feedback(
        self,
        conversation_messages: list,
        patient_context: str,
        extraction_status: dict,
    ) -> Optional[str]:
        """Generate checklist-based supervisor feedback via external API.
        
        Args:
            conversation_messages: Full conversation so far
            patient_context: Patient's background context
            extraction_status: Current extraction checklist status
            
        Returns:
            Supervisor feedback text, or None on failure
        """
        # Find the last therapist response and patient message
        last_therapist = ""
        last_patient = ""
        for msg in reversed(conversation_messages):
            if msg["role"] == "assistant" and not last_therapist:
                last_therapist = msg["content"]
            elif msg["role"] == "user" and not last_patient:
                last_patient = msg["content"]
            if last_therapist and last_patient:
                break

        extraction_text = format_extraction_status(extraction_status)
        next_pri = get_next_priority(extraction_status)

        sup_system_prompt = get_supervisor_system_prompt()
        supervisor_messages = [
            {"role": "system", "content": sup_system_prompt},
            {"role": "user", "content": (
                f"INFORMATION EXTRACTION CHECKLIST (what the therapist still needs to obtain):\n"
                f"{extraction_text}\n\n"
                f"NEXT PRIORITY AREA: {next_pri or 'All extraction items covered!'}\n\n"
                f"PATIENT CONTEXT:\n{patient_context}\n\n"
                f"PATIENT SAID:\n\"{last_patient}\"\n\n"
                f"THERAPIST RESPONDED:\n\"{last_therapist}\"\n\n"
                "EVALUATION TASK:\n"
                "1. Note which extraction checklist items were addressed in this exchange\n"
                "2. Identify 1-2 strengths in the response\n"
                "3. Suggest what the therapist should ask about next based on pending extraction items\n"
                "4. Provide a concrete reframed response example if improvement is needed"
            )},
        ]

        try:
            # Smaller budget than the prompt suggests — the feedback is
            # bullet-style ("1) noted: ... 2) strengths: ... 3) ask next: ...");
            # 200 tokens is enough and saves ~40% decode latency per call.
            response = await self.supervisor_client.chat_completion(
                messages=supervisor_messages,
                max_tokens=200,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Supervisor model API call failed: {e}")
            return None
    
    def _get_session_phase(self, turn: int) -> str:
        """Determine therapy session phase based on turn."""
        return get_session_phase(turn, self.max_turns)
    
    def _should_end_session(self, last_response: str) -> bool:
        """Check if session should naturally end."""
        return should_end_session(last_response)
    
    # ========== Combined Supervisor Turn (preferred) ==========

    async def _supervise_turn(
        self,
        conversation_messages: list,
        patient_context: str,
        extraction_status: dict,
        turn: int,
    ) -> tuple[dict, Optional[str]]:
        """One supervisor LLM call that returns BOTH:
          * newly covered checklist items (parsed into extraction_status), AND
          * free-text feedback for the next therapist turn.

        Replaces the prior two sequential supervisor calls
        (_update_extraction_status + _generate_supervisor_feedback). Both old
        prompts shared the same long transcript prefix, so combining halves
        prefix prefill cost and removes a round-trip per turn.

        Returns (updated_extraction_status, feedback_or_none).
        """
        if not self.supervisor_client:
            return extraction_status, None

        # Build pending items
        pending_items = {}
        for category, data in extraction_status.items():
            pending = [item for item, st in data["items"].items() if not st["covered"]]
            if pending:
                pending_items[data["name"]] = pending

        # Conversation text + last exchange (for the feedback section).
        # Cap convo_text to the most recent SUP_CONVO_TURNS exchanges so the
        # supervisor prompt does not overflow the patient/supervisor vLLM's
        # max_model_len on long sessions (we saw 400 Bad Request at turn 18+
        # with full history). The supervisor still gets the explicit "PATIENT
        # JUST SAID" / "THERAPIST RESPONDED" fields below for the latest
        # exchange, and extraction_status persists across turns, so it can
        # still correctly score what changed this turn.
        # Reduced 6 → 3 to halve the supervisor prompt token count. The
        # supervisor was the per-turn wall-time bottleneck (~70% SM util while
        # the trainer sat at 0-12%); smaller prompt → more concurrent
        # supervisor requests fit into `--max-num-batched-tokens` → faster
        # gen phase. extraction_status persists across turns, so the
        # supervisor still sees the full checklist state — only the verbatim
        # transcript window shrinks. Override via SUP_CONVO_TURNS env.
        import os as _os_sup
        SUP_CONVO_TURNS = int(_os_sup.environ.get("SUP_CONVO_TURNS", "3"))
        non_system = [m for m in conversation_messages if m["role"] != "system"]
        recent = non_system[-(2 * SUP_CONVO_TURNS):]
        truncated_prefix = ""
        if len(non_system) > len(recent):
            truncated_prefix = f"[... {len(non_system) - len(recent)} earlier messages omitted for brevity ...]\n"
        convo_text = truncated_prefix
        last_therapist = ""
        last_patient = ""
        for msg in recent:
            role = "Therapist" if msg["role"] == "assistant" else "Patient"
            convo_text += f"{role}: {msg['content']}\n"
        for msg in reversed(conversation_messages):
            if msg["role"] == "assistant" and not last_therapist:
                last_therapist = msg["content"]
            elif msg["role"] == "user" and not last_patient:
                last_patient = msg["content"]
            if last_therapist and last_patient:
                break

        # Build a numbered list of pending items so the supervisor can return
        # IDs instead of free-text. Free-text was unreliable (the 4B supervisor
        # paraphrases item names, e.g. returns "PHQ-9 mood" for the checklist
        # key "Persistent low mood (PHQ-9 item 1)"); fuzzy matching against
        # such paraphrases was the cause of ir_frac=0 across the previous run.
        id_to_item = []  # list of (cat_data, item_key) — supervisor's IDs are 1-indexed positions
        pending_text = ""
        if pending_items:
            for cat_name, items in pending_items.items():
                pending_text += f"\n{cat_name}:\n"
                for item in items:
                    # Map back to cat_data so we can mark covered later.
                    for cd in extraction_status.values():
                        if cd["name"] == cat_name and item in cd["items"]:
                            id_to_item.append((cd, item))
                            pending_text += f"  [{len(id_to_item)}] {item}\n"
                            break
        else:
            pending_text = "(all items covered)"

        next_pri = get_next_priority(extraction_status)

        messages = [
            {"role": "system",
             "content": "You are a clinical supervisor. Respond ONLY in valid JSON."},
            {"role": "user", "content": (
                f"CONVERSATION SO FAR:\n{convo_text}\n"
                f"CHECKLIST ITEMS STILL PENDING (each prefixed with [ID]):\n{pending_text}\n"
                f"NEXT PRIORITY AREA: {next_pri or 'All extraction items covered.'}\n\n"
                f"PATIENT JUST SAID:\n\"{last_patient}\"\n"
                f"THERAPIST RESPONDED:\n\"{last_therapist}\"\n\n"
                "Return JSON with TWO fields:\n"
                '{\n'
                '  "covered_ids": [<list of integer IDs from the pending list above that were covered in this exchange>],\n'
                '  "feedback": "<2-4 short bullet points to coach the next therapist turn>"\n'
                '}\n'
                'If nothing new was covered this turn, use "covered_ids": [].\n'
                'Only return IDs from the pending list (do not invent new IDs).'
            )},
        ]

        try:
            response = await self.supervisor_client.chat_completion(
                messages=messages,
                max_tokens=256,        # IDs JSON (~24 tok) + feedback (3–4 bullets, ~200 tok, with headroom)
                temperature=0.2,
            )
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            feedback = None
            if start >= 0 and end > start:
                try:
                    result = json.loads(text[start:end])
                except json.JSONDecodeError:
                    result = {}
                # Primary path: covered_ids (1-indexed into the numbered list above).
                raw_ids = result.get("covered_ids", []) if isinstance(result, dict) else []
                if not isinstance(raw_ids, list):
                    raw_ids = []
                for rid in raw_ids:
                    try:
                        idx = int(rid) - 1
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(id_to_item):
                        cat_data, item_key = id_to_item[idx]
                        cat_data["items"][item_key]["covered"] = True
                        cat_data["items"][item_key]["turn_covered"] = turn
                # Backward-compat fallback: legacy covered_items free-text shape
                # (older prompts or supervisors that ignore the new schema).
                # Uses the robust matcher; we still want IR to climb even if
                # the supervisor reverts to paraphrased item names.
                raw_items = result.get("covered_items", []) if isinstance(result, dict) else []
                if isinstance(raw_items, list):
                    for entry in raw_items:
                        item_text = entry.get("item", "") if isinstance(entry, dict) else (entry if isinstance(entry, str) else "")
                        if not item_text or not isinstance(item_text, str):
                            continue
                        match = _best_checklist_match(item_text, extraction_status)
                        if match is not None:
                            cat_data, item_key = match
                            cat_data["items"][item_key]["covered"] = True
                            cat_data["items"][item_key]["turn_covered"] = turn
                fb = result.get("feedback", None) if isinstance(result, dict) else None
                if isinstance(fb, str) and fb.strip():
                    feedback = fb.strip()
                elif isinstance(fb, list):
                    # Some responses give feedback as a list of bullet points.
                    fb_str = "\n".join(s for s in fb if isinstance(s, str))
                    if fb_str.strip():
                        feedback = fb_str.strip()
        except Exception as e:
            logger.warning(f"Combined supervisor call failed: {e}")
            feedback = None

        return extraction_status, feedback

    # ========== Extraction Checklist Tracking (legacy, kept for parity) ==========

    async def _update_extraction_status(
        self, conversation_messages: list, extraction_status: dict, turn: int
    ) -> dict:
        """Use supervisor LLM to determine which extraction items were covered."""
        if not self.supervisor_client:
            return extraction_status

        # Build pending items
        pending_items = {}
        for category, data in extraction_status.items():
            pending = [item for item, st in data["items"].items() if not st["covered"]]
            if pending:
                pending_items[data["name"]] = pending
        
        if not pending_items:
            return extraction_status

        # Build conversation text
        convo_text = ""
        for msg in conversation_messages:
            if msg["role"] == "system":
                continue
            role = "Therapist" if msg["role"] == "assistant" else "Patient"
            convo_text += f"{role}: {msg['content']}\n"
        
        # Numbered list — same ID-based scheme as _supervise_turn (see comments
        # there). Eliminates fuzzy paraphrase matching that caused ir_frac=0.
        id_to_item = []
        pending_text = ""
        for cat_name, items in pending_items.items():
            pending_text += f"\n{cat_name}:\n"
            for item in items:
                for cd in extraction_status.values():
                    if cd["name"] == cat_name and item in cd["items"]:
                        id_to_item.append((cd, item))
                        pending_text += f"  [{len(id_to_item)}] {item}\n"
                        break

        messages = [
            {"role": "system", "content": "You are a clinical supervisor tracking what information has been obtained from a patient. Respond ONLY in valid JSON."},
            {"role": "user", "content": (
                f"CONVERSATION SO FAR:\n{convo_text}\n\n"
                f"CHECKLIST ITEMS STILL PENDING (each prefixed with [ID]):\n{pending_text}\n\n"
                "Which of these pending items were covered (even partially) by the patient?\n"
                'Respond in JSON: {"covered_ids": [<list of integer IDs from the pending list above>]}\n'
                'If nothing new was covered: {"covered_ids": []}\n'
                'Only return IDs from the pending list (do not invent new IDs).'
            )},
        ]

        try:
            # JSON of newly-covered IDs is short; 256 tokens is plenty.
            response = await self.supervisor_client.chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.1,
            )
            # Parse JSON
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                # Primary: covered_ids (1-indexed integers).
                for rid in result.get("covered_ids", []):
                    try:
                        idx = int(rid) - 1
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(id_to_item):
                        cat_data, item_key = id_to_item[idx]
                        cat_data["items"][item_key]["covered"] = True
                        cat_data["items"][item_key]["turn_covered"] = turn
                # Backward-compat fallback for legacy covered_items shape.
                for entry in result.get("covered_items", []):
                    item_text = entry.get("item", "") if isinstance(entry, dict) else (entry if isinstance(entry, str) else "")
                    if not item_text:
                        continue
                    match = _best_checklist_match(item_text, extraction_status)
                    if match is not None:
                        cat_data, item_key = match
                        cat_data["items"][item_key]["covered"] = True
                        cat_data["items"][item_key]["turn_covered"] = turn
        except Exception as e:
            logger.warning(f"Extraction status update failed: {e}")
        
        return extraction_status

    def _should_end_session(self, last_response: str) -> bool:
        """Check if session should naturally end."""
        return should_end_session(last_response)
