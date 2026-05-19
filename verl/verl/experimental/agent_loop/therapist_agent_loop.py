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
    EXTRACTION_CHECKLIST,
    EXTRACTION_PRIORITY_ORDER,
    get_therapist_system_prompt,
    get_patient_system_prompt,
    get_supervisor_system_prompt,
    get_session_phase,
    should_end_session,
    init_extraction_status,
    compute_extraction_summary,
    format_extraction_status,
    get_next_priority,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ExternalModelClient:
    """Async client for OpenAI-compatible API endpoints (vLLM, TGI, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
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
        """Call the /v1/chat/completions endpoint and return the text response."""
        session = await self._get_session()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

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

        # External model config for patient
        patient_cfg = self.config.data.get("patient_model", {})
        self.patient_client = ExternalModelClient(
            base_url=patient_cfg.get("base_url", "http://localhost:8001"),
            model=patient_cfg.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            api_key=patient_cfg.get("api_key", "EMPTY"),
        )
        self.patient_max_tokens = patient_cfg.get("max_tokens", 256)
        self.patient_temperature = patient_cfg.get("temperature", 0.7)

        # External model config for supervisor (agent_2)
        supervisor_cfg = self.config.data.get("supervisor_model", {})
        self.supervisor_client = ExternalModelClient(
            base_url=supervisor_cfg.get("base_url", "http://localhost:8002"),
            model=supervisor_cfg.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            api_key=supervisor_cfg.get("api_key", "EMPTY"),
        ) if self.enable_supervisor else None

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
                
                # Generate patient response if not final turn
                if turn_idx < self.max_turns - 1:
                    patient_text = await self._generate_patient_response(
                        messages=conversation_messages,
                        patient_context=patient_context,
                        patient_profile=patient_profile,
                    )
                    
                    if patient_text:
                        conversation_messages.append({"role": "user", "content": patient_text})
                        
                        # Tokenize patient text to track in response sequence
                        patient_token_ids = self.tokenizer.encode(patient_text, add_special_tokens=False)
                        # Mark patient tokens as mask=0 (NOT trained on)
                        all_response_ids.extend(patient_token_ids)
                        all_response_masks.extend([0] * len(patient_token_ids))
                
                num_turns += 1
                
                # Update extraction checklist and get supervisor feedback
                last_supervisor_feedback = None
                if self.enable_supervisor and self.supervisor_client:
                    # Update what's been extracted so far
                    extraction_status = await self._update_extraction_status(
                        conversation_messages, extraction_status, turn_idx
                    )
                    
                    # Get checklist-based feedback for the next turn
                    last_supervisor_feedback = await self._generate_supervisor_feedback(
                        conversation_messages=conversation_messages,
                        patient_context=patient_context,
                        extraction_status=extraction_status,
                    )
                    if last_supervisor_feedback:
                        metrics[f"supervisor_feedback_turn_{turn_idx}"] = last_supervisor_feedback
                    
                    metrics[f"extraction_status_turn_{turn_idx}"] = compute_extraction_summary(extraction_status)
                
                # Check for natural session ending
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
            }
        )
        
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
                    **self.apply_chat_template_kwargs
                ),
            )
        
        # Generate
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
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
            response = await self.supervisor_client.chat_completion(
                messages=supervisor_messages,
                max_tokens=400,
                temperature=0.7,
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
    
    # ========== Extraction Checklist Tracking ==========

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
        
        pending_text = ""
        for cat_name, items in pending_items.items():
            pending_text += f"\n{cat_name}:\n"
            for item in items:
                pending_text += f"  - {item}\n"

        messages = [
            {"role": "system", "content": "You are a clinical supervisor tracking what information has been obtained from a patient. Respond ONLY in valid JSON."},
            {"role": "user", "content": (
                f"CONVERSATION SO FAR:\n{convo_text}\n\n"
                f"CHECKLIST ITEMS STILL PENDING:\n{pending_text}\n\n"
                "Which of these pending items were covered (even partially)?\n"
                'Respond in JSON: {"covered_items": [{"item": "exact item text", "evidence": "brief quote"}]}\n'
                'If nothing new was covered: {"covered_items": []}'
            )},
        ]

        try:
            response = await self.supervisor_client.chat_completion(
                messages=messages,
                max_tokens=500,
                temperature=0.3,
            )
            # Parse JSON
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                for entry in result.get("covered_items", []):
                    item_text = entry.get("item", "")
                    for cat_data in extraction_status.values():
                        for item_key in cat_data["items"]:
                            if item_key.lower() in item_text.lower() or item_text.lower() in item_key.lower():
                                cat_data["items"][item_key]["covered"] = True
                                cat_data["items"][item_key]["turn_covered"] = turn
                                break
        except Exception as e:
            logger.warning(f"Extraction status update failed: {e}")
        
        return extraction_status

    def _should_end_session(self, last_response: str) -> bool:
        """Check if session should naturally end."""
        return should_end_session(last_response)
