"""
Patient module — simulates a patient responding to a therapist based on a Reddit post context.

Uses a shared vLLM client (see shared_vllm.py) so Patient and Agent 2 share one model load.
"""

import time
from typing import Dict, List, Optional

from shared_vllm import DEFAULT_MODEL, SharedVLLM, get_shared_vllm


class Patient:
    """Represents a patient in therapy, simulating behavior based on a Reddit post."""

    def __init__(
        self,
        reddit_post: List[str],
        patient_profile: Optional[Dict] = None,
        model_name: str = DEFAULT_MODEL,
        llm: Optional[SharedVLLM] = None,
        load_model: bool = True,
    ):
        """
        Args:
            reddit_post: Reddit post history (context for patient to roleplay)
            patient_profile: Optional patient demographics and history
            model_name: HF model id used when no shared client is supplied
            llm: Pre-loaded shared vLLM client (used as-is when given — its
                model_name overrides this constructor's `model_name`)
            load_model: If False, skip loading the LLM (for data prep / VERL mode)
        """
        self.reddit_post = reddit_post
        self.patient_profile = patient_profile or self._default_profile()
        self.conversation_history: List[Dict] = []
        self.model_name = model_name
        self._llm = llm
        if load_model and self._llm is None:
            self._llm = get_shared_vllm(model_name)

    @property
    def llm(self) -> SharedVLLM:
        if self._llm is None:
            self._llm = get_shared_vllm(self.model_name)
        return self._llm

    def _default_profile(self) -> Dict:
        return {
            "name": "Patient",
            "age": 30,
            "occupation": "Not specified",
            "family_structure": "Not specified",
            "primary_concern": "Mental health support",
        }

    def get_system_prompt(self) -> str:
        """Generate system prompt for patient based on the Reddit post."""
        post_context = "\n".join(self.reddit_post) if isinstance(self.reddit_post, list) else self.reddit_post

        return f"""You are a patient in therapy. You are based on the following real Reddit post describing your situation:

REDDIT POST (Your Situation):
{post_context}

PATIENT PROFILE:
Name: {self.patient_profile.get('name', 'Patient')}
Age: {self.patient_profile.get('age', '30')}
Occupation: {self.patient_profile.get('occupation', 'Not specified')}
Family Structure: {self.patient_profile.get('family_structure', 'Not specified')}

INSTRUCTIONS:
1. Respond authentically as someone experiencing the situation described in the Reddit post.
2. Express genuine emotions, concerns, and worries about your situation.
3. Be natural and conversational — not overly formal.
4. Share relevant details from your experience when asked.
5. Show vulnerability and openness as appropriate.
6. If asked questions you don't have info for, say so naturally.
7. Maintain consistency with the Reddit post narrative throughout the conversation.
"""

    def _build_messages(
        self,
        therapist_message: str,
        transcript: Optional[List[Dict]],
        context_window: int,
    ) -> List[Dict[str, str]]:
        """Convert the shared transcript into chat-template messages.

        The LLM is roleplaying the patient, so patient turns map to "assistant"
        and therapist turns map to "user".
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.get_system_prompt()}
        ]

        if transcript:
            for msg in transcript[-context_window:]:
                text = (msg.get("message") or "").strip()
                if not text:
                    continue
                if msg.get("role") == "therapist":
                    messages.append({"role": "user", "content": text})
                else:
                    messages.append({"role": "assistant", "content": text})
            # If the latest therapist utterance isn't already the trailing user turn, add it.
            if (
                len(messages) == 1
                or messages[-1]["role"] != "user"
                or messages[-1]["content"] != therapist_message
            ):
                messages.append({"role": "user", "content": therapist_message})
        else:
            messages.append({"role": "user", "content": therapist_message})

        return messages

    def generate_response(
        self,
        therapist_message: str,
        # Matches training cmdline (+data.patient_model.max_tokens=192). Eval
        # used to default to 256 — let the patient ramble longer than training
        # ever saw, creating a distribution shift on every turn's input.
        max_tokens: int = 192,
        temperature: float = 0.7,
        transcript: Optional[List[Dict]] = None,
        context_window: int = 12,
    ) -> str:
        """
        Generate patient's response to the therapist's latest utterance.

        Args:
            therapist_message: The therapist's most recent utterance (the turn being responded to)
            max_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            transcript: Shared session transcript (entries with role + message).
                When given, the full bilateral conversation history is rendered
                so the patient stays consistent across turns.
            context_window: How many trailing transcript entries to include.

        Returns:
            Patient's response text
        """
        messages = self._build_messages(therapist_message, transcript, context_window)
        response = self.llm.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )

        self.conversation_history.append(
            {"role": "patient", "message": response, "timestamp": time.time()}
        )
        return response

    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history

    def reset_conversation(self) -> None:
        self.conversation_history = []
