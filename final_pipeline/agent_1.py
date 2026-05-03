"""
Agent 1 module - Primary Therapist (VERL-Compatible)
Engages with the patient and follows therapeutic best practices.
Can be used standalone or integrated with VERL training pipeline.
"""

import json
import time
import torch
from transformers import pipeline
from typing import Dict, List, Optional, Tuple

from shared_config import get_therapist_system_prompt


class Agent1_PrimaryTherapist:
    """Primary therapist agent that conducts the therapy session."""
    
    THERAPEUTIC_GUIDELINES = {
        "opening": "Establish rapport, create safe space, understand chief complaint",
        "active_listening": "Reflect back what you hear, validate emotions",
        "questioning": "Use open-ended questions, explore root causes",
        "empathy": "Show genuine understanding and compassion",
        "safety": "Assess for safety concerns, risk factors",
        "planning": "Collaborate on next steps and coping strategies"
    }
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", session_type: str = "counseling", load_model: bool = True):
        """
        Initialize primary therapist agent.
        
        Args:
            model_name: LLM model to use
            session_type: Type of therapy session (counseling, crisis, assessment, etc.)
            load_model: If False, skip loading the HF model (for data prep / vLLM mode)
        """
        self.model_name = model_name
        self.session_type = session_type
        self.conversation_history = []
        self.turn_count = 0
        self._pipeline = None
        
        if load_model:
            self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the HF text-generation pipeline on demand."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": torch.float16},
                device_map="auto",
            )
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline
    
    def get_system_prompt(self, session_phase: str = "opening") -> str:
        """Generate system prompt for therapist based on session phase."""
        return get_therapist_system_prompt(session_phase, self.session_type)
    
    def generate_opening(self) -> str:
        """Generate the opening of the therapy session."""
        self.turn_count += 1
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>

{self.get_system_prompt("opening")}<|eot_id|><|start_header_id|>user<|end_header_id|>

You are beginning a new therapy session. Open the session warmly and ask the patient what brings them in today.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        output = self.pipeline(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
        )
        
        response = output[0]["generated_text"].strip()
        if "assistant<|end_header_id|>" in response:
            response = response.split("assistant<|end_header_id|>")[-1].strip()
        
        self.conversation_history.append({
            "role": "therapist",
            "message": response,
            "phase": "opening",
            "turn": self.turn_count,
            "timestamp": time.time()
        })
        
        return response
    
    def respond_to_patient(self, patient_message: str, session_phase: str = "assessment",
                           supervisor_feedback: Optional[str] = None) -> str:
        """
        Generate therapist's response to patient message.
        
        Args:
            patient_message: The patient's statement or response
            session_phase: Current phase of therapy session
            supervisor_feedback: Optional feedback from the supervisor on the previous response
            
        Returns:
            Therapist's response
        """
        self.turn_count += 1
        
        # Build conversation context from history
        context_lines = []
        for msg in self.conversation_history[-4:]:  # Use last 4 messages for context
            role = "Therapist" if msg["role"] == "therapist" else "Patient"
            context_lines.append(f"{role}: {msg['message']}")
        
        context_lines.append(f"Patient: {patient_message}")
        context = "\n".join(context_lines)
        
        system_prompt = self.get_system_prompt(session_phase)
        
        # Add supervisor guidance if available
        supervisor_section = ""
        if supervisor_feedback:
            supervisor_section = f"""

SUPERVISOR GUIDANCE (use this to improve your next response):
{supervisor_feedback}

Apply the supervisor's suggestions while responding naturally to the patient. Do not mention the supervisor."""
        
        formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_prompt}{supervisor_section}<|eot_id|><|start_header_id|>user<|end_header_id|>

Conversation so far:
{context}

Therapist:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        output = self.pipeline(
            formatted_prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        response = output[0]["generated_text"].strip()
        if "assistant<|end_header_id|>" in response:
            response = response.split("assistant<|end_header_id|>")[-1].strip()
        
        self.conversation_history.append({
            "role": "therapist",
            "message": response,
            "phase": session_phase,
            "turn": self.turn_count,
            "timestamp": time.time()
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Return therapist's conversation history."""
        return self.conversation_history
    
    def reset_session(self):
        """Reset session for a new interaction."""
        self.conversation_history = []
        self.turn_count = 0
    
    # ========== VERL Training Integration Methods ==========
    
    def get_training_sample(self) -> Dict:
        """
        Convert current session into a training sample for VERL.
        
        Returns:
            Dictionary with training data:
            - messages: Formatted as [system, user, assistant, user, assistant, ...]
            - conversation_text: Each turn as structured conversation
            - metrics: Quality metrics for reward model
            - raw_prompt: Original prompt for reference
        """
        training_data = {
            "messages": self._format_messages_for_training(),
            "conversation_record": [
                {
                    "role": msg["role"],
                    "content": msg["message"],
                    "phase": msg.get("phase", "unknown"),
                    "timestamp": msg.get("timestamp", time.time())
                }
                for msg in self.conversation_history
            ],
            "turn_count": self.turn_count,
            "therapist_metrics": self._compute_therapist_metrics(),
        }
        return training_data
    
    def _format_messages_for_training(self) -> List[Dict]:
        """Format conversation history for VERL training format."""
        messages = []
        
        # Add initial system prompt
        messages.append({
            "role": "system",
            "content": self.get_system_prompt("opening")
        })
        
        # Add all conversation turns
        for entry in self.conversation_history:
            role = "assistant" if entry["role"] == "therapist" else "user"
            messages.append({
                "role": role,
                "content": entry["message"]
            })
        
        return messages
    
    def _compute_therapist_metrics(self) -> Dict:
        """
        Compute metrics about therapist performance for reward model.
        
        Returns:
            Dictionary with therapeutic quality indicators
        """
        metrics = {
            "total_turns": len(self.conversation_history),
            "avg_response_length": 0.0,
            "therapeutic_quality_indicators": {
                "uses_reflection": 0,
                "validates_emotion": 0,
                "asks_open_ended": 0,
                "shows_empathy": 0,
                "maintains_boundaries": 0,
            },
            "phase_distribution": {}
        }
        
        if not self.conversation_history:
            return metrics
        
        # Compute average response length
        therapist_responses = [
            len(msg["message"].split()) 
            for msg in self.conversation_history 
            if msg["role"] == "therapist"
        ]
        metrics["avg_response_length"] = (
            sum(therapist_responses) / len(therapist_responses) 
            if therapist_responses else 0.0
        )
        
        # Count therapeutic quality indicators in responses
        indicators = metrics["therapeutic_quality_indicators"]
        for msg in self.conversation_history:
            if msg["role"] == "therapist":
                content = msg["message"].lower()
                
                # Check for reflective listening
                if "it sounds like" in content or "so you're saying" in content:
                    indicators["uses_reflection"] += 1
                
                # Check for validation
                if "valid" in content or "understandable" in content or "makes sense" in content:
                    indicators["validates_emotion"] += 1
                
                # Check for open-ended questions (ends with ?)
                if content.strip().endswith("?"):
                    indicators["asks_open_ended"] += 1
                
                # Check for empathy markers
                if ("understand" in content or "care" in content or 
                    "empathi" in content or "compassion" in content):
                    indicators["shows_empathy"] += 1
                
                # Boundary maintenance
                if ("i can't" in content or "that's not my role" in content or
                    "we should focus" in content):
                    indicators["maintains_boundaries"] += 1
            
            # Track phase distribution
            phase = msg.get("phase", "unknown")
            metrics["phase_distribution"][phase] = metrics["phase_distribution"].get(phase, 0) + 1
        
        return metrics
    
    def to_verl_format(self) -> Dict:
        """
        Convert session to VERL-compatible format for rewards/training.
        
        Returns:
            Dictionary compatible with VERL training pipeline
        """
        messages = self._format_messages_for_training()
        
        return {
            "prompt": messages[:-1] if len(messages) > 1 else messages,  # All but last
            "response": messages[-1] if len(messages) > 1 else {"role": "assistant", "content": ""},
            "conversation_turns": len(self.conversation_history),
            "metrics": self._compute_therapist_metrics(),
            "raw_conversation": [msg for msg in self.conversation_history],
        }
    
    @classmethod
    def create_from_verl_messages(
        cls, 
        messages: List[Dict],
        session_type: str = "counseling",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ) -> "Agent1_PrimaryTherapist":
        """
        Create a therapist agent from VERL message format.
        
        Args:
            messages: VERL-format messages (system, user, assistant, ...)
            session_type: Type of therapy session
            model_name: LLM model name
            
        Returns:
            Agent1_PrimaryTherapist instance with conversation loaded
        """
        agent = cls(model_name=model_name, session_type=session_type)
        
        # Load conversation history from messages
        # Skip system message
        for msg in messages:
            if msg["role"] == "system":
                continue
            
            role = "therapist" if msg["role"] == "assistant" else "patient"
            agent.conversation_history.append({
                "role": role,
                "message": msg.get("content", ""),
                "phase": "loaded",
                "turn": len(agent.conversation_history),
                "timestamp": time.time()
            })
        
        agent.turn_count = len(agent.conversation_history)
        return agent
