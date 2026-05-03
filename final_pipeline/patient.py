"""
Patient module - Simulates a patient responding to a therapist based on a Reddit post context.
The patient acts out the scenario described in the Reddit post.
"""

import json
import time
import torch
from transformers import pipeline
from typing import Dict, List, Optional


class Patient:
    """Represents a patient in therapy, simulating behavior based on a Reddit post."""
    
    def __init__(self, reddit_post: List[str], patient_profile: Optional[Dict] = None, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", load_model: bool = True):
        """
        Initialize the patient.
        
        Args:
            reddit_post: Reddit post history (context for patient to roleplay)
            patient_profile: Optional patient demographics and history
            model_name: LLM model to use
            load_model: If False, skip loading the HF model (for data prep / vLLM mode)
        """
        self.reddit_post = reddit_post
        self.patient_profile = patient_profile or self._default_profile()
        self.conversation_history = []
        self.model_name = model_name
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
    
    def _default_profile(self) -> Dict:
        """Return a default patient profile."""
        return {
            "name": "Patient",
            "age": 30,
            "occupation": "Not specified",
            "family_structure": "Not specified",
            "primary_concern": "Mental health support"
        }
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for patient based on Reddit post."""
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
3. Be natural and conversational - not overly formal.
4. Share relevant details from your experience when asked.
5. Show vulnerability and openness as appropriate.
6. If asked questions you don't have info for, say so naturally.
7. Maintain consistency with the Reddit post narrative throughout conversation.
"""
    
    def generate_response(self, therapist_message: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Generate patient's response to therapist input.
        
        Args:
            therapist_message: The therapist's question or statement
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for response diversity
            
        Returns:
            Patient's response
        """
        system_prompt = self.get_system_prompt()
        
        # Build conversation context
        context = "Therapist: " + therapist_message + "\n\nPatient:"
        
        # Format prompt in Llama3 chat format
        formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        # Generate response
        output = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        
        response = output[0]["generated_text"].strip()
        
        # Extract just the patient's response (remove system/user parts)
        if "assistant<|end_header_id|>" in response:
            response = response.split("assistant<|end_header_id|>")[-1].strip()
        
        self.conversation_history.append({
            "role": "patient",
            "message": response,
            "timestamp": time.time()
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Return the patient's conversation history."""
        return self.conversation_history
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
