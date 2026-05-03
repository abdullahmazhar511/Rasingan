"""
Agent 2 module - Senior Therapist Supervisor
Reviews Agent 1's responses against therapeutic best practices and provides suggestions.
Tracks checklist progress over time for inference sessions.
"""

import json
import time
import torch
from pathlib import Path
from transformers import pipeline
from typing import Dict, List, Optional

from shared_config import (
    EXTRACTION_CHECKLIST,
    EXTRACTION_PRIORITY_ORDER,
    get_supervisor_system_prompt,
    init_extraction_status,
    compute_extraction_summary,
    format_extraction_status,
    get_pending_items,
    get_next_priority,
)


class Agent2_SeniorTherapist:
    """Senior therapist agent that supervises and coaches Agent 1.
    
    Maintains an EXTRACTION CHECKLIST of information the junior therapist
    must obtain from the patient, and tracks progress toward completing it.
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", load_model: bool = True, 
                 session_id: Optional[str] = None, progress_dir: Optional[Path] = None):
        """
        Initialize senior therapist supervisor.
        
        Args:
            model_name: LLM model to use for generating feedback
            load_model: If False, skip loading the HF model (for data prep / vLLM mode)
            session_id: Unique ID for this supervision session (for tracking progress)
            progress_dir: Directory to save checklist progress (default: ./supervisor_progress/)
        """
        self.model_name = model_name
        self.feedback_history = []
        self._pipeline = None
        
        # Progress tracking
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.progress_dir = Path(progress_dir or "./supervisor_progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        self.extraction_status = init_extraction_status()
        self.progress_log_path = self.progress_dir / f"session_{self.session_id}_checklist.json"
        self._load_progress()
        
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
    
    def _load_progress(self):
        """Load existing checklist progress from file if it exists."""
        if self.progress_log_path.exists():
            with open(self.progress_log_path, 'r') as f:
                data = json.load(f)
                self.feedback_history = data.get("feedback_history", [])
                saved_extraction = data.get("extraction_status", {})
                if saved_extraction:
                    self.extraction_status = saved_extraction
    
    def _save_progress(self):
        """Save checklist progress to file."""
        progress_data = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "extraction_status": self.extraction_status,
            "feedback_history": self.feedback_history,
            "extraction_summary": compute_extraction_summary(self.extraction_status)
        }
        with open(self.progress_log_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def update_extraction_status(self, conversation_history: List[Dict], turn_number: int) -> Dict:
        """Ask the LLM to determine which extraction items were covered in the conversation.
        
        Args:
            conversation_history: List of {"role": "therapist"/"patient", "message": "..."}
            turn_number: Current turn number
            
        Returns:
            Updated extraction summary
        """
        # Build the conversation text
        convo_text = ""
        for msg in conversation_history:
            role = "Therapist" if msg.get("role") == "therapist" else "Patient"
            convo_text += f"{role}: {msg.get('message', '')}\n"
        
        # Build pending items list
        pending_items = {}
        for category, data in self.extraction_status.items():
            pending = [item for item, status in data["items"].items() if not status["covered"]]
            if pending:
                pending_items[data["name"]] = pending
        
        if not pending_items:
            return compute_extraction_summary(self.extraction_status)
        
        pending_text = ""
        for cat_name, items in pending_items.items():
            pending_text += f"\n{cat_name}:\n"
            for item in items:
                pending_text += f"  - {item}\n"
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a clinical supervisor tracking what information has been obtained from a patient during therapy.
Respond ONLY in valid JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>

CONVERSATION SO FAR:
{convo_text}

CHECKLIST ITEMS STILL PENDING:
{pending_text}

Which of these pending items were covered (even partially) in the conversation above?
For each covered item, provide a brief quote or evidence from the conversation.

Respond in this exact JSON format:
{{
  "covered_items": [
    {{"item": "exact item text from the list", "evidence": "brief quote from conversation"}}
  ]
}}

If nothing new was covered, respond: {{"covered_items": []}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        output = self.pipeline(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.3,  # Low temp for structured output
        )
        
        response_text = output[0]["generated_text"].strip()
        if "assistant<|end_header_id|>" in response_text:
            response_text = response_text.split("assistant<|end_header_id|>")[-1].strip()
        
        # Parse JSON response
        try:
            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])
                covered = result.get("covered_items", [])
                
                for entry in covered:
                    item_text = entry.get("item", "")
                    evidence = entry.get("evidence", "")
                    
                    # Find and mark the item as covered
                    for category, data in self.extraction_status.items():
                        for item_key in data["items"]:
                            if item_key.lower() in item_text.lower() or item_text.lower() in item_key.lower():
                                data["items"][item_key]["covered"] = True
                                data["items"][item_key]["turn_covered"] = turn_number
                                data["items"][item_key]["evidence"] = evidence
                                break
        except (json.JSONDecodeError, KeyError) as e:
            pass  # If parsing fails, extraction status stays unchanged
        
        self._save_progress()
        return compute_extraction_summary(self.extraction_status)
    
    def get_evaluation_prompt(self, patient_message: str, therapist_response: str, session_context: str = "") -> str:
        """
        Generate prompt for senior therapist to evaluate Agent 1's response.
        
        Args:
            patient_message: What the patient just said
            therapist_response: Agent 1's response to the patient
            session_context: Brief context about the session
            
        Returns:
            Formatted prompt for evaluation
        """
        extraction_text = format_extraction_status(self.extraction_status)
        next_pri = get_next_priority(self.extraction_status)
        
        return f"""You are an experienced senior therapist/clinical supervisor reviewing a therapy session.
A primary therapist has just responded to a patient. Your job is to evaluate whether the therapist
is making progress on the information extraction checklist.

INFORMATION EXTRACTION CHECKLIST (what the therapist still needs to obtain from the patient):
{extraction_text}

NEXT PRIORITY AREA: {next_pri or 'All extraction items covered!'}

PATIENT'S STATEMENT:
"{patient_message}"

PRIMARY THERAPIST'S RESPONSE:
"{therapist_response}"

SESSION CONTEXT: {session_context if session_context else "Initial assessment/counseling session"}

EVALUATION TASK:
1. Note which extraction checklist items were addressed in this exchange
2. Identify 1-2 strengths in the response
3. Suggest what the therapist should ask about next based on pending extraction items
4. Provide a concrete reframed response example if improvement is needed

Format your feedback clearly and constructively. Focus on coaching, not criticism.
"""
    
    def _format_extraction_status(self) -> str:
        """Format extraction checklist showing covered vs pending items."""
        return format_extraction_status(self.extraction_status)
    
    def evaluate_response(self, patient_message: str, therapist_response: str, 
                         session_context: str = "", turn_number: Optional[int] = None) -> Dict:
        """
        Evaluate Agent 1's response and provide feedback.
        Tracks checklist scores for progress monitoring.
        
        Args:
            patient_message: Patient's statement
            therapist_response: Agent 1's response
            session_context: Context about the session
            turn_number: Turn number (for tracking across full session)
            
        Returns:
            Dictionary with evaluation results and suggestions
        """
        eval_prompt = self.get_evaluation_prompt(patient_message, therapist_response, session_context)
        
        formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a compassionate senior therapist supervisor. Provide constructive, actionable feedback.<|eot_id|><|start_header_id|>user<|end_header_id|>

{eval_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        output = self.pipeline(
            formatted_prompt,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        feedback_text = output[0]["generated_text"].strip()
        if "assistant<|end_header_id|>" in feedback_text:
            feedback_text = feedback_text.split("assistant<|end_header_id|>")[-1].strip()
        
        feedback_dict = {
            "patient_message": patient_message,
            "therapist_response": therapist_response,
            "feedback": feedback_text,
            "timestamp": time.time(),
            "turn": turn_number
        }
        
        self.feedback_history.append(feedback_dict)
        
        self._save_progress()
        
        return feedback_dict
    
    def generate_suggestions(self, patient_message: str, therapist_response: str) -> str:
        """
        Generate specific suggestions for how Agent 1 could improve.
        
        Args:
            patient_message: Patient's statement
            therapist_response: Agent 1's response (to be improved)
            
        Returns:
            Suggestions for improvement
        """
        eval_result = self.evaluate_response(patient_message, therapist_response)
        return eval_result["feedback"]
    

    
    def get_feedback_history(self) -> List[Dict]:
        """Return supervisor's feedback history."""
        return self.feedback_history
    
    def reset_history(self):
        """Reset feedback history."""
        self.feedback_history = []
    
    def get_progress_report(self) -> Dict:
        """Get a formatted progress report of all evaluations."""
        report = {
            "session_id": self.session_id,
            "evaluations_completed": len(self.feedback_history),
            "extraction_summary": compute_extraction_summary(self.extraction_status),
            "recent_feedback": self.feedback_history[-3:] if self.feedback_history else [],
            "progress_log_path": str(self.progress_log_path)
        }
        return report
    
    def print_progress_report(self):
        """Print a human-readable progress report."""
        report = self.get_progress_report()
        
        print("\n" + "="*60)
        print("SUPERVISOR PROGRESS REPORT")
        print("="*60)
        print(f"Session ID: {report['session_id']}")
        print(f"Evaluations Completed: {report['evaluations_completed']}")
        
        extraction = report.get('extraction_summary', {})
        if extraction:
            print(f"\nExtraction Progress: {extraction.get('covered_items', 0)}/{extraction.get('total_items', 0)} items ({extraction.get('overall_pct', 0)}%)")
            for cat in extraction.get('categories', {}).values():
                mark = "✓" if cat['covered'] == cat['total'] else "◐" if cat['covered'] > 0 else "☐"
                print(f"  {mark} {cat['name']}: {cat['covered']}/{cat['total']}")
            pending = extraction.get('pending_categories', [])
            if pending:
                print(f"\nStill pending: {', '.join(pending)}")
        
        print(f"\nProgress saved to: {report['progress_log_path']}")
        print("="*60 + "\n")
