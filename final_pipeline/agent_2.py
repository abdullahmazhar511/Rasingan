"""
Agent 2 module - Senior Therapist Supervisor
Reviews Agent 1's responses against therapeutic best practices and provides suggestions.
Tracks checklist progress over time for inference sessions.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from shared_config import get_supervisor_system_prompt
from shared_vllm import DEFAULT_MODEL, SharedVLLM, get_shared_vllm
from checklists import (
    get_checklist,
    init_status,
    compute_summary,
    format_status,
    get_pending_items,
    get_next_priority,
)


# NOTE: the supervisor's per-turn judge now only asks the LLM to list which
# pending items the *latest patient utterance* satisfied (returned as a JSON
# array of item strings). No evidence quote, no transcript dump, no absence-
# narration to filter — the response is just names matched back to the
# pending list. Anything outside that list is silently rejected by the
# substring match in update_extraction_status.


class Agent2_SeniorTherapist:
    """Senior therapist agent that supervises and coaches Agent 1.
    
    Maintains an EXTRACTION CHECKLIST of information the junior therapist
    must obtain from the patient, and tracks progress toward completing it.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, load_model: bool = True,
                 session_id: Optional[str] = None, progress_dir: Optional[Path] = None,
                 checklist_name: str = "combined",
                 llm: Optional[SharedVLLM] = None):
        """
        Initialize senior therapist supervisor.

        Args:
            model_name: HF model id used when no shared client is supplied
            load_model: If False, skip loading the LLM (for data prep / VERL mode)
            session_id: Unique ID for this supervision session (for tracking progress)
            progress_dir: Directory to save checklist progress (default: ./supervisor_progress/)
            checklist_name: Which checklist to track — "phq9", "gad7", or "combined" (default)
            llm: Pre-loaded shared vLLM client (preferred — share one model with Patient)
        """
        self.model_name = model_name
        self.feedback_history = []
        self._llm = llm

        # Progress tracking
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.progress_dir = Path(progress_dir or "./supervisor_progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        bundle = get_checklist(checklist_name)
        self.checklist_name = checklist_name
        self.checklist = bundle["checklist"]
        self.priority_order = bundle["priority_order"]
        self.instrument_label = bundle.get("instrument", checklist_name)
        self.extraction_status = init_status(self.checklist)
        self.progress_log_path = self.progress_dir / f"session_{self.session_id}_checklist.json"
        self._load_progress()

        if load_model and self._llm is None:
            self._llm = get_shared_vllm(model_name)

    @property
    def llm(self) -> SharedVLLM:
        if self._llm is None:
            self._llm = get_shared_vllm(self.model_name)
        return self._llm
    
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
            "extraction_summary": compute_summary(self.extraction_status)
        }
        with open(self.progress_log_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def update_extraction_status(self, conversation_history: List[Dict], turn_number: int) -> Dict:
        """Mark any pending checklist items the patient just disclosed.

        Simple per-turn judge: look at the *latest patient utterance*, ask the
        LLM which (if any) of the still-pending items it satisfies, return a
        JSON list of item names. No evidence quote, no full transcript, no
        absence-narration handling required — the LLM can only respond with
        items from the pending list, and anything else is rejected by the
        substring match.

        Args:
            conversation_history: full session transcript
            turn_number: current turn number
        Returns:
            Updated extraction summary
        """
        # Find the latest patient utterance (what we want to judge against the checklist).
        latest_patient = ""
        for msg in reversed(conversation_history):
            if msg.get("role") == "patient":
                latest_patient = (msg.get("message") or "").strip()
                break
        if not latest_patient:
            return compute_summary(self.extraction_status)

        # Collect pending items as a flat list — that's all the LLM needs to see.
        pending: List[str] = []
        item_to_category: Dict[str, str] = {}
        for category, data in self.extraction_status.items():
            for item, status in data["items"].items():
                if not status["covered"]:
                    pending.append(item)
                    item_to_category[item] = category
        if not pending:
            return compute_summary(self.extraction_status)

        pending_block = "\n".join(f"- {it}" for it in pending)

        user_prompt = (
            f'PATIENT JUST SAID:\n"{latest_patient}"\n\n'
            f"PENDING CHECKLIST ITEMS:\n{pending_block}\n\n"
            "Which of these items did the patient just disclose in the statement above?\n"
            "Reply with a JSON array of the exact item strings from the list, e.g.:\n"
            '  ["Feeling down, depressed, or hopeless"]\n'
            "If the patient did not disclose any of these items, reply: []"
        )

        try:
            response_text = self.llm.chat(
                [
                    {"role": "system", "content": (
                        "You are a clinical supervisor. Reply with a JSON array of "
                        "checklist item strings the patient just disclosed. Nothing else."
                    )},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.1,
                top_p=0.9,
            )
        except Exception as e:
            print(f"[supervisor] update_extraction_status LLM call failed at turn {turn_number}: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            self._save_progress()
            return compute_summary(self.extraction_status)

        # Parse: find the first JSON array in the response.
        try:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start >= 0 and end > start:
                covered_names = json.loads(response_text[start:end])
                if isinstance(covered_names, list):
                    for name in covered_names:
                        if not isinstance(name, str) or not name.strip():
                            continue
                        # Substring match against pending items (either direction)
                        n_low = name.lower()
                        for item in pending:
                            if item.lower() in n_low or n_low in item.lower():
                                cat = item_to_category[item]
                                self.extraction_status[cat]["items"][item]["covered"] = True
                                self.extraction_status[cat]["items"][item]["turn_covered"] = turn_number
                                self.extraction_status[cat]["items"][item]["evidence"] = latest_patient
                                break
        except (json.JSONDecodeError, KeyError):
            pass  # Parsing failed → keep current state.

        self._save_progress()
        return compute_summary(self.extraction_status)
    
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
        extraction_text = format_status(self.extraction_status)
        next_pri = get_next_priority(self.extraction_status, self.priority_order)
        
        return f"""You are an experienced senior therapist/clinical supervisor reviewing a therapy session.
A primary therapist has just responded to a patient. Your job is to evaluate whether the therapist
is making progress on the {self.instrument_label} information extraction checklist.

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
        return format_status(self.extraction_status)
    
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

        # Graceful fallback: if the supervisor LLM call fails (context length,
        # network, etc.) we still want the session to keep going. Record the
        # failure as the "feedback" instead of crashing.
        try:
            feedback_text = self.llm.chat(
                [
                    {"role": "system", "content": (
                        "You are a compassionate senior therapist supervisor. "
                        "Provide constructive, actionable feedback."
                    )},
                    {"role": "user", "content": eval_prompt},
                ],
                max_tokens=400,
                temperature=0.7,
                top_p=0.9,
            )
        except Exception as e:
            print(f"[supervisor] evaluate_response LLM call failed at turn {turn_number}: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            feedback_text = f"[supervisor unavailable: {type(e).__name__}]"
        
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
            "checklist": self.instrument_label,
            "evaluations_completed": len(self.feedback_history),
            "extraction_summary": compute_summary(self.extraction_status),
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
        print(f"Checklist: {report.get('checklist', 'n/a')}")
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
