"""
Agent 2 module - Senior Therapist Supervisor
Reviews Agent 1's responses against therapeutic best practices and provides suggestions.
Tracks checklist progress over time for inference sessions.
"""

import json
import os
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

        ID-based scheme (matches training's _supervise_turn in
        verl/.../therapist_agent_loop.py): the supervisor sees the pending
        items numbered [1], [2], ... and replies with an integer-id JSON
        array {"covered_ids": [1, 3]}. We map IDs back exactly — no string
        fuzziness, no paraphrase-induced misses.

        Why: the previous free-text-name scheme was the explicit cause of
        ir_frac ≈ 0 (4B supervisors paraphrase canonical item names like
        "Persistent low mood (PHQ-9 item 1)" as "PHQ-9 mood"; substring match
        against the paraphrase fails and nothing gets marked covered).

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

        # Build a numbered list of pending items so the supervisor can return
        # IDs instead of paraphrased names. id_to_item[i] gives back the
        # (category, item_key) pair for the supervisor's 1-indexed [i+1].
        id_to_item: List[tuple] = []
        pending_block = ""
        for category, data in self.extraction_status.items():
            cat_pending = [item for item, st in data["items"].items() if not st["covered"]]
            if not cat_pending:
                continue
            pending_block += f"\n{data['name']}:\n"
            for item in cat_pending:
                id_to_item.append((category, item))
                pending_block += f"  [{len(id_to_item)}] {item}\n"
        if not id_to_item:
            return compute_summary(self.extraction_status)

        user_prompt = (
            f'PATIENT JUST SAID:\n"{latest_patient}"\n\n'
            f"PENDING CHECKLIST ITEMS (each prefixed with [ID]):\n{pending_block}\n"
            "Which of these items did the patient just disclose in the statement above?\n"
            'Reply with JSON: {"covered_ids": [<list of integer IDs from above>]}.\n'
            'If nothing was disclosed, reply: {"covered_ids": []}\n'
            "Only return IDs from the list (do not invent new IDs)."
        )

        try:
            response_text = self.llm.chat(
                [
                    {"role": "system", "content": (
                        "You are a clinical supervisor. Respond ONLY in valid JSON "
                        'of the form {"covered_ids": [<int>...]}. Nothing else.'
                    )},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=128,
                temperature=0.1,
                top_p=0.9,
            )
        except Exception as e:
            print(f"[supervisor] update_extraction_status LLM call failed at turn {turn_number}: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            self._save_progress()
            return compute_summary(self.extraction_status)

        # Parse: first JSON object in the response, pull covered_ids.
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])
                raw_ids = result.get("covered_ids", []) if isinstance(result, dict) else []
                if isinstance(raw_ids, list):
                    for rid in raw_ids:
                        try:
                            idx = int(rid) - 1
                        except (TypeError, ValueError):
                            continue
                        if 0 <= idx < len(id_to_item):
                            cat, item = id_to_item[idx]
                            self.extraction_status[cat]["items"][item]["covered"] = True
                            self.extraction_status[cat]["items"][item]["turn_covered"] = turn_number
                            self.extraction_status[cat]["items"][item]["evidence"] = latest_patient
        except (json.JSONDecodeError, KeyError):
            pass  # Parsing failed → keep current state.

        self._save_progress()
        return compute_summary(self.extraction_status)
    
    # Mirrors training (verl/.../therapist_agent_loop.py:_supervise_turn) —
    # cap the supervisor's transcript context at the last N exchanges so the
    # prompt doesn't overflow the patient/supervisor vLLM's max_model_len on
    # long sessions, AND so train + eval see the same context window when
    # judging coverage. Training sets SUP_CONVO_TURNS=3 (and exposes the
    # SUP_CONVO_TURNS env override); eval mirrors that exact value.
    SUP_CONVO_TURNS = int(os.environ.get("SUP_CONVO_TURNS", "3"))

    # The exact override training applies when the checklist becomes complete
    # (verl/.../therapist_agent_loop.py:363-371). Reproduced verbatim so that
    # the trained-model's "wrap up" trigger fires identically in eval.
    _CHECKLIST_COMPLETE_OVERRIDE = (
        "[CHECKLIST COMPLETE] Every required information-extraction "
        "item has been gathered. Do NOT introduce new probes. "
        "Begin wrapping up the session: briefly summarize what the "
        "client shared, validate their effort, and close the session "
        "naturally (e.g., 'Thank you for sharing this with me today. "
        "Take care, and I'll see you next time.'). End the session "
        "with clear closing language."
    )

    def supervise_turn(self, conversation_history: List[Dict], turn_number: int) -> Dict:
        """One combined supervisor call — matches training's _supervise_turn.

        Returns:
            {
              "feedback": str (with [CHECKLIST COMPLETE] override applied if all covered),
              "raw_feedback": str (pre-override, for logging),
              "extraction_summary": dict,
              "checklist_complete": bool,
              "patient_message": str,  # last patient evidence the supervisor saw
              "therapist_response": str,
              "timestamp": float,
              "turn": int,
            }

        Does ONE LLM call (vs the prior pair of update_extraction_status +
        evaluate_response calls). Uses the ID-based numbered scheme so the 4B
        supervisor can return integer IDs that map back to checklist items
        exactly (avoids paraphrase-induced misses).
        """
        # Identify last therapist + last patient from the FULL transcript
        # (for the explicit "PATIENT JUST SAID" / "THERAPIST RESPONDED" fields).
        last_therapist = ""
        last_patient = ""
        for msg in reversed(conversation_history):
            role = msg.get("role")
            text = (msg.get("message") or msg.get("content") or "").strip()
            if not text:
                continue
            if role == "therapist" and not last_therapist:
                last_therapist = text
            elif role == "patient" and not last_patient:
                last_patient = text
            if last_therapist and last_patient:
                break

        # Build pending items as a numbered list (1-indexed IDs).
        id_to_item: List[tuple] = []
        pending_block = ""
        for category, data in self.extraction_status.items():
            cat_pending = [item for item, st in data["items"].items() if not st["covered"]]
            if not cat_pending:
                continue
            pending_block += f"\n{data['name']}:\n"
            for item in cat_pending:
                id_to_item.append((category, item))
                pending_block += f"  [{len(id_to_item)}] {item}\n"
        if not pending_block:
            pending_block = "(all items covered)"

        # Truncate conversation to most recent SUP_CONVO_TURNS exchanges
        # (matches training's prompt-overflow guard).
        non_system = [m for m in conversation_history
                      if m.get("role") in ("patient", "therapist")]
        recent = non_system[-(2 * self.SUP_CONVO_TURNS):]
        truncated_prefix = ""
        if len(non_system) > len(recent):
            truncated_prefix = (
                f"[... {len(non_system) - len(recent)} earlier messages omitted "
                f"for brevity ...]\n"
            )
        convo_text = truncated_prefix
        for msg in recent:
            label = "Therapist" if msg.get("role") == "therapist" else "Patient"
            convo_text += f"{label}: {(msg.get('message') or '').strip()}\n"

        next_pri = get_next_priority(self.extraction_status, self.priority_order)

        messages = [
            {"role": "system",
             "content": "You are a clinical supervisor. Respond ONLY in valid JSON."},
            {"role": "user", "content": (
                f"CONVERSATION SO FAR:\n{convo_text}\n"
                f"CHECKLIST ITEMS STILL PENDING (each prefixed with [ID]):\n{pending_block}\n"
                f"NEXT PRIORITY AREA: {next_pri or 'All extraction items covered.'}\n\n"
                f"PATIENT JUST SAID:\n\"{last_patient}\"\n"
                f"THERAPIST RESPONDED:\n\"{last_therapist}\"\n\n"
                "Return JSON with TWO fields:\n"
                "{\n"
                "  \"covered_ids\": [<integer IDs from the pending list above that were covered in this exchange>],\n"
                "  \"feedback\": \"<2-4 short bullet points to coach the next therapist turn>\"\n"
                "}\n"
                "If nothing new was covered this turn, use \"covered_ids\": [].\n"
                "Only return IDs from the pending list (do not invent new IDs)."
            )},
        ]

        raw_feedback: Optional[str] = None
        try:
            # Sampling matches training (_supervise_turn @ therapist_agent_loop.py:909-914)
            # exactly: max_tokens=256, temperature=0.2, no top_p (vLLM default 1.0).
            # Previously eval passed top_p=0.9 which shifted the supervisor's
            # output distribution vs training and was a hidden source of
            # train/eval IR divergence.
            response_text = self.llm.chat(
                messages,
                max_tokens=256,
                temperature=0.2,
            )
        except Exception as e:
            print(f"[supervisor] supervise_turn LLM call failed at turn {turn_number}: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            response_text = ""

        # Parse the JSON envelope.
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])
                if isinstance(result, dict):
                    raw_ids = result.get("covered_ids", [])
                    if isinstance(raw_ids, list):
                        for rid in raw_ids:
                            try:
                                idx = int(rid) - 1
                            except (TypeError, ValueError):
                                continue
                            if 0 <= idx < len(id_to_item):
                                cat, item = id_to_item[idx]
                                self.extraction_status[cat]["items"][item]["covered"] = True
                                self.extraction_status[cat]["items"][item]["turn_covered"] = turn_number
                                self.extraction_status[cat]["items"][item]["evidence"] = last_patient
                    fb = result.get("feedback")
                    if isinstance(fb, str) and fb.strip():
                        raw_feedback = fb.strip()
                    elif isinstance(fb, list):
                        joined = "\n".join(s for s in fb if isinstance(s, str))
                        if joined.strip():
                            raw_feedback = joined.strip()
        except (json.JSONDecodeError, KeyError):
            pass  # parsing failure → keep state, raw_feedback stays None

        summary = compute_summary(self.extraction_status)
        covered = summary.get("covered_items", 0) or 0
        total = summary.get("total_items", 0) or 0
        checklist_complete = total > 0 and covered >= total

        # Training's override (verl/.../therapist_agent_loop.py:363): when the
        # checklist is fully covered, replace the supervisor's organic
        # feedback with an explicit wrap-up instruction. This is what the
        # therapist was trained to receive at that point.
        final_feedback = self._CHECKLIST_COMPLETE_OVERRIDE if checklist_complete else (raw_feedback or "")

        feedback_dict = {
            "patient_message": last_patient,
            "therapist_response": last_therapist,
            "feedback": final_feedback,
            "raw_feedback": raw_feedback,
            "extraction_summary": summary,
            "checklist_complete": checklist_complete,
            "timestamp": time.time(),
            "turn": turn_number,
        }
        self.feedback_history.append(feedback_dict)
        self._save_progress()
        return feedback_dict

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
