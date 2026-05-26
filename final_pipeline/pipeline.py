"""
Main Pipeline Module
Orchestrates the interaction between Patient, Agent 1 (Primary Therapist), 
and Agent 2 (Senior Therapist Supervisor).
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from tqdm.auto import tqdm

from patient import Patient
from agent_1 import Agent1_PrimaryTherapist
from agent_2 import Agent2_SeniorTherapist
from shared_config import get_session_phase, should_end_session
from shared_vllm import DEFAULT_MODEL, RemoteVLLM, get_shared_vllm
from hf_therapist import HFTherapist


class TherapyPipeline:
    """Main pipeline orchestrating the three-way therapeutic interaction."""
    
    def __init__(self, reddit_posts: List[str], patient_profile: Optional[Dict] = None,
                 output_dir: str = "sessions", max_turns: int = 10,
                 enable_supervisor: bool = True,
                 shared_model: str = DEFAULT_MODEL,
                 therapist_model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 therapist_llm: Optional[HFTherapist] = None,
                 supervisor_endpoint: Optional[str] = None,
                 scenario_meta: Optional[Dict] = None):
        """
        Initialize the therapy pipeline.

        Backends per role:
          Patient + Supervisor (Agent 2)  →  vLLM (in-process or RemoteVLLM)
          Therapist (Agent 1)             →  HuggingFace transformers, in-process
                                              (HFTherapist) — NO vLLM.

        Args:
            reddit_posts:     Patient roleplay seed.
            patient_profile:  Patient demographics.
            output_dir:       Where to save session transcripts.
            max_turns:        Max conversation turns.
            enable_supervisor: If False, skip Agent 2 entirely.
            shared_model:     HF id for Patient + Supervisor (one vLLM client).
            therapist_model:  HF id or local dir for Agent 1. Stored for metadata
                              only — the actual model has already been loaded
                              into `therapist_llm` by the caller.
            therapist_llm:    Pre-built HFTherapist instance. The caller (run.py)
                              constructs ONE per process and passes it into every
                              TherapyPipeline. If None, raises — we don't load
                              the model lazily inside the pipeline anymore.
            supervisor_endpoint: If set, route Patient+Supervisor through this
                              vLLM HTTP server URL instead of an in-process load.
            scenario_meta:    Extra fields written into the session metadata.
        """
        if therapist_llm is None:
            raise ValueError(
                "TherapyPipeline requires a pre-built `therapist_llm` (HFTherapist). "
                "Construct it once in run.py.main() and pass it in."
            )
        self.reddit_posts = reddit_posts
        self.patient_profile = patient_profile
        self.output_dir = Path(output_dir)
        self.max_turns = max_turns
        self.enable_supervisor = enable_supervisor
        self.shared_model = shared_model
        self.therapist_model = therapist_model
        self.scenario_meta = scenario_meta or {}

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Session tracking — assigned before agents so the supervisor can share it.
        # Includes microseconds + random suffix so concurrent sessions never collide.
        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + "_" + uuid.uuid4().hex[:6]
        )

        # ---- LLM backends ---------------------------------------------------
        # Patient + Supervisor share ONE vLLM client (in-process or RemoteVLLM
        # when a supervisor_endpoint URL is supplied — the latter lets the
        # supervisor model run in another Python env).
        #
        # instruct_mode mirrors training (verl/.../therapist_agent_loop.py)
        # ExternalModelClient(instruct_mode=True): sets sampling to the
        # Qwen3-Instruct-2507 model-card values AND disables Qwen3 thinking
        # via chat_template_kwargs. Train run used INSTRUCT_MODE=1 (see
        # scripts/train_multiturn.log:18). Eval defaults the same; override
        # with EVAL_INSTRUCT_MODE=0 for legacy sampling.
        import os as _os
        self.supervisor_endpoint = supervisor_endpoint
        _instruct = _os.environ.get("EVAL_INSTRUCT_MODE", "1").strip().lower() not in ("0", "false", "no", "off")
        if supervisor_endpoint:
            shared_llm = RemoteVLLM(
                base_url=supervisor_endpoint,
                model_name=shared_model,
                instruct_mode=_instruct,
            )
        else:
            shared_llm = get_shared_vllm(shared_model)

        # Therapist (Agent 1) uses the HFTherapist that the CALLER built once
        # for the whole run.py process. We just hand it down to Agent 1.
        self.patient = Patient(reddit_posts, patient_profile, llm=shared_llm)
        self.therapist = Agent1_PrimaryTherapist(model_name=therapist_model, llm=therapist_llm)
        if enable_supervisor:
            self.supervisor = Agent2_SeniorTherapist(
                session_id=self.session_id,
                progress_dir=self.output_dir / "supervisor_progress",
                llm=shared_llm,
            )
        else:
            self.supervisor = None
        self.session_transcript = []
        self.supervisor_feedback = []
        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "reddit_posts": reddit_posts,
            "patient_profile": patient_profile,
            "max_turns": max_turns,
            "turn_count": 0,
            **self.scenario_meta,
        }
    
    def run_session(self, enable_supervisor: Optional[bool] = None,
                   enable_suggestions: bool = True,
                   verbose: bool = True) -> Dict:
        """
        Run a complete therapy session.

        Args:
            enable_supervisor: Whether to use supervisor feedback this run.
                Defaults to the pipeline's constructor setting. Cannot be True
                if the pipeline was built with enable_supervisor=False (no
                supervisor was instantiated).
            enable_suggestions: Whether to print suggestions in real-time
            verbose: Whether to print detailed output

        Returns:
            Session results dictionary
        """
        if enable_supervisor is None:
            enable_supervisor = self.enable_supervisor
        if enable_supervisor and self.supervisor is None:
            raise RuntimeError(
                "Pipeline was constructed with enable_supervisor=False; "
                "supervisor was never instantiated. Re-create the pipeline "
                "with enable_supervisor=True to enable feedback."
            )
        if verbose:
            print("=" * 80)
            print(f"THERAPY SESSION: {self.session_id}")
            print(f"Patient Context: {self.reddit_posts[0][:100]}...")
            print("=" * 80)
        
        # Phase 1: Opening
        therapist_opening = self.therapist.generate_opening()
        if verbose:
            print(f"\n[THERAPIST]: {therapist_opening}\n")
        
        self.session_transcript.append({
            "turn": 0,
            "role": "therapist",
            "message": therapist_opening,
            "phase": "opening"
        })
        
        time.sleep(0.5)  # Small delay to avoid rate limiting
        
        # Main conversation loop — restructured to match training
        # (verl/.../therapist_agent_loop.py): therapist-first inside each
        # iteration, with the NEXT patient response and the CURRENT supervisor
        # call running in parallel. The supervisor therefore sees the PRIOR
        # patient turn as evidence (not the current one), giving the same
        # one-turn-lag in extraction as training did.
        #
        # Pre-loop: generate the very first patient response (in reply to the
        # opening). Inside the loop, each iteration k:
        #   1. therapist_k = respond(patient_k, feedback_{k-1})
        #   2. if not last turn: parallel(patient_{k+1}, supervisor_k)
        #       — supervisor_k snapshot contains [..., patient_k, therapist_k]
        #         BUT we strip patient_k from the snapshot so the supervisor's
        #         "PATIENT JUST SAID" is patient_{k-1}, matching training.
        #
        # tqdm shows progress in quiet mode; disabled in verbose mode.
        last_feedback = None
        sup_pool = ThreadPoolExecutor(max_workers=2) if enable_supervisor else None

        # First patient response (responds to the opening). Generated up-front
        # so the loop can flip to therapist-first.
        patient_response = self.patient.generate_response(
            therapist_opening, transcript=self.session_transcript,
        )
        if verbose:
            print(f"[PATIENT]: {patient_response}\n")
        # The patient turn is appended INSIDE the loop body (at iter k=1) so
        # transcript ordering stays [opening, patient_1, therapist_1, ...].
        prior_patient_msg = patient_response

        turn_bar = tqdm(
            range(1, self.max_turns),
            desc=f"{self.session_id} turns",
            unit="turn",
            disable=verbose,
            leave=False,
        )
        try:
            for turn in turn_bar:
                if verbose:
                    print(f"\n--- Turn {turn} ---")

                # Append THIS turn's patient (generated in the previous iter,
                # or pre-loop for turn 1).
                self.session_transcript.append({
                    "turn": turn,
                    "role": "patient",
                    "message": prior_patient_msg,
                    "phase": "conversation",
                })

                # Therapist responds to the just-appended patient using the
                # supervisor feedback from the previous turn.
                session_phase = self._get_session_phase(turn)
                therapist_response = self.therapist.respond_to_patient(
                    prior_patient_msg, session_phase,
                    supervisor_feedback=last_feedback,
                    transcript=self.session_transcript,
                )
                if verbose:
                    print(f"[THERAPIST]: {therapist_response}")
                self.session_transcript.append({
                    "turn": turn,
                    "role": "therapist",
                    "message": therapist_response,
                    "phase": session_phase,
                })

                # Last in-loop iteration: no further patient or supervisor.
                # Matches training's `if turn_idx < max_turns - 1` guard.
                is_last_iter = (turn == self.max_turns - 1)
                if is_last_iter:
                    if self._should_end_session(therapist_response) and verbose:
                        print("\n[Session naturally concluding]")
                    break

                # Snapshot for the supervisor that EXCLUDES the just-appended
                # patient — so "PATIENT JUST SAID" surfaces patient_{k-1}, as
                # in training where patient_k is being generated concurrently.
                # session_transcript order at this moment:
                #   [..., patient_{k-1}, therapist_{k-1}, patient_k, therapist_k]
                # Drop the patient_k entry (index -2) to construct the view.
                sup_snapshot = self.session_transcript[:-2] + self.session_transcript[-1:]

                # Parallel: next patient generation + supervisor for this turn.
                patient_future = sup_pool.submit(
                    self.patient.generate_response,
                    therapist_response,
                    transcript=list(self.session_transcript),
                ) if sup_pool else None
                sup_future = sup_pool.submit(
                    self.supervisor.supervise_turn,
                    sup_snapshot, turn,
                ) if (sup_pool and enable_supervisor) else None

                # Sync fallback if pool not in use.
                if patient_future is None:
                    next_patient_msg = self.patient.generate_response(
                        therapist_response, transcript=list(self.session_transcript),
                    )
                else:
                    next_patient_msg = patient_future.result()

                if sup_future is not None:
                    sup_result = sup_future.result()
                    self.supervisor_feedback.append(sup_result)
                    last_feedback = sup_result.get("feedback") or None
                    if enable_suggestions and verbose:
                        print(f"\n[SUPERVISOR FEEDBACK]:\n{last_feedback}\n")
                else:
                    last_feedback = None

                prior_patient_msg = next_patient_msg
                if verbose:
                    print(f"[PATIENT]: {next_patient_msg}\n")

                # Natural-ending check on therapist response (post-supervisor
                # so the closing-language check still runs after wrap-up
                # override has had a chance to fire).
                if self._should_end_session(therapist_response):
                    if verbose:
                        print("\n[Session naturally concluding]")
                    break
        finally:
            if sup_pool is not None:
                sup_pool.shutdown(wait=False)
        
        # Save session
        session_results = self._save_session()
        
        if verbose:
            print("\n" + "=" * 80)
            print(f"Session saved to: {session_results['transcript_file']}")
            print("=" * 80)
        
        return session_results
    
    def _get_session_phase(self, turn: int) -> str:
        """Determine the session phase based on turn number."""
        return get_session_phase(turn, self.max_turns)
    
    def _get_session_context(self) -> str:
        """Generate brief context about the current session."""
        if len(self.session_transcript) <= 2:
            return "Initial assessment phase"
        elif len(self.session_transcript) <= self.max_turns:
            return "Middle phase - exploring and intervening"
        else:
            return "Closing phase"
    
    def _should_end_session(self, last_response: str) -> bool:
        """Check if the session should naturally end."""
        return should_end_session(last_response)
    
    def _save_session(self) -> Dict:
        """Save the session transcript and results."""
        timestamp = datetime.now().isoformat()
        
        # Pull checklist coverage from the supervisor (already judged turn-by-turn during the session).
        # When the supervisor is disabled, both fields stay empty.
        from checklists import compute_summary as _compute_summary  # local import to avoid cycles
        extraction_status = getattr(self.supervisor, "extraction_status", {}) if self.supervisor else {}
        extraction_summary = _compute_summary(extraction_status) if extraction_status else {}

        # Prepare full session data
        session_data = {
            "metadata": {
                **self.session_metadata,
                "end_time": timestamp,
                "duration_turns": len(self.session_transcript),
                "supervisor_feedback_count": len(self.supervisor_feedback),
                "checklist": getattr(self.supervisor, "instrument_label", None) if self.supervisor else None,
                "supervisor_enabled": self.enable_supervisor,
            },
            "transcript": self.session_transcript,
            "supervisor_feedback": self.supervisor_feedback,
            "extraction_status": extraction_status,
            "extraction_summary": extraction_summary,
        }
        
        # Save as JSON
        output_file = self.output_dir / f"session_{self.session_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=4, ensure_ascii=False)
        
        # Save summary
        summary_file = self.output_dir / f"summary_{self.session_id}.txt"
        self._save_summary(summary_file, session_data)
        
        return {
            "session_id": self.session_id,
            "transcript_file": str(output_file),
            "summary_file": str(summary_file),
            "duration_turns": len(self.session_transcript),
            "supervisor_feedbacks": len(self.supervisor_feedback)
        }
    
    def _save_summary(self, filepath: Path, session_data: Dict):
        """Save a human-readable summary."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("THERAPY SESSION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            meta = session_data["metadata"]
            f.write(f"Session ID: {meta['session_id']}\n")
            f.write(f"Start Time: {meta['start_time']}\n")
            f.write(f"End Time: {meta['end_time']}\n")
            f.write(f"Total Turns: {meta['duration_turns']}\n")
            f.write(f"Patient Profile: {meta['patient_profile']}\n")
            f.write(f"\nReddit Post Context:\n{meta['reddit_posts'][0]}\n")
            
            f.write("\n" + "=" * 80 + "\nTRANSCRIPT\n" + "=" * 80 + "\n\n")
            
            for entry in session_data["transcript"]:
                role = entry["role"].upper()
                message = entry["message"]
                f.write(f"[{role}]: {message}\n\n")
            
            if session_data["supervisor_feedback"]:
                f.write("\n" + "=" * 80 + "\nSUPERVISOR FEEDBACK SUMMARY\n" + "=" * 80 + "\n\n")
                for i, feedback in enumerate(session_data["supervisor_feedback"], 1):
                    f.write(f"Feedback #{i}:\n{feedback['feedback']}\n\n")
    
    def get_transcript(self) -> List[Dict]:
        """Get the session transcript."""
        return self.session_transcript
    
    def get_supervisor_feedback(self) -> List[Dict]:
        """Get all supervisor feedback."""
        return self.supervisor_feedback
    
    def export_session(self, format: str = "json") -> str:
        """
        Export session in different formats.
        
        Args:
            format: 'json', 'txt', or 'html'
            
        Returns:
            Formatted session data as string
        """
        if format == "json":
            return json.dumps({
                "transcript": self.session_transcript,
                "feedback": self.supervisor_feedback
            }, indent=2, ensure_ascii=False)
        
        elif format == "txt":
            lines = []
            lines.append("THERAPY SESSION TRANSCRIPT")
            lines.append("=" * 80)
            
            for entry in self.session_transcript:
                lines.append(f"\n[{entry['role'].upper()}: {entry['phase']}]")
                lines.append(entry['message'])
            
            return "\n".join(lines)
    
    def get_session_stats(self) -> Dict:
        """Get statistics about the session."""
        therapist_turns = sum(1 for e in self.session_transcript if e['role'] == 'therapist')
        patient_turns = sum(1 for e in self.session_transcript if e['role'] == 'patient')
        
        avg_therapist_length = sum(
            len(e['message'].split()) for e in self.session_transcript if e['role'] == 'therapist'
        ) / max(therapist_turns, 1)
        
        avg_patient_length = sum(
            len(e['message'].split()) for e in self.session_transcript if e['role'] == 'patient'
        ) / max(patient_turns, 1)
        
        return {
            "total_transcript_entries": len(self.session_transcript),
            "therapist_turns": therapist_turns,
            "patient_turns": patient_turns,
            "avg_therapist_response_length": round(avg_therapist_length, 1),
            "avg_patient_response_length": round(avg_patient_length, 1),
            "supervisor_feedbacks": len(self.supervisor_feedback)
        }
