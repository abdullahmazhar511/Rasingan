"""
Main Pipeline Module
Orchestrates the interaction between Patient, Agent 1 (Primary Therapist), 
and Agent 2 (Senior Therapist Supervisor).
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from patient import Patient
from agent_1 import Agent1_PrimaryTherapist
from agent_2 import Agent2_SeniorTherapist
from shared_config import get_session_phase, should_end_session


class TherapyPipeline:
    """Main pipeline orchestrating the three-way therapeutic interaction."""
    
    def __init__(self, reddit_posts: List[str], patient_profile: Optional[Dict] = None,
                 output_dir: str = "sessions", max_turns: int = 10):
        """
        Initialize the therapy pipeline.
        
        Args:
            reddit_posts: List of Reddit posts for patient to roleplay
            patient_profile: Optional patient demographics
            output_dir: Directory to save session transcripts
            max_turns: Maximum number of conversation turns
        """
        self.reddit_posts = reddit_posts
        self.patient_profile = patient_profile
        self.output_dir = Path(output_dir)
        self.max_turns = max_turns
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Session tracking — assigned before agents so the supervisor can share it
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize agents
        self.patient = Patient(reddit_posts, patient_profile)
        self.therapist = Agent1_PrimaryTherapist()
        self.supervisor = Agent2_SeniorTherapist(
            session_id=self.session_id,
            progress_dir=self.output_dir / "supervisor_progress",
        )
        self.session_transcript = []
        self.supervisor_feedback = []
        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "reddit_posts": reddit_posts,
            "patient_profile": patient_profile,
            "max_turns": max_turns,
            "turn_count": 0
        }
    
    def run_session(self, enable_supervisor: bool = True, 
                   enable_suggestions: bool = True,
                   verbose: bool = True) -> Dict:
        """
        Run a complete therapy session.
        
        Args:
            enable_supervisor: Whether to enable supervisor feedback
            enable_suggestions: Whether to print suggestions in real-time
            verbose: Whether to print detailed output
            
        Returns:
            Session results dictionary
        """
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
        
        # Main conversation loop
        last_feedback = None  # Track supervisor feedback for next turn
        for turn in range(1, self.max_turns + 1):
            if verbose:
                print(f"\n--- Turn {turn} ---")
            
            # Patient responds
            patient_response = self.patient.generate_response(therapist_opening if turn == 1 else therapist_response)
            if verbose:
                print(f"[PATIENT]: {patient_response}\n")
            
            self.session_transcript.append({
                "turn": turn,
                "role": "patient",
                "message": patient_response,
                "phase": "conversation"
            })
            
            time.sleep(0.5)
            
            # Therapist responds (with supervisor feedback from previous turn)
            session_phase = self._get_session_phase(turn)
            therapist_response = self.therapist.respond_to_patient(
                patient_response, session_phase,
                supervisor_feedback=last_feedback
            )
            
            if verbose:
                print(f"[THERAPIST]: {therapist_response}")
            
            self.session_transcript.append({
                "turn": turn,
                "role": "therapist",
                "message": therapist_response,
                "phase": session_phase
            })
            
            # Supervisor provides feedback (used by therapist in next turn)
            last_feedback = None
            if enable_supervisor:
                # Update extraction checklist based on conversation so far
                self.supervisor.update_extraction_status(
                    self.session_transcript, turn_number=turn
                )
                
                feedback = self.supervisor.evaluate_response(
                    patient_response, 
                    therapist_response,
                    session_context=self._get_session_context(),
                    turn_number=turn
                )
                
                self.supervisor_feedback.append(feedback)
                last_feedback = feedback.get("feedback")
                
                if enable_suggestions and verbose:
                    print(f"\n[SUPERVISOR FEEDBACK]:\n{feedback['feedback']}\n")
            
            time.sleep(0.5)
            
            # Check for natural session ending
            if self._should_end_session(therapist_response):
                if verbose:
                    print("\n[Session naturally concluding]")
                break
        
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
        
        # Pull checklist coverage from the supervisor (already judged turn-by-turn during the session)
        from checklists import compute_summary as _compute_summary  # local import to avoid cycles
        extraction_status = getattr(self.supervisor, "extraction_status", {})
        extraction_summary = _compute_summary(extraction_status) if extraction_status else {}

        # Prepare full session data
        session_data = {
            "metadata": {
                **self.session_metadata,
                "end_time": timestamp,
                "duration_turns": len(self.session_transcript),
                "supervisor_feedback_count": len(self.supervisor_feedback),
                "checklist": getattr(self.supervisor, "instrument_label", None),
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
