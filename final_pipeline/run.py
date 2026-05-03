#!/usr/bin/env python
"""
Main runner script for the therapeutic agent pipeline.
Example usage of the three-agent system: Patient, Primary Therapist, Senior Therapist Supervisor.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from pipeline import TherapyPipeline
from reddit_posts import get_reddit_post, get_patient_profile, list_scenarios


def run_therapy_session(
    scenario: str,
    max_turns: int = 8,
    enable_supervisor: bool = True,
    enable_suggestions: bool = True,
    verbose: bool = True,
    output_dir: str = "sessions",
    custom_post: Optional[str] = None,
    custom_profile: Optional[dict] = None
) -> None:
    """
    Run a therapy session with the specified configuration.
    
    Args:
        scenario: Predefined scenario from reddit_posts
        max_turns: Maximum conversation turns
        enable_supervisor: Enable supervisor feedback
        enable_suggestions: Print suggestions during session
        verbose: Print detailed output
        output_dir: Directory to save sessions
        custom_post: Optional custom Reddit post
        custom_profile: Optional custom patient profile
    """
    
    # Get Reddit post and patient profile
    if custom_post:
        reddit_posts = [custom_post]
    else:
        reddit_posts = get_reddit_post(scenario)
    
    if custom_profile:
        patient_profile = custom_profile
    else:
        patient_profile = get_patient_profile(scenario)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"STARTING THERAPY SESSION: {scenario}")
        print(f"{'='*80}")
        print(f"Patient: {patient_profile['name']}")
        print(f"Age: {patient_profile['age']}, Occupation: {patient_profile['occupation']}")
        print(f"Primary Concern: {patient_profile['primary_concern']}")
        print(f"Max Turns: {max_turns}")
        print(f"Supervisor Feedback: {'Enabled' if enable_supervisor else 'Disabled'}")
        print(f"{'='*80}\n")
    
    # Create and run pipeline
    pipeline = TherapyPipeline(
        reddit_posts=reddit_posts,
        patient_profile=patient_profile,
        output_dir=output_dir,
        max_turns=max_turns
    )
    
    # Run the session
    results = pipeline.run_session(
        enable_supervisor=enable_supervisor,
        enable_suggestions=enable_suggestions,
        verbose=verbose
    )
    
    # Print statistics
    if verbose:
        stats = pipeline.get_session_stats()
        print(f"\n{'='*80}")
        print("SESSION STATISTICS")
        print(f"{'='*80}")
        print(f"Total Exchanges: {stats['total_transcript_entries']}")
        print(f"Therapist Turns: {stats['therapist_turns']}")
        print(f"Patient Turns: {stats['patient_turns']}")
        print(f"Avg Therapist Response: {stats['avg_therapist_response_length']} words")
        print(f"Avg Patient Response: {stats['avg_patient_response_length']} words")
        print(f"Supervisor Feedbacks: {stats['supervisor_feedbacks']}")
        print(f"{'='*80}\n")
        
        print(f"\nResults saved to:")
        print(f"  - Full transcript: {results['transcript_file']}")
        print(f"  - Summary: {results['summary_file']}\n")


def run_interactive_mode() -> None:
    """Run the pipeline in interactive mode."""
    print("\n" + "="*80)
    print("THERAPEUTIC AGENT PIPELINE - INTERACTIVE MODE")
    print("="*80 + "\n")
    
    print("Available scenarios:")
    scenarios = list_scenarios()
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario}")
    
    # Get user input
    try:
        choice = int(input("\nSelect scenario (number): "))
        if 1 <= choice <= len(scenarios):
            scenario = scenarios[choice - 1]
        else:
            print("Invalid choice!")
            return
    except ValueError:
        print("Invalid input!")
        return
    
    # Get parameters
    turns = input("\nMax turns (default 8): ")
    max_turns = int(turns) if turns.isdigit() else 8
    
    supervisor = input("Enable supervisor feedback? (y/n, default y): ").lower() != 'n'
    suggestions = input("Show suggestions in real-time? (y/n, default y): ").lower() != 'n'
    
    # Run session
    run_therapy_session(
        scenario=scenario,
        max_turns=max_turns,
        enable_supervisor=supervisor,
        enable_suggestions=suggestions,
        verbose=True,
        output_dir="sessions"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Therapeutic Agent Pipeline - Three-agent therapy simulation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a predefined scenario
  python run.py --scenario anxiety_workplace
  
  # Run with custom settings
  python run.py --scenario depression_isolation --max-turns 12 --no-supervisor
  
  # Run interactive mode
  python run.py --interactive
  
  # List available scenarios
  python run.py --list-scenarios
        """
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list_scenarios(),
        help="Choose a predefined scenario"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum number of conversation turns (default: 8)"
    )
    
    parser.add_argument(
        "--no-supervisor",
        action="store_true",
        help="Disable supervisor feedback"
    )
    
    parser.add_argument(
        "--no-suggestions",
        action="store_true",
        help="Don't print suggestions during session"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sessions",
        help="Directory to save session transcripts (default: sessions)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    parser.add_argument(
        "--custom-post",
        type=str,
        help="Path to JSON file with custom Reddit post"
    )
    
    parser.add_argument(
        "--custom-profile",
        type=str,
        help="Path to JSON file with custom patient profile"
    )
    
    args = parser.parse_args()
    
    # Handle list scenarios
    if args.list_scenarios:
        print("\nAvailable Scenarios:")
        for scenario in list_scenarios():
            profile = get_patient_profile(scenario)
            print(f"  • {scenario}")
            print(f"    Patient: {profile['name']}")
            print(f"    Concern: {profile['primary_concern']}\n")
        return
    
    # Handle interactive mode
    if args.interactive:
        run_interactive_mode()
        return
    
    # Require scenario for non-interactive mode
    if not args.scenario:
        parser.print_help()
        print("\nError: Please specify a --scenario or use --interactive mode")
        print(f"\nAvailable scenarios: {', '.join(list_scenarios())}")
        sys.exit(1)
    
    # Load custom post if provided
    custom_post = None
    if args.custom_post:
        with open(args.custom_post, 'r') as f:
            custom_post = json.load(f)
    
    # Load custom profile if provided
    custom_profile = None
    if args.custom_profile:
        with open(args.custom_profile, 'r') as f:
            custom_profile = json.load(f)
    
    # Run therapy session
    run_therapy_session(
        scenario=args.scenario,
        max_turns=args.max_turns,
        enable_supervisor=not args.no_supervisor,
        enable_suggestions=not args.no_suggestions,
        verbose=not args.quiet,
        output_dir=args.output_dir,
        custom_post=custom_post,
        custom_profile=custom_profile
    )


if __name__ == "__main__":
    main()
