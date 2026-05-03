#!/usr/bin/env python
"""
Generate VERL training data from final_pipeline therapy sessions.
Converts standalone pipeline output to VERL parquet format for training.

Usage:
    python generate_verl_training_data.py \
        --output_dir data/therapy_conversations \
        --num_samples_per_scenario 5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Import from final_pipeline
try:
    from pipeline import TherapyPipeline
    from patient import PatientSimulator
    from agent_1 import Agent1_PrimaryTherapist
    from agent_2 import Agent2_Supervisor
    import reddit_posts
except ImportError:
    print("Error: Please run this script from the final_pipeline directory")
    print("cd /home/umairai/faithfulness_emnlp/Rasingan/final_pipeline")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standalone_session_to_verl_format(
    session_data: Dict,
    scenario_name: str
) -> Dict:
    """
    Convert a standalone pipeline session to VERL training format.
    
    Args:
        session_data: Output from TherapyPipeline.run_session()
        scenario_name: Name of the scenario (anxiety_workplace, etc.)
        
    Returns:
        VERL-compatible data dictionary
    """
    transcript = session_data.get("transcript", [])
    metadata = session_data.get("metadata", {})
    
    # Extract therapist and patient messages
    messages = [
        {
            "role": "system",
            "content": "You are a compassionate therapist conducting a therapeutic session. "
                      "Use person-centered therapy: empathy, unconditional positive regard, and genuineness."
        }
    ]
    
    for entry in transcript:
        if entry["role"] == "therapist":
            messages.append({
                "role": "assistant",
                "content": entry["message"]
            })
        elif entry["role"] == "patient":
            messages.append({
                "role": "user",
                "content": entry["message"]
            })
    
    # Get patient context from reddit posts
    reddit_posts_list = reddit_posts.REDDIT_POSTS.get(scenario_name, [""])[0]
    
    # Create VERL format
    verl_data = {
        "raw_prompt": messages[:-1],  # Everything except last assistant message
        "response": messages[-1]["content"] if len(messages) > 1 else "",
        "patient_context": reddit_posts_list,
        "patient_profile": metadata.get("patient_profile", {}),
        "interaction_kwargs": {
            "scenario": scenario_name,
            "num_turns": len([e for e in transcript if e["role"] == "therapist"]),
            "session_id": metadata.get("session_id", ""),
        },
        "session_id": metadata.get("session_id", ""),
        "supervisor_feedback": session_data.get("supervisor_feedback", []),
        "duration_turns": metadata.get("duration_turns", 0),
    }
    
    return verl_data


def generate_training_data(
    output_dir: Path,
    num_samples_per_scenario: int = 3,
    train_split: float = 0.8,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> tuple:
    """
    Generate VERL training data by running therapy sessions through final_pipeline.
    
    Args:
        output_dir: Directory to save parquet files
        num_samples_per_scenario: Number of sessions per scenario
        train_split: Train/val split ratio (default 80/20)
        model_name: Model to use for agent_1
        
    Returns:
        Tuple of (train_path, val_path, num_samples_generated)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating VERL training data in {output_dir}")
    logger.info(f"Samples per scenario: {num_samples_per_scenario}")
    
    # Initialize pipeline components
    logger.info("Initializing therapy pipeline components...")
    therapist = Agent1_PrimaryTherapist(model_name=model_name)
    supervisor = Agent2_Supervisor()
    
    train_data = []
    val_data = []
    
    # Get all scenarios from reddit_posts
    scenarios = list(reddit_posts.REDDIT_POSTS.keys())
    logger.info(f"Found {len(scenarios)} scenarios: {scenarios}")
    
    total_generated = 0
    
    for scenario_name in scenarios:
        logger.info(f"\nGenerating sessions for scenario: {scenario_name}")
        
        for sample_idx in range(num_samples_per_scenario):
            try:
                # Create patient simulator for this scenario
                patient = PatientSimulator(
                    scenario=scenario_name,
                    use_llm=True,
                    model_name=model_name,
                )
                
                # Run therapy session through pipeline
                logger.info(f"  Running session {sample_idx + 1}/{num_samples_per_scenario}...")
                pipeline = TherapyPipeline(
                    agent_1=therapist,
                    agent_2=supervisor,
                    patient=patient,
                )
                
                session_result = pipeline.run_session(
                    max_turns=5,
                    include_supervisor=True,
                )
                
                # Convert to VERL format
                verl_entry = standalone_session_to_verl_format(session_result, scenario_name)
                
                # Split into train/val
                if sample_idx % int(1.0 / (1.0 - train_split)) == 0:
                    val_data.append(verl_entry)
                else:
                    train_data.append(verl_entry)
                
                total_generated += 1
                logger.info(f"    ✓ Session generated ({len(train_data)} train, {len(val_data)} val so far)")
                
            except Exception as e:
                logger.warning(f"  ✗ Error generating session: {e}")
                continue
    
    # Save training data
    if train_data:
        train_table = pa.Table.from_pylist(train_data)
        train_path = output_dir / "train.parquet"
        pq.write_table(train_table, str(train_path))
        logger.info(f"✓ Saved {len(train_data)} training examples to {train_path}")
    else:
        logger.error("No training data generated!")
        return None, None, 0
    
    # Save validation data
    if val_data:
        val_table = pa.Table.from_pylist(val_data)
        val_path = output_dir / "val.parquet"
        pq.write_table(val_table, str(val_path))
        logger.info(f"✓ Saved {len(val_data)} validation examples to {val_path}")
    else:
        logger.warning("No validation data generated, using training data for validation")
        val_path = train_path
    
    logger.info(f"\n✅ Generated {total_generated} total therapy sessions")
    logger.info(f"   Train: {len(train_data)} samples")
    logger.info(f"   Val:   {len(val_data)} samples")
    
    return train_path, val_path, total_generated


def validate_dataset(parquet_path: Path) -> bool:
    """Validate that parquet file has correct VERL format."""
    try:
        table = pq.read_table(str(parquet_path))
        required_columns = ["raw_prompt", "response", "patient_context"]
        
        missing = [col for col in required_columns if col not in table.column_names]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        
        logger.info(f"✓ Dataset valid: {len(table)} examples with columns: {table.column_names}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate VERL training data from final_pipeline therapy sessions"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/therapy_conversations"),
        help="Directory to save parquet files"
    )
    
    parser.add_argument(
        "--num_samples_per_scenario",
        type=int,
        default=3,
        help="Number of therapy sessions per scenario"
    )
    
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Training/validation split ratio"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use for agent_1"
    )
    
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate existing dataset"
    )
    
    parser.add_argument(
        "--validate_path",
        type=Path,
        help="Path to parquet file to validate"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        if not args.validate_path:
            logger.error("--validate_path required with --validate_only")
            return
        validate_dataset(args.validate_path)
        return
    
    # Generate training data
    train_path, val_path, num_generated = generate_training_data(
        output_dir=args.output_dir,
        num_samples_per_scenario=args.num_samples_per_scenario,
        train_split=args.train_split,
        model_name=args.model,
    )
    
    if train_path:
        # Validate generated data
        logger.info("\nValidating generated dataset...")
        if validate_dataset(train_path) and validate_dataset(val_path):
            logger.info("✅ All validation passed!")
            logger.info(f"\nReady for training! Use these paths in run_multiturn.sh:")
            logger.info(f"  TRAIN_FILES={train_path}")
            logger.info(f"  VAL_FILES={val_path}")
        else:
            logger.error("❌ Validation failed!")


if __name__ == "__main__":
    main()
