import json
import random
import time
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import openai
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

# ========================================================
# CONFIGURATION
# ========================================================

MODEL_ID = "gpt-5.4-mini"  # User specified model
TEMPERATURE = 0.3
MAX_TOKENS = 180
MAX_RETRIES = 5

# Debug / Monitoring
DEBUG_SAVE_RAW_RESPONSE = False  # Set to True to save raw API responses
MIN_EXPLANATION_WORDS = 30       # Reject if shorter than this

# Sleep Ranges (seconds)
FULL_SLEEP_RANGE = (5, 10)
DEBUG_SLEEP_RANGE = (1, 3)

# Paths
DEFAULT_INPUT_DIR = "faith_data/04_clustered_prototypes/prototypes"
DEFAULT_OUTPUT_DIR = "faith_data/05_generated_explanations"
TOKEN_USAGE_FILE = "token_usage.json"
FAILED_FILES_LOG = "failed_files.txt"

# ========================================================
# PROMPTS
# ========================================================

SYSTEM_PROMPT = """
You are an expert evaluator of therapeutic communication in mental health counseling.

Your task is to analyze therapist responses according to therapeutic alignment dimensions.

You must reason carefully about:
- emotional attunement
- interpersonal impact
- therapeutic effectiveness
- reflective depth
- conversational alignment

Your explanations should be clinically grounded, behavior-focused, and precise.

Do not provide generic praise or criticism.
Focus specifically on the therapist's communication behavior and its likely therapeutic impact.
"""

POSITIVE_PROMPT_TEMPLATE = """
Therapeutic Dimension:
{dimension}

Patient Utterance:
{patient}

Therapist Response:
{therapist}

Task:
Explain why the therapist response demonstrates strong alignment with the therapeutic dimension.

Focus on:
- how the therapist supports the patient emotionally
- therapeutic communication strengths
- reflective or supportive behaviors
- why the response may help the patient feel understood, supported, or emotionally safe

Requirements:
- 80–120 words
- clinically grounded
- behavior-focused
- concise but detailed
- avoid generic statements
- do not use bullet points
"""

NEGATIVE_PROMPT_TEMPLATE = """
Therapeutic Dimension:
{dimension}

Patient Utterance:
{patient}

Therapist Response:
{therapist}

Task:
Explain why the therapist response demonstrates weak, ineffective, misaligned, or therapeutically problematic behavior for the specified therapeutic dimension.

Focus on:
- missed emotional attunement
- lack of reflective depth
- conversational misalignment
- potentially unhelpful therapist behavior
- why the response may fail to support the patient effectively

Do NOT be harsh or judgmental.
Provide a clinically grounded explanation of the therapeutic limitation or weakness.

Requirements:
- 80–120 words
- clinically grounded
- behavior-focused
- concise but detailed
- avoid generic criticism
- do not use bullet points
"""

# ========================================================
# TRACKING & UTILITIES
# ========================================================

class PipelineStats:
    def __init__(self):
        self.processed = 0
        self.successful = 0
        self.skipped = 0
        self.failed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def update_usage(self, prompt, completion, total):
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total
        self.save_usage_to_disk()

    def save_usage_to_disk(self):
        usage_data = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "num_requests": self.successful
        }
        try:
            # Atomic save for usage stats too
            tmp_file = Path(TOKEN_USAGE_FILE).with_suffix(".json.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(usage_data, f, indent=2)
            tmp_file.replace(TOKEN_USAGE_FILE)
        except Exception as e:
            logging.error(f"Failed to save token usage: {e}")

    def print_summary(self):
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        print(f"Processed:  {self.processed}")
        print(f"Successful: {self.successful}")
        print(f"Skipped:    {self.skipped}")
        print(f"Failed:     {self.failed}")
        print("-" * 20)
        print(f"Prompt Tokens:     {self.total_prompt_tokens}")
        print(f"Completion Tokens: {self.total_completion_tokens}")
        print(f"Total Tokens:      {self.total_tokens}")
        print("="*50 + "\n")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("explanation_generation.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def log_failure(file_path: Path, error_msg: str):
    """Logs failed file paths to a dedicated tracking file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FAILED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {file_path} | {error_msg}\n")

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_atomic(data: Dict[str, Any], path: Path):
    """Saves JSON to a temporary file then renames to avoid corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)

# ========================================================
# CORE API LOGIC
# ========================================================

def validate_model(client: OpenAI):
    """Startup check to ensure model is available."""
    print(f"Starting pipeline with model: {MODEL_ID}")
    try:
        # A simple model list call or a dummy request to check validity
        # We'll just print it clearly as requested, and handle errors during first call
        logging.info(f"Model set to: {MODEL_ID}")
    except Exception as e:
        logging.error(f"Model validation failed: {e}")
        exit(1)

def is_polarity_consistent(explanation: str, polarity: str) -> bool:
    """Checks if the explanation content matches the expected polarity."""
    if polarity == "negative":
        forbidden_words = ["supports", "validates", "emotionally safe", "helpful"]
        explanation_lower = explanation.lower()
        for word in forbidden_words:
            if word in explanation_lower:
                return False
    return True

def generate_explanation(client: OpenAI, dimension: str, polarity: str, patient: str, therapist: str, is_debug: bool) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
    """Generates an explanation with quality checks and detailed retry logic."""
    
    # Polarity-aware prompt selection
    if polarity == "positive":
        template = POSITIVE_PROMPT_TEMPLATE
    else:
        template = NEGATIVE_PROMPT_TEMPLATE

    user_content = template.format(
        dimension=dimension.replace("_", " "),
        patient=patient,
        therapist=therapist
    )
    
    base_wait = 10
    sleep_range = DEBUG_SLEEP_RANGE if is_debug else FULL_SLEEP_RANGE
    
    for attempt in range(MAX_RETRIES):
        try:
            # Randomized delay
            time.sleep(random.uniform(*sleep_range))
            
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS
            )
            
            content = response.choices[0].message.content.strip()
            usage = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
            raw_res = response.model_dump_json() if DEBUG_SAVE_RAW_RESPONSE else None

            # Quality Check: Length
            word_count = len(content.split())
            if not content or word_count < MIN_EXPLANATION_WORDS:
                logging.warning(f"Rejecting explanation: too short ({word_count} words). Retrying...")
                continue

            # Polarity Check: Ensure negative models don't sound positive
            if not is_polarity_consistent(content, polarity):
                logging.warning(f"Rejecting explanation: polarity inconsistency for {polarity} model. Retrying...")
                continue
                
            return content, usage, raw_res
            
        except openai.NotFoundError:
            msg = f"Model '{MODEL_ID}' not found. Stopping execution."
            logging.error(msg)
            raise SystemExit(msg)
        except openai.RateLimitError:
            wait = base_wait * (2 ** attempt) + random.uniform(1, 5)
            logging.warning(f"Rate limit hit. Attempt {attempt+1}/{MAX_RETRIES}. Waiting {wait:.2f}s...")
            time.sleep(wait)
        except Exception as e:
            logging.error(f"API Error (Attempt {attempt+1}): {e}")
            if attempt == MAX_RETRIES - 1:
                return None, None, str(e)
            time.sleep(random.uniform(2, 5))
            
    return None, None, "Max retries reached or quality check failed."

def process_single_file(client: OpenAI, input_path: Path, output_path: Path, stats: PipelineStats, is_debug: bool):
    """Processes one prototype with detailed logging and usage tracking."""
    stats.processed += 1
    
    # 1. Resumable Skip
    if output_path.exists():
        stats.skipped += 1
        return

    try:
        # Extract metadata for better logging
        # Path: k_12/Dimension_Name/positive/cluster_0.json
        parts = input_path.relative_to(input_path.parents[3]).parts
        k_folder = parts[0]
        dimension = parts[1]
        cluster_id = input_path.stem
        
        log_prefix = f"[{k_folder}][{dimension}][{cluster_id}]"

        data = load_json(input_path)
        
        explanation, usage, raw_res_or_error = generate_explanation(
            client, 
            data.get("dimension", dimension),
            data.get("polarity", "Unknown"),
            data.get("patient", ""),
            data.get("therapist", ""),
            is_debug
        )
        
        if explanation:
            data["explanation"] = explanation
            data["token_usage"] = usage
            
            # Save enriched JSON atomically
            save_json_atomic(data, output_path)
            
            # Save raw response if enabled
            if DEBUG_SAVE_RAW_RESPONSE and raw_res_or_error:
                raw_path = output_path.with_suffix(".raw.json")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(raw_res_or_error)
            
            # Update Statistics
            stats.successful += 1
            stats.update_usage(usage["prompt"], usage["completion"], usage["total"])
            
            logging.info(f"{log_prefix} SUCCESS (Words: {len(explanation.split())})")
        else:
            stats.failed += 1
            log_failure(input_path, raw_res_or_error or "Unknown failure")
            logging.error(f"{log_prefix} FAILED: {raw_res_or_error}")
            
    except Exception as e:
        stats.failed += 1
        log_failure(input_path, f"Code Error: {str(e)}")
        logging.error(f"Critical error on {input_path}: {e}")

# ========================================================
# MAIN
# ========================================================

def main():
    parser = argparse.ArgumentParser(description="Fault-tolerant, production-ready explanation generator.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug_count", type=int, default=5, help="Samples for debug")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    setup_logging()
    
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY missing.")
        return

    client = OpenAI()
    validate_model(client)
    
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    stats = PipelineStats()
    
    if not input_root.exists():
        logging.error(f"Input dir {input_root} not found.")
        return

    # Collect files
    all_files = sorted(list(input_root.glob("k_*/*/*/*.json")))
    if args.debug:
        logging.info(f"DEBUG: Processing {args.debug_count} random files.")
        all_files = random.sample(all_files, min(args.debug_count, len(all_files)))
    
    logging.info(f"Starting run: {len(all_files)} files to check.")

    # Main Loop
    pbar = tqdm(all_files, desc="Overall Progress")
    for file_path in pbar:
        try:
            rel_path = file_path.relative_to(input_root)
            output_path = output_root / rel_path
            
            k_folder = rel_path.parts[0]
            pbar.set_description(f"Running {k_folder}")
            
            process_single_file(client, file_path, output_path, stats, args.debug)
            
            # Periodic usage print
            if stats.successful % 20 == 0 and stats.successful > 0:
                print(f"\nRunning Total: {stats.total_tokens} tokens used across {stats.successful} requests.")
                
        except Exception as e:
            logging.error(f"Loop error: {e}")

    # Final Summary
    stats.print_summary()

if __name__ == "__main__":
    main()
