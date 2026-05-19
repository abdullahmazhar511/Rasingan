import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


RASINGAN_ROOT = Path(__file__).resolve().parents[4]
UTILS_DIR = RASINGAN_ROOT / "utils"
sys.path.insert(0, str(UTILS_DIR))

from hfDataset import MHCoPilot_Dataset


SYSTEM_PROMPT = """You are a compassionate, client-centered therapist.

Respond with empathy, warmth, and non-judgmental understanding. Reflect the
client's emotions and perspective using reflective listening (e.g., \"It sounds like...\", 
\"I hear that...\", \"You're feeling...\").

Encourage gentle exploration through open-ended questions and support the
client's autonomy.

Guidelines:
- Focus on the client's feelings and lived experience.
- Be concise, calm, and emotionally attuned.
- Do NOT give advice, instructions, or solutions.
- Do NOT judge, confront, diagnose, or moralize.
- Do NOT assume information not expressed by the client.

Task: Write the next therapist response."""

CARE_COLUMNS = {
    "Non-Judgmental Language": 0,
    "Warmth and Encouragement": 0,
    "Respect for Autonomy": 0,
    "Active Listening": 0,
    "Reflecting Feelings": 0,
    "Situational Appropriateness": 0,
}


def context_to_chat_messages(context):
    messages = []
    for raw_line in str(context).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Patient:"):
            content = line[len("Patient:") :].strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif line.startswith("Therapist:"):
            content = line[len("Therapist:") :].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
    return messages


def make_prompt_messages(example):
    history_messages = context_to_chat_messages(example.get("context", ""))

    # Match train.py formatting logic for strict chat templates.
    while history_messages and history_messages[0]["role"] == "assistant":
        history_messages.pop(0)

    merged = []
    for msg in history_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))
    history_messages = merged

    if not history_messages or history_messages[-1]["role"] != "user":
        history_messages.append({"role": "user", "content": "Please continue the session."})

    # Match train.py: fold system prompt into the first user turn.
    history_messages[0]["content"] = f"{SYSTEM_PROMPT}\n\n{history_messages[0]['content']}"

    return history_messages


def map_dataset(split_name):
    def process_fn(example, idx):
        ground_truth = {col: example.get(col, 0) for col in CARE_COLUMNS.keys()}
        data_source = example.get("data_source", "faith_dataset")
        return {
            "data_source": data_source,
            "prompt": make_prompt_messages(example),
            "ability": "dialogue",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "id": example.get("ID", idx),
            },
        }

    return process_fn


def to_processed_dataset(hf_dataset, split_name):
    return hf_dataset.map(function=map_dataset(split_name), with_indices=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_dir",
        default=str(RASINGAN_ROOT / "respair_mhcopilot_format"),
        help="Directory containing train.csv, val.csv, test.csv",
    )
    parser.add_argument("--context_window", type=int, default=6)
    parser.add_argument(
        "--train_csv",
        default=str(RASINGAN_ROOT / "respair_mhcopilot_format" / "train_rl_single.csv"),
        help="Path to training CSV file (defaults to curated train_rl_single.csv)",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default=str(RASINGAN_ROOT / "verl" / "examples" / "faith" / "data"),
        help="Local directory to save train/val/test parquet files",
    )
    args = parser.parse_args()

    mhcopilot = MHCoPilot_Dataset(args.csv_dir, context_window=args.context_window)
    mhcopilot.get_data()

    train_csv_path = Path(args.train_csv)
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")

    # Override only the train split with curated RL training conversations.
    train_df = pd.read_csv(train_csv_path)
    train_df["Utterance"] = train_df["Utterance"].fillna("")
    train_df.fillna(CARE_COLUMNS, inplace=True)
    mhcopilot.train_dataset = mhcopilot.preprocess(train_df)

    train_dataset = to_processed_dataset(mhcopilot.train_dataset, "train")
    val_dataset = to_processed_dataset(mhcopilot.val_dataset, "val")
    test_dataset = to_processed_dataset(mhcopilot.test_dataset, "test")

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.local_save_dir, "val.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))

    print(f"Preprocessing complete! Datasets saved to: {args.local_save_dir}")
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
