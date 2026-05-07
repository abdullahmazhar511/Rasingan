"""
CARE_v2 single-sample prediction CLI.

Run:
    python -m care_v2.predict \
        --context "Patient: I've been feeling really anxious lately." \
        --therapist "It sounds like this anxiety has been weighing heavily on you."
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from care_v2.configs.config import CAREv2Config
from care_v2.inference.predictor import CAREv2Predictor
from care_v2.models.care_classifier import CAREv2Classifier
from care_v2.retrieval.prototype_retriever import PrototypeRetriever
from care_v2.utils.io_utils import setup_logging

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CARE_v2 Single-Sample Predictor")
    parser.add_argument("--context", type=str, required=True, help="Conversation context")
    parser.add_argument("--therapist", type=str, required=True, help="Therapist utterance to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint dir (overrides config)")
    args = parser.parse_args()

    config = CAREv2Config()
    setup_logging(config.output_dir)

    import os
    device_id = os.environ.get("DEVICE_ID") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    primary_gpu = device_id.split(",")[0].strip()
    device = f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu"
    
    log.info(f"Using device: {device}")

    # Load artefacts
    final_dir = Path(args.checkpoint or Path(config.output_dir) / "final")
    log.info(f"Loading model from {final_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(str(final_dir / "tokenizer"), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    retriever = PrototypeRetriever(config, device=device)
    model = CAREv2Classifier(config, device=device)
    model.load_heads(str(final_dir))
    model.to(device)

    predictor = CAREv2Predictor(model, retriever, tokenizer, config)

    # Predict
    results = predictor.predict_single(
        context=args.context,
        therapist_utterance=args.therapist,
    )

    print("\n" + "=" * 50)
    print("CARE_v2 Predictions")
    print("=" * 50)
    print(f"Therapist: \"{args.therapist}\"\n")
    for dim, score in results.items():
        sign = "+" if score > 0 else ""
        print(f"  {dim:<35}: {sign}{score}")
    print("=" * 50)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
