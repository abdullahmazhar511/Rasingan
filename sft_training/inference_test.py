import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import sys
import numpy as np

# Add Rasingan utils to path for hfDataset
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "utils"))
sys.path.insert(0, UTILS_DIR)
from hfDataset import MHCoPilot_Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT Adapter on Test Data")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, required=False, help="Path to the saved LoRA adapter")
    parser.add_argument("--data_dir", type=str, default="/home/umairai/faith_data/dataset")
    parser.add_argument("--context_window", type=int, default=8, help="Number of prior turns to include when building HF dataset context")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=768, help="Max input length; should match train max_length")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (disabled by default for stable eval metrics)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="1", help="CUDA device index")
    return parser.parse_args()

def context_to_chat_messages(context):
    messages = []
    for raw_line in str(context).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Patient:"):
            content = line[len("Patient:"):].strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif line.startswith("Therapist:"):
            content = line[len("Therapist:"):].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
    return messages

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer_source = args.base_model
    if args.adapter_path and os.path.isdir(args.adapter_path):
        adapter_has_tokenizer = any(
            os.path.exists(os.path.join(args.adapter_path, f))
            for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        )
        if adapter_has_tokenizer:
            tokenizer_source = args.adapter_path

    print(f"Loading Tokenizer: {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(f"Loading Base Model: {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"Loading Adapter from: {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    else:
        if args.adapter_path:
            print(f"Warning: Adapter path {args.adapter_path} not found. Running base model inference.")

    model.eval()

    print("Loading Test Dataset...")
    mhcopilot = MHCoPilot_Dataset(args.data_dir, context_window=args.context_window)
    mhcopilot.get_data()
    test_data = mhcopilot.test_dataset

    system_prompt = """You are a compassionate, client-centered therapist.

Respond with empathy, warmth, and non-judgmental understanding. Reflect the
client’s emotions and perspective using reflective listening (e.g., “It sounds like…”, 
“I hear that…”, “You’re feeling…”).

Encourage gentle exploration through open-ended questions and support the
client’s autonomy.

Guidelines:
- Focus on the client’s feelings and lived experience.
- Be concise, calm, and emotionally attuned.
- Do NOT give advice, instructions, or solutions.
- Do NOT judge, confront, diagnose, or moralize.
- Do NOT assume information not expressed by the client.

Task: Write the next therapist response."""

    predictions = []
    references = []

    print(f"Running Inference on {len(test_data)} examples...")
    for i in tqdm(range(0, len(test_data), args.batch_size)):
        end_idx = min(i + args.batch_size, len(test_data))
        batch = test_data.select(range(i, end_idx))
        
        prompts = []
        for context in batch['context']:
            history_messages = context_to_chat_messages(context)

            # Drop leading assistant turns; strict templates (Gemma) require first role=user.
            while history_messages and history_messages[0]["role"] == "assistant":
                history_messages.pop(0)

            # Merge consecutive same-role turns so the sequence strictly alternates.
            merged = []
            for msg in history_messages:
                if merged and merged[-1]["role"] == msg["role"]:
                    merged[-1]["content"] += "\n" + msg["content"]
                else:
                    merged.append(dict(msg))
            history_messages = merged

            if not history_messages or history_messages[-1]["role"] != "user":
                history_messages.append({"role": "user", "content": "Please continue the session."})
            # Fold system prompt into the first user turn for chat-template compatibility
            # across models that do/don't support a "system" role (e.g., Gemma-3).
            history_messages[0]["content"] = f"{system_prompt}\n\n{history_messages[0]['content']}"
            messages = history_messages
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        # Move inputs to model device (handles multi-GPU setup)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "repetition_penalty": 1.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.do_sample:
                gen_kwargs.update({
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                })
            outputs = model.generate(
                **inputs, 
                **gen_kwargs,
            )
        
        generated_texts = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predictions.extend([t.strip() for t in generated_texts])
        references.extend([t.strip() for t in batch['Utterance']])

    print("Calculating Metrics...")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("bleu")

    rouge_results = rouge.compute(predictions=predictions, references=references)
    meteor_results = meteor.compute(predictions=predictions, references=references)
    # BLEU expects references as list of lists
    bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased", device="cuda")

    print("\n--- FINAL TEST METRICS ---")
    print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
    print(f"METEOR:  {meteor_results['meteor']:.4f}")
    print(f"BLEU:    {bleu_results['bleu']:.4f}")
    print(f"BERTScore Precision (Avg): {np.mean(bert_results['precision']):.4f}")
    print(f"BERTScore Recall    (Avg): {np.mean(bert_results['recall']):.4f}")
    print(f"BERTScore F1        (Avg): {np.mean(bert_results['f1']):.4f}")

    # Optional: Save results to CSV
    if args.adapter_path:
        output_csv = args.adapter_path.replace("/", "_") + "_test_predictions.csv"
    else:         
        output_csv = "test_predictions.csv"
    pd.DataFrame({"reference": references, "prediction": predictions}).to_csv(output_csv, index=False)
    print(f"Success! Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
