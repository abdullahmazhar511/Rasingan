import argparse
import json
import os
import sys

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RASINGAN_ROOT = os.path.dirname(SCRIPT_DIR)
UTILS_DIR = os.path.join(RASINGAN_ROOT, "utils")
sys.path.insert(0, UTILS_DIR)

from hfDataset import MHCoPilot_Dataset


EVAL_ROOT = os.environ.get("EVAL_ROOT", os.path.join(RASINGAN_ROOT, "evaluation_pipeline"))
DATASET_PATH = os.environ.get("DATASET_PATH", os.path.join(RASINGAN_ROOT, "respair_mhcopilot_format"))
MODELS_CONFIG = os.path.join(EVAL_ROOT, "models_config.json")
CARE_CONTEXT_WINDOW_DEFAULT = 6


system_prompt = """You are a compassionate, client-centered therapist.

Respond with empathy, warmth, and non-judgmental understanding. Reflect the
client's emotions and perspective using reflective listening (e.g., "It sounds like…",
"I hear that…", "You're feeling…").

Encourage gentle exploration through open-ended questions and support the
client's autonomy.

Guidelines:
- Focus on the client's feelings and lived experience.
- Be concise, calm, and emotionally attuned.
- Do NOT give advice, instructions, or solutions.
- Do NOT judge, confront, diagnose, or moralize.
- Do NOT assume information not expressed by the client.

Task: Write the next therapist response."""


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


def build_prompt_from_context(context, tokenizer):
    history_messages = context_to_chat_messages(context)

    # Strict templates can require the first role to be user.
    while history_messages and history_messages[0]["role"] == "assistant":
        history_messages.pop(0)

    # Merge consecutive same-role turns so roles strictly alternate.
    merged = []
    for msg in history_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))
    history_messages = merged

    if not history_messages or history_messages[-1]["role"] != "user":
        history_messages.append({"role": "user", "content": "Please continue the session."})

    # Fold system prompt into first user turn for template compatibility across models.
    history_messages[0]["content"] = f"{system_prompt}\n\n{history_messages[0]['content']}"

    return tokenizer.apply_chat_template(history_messages, tokenize=False, add_generation_prompt=True)


def load_test_dataset(data_dir, context_window):
    mhcopilot = MHCoPilot_Dataset(data_dir, context_window=context_window)
    mhcopilot.get_data()
    return mhcopilot.test_dataset


def generate_for_model(
    model_cfg,
    eval_root,
    data_dir,
    context_window,
    batch_size,
    max_input_length,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    device,
):
    model_name = model_cfg["name"]
    base_model = model_cfg["base_model"]
    adapter_path = model_cfg.get("adapter_path")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    tokenizer_source = base_model
    if adapter_path and os.path.isdir(adapter_path):
        adapter_has_tokenizer = any(
            os.path.exists(os.path.join(adapter_path, f))
            for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        )
        if adapter_has_tokenizer:
            tokenizer_source = adapter_path

    print(f"\n[{model_name}] Loading Tokenizer: {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(f"[{model_name}] Loading Base Model: {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"[{model_name}] Loading SFT Adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    elif adapter_path:
        print(f"[{model_name}] Warning: Adapter path not found, running base model inference.")

    model.eval()

    print(f"[{model_name}] Loading test dataset from {data_dir} (context_window={context_window})...")
    test_data = load_test_dataset(data_dir, context_window=context_window)

    preds = []
    output_rows = []
    keep_cols = [c for c in test_data.column_names if c != "__index_level_0__"]

    print(f"[{model_name}] Generating responses for {len(test_data)} examples...")
    for i in tqdm(range(0, len(test_data), batch_size)):
        end_idx = min(i + batch_size, len(test_data))
        batch = test_data.select(range(i, end_idx))

        prompts = [build_prompt_from_context(context, tokenizer) for context in batch["context"]]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p})

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated_texts = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        generated_texts = [t.strip() for t in generated_texts]
        preds.extend(generated_texts)

        for j in range(len(generated_texts)):
            row = {col: batch[col][j] for col in keep_cols}
            row["model_prediction"] = generated_texts[j]
            output_rows.append(row)

    output_dir = os.path.join(eval_root, model_name, "responses")
    os.makedirs(output_dir, exist_ok=True)

    out_df = pd.DataFrame(output_rows)
    if "ID" in out_df.columns:
        out_df["conversation_id"] = out_df["ID"].astype(str).str.replace(r"_\d+$", "", regex=True)
        for conv_id, conv_df in out_df.groupby("conversation_id", sort=False):
            conv_df = conv_df.drop(columns=["conversation_id"])
            conv_df.to_csv(os.path.join(output_dir, f"{conv_id}.csv"), index=False)
    else:
        out_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    del model, tokenizer
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate model responses using inference_test-compatible logic")
    parser.add_argument("--model-name", type=str, help="Model name (folder in eval root)")
    parser.add_argument("--base-model", "--base_model", dest="base_model", type=str, help="Base model path")
    parser.add_argument("--adapter-path", "--adapter_path", dest="adapter_path", type=str, default=None)
    parser.add_argument("--dataset-path", "--data-dir", "--data_dir", dest="data_dir", type=str, default=DATASET_PATH)
    parser.add_argument("--context-window", "--context_window", dest="context_window", type=int, default=CARE_CONTEXT_WINDOW_DEFAULT)
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=128)
    parser.add_argument("--max-input-length", "--max_input_length", dest="max_input_length", type=int, default=768)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_name and args.base_model:
        model_cfg = {
            "name": args.model_name,
            "base_model": args.base_model,
            "adapter_path": None if args.adapter_path in [None, "null", "None"] else args.adapter_path,
        }

        try:
            generate_for_model(
                model_cfg=model_cfg,
                eval_root=args.eval_root,
                data_dir=args.data_dir,
                context_window=args.context_window,
                batch_size=args.batch_size,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
            print(f"[OK] Successfully generated for {args.model_name}")
        except Exception as e:
            print(f"[ERROR] Generation failed for {args.model_name}: {e}")
            raise
        return

    if not os.path.exists(MODELS_CONFIG):
        raise FileNotFoundError(
            f"No models_config.json at {MODELS_CONFIG}. Provide --model-name and --base-model."
        )

    with open(MODELS_CONFIG, "r") as f:
        models = json.load(f)

    for model_cfg in models:
        model_name = model_cfg["name"]
        expected_dir = os.path.join(args.eval_root, model_name, "responses")
        if os.path.exists(expected_dir) and len(os.listdir(expected_dir)) > 0:
            print(f"Skipping {model_name} generation (responses already exist)")
            continue

        try:
            generate_for_model(
                model_cfg=model_cfg,
                eval_root=args.eval_root,
                data_dir=args.data_dir,
                context_window=args.context_window,
                batch_size=args.batch_size,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
        except Exception as e:
            print(f"Error generating for {model_name}: {e}")


if __name__ == "__main__":
    main()
