import os
import glob
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from peft import PeftModel
import argparse

import sys
sys.path.append("/home/umairai/faithfulness_emnlp/sft_training")
# import prompts

EVAL_ROOT = os.environ.get("EVAL_ROOT", "/home/umairai/faith_data/evaluation_pipeline")
DATASET_PATH = os.environ.get("DATASET_PATH", "/home/umairai/faith_data/dataset/llm_test")
MODELS_CONFIG = os.path.join(EVAL_ROOT, "models_config.json")

context_length_eval = 1024
eval_batch = 12

def load_eval_split(path, context_window=4):
    role_dict = {"T": "Therapist", "P": "Patient"}
    csv_files = glob.glob(os.path.join(path, "*.csv")) if os.path.isdir(path) else [path]
    
    all_rows = []
    for file in tqdm(csv_files, desc=f"Loading test data"):
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)
            
            if 'Utterance' not in df.columns or 'Type' not in df.columns: continue
                
            df['Utterance'] = df['Utterance'].fillna('')
            contexts = [""] * len(df)
            
            for i in range(1, len(df)):
                start_idx = max(0, i - context_window)
                window_slice = df.iloc[start_idx:i]
                contexts[i] = "\n".join([f"{role_dict.get(r['Type'], str(r['Type']))}: {r['Utterance']}" for _, r in window_slice.iterrows()])
                
            df['context'] = contexts
            df_target = df[df['Type'] == 'T'].copy()
            if len(df_target) > 1: df_target = df_target.iloc[1:]
            all_rows.append(df_target)
        except Exception:
            continue
            
    if not all_rows: return HFDataset.from_pandas(pd.DataFrame(columns=['context', 'Utterance']))
    
    final_df = pd.concat(all_rows, ignore_index=True)
    for col in final_df.select_dtypes(include=['object']).columns:
        final_df[col] = final_df[col].astype(str)
            
    return HFDataset.from_pandas(final_df)

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

def prepare_eval(row, tokenizer):
    context = row['context']
    prompt_str = f"Context: {context}\nTherapist:"
    
    # Use tokenizer's chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_str}
    ]
    
    row['test'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    row['answer'] = row['Utterance']
    return row

def generate_for_model(model_cfg, dataset, eval_root):
    model_name = model_cfg['name']
    base_model = model_cfg['base_model']
    
    print(f"\n[{model_name}] Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    )

    if model_cfg.get('adapter_path'):
        print(f"[{model_name}] Loading SFT Adapter...")
        model = PeftModel.from_pretrained(model, model_cfg['adapter_path'])
        model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    def _map(r): return prepare_eval(r, tokenizer)
    mapped_dataset = dataset.map(_map, num_proc=4)
    
    eval_dataset = mapped_dataset.remove_columns([c for c in mapped_dataset.column_names if c not in ['test', 'answer']])
    dataloader = DataLoader(eval_dataset, batch_size=eval_batch)
    preds = []

    print(f"[{model_name}] Generating responses...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = tokenizer(batch['test'], return_tensors='pt', max_length=context_length_eval, truncation=True, padding=True).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, use_cache=False)
            decoded = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            preds.extend([t.strip() for t in decoded])

    # Save to eval_root / model_name / responses
    output_dir = os.path.join(eval_root, model_name, "responses")
    os.makedirs(output_dir, exist_ok=True)
    
    df_eval = mapped_dataset.to_pandas()
    df_eval['model_prediction'] = preds
    
    for sf in df_eval['source_file'].unique():
        sf_df = df_eval[df_eval['source_file'] == sf].copy()
        sf_df = sf_df.drop(columns=['test', 'answer'], errors='ignore')
        sf_df.to_csv(os.path.join(output_dir, sf), index=False)

    del model, tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses for a model")
    parser.add_argument("--model-name", type=str, help="Model name (folder in EVAL_ROOT)")
    parser.add_argument("--base-model", type=str, help="Base model path")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional adapter path")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH, help="Dataset path")
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT, help="Evaluation root directory")
    
    args = parser.parse_args()
    
    # Override with CLI args if provided
    if args.eval_root:
        EVAL_ROOT = args.eval_root
    if args.dataset_path:
        DATASET_PATH = args.dataset_path
    
    # If specific model provided via CLI, generate for that model only
    if args.model_name and args.base_model:
        model_cfg = {
            'name': args.model_name,
            'base_model': args.base_model,
            'adapter_path': None if args.adapter_path in [None, 'null', 'None'] else args.adapter_path
        }
        
        dataset = load_eval_split(DATASET_PATH)
        print(f"Generating for {args.model_name}...")
        try:
            generate_for_model(model_cfg, dataset, args.eval_root)
            print(f"✓ Successfully generated for {args.model_name}")
        except Exception as e:
            print(f"✗ Error generating for {args.model_name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Generate for all models in config
        with open(MODELS_CONFIG, "r") as f:
            models = json.load(f)
            
        dataset = load_eval_split(DATASET_PATH)
        
        for m in models:
            # Check if already generated
            expected_dir = os.path.join(args.eval_root, m['name'], "responses")
            if os.path.exists(expected_dir) and len(os.listdir(expected_dir)) > 0:
                print(f"Skipping {m['name']} Generation (already exists)")
                continue
            try:
                generate_for_model(m, dataset, args.eval_root)
            except Exception as e:
                print(f"Error generating for {m['name']}: {e}")
