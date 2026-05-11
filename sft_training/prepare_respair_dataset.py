"""
Prepare RESPAIR dataset for SFT training.
Converts individual CSV files into formatted training data.
"""
import os
import pandas as pd
import glob
from pathlib import Path
from collections import defaultdict
import argparse

SYSTEM_PROMPT = """You are a compassionate, client-centered therapist.

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


def load_respair_splits(data_dir):
    """Load all RESPAIR CSV files from train/val/test splits."""
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found")
            continue
            
        csv_files = sorted(glob.glob(os.path.join(split_dir, "*.csv")))
        print(f"Found {len(csv_files)} files in {split} split")
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        if all_data:
            datasets[split] = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(datasets[split])} rows for {split} split")
    
    return datasets


def build_conversation_context(df):
    """Build conversation context for each therapist utterance."""
    conversations = []
    current_id = None
    current_context = []
    
    for idx, row in df.iterrows():
        utterance_id = row.get('ID', '')
        # Extract conversation ID (remove the suffix like _1, _2, etc.)
        conv_id = '_'.join(utterance_id.split('_')[:-1]) if '_' in utterance_id else utterance_id
        
        if conv_id != current_id:
            # New conversation
            current_context = []
            current_id = conv_id
        
        if row['Type'] == 'T':
            # Therapist utterance - this is what we want to predict
            context_str = "\n".join(current_context) if current_context else "[Beginning of conversation]"
            
            conversations.append({
                'id': utterance_id,
                'context': context_str,
                'therapist_response': row['Utterance'],
                'quality_score': row.get('NJ', 1.0) if pd.notna(row.get('NJ')) else 1.0
            })
        else:
            # Patient utterance - add to context
            if pd.notna(row.get('Utterance')):
                current_context.append(f"Client: {row['Utterance']}")
    
    return conversations


def format_for_sft(conversations):
    """Format conversations for SFT training with messages format."""
    formatted = []
    
    for conv in conversations:
        formatted.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Context: {conv['context']}\n\nTherapist:"},
                {'role': 'assistant', 'content': conv['therapist_response']}
            ],
            'quality_score': conv['quality_score']
        })
    
    return formatted


def create_curriculum_splits(all_conversations, train_ratio=0.8, val_ratio=0.1):
    """
    Create curriculum learning splits based on quality scores.
    Phase 1: High quality examples
    Phase 2: All examples
    """
    # Sort by quality score
    sorted_convs = sorted(all_conversations, 
                         key=lambda x: x['quality_score'], 
                         reverse=True)
    
    total = len(sorted_convs)
    phase1_size = int(total * 0.5)  # Top 50% quality for phase 1
    
    phase1_data = sorted_convs[:phase1_size]
    phase2_data = sorted_convs
    
    # Split each phase into train/val
    phase1_train_size = int(len(phase1_data) * train_ratio)
    phase1_train = phase1_data[:phase1_train_size]
    phase1_val = phase1_data[phase1_train_size:]
    
    phase2_train_size = int(len(phase2_data) * train_ratio)
    phase2_train = phase2_data[:phase2_train_size]
    phase2_val = phase2_data[phase2_train_size:]
    
    return {
        'phase1_train': phase1_train,
        'phase1_val': phase1_val,
        'phase2_train': phase2_train,
        'phase2_val': phase2_val
    }


def save_training_data(curriculum_data, output_dir):
    """Save curriculum data as CSV files for training."""
    os.makedirs(output_dir, exist_ok=True)
    
    for phase_name, data in curriculum_data.items():
        if not data:
            continue
        
        # Convert to DataFrames
        rows = []
        for item in data:
            rows.append({
                'context': item['messages'][1]['content'],
                'Utterance': item['messages'][2]['content'],
                'quality_score': item['quality_score'],
                'dataset_source': 'RESPAIR'
            })
        
        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, f"{phase_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare RESPAIR dataset for SFT training")
    parser.add_argument("--data_dir", type=str, 
                       default="/raid/home/pushpendra/asbah/EMNLP_RESPAIR/care_v2/faith_data/RESPAIR",
                       help="Path to RESPAIR dataset")
    parser.add_argument("--output_dir", type=str,
                       default="/raid/home/pushpendra/asbah/EMNLP_RESPAIR/sft_training/respair_prepared",
                       help="Output directory for prepared data")
    args = parser.parse_args()
    
    print("=" * 60)
    print("RESPAIR Dataset Preparation for SFT Training")
    print("=" * 60)
    
    # Load all splits
    print(f"\nLoading RESPAIR dataset from {args.data_dir}...")
    datasets = load_respair_splits(args.data_dir)
    
    # Process each split and combine
    all_conversations = []
    for split, df in datasets.items():
        print(f"\nProcessing {split} split...")
        conversations = build_conversation_context(df)
        formatted = format_for_sft(conversations)
        all_conversations.extend(formatted)
        print(f"Generated {len(formatted)} training examples from {split} split")
    
    print(f"\nTotal training examples: {len(all_conversations)}")
    
    # Create curriculum splits
    print("\nCreating curriculum learning splits...")
    curriculum_data = create_curriculum_splits(
        [{'messages': x['messages'], 'quality_score': x['quality_score']} 
         for x in all_conversations]
    )
    
    # Save training data
    print(f"\nSaving prepared data to {args.output_dir}...")
    save_training_data(curriculum_data, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Update your training script to use: {args.output_dir}")
    print(f"2. Or manually copy CSV files from {args.output_dir} to your training directory")


if __name__ == "__main__":
    main()
