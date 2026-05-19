"""
Convert RESPAIR dataset into MHCoPilot format (single train.csv, val.csv, test.csv).
"""
import os
import pandas as pd
import glob
from pathlib import Path

def merge_respair_to_mhcopilot_format(respair_dir, output_dir):
    """
    Convert RESPAIR individual CSV files to MHCoPilot format.
    RESPAIR: individual files in train/, val/, test/ directories
    MHCoPilot: single train.csv, val.csv, test.csv files
    """
    print(os.path.abspath(respair_dir))
    print(os.path.abspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(respair_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found")
            continue
        
        # Collect all CSV files in this split
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
        
        if not all_data:
            print(f"No data found for {split} split")
            continue
        
        # Merge all files
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure required columns exist
        required_cols = ['Utterance', 'Type', 'ID']
        for col in required_cols:
            if col not in merged_df.columns:
                if col == 'Type':
                    # Default to 'T' if missing
                    merged_df[col] = 'T'
                else:
                    merged_df[col] = ''
        
        # Rename quality columns to match MHCoPilot naming
        quality_cols_mapping = {
            'NJ': 'Non-Judgmental Language',
            'WE': 'Warmth and Encouragement',
            'RA': 'Respect for Autonomy',
            'AL': 'Active Listening',
            'RF': 'Reflecting Feelings',
            'SA': 'Situational Appropriateness'
        }
        
        for old_col, new_col in quality_cols_mapping.items():
            if old_col in merged_df.columns:
                merged_df[new_col] = merged_df[old_col]
        
        # Fill missing quality scores with 0
        for new_col in quality_cols_mapping.values():
            if new_col not in merged_df.columns:
                merged_df[new_col] = 0
            merged_df[new_col].fillna(0, inplace=True)
        
        # Save to single CSV file
        output_path = os.path.join(output_dir, f"{split}.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Saved {len(merged_df)} rows to {output_path}")
    
    print(f"\nMHCoPilot format data saved to: {output_dir}")
    print("Ready to use with MHCoPilot_Dataset!")

if __name__ == "__main__":
    respair_dir = "../RESPAIR"
    output_dir = "./respair_mhcopilot_format"
    
    merge_respair_to_mhcopilot_format(respair_dir, output_dir)
