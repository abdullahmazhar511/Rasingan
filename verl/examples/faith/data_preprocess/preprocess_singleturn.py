import argparse
import os
import re
from tqdm import tqdm
import pandas as pd

import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


role_dict = {
    "T": "Therapist",
    "P": "Patient"
}

cols_dict={
    "Non-Judgmental Language":0,
    "Warmth and Encouragement":0,
    "Respect for Autonomy":0,
    "Active Listening":0,
    "Reflecting Feelings":0,
    "Situational Appropriateness":0 
}

class Faith_Dataset:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.context_window = 4
        
        # Load CSVs from the specified directory
        self.train_df = pd.read_csv(os.path.join(csv_dir, 'train.csv'))
        self.val_df = pd.read_csv(os.path.join(csv_dir, 'val.csv'))
        self.test_df = pd.read_csv(os.path.join(csv_dir, 'test.csv'))
        
        # Fill NaN utterances with empty string
        self.train_df['Utterance'].fillna('', inplace=True)
        self.val_df['Utterance'].fillna('', inplace=True)
        self.test_df['Utterance'].fillna('', inplace=True)
        
        self.train_df.fillna(cols_dict, inplace=True)
        self.val_df.fillna(cols_dict, inplace=True)
        self.test_df.fillna(cols_dict, inplace=True)
        
        self.train_dataset = self.process_df(self.train_df)
        self.val_dataset = self.process_df(self.val_df)
        self.test_dataset = self.process_df(self.test_df)
        
    def process_df(self, df):
        for i in tqdm(range(1,len(df))): 
            df.at[i, 'context'] = '\n'.join(
                df.loc[max(0, i - self.context_window):i - 1]
                .apply(lambda row: f"{role_dict[row['Type']]}: {row['Utterance']}", axis=1)
                .tolist()
            )
        df = df[df['Type'] == 'T']
        df=Dataset.from_pandas(df[1:])
        return df    
    
    def make_map_fn(self, split):
        def process_fn(example, idx):
            
            # Extract context from the example
            context = example.get('context', '')
            
            # Extract ground truth as dict of evaluation scores from cols_dict
            ground_truth = {col: example.get(col, 0) for col in cols_dict.keys()}
            data_source = example.get('data_source', 'faith_dataset')
            
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                        "You are a compassionate, client-centered therapist.\n\n"
                        "Respond with empathy, warmth, and non-judgmental understanding. Reflect the\n"
                        "client’s emotions and perspective using reflective listening (e.g., “It sounds like…”, \n"
                        "“I hear that…”, “You’re feeling…”).\n\n"
                        "Encourage gentle exploration through open-ended questions and support the\n"
                        "client’s autonomy.\n\n"
                        "Guidelines:\n"
                        "- Focus on the client’s feelings and lived experience.\n"
                        "- Be concise, calm, and emotionally attuned.\n"
                        "- Do NOT give advice, instructions, or solutions.\n"
                        "- Do NOT judge, confront, diagnose, or moralize.\n"
                        "- Do NOT assume information not expressed by the client."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Task: Write the next therapist response.\n\n Context: {context}\nTherapist:"
                    },
                ],
                "ability": "dialogue",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data
        
        return process_fn
    
    def get_data(self):
        # Apply the map function to transform the datasets
        self.train_dataset = self.train_dataset.map(function=self.make_map_fn("train"), with_indices=True)
        self.val_dataset = self.val_dataset.map(function=self.make_map_fn("val"), with_indices=True)
        self.test_dataset = self.test_dataset.map(function=self.make_map_fn("test"), with_indices=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="/home/umairai/faith_data/dataset", help="The directory containing CSV files (train.csv, val.csv, test.csv).")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir", default="../data", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()
    
    dataset_processor = Faith_Dataset(args.csv_dir)
    dataset_processor.get_data()
    
    train_dataset = dataset_processor.train_dataset
    val_dataset = dataset_processor.val_dataset
    test_dataset = dataset_processor.test_dataset
    
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.local_save_dir, "val.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_save_dir, dst=args.hdfs_dir)
