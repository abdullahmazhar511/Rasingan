"""
Dataset utilities for server evaluation
Mirrors MHCoPilot_Dataset preprocessing exactly
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import os

logger = logging.getLogger(__name__)

CARE_LABELS = [
    "Non-Judgmental Language", "Warmth and Encouragement", 
    "Respect for Autonomy", "Active Listening", 
    "Reflecting Feelings", "Situational Appropriateness"
]

ROLE_DICT = {
    "T": "Therapist",
    "P": "Patient"
}


class SimpleDataset:
    """Load test data from CSV file - mirrors MHCoPilot_Dataset preprocessing exactly"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to test.csv file
        """
        self.data_path = Path(data_path)
        self.context_window = 4  # Match MHCoPilot_Dataset
    
    def load(self) -> Tuple[List[str], List[str], List[Dict]]:
        """Load data and return contexts, utterances, ground_truths"""
        
        # Load CSV
        if self.data_path.is_file() and self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.is_dir():
            # Load all CSV files in directory
            csv_files = list(self.data_path.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")
            
            dfs = [pd.read_csv(f) for f in sorted(csv_files)]
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(csv_files)} CSV files from {self.data_path}")
        else:
            raise ValueError(f"Invalid path: {self.data_path}")
        
        logger.info(f"Loaded CSV with {len(df)} rows")
        
        # EXACTLY match MHCoPilot_Dataset preprocessing
        # 1. Fill NaN utterances with empty string
        df['Utterance'] = df['Utterance'].fillna('')
        
        # 2. Fill NaN care labels with 0
        cols_dict = {label: 0 for label in CARE_LABELS}
        for col in CARE_LABELS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 3. Build context for each row (same as preprocess)
        df['context'] = ''
        for i in range(1, len(df)):
            # Get previous context_window rows
            start_idx = max(0, i - self.context_window)
            end_idx = i
            
            # Build context from those rows
            context_rows = df.loc[start_idx:end_idx-1]
            context_lines = []
            for _, row in context_rows.iterrows():
                role = ROLE_DICT.get(row['Type'], 'Unknown')
                utterance = row['Utterance']
                context_lines.append(f"{role}: {utterance}")
            
            df.at[i, 'context'] = '\n'.join(context_lines)
        
        # 4. Filter to only therapist utterances
        df = df[df['Type'] == 'T']
        
        # 5. Skip first row (df[1:])
        df = df[1:]
        
        logger.info(f"After preprocessing: {len(df)} therapist utterances")
        
        # Extract data
        contexts = df['context'].tolist()
        utterances = df['Utterance'].tolist()
        
        ground_truths = []
        for idx, row in df.iterrows():
            ground_truth = {}
            for label in CARE_LABELS:
                if label in row and pd.notna(row[label]):
                    value = float(row[label])
                    ground_truth[label] = int(value)  # Keep original scale (-2 to 2)
                else:
                    ground_truth[label] = None
            ground_truths.append(ground_truth)
        
        return contexts, utterances, ground_truths


def load_test_data(data_path: str) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load test data from CSV file(s) - matches MHCoPilot_Dataset preprocessing exactly
    
    Args:
        data_path: Path to test.csv file or directory (will look for test.csv inside)
    
    Returns:
        (contexts, utterances, ground_truths)
    """
    dataset = SimpleDataset(data_path)
    return dataset.load()


if __name__ == "__main__":
    # Test
    server_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    default_test_csv = server_dir.parent / "respair_mhcopilot_format" / "test.csv"
    contexts, utterances, ground_truths = load_test_data(str(default_test_csv))
    print(f"Loaded {len(utterances)} samples")
    print(f"First context: {contexts[0][:100]}...")
    print(f"First utterance: {utterances[0][:100]}...")
    print(f"First ground truth: {ground_truths[0]}")
