from datasets import Dataset
import pandas as pd
import os
from tqdm import tqdm
import sys


role_dict={
    "T":"Therapist",
    "P":"Patient"
}
class MHCoPilot_Dataset():
    def __init__(self, path):
        self.train_df=pd.read_csv(os.path.join(path, 'train.csv'))
        self.val_df=pd.read_csv(os.path.join(path, 'val.csv'))
        self.test_df=pd.read_csv(os.path.join(path, 'test.csv'))
        self.context_window=8

        self.train_df['Utterance'] = self.train_df['Utterance'].fillna('')
        self.val_df['Utterance'] = self.val_df['Utterance'].fillna('')
        self.test_df['Utterance'] = self.test_df['Utterance'].fillna('')
        
        cols_dict={
            "Non-Judgmental Language":0,
            "Warmth and Encouragement":0,
            "Respect for Autonomy":0,
            "Active Listening":0,
            "Reflecting Feelings":0,
            "Situational Appropriateness":0 
        }
        
        self.train_df.fillna(cols_dict, inplace=True)
        self.val_df.fillna(cols_dict, inplace=True)
        self.test_df.fillna(cols_dict, inplace=True)
        
    def preprocess(self,df):
        df = df.copy()
        # Keep context within the same conversation; IDs are like <conv_id>_<turn_id>.
        df['base_conv_id'] = df['ID'].astype(str).str.replace(r'_\d+$', '', regex=True)
        df['context'] = ''

        for i in tqdm(range(len(df))):
            if i == 0:
                continue
            start = max(0, i - self.context_window)
            prev = df.iloc[start:i]
            same_conv_prev = prev[prev['base_conv_id'] == df.at[i, 'base_conv_id']]
            context_lines = []
            for row in same_conv_prev.itertuples(index=False):
                context_lines.append(f"{role_dict.get(row.Type, row.Type)}: {row.Utterance}")
            df.at[i, 'context'] = '\n'.join(context_lines)

        df = df[df['Type'] == 'T'].drop(columns=['base_conv_id'])
        df = Dataset.from_pandas(df)
        return df
    
    def get_data(self):
        self.train_dataset = self.preprocess(self.train_df)
        self.val_dataset = self.preprocess(self.val_df)
        self.test_dataset = self.preprocess(self.test_df)