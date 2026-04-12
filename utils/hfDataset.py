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
        self.context_window=4
        
        self.train_df['Utterance'].fillna('',inplace=True)
        self.val_df['Utterance'].fillna('',inplace=True)
        self.test_df['Utterance'].fillna('',inplace=True)
        
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
        for i in tqdm(range(1,len(df))): 
            df.at[i, 'context'] = '\n'.join(
                df.loc[max(0, i - self.context_window):i - 1]
                .apply(lambda row: f"{role_dict[row['Type']]}: {row['Utterance']}", axis=1)
                .tolist()
            )
        df = df[df['Type'] == 'T']
        df=Dataset.from_pandas(df[1:])
        return df
    
    def get_data(self):
        self.train_dataset = self.preprocess(self.train_df)
        self.val_dataset = self.preprocess(self.val_df)
        self.test_dataset = self.preprocess(self.test_df)