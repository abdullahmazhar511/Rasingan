import glob
import pandas as pd
from pathlib import Path
from typing import List

from care_v2.configs.config import CAREv2Config


def clean_label_column(series: pd.Series) -> pd.Series:
    """Cleans and validates a label column to the range [-2, 2]."""
    def clean_val(x):
        try:
            s = str(x).strip().replace('`', '').replace("'", "").replace('"', '')
            val = int(float(s))
            return val if -2 <= val <= 2 else 0
        except Exception:
            return 0
    return series.apply(clean_val)


def prepare_context(df: pd.DataFrame, m: int = 4) -> pd.DataFrame:
    """Builds the conversational context for each therapist utterance."""
    df = df.copy()
    df['ID'] = df['ID'].astype(str)
    df['ConvID'] = df['ID'].apply(lambda x: x.split('_')[0])
    contexts = []

    for _, group in df.groupby('ConvID', sort=False):
        utterances = group['Utterance'].tolist()
        types = group['Type'].tolist()
        for i in range(len(group)):
            start = max(0, i - m)
            ctx_parts = []
            for t, u in zip(types[start:i], utterances[start:i]):
                role = "Therapist" if t == 'T' else "Patient"
                ctx_parts.append(f"{role}: {u}")
            contexts.append("\n".join(ctx_parts))

    df['Context'] = contexts
    return df


def load_csv_data(folder_path: str, config: CAREv2Config) -> pd.DataFrame:
    """Loads all CSVs from a folder, applies label cleaning, and prepares context."""
    files = glob.glob(str(Path(folder_path) / "*.csv"))
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()

            # Normalise label column names
            col_map = {
                "NJ": "Non-Judgmental Language",
                "WE": "Warmth and Encouragement",
                "RA": "Respect for Autonomy",
                "AL": "Active Listening",
                "RF": "Reflecting Feelings",
                "SA": "Situational Appropriateness",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

            for lbl in config.labels:
                if lbl in df.columns:
                    df[lbl] = clean_label_column(df[lbl])
                else:
                    df[lbl] = 0

            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Skipping {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = prepare_context(full_df, m=config.context_window)
    return (
        full_df[full_df['Type'] == 'T']
        .dropna(subset=['Utterance'])
        .reset_index(drop=True)
    )
