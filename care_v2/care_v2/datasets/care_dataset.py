"""
CARE_v2 HuggingFace Dataset — with embedding caching.

Key improvements over v1:
- All therapist utterance embeddings are pre-computed ONCE at __init__ time
- Cached embeddings are passed directly to the retriever (no re-encoding per item)
- Retrieval metadata (cluster IDs + similarity scores) are stored per sample
- Analysis text construction is fully ablation-flag-aware
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from care_v2.configs.config import CAREv2Config, LABEL_TO_IDX
from care_v2.retrieval.prototype_retriever import PrototypeRetriever, RetrievalResult

log = logging.getLogger(__name__)


class CAREDataset(Dataset):
    """
    PyTorch Dataset for CARE_v2 with embedding caching.

    At __init__ time:
    - Batch-encodes ALL therapist utterances using the retriever's embedding model
    - Stores embeddings as a float32 numpy array on CPU

    At __getitem__ time:
    - Retrieves the cached embedding for the sample (O(1) lookup)
    - Passes it directly to the retriever — NO re-encoding
    - Retrieval metadata is stored in the output for interpretability

    This dramatically reduces training overhead when iterating over the dataset
    across multiple epochs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        retriever: PrototypeRetriever,
        tokenizer: PreTrainedTokenizerBase,
        config: CAREv2Config,
    ):
        self.df = df.reset_index(drop=True)
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.config = config

        # Pre-compute and cache all utterance embeddings
        self.cached_embeddings: np.ndarray = self._precompute_embeddings()

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def _precompute_embeddings(self) -> np.ndarray:
        """
        Encodes all therapist utterances in one batched pass.
        Returns a float32 numpy array of shape [N, D].
        """
        utterances: List[str] = self.df["Utterance"].astype(str).tolist()
        log.info(f"Pre-computing embeddings for {len(utterances)} utterances...")

        embeddings = self.retriever.encode_batch(
            utterances,
            batch_size=self.config.embed_batch_size,
        )
        log.info(f"Embedding cache ready — shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        return embeddings  # [N, D], already normalised float32

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        context = row.get("Context", "")
        utterance = str(row["Utterance"])

        # 1. Use cached embedding directly (no API call)
        query_emb: np.ndarray = self.cached_embeddings[idx]   # [D]

        # 2. Build ablation-aware analysis text + capture retrieval metadata
        text, retrieval_meta = self.retriever.build_analysis_text(
            context=context,
            therapist_utterance=utterance,
            dimensions=self.config.labels,
            query_embedding=query_emb,
        )

        # 3. Tokenize
        enc = self.tokenizer(
            text,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 4. Build label tensor [6]
        targets = []
        for lbl in self.config.labels:
            raw = int(row.get(lbl, 0))
            targets.append(LABEL_TO_IDX.get(raw, LABEL_TO_IDX[0]))

        # 5. Build retrieval metadata dict (serialisable)
        meta_summary = self._build_meta_summary(retrieval_meta)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(targets, dtype=torch.long),
            "retrieval_meta": meta_summary,    # For interpretability / analysis
        }

    # ------------------------------------------------------------------
    # Metadata helper
    # ------------------------------------------------------------------

    def _build_meta_summary(
        self,
        retrieval_meta: Dict[str, Dict[str, RetrievalResult]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Converts RetrievalResult objects into a plain serialisable dict.

        Structure:
        {
            "Non-Judgmental Language": {
                "positive_cluster_id": 3,
                "positive_similarity": 0.843,
                "negative_cluster_id": 7,
                "negative_similarity": 0.791,
            },
            ...
        }
        """
        summary = {}
        for dim, polarity_results in retrieval_meta.items():
            pos = polarity_results["positive"]
            neg = polarity_results["negative"]
            summary[dim] = {
                "positive_cluster_id": pos.cluster_id,
                "positive_similarity": round(float(pos.similarity), 4),
                "negative_cluster_id": neg.cluster_id,
                "negative_similarity": round(float(neg.similarity), 4),
            }
        return summary
