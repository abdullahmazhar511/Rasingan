"""
Inference engine for CARE_v2.

predict_single() accepts a context + therapist utterance and returns
a dict of {dimension: predicted_score} using the trained model.
"""

import logging
from typing import Dict

import torch
import numpy as np

from care_v2.configs.config import CAREv2Config, LABELS, IDX_TO_LABEL
from care_v2.models.care_classifier import CAREv2Classifier
from care_v2.retrieval.prototype_retriever import PrototypeRetriever

log = logging.getLogger(__name__)


class CAREv2Predictor:
    """
    Wraps CAREv2Classifier + PrototypeRetriever for single-sample inference.
    """

    def __init__(
        self,
        model: CAREv2Classifier,
        retriever: PrototypeRetriever,
        tokenizer,
        config: CAREv2Config,
    ):
        self.model = model
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.config = config

        self._device = next(model.parameters()).device
        model.eval()

    def predict_single(
        self,
        context: str,
        therapist_utterance: str,
    ) -> Dict[str, int]:
        """
        End-to-end inference for a single therapist utterance.

        Returns:
            dict mapping dimension name → predicted therapeutic score in [-2, -1, 0, 1, 2]
        """
        # 1. Build analysis text + capture retrieval metadata
        text, retrieval_meta = self.retriever.build_analysis_text(
            context=context,
            therapist_utterance=therapist_utterance,
            dimensions=LABELS,
        )

        # 2. Optionally print retrieval metadata for interpretability
        if self.config.debug_retrieval:
            print("\n[PREDICTOR] Retrieval Metadata:")
            for dim, meta in retrieval_meta.items():
                pos = meta["positive"]
                neg = meta["negative"]
                print(
                    f"  {dim}\n"
                    f"    + Cluster {pos.cluster_id:>3} | sim={pos.similarity:.4f}\n"
                    f"    - Cluster {neg.cluster_id:>3} | sim={neg.similarity:.4f}"
                )

        # 3. Tokenize
        enc = self.tokenizer(
            text,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        # 4. Forward pass
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = out["logits"]      # [1, 6, 5]
        preds_idx = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()  # [6]

        # 5. Map indices → real labels
        result = {}
        for i, dim in enumerate(LABELS):
            result[dim] = IDX_TO_LABEL[int(preds_idx[i])]

        return result

    def predict_batch(self, samples: list) -> list:
        """
        Runs predict_single on a list of {"context": ..., "therapist": ...} dicts.
        """
        return [
            self.predict_single(s["context"], s["therapist"])
            for s in samples
        ]
