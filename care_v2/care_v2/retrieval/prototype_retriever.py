"""
Prototype Retrieval Engine for CARE_v2.

At init time, loads ALL prototype JSONs from the generated_explanations dir,
stores embeddings in memory (organized by dimension + polarity), and provides
fast cosine-similarity-based retrieval at inference time.

Supports:
- Pre-computed query embeddings (for Dataset caching)
- Retrieval metadata (cluster_id + similarity score)
- Ablation-aware analysis text construction
- Debug mode logging
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from care_v2.configs.config import CAREv2Config

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Metadata returned alongside a retrieved prototype."""
    entry: Optional["PrototypeEntry"]
    similarity: float = 0.0

    @property
    def cluster_id(self) -> int:
        return self.entry.cluster_id if self.entry else -1

    @property
    def explanation(self) -> str:
        return self.entry.explanation if self.entry else "No reference available."


class PrototypeEntry:
    """Lightweight struct for a single stored prototype."""
    __slots__ = ("dimension", "polarity", "cluster_id", "patient", "therapist", "explanation", "embedding")

    def __init__(self, d: dict):
        self.dimension = d.get("dimension", "")
        self.polarity = d.get("polarity", "")
        self.cluster_id = d.get("cluster_id", -1)
        self.patient = d.get("patient", "")
        self.therapist = d.get("therapist", "")
        self.explanation = d.get("explanation", "")
        self.embedding: Optional[np.ndarray] = (
            np.array(d["embedding"], dtype=np.float32) if "embedding" in d else None
        )


class PrototypeRetriever:
    """
    Semantic prototype retrieval engine.

    For each therapeutic dimension, stores separate pools of:
    - positive prototypes
    - negative prototypes

    Retrieves the single BEST matching prototype per polarity per dimension
    based on cosine similarity with the query utterance.
    """

    def __init__(self, config: CAREv2Config, device: Optional[str] = None):
        self.config = config
        self.embed_model = SentenceTransformer(config.embedding_model_id, device=device)

        # Pool: {dimension -> {polarity -> List[PrototypeEntry]}}
        self._pool: Dict[str, Dict[str, List[PrototypeEntry]]] = {}
        self._pool_tensors: Dict[str, Dict[str, np.ndarray]] = {}

        self._load_all_prototypes()
        self._build_embedding_matrices()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_all_prototypes(self):
        """Recursively loads every JSON from exactly ONE prototype k-folder."""
        root = Path(self.config.prototype_dir)
        k_dir = root / self.config.prototype_k
        
        if not k_dir.exists():
            raise FileNotFoundError(f"Selected prototype folder {k_dir} not found. Check config.prototype_k.")

        print("\n" + "=" * 40)
        print("PROTOTYPE CONFIG")
        print("=" * 40)
        print(f"Prototype Set: {self.config.prototype_k}")

        for json_path in k_dir.glob("*/*/*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entry = PrototypeEntry(data)
                if not entry.explanation or entry.embedding is None:
                    continue

                dim = entry.dimension
                pol = entry.polarity

                self._pool.setdefault(dim, {}).setdefault(pol, []).append(entry)
            except Exception as e:
                log.debug(f"Skipping {json_path}: {e}")

        # Summary statistics
        total_dimensions = len(self._pool)
        total_prototypes = sum(
            len(entries)
            for pols in self._pool.values()
            for entries in pols.values()
        )
        
        print(f"Total Dimensions: {total_dimensions}")
        print(f"Total Prototypes: {total_prototypes}")
        print("=" * 40 + "\n")

        # Detailed breakdown
        log.info(f"Loaded {total_prototypes} prototypes from {self.config.prototype_k} across {total_dimensions} dimensions.")
        for dim, pols in self._pool.items():
            pos_count = len(pols.get("positive", []))
            neg_count = len(pols.get("negative", []))
            log.info(f"  {dim:<30} | pos: {pos_count:>3} | neg: {neg_count:>3}")

    def _build_embedding_matrices(self):
        """Pre-stacks embeddings into numpy arrays for fast batch cosine similarity."""
        for dim, polarity_dict in self._pool.items():
            self._pool_tensors[dim] = {}
            for pol, entries in polarity_dict.items():
                mat = np.stack([e.embedding for e in entries], axis=0)  # [N, D]
                # L2 normalise for fast cosine via dot product
                norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
                self._pool_tensors[dim][pol] = mat / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """Encodes a single utterance into a unit-normalised float32 embedding."""
        vec = self.embed_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Batch-encodes a list of utterances. Returns [N, D] float32 array."""
        log.info(f"Encoding {len(texts)} utterances in batches of {batch_size}...")
        vecs = self.embed_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vecs.astype(np.float32)

    def retrieve_best_with_meta(
        self,
        query_embedding: np.ndarray,
        dimension: str,
        polarity: str,
    ) -> RetrievalResult:
        """
        Returns a RetrievalResult containing:
        - the best matching PrototypeEntry
        - the cosine similarity score
        """
        if dimension not in self._pool_tensors or polarity not in self._pool_tensors[dimension]:
            return RetrievalResult(entry=None, similarity=0.0)

        mat = self._pool_tensors[dimension][polarity]     # [N, D]
        scores = mat @ query_embedding                    # [N]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        return RetrievalResult(
            entry=self._pool[dimension][polarity][best_idx],
            similarity=best_score,
        )

    def retrieve_for_all_dimensions(
        self,
        query_embedding: np.ndarray,
        dimensions: List[str],
    ) -> Dict[str, Dict[str, RetrievalResult]]:
        """
        Given a pre-computed query embedding, retrieves ONE best positive +
        ONE best negative RetrievalResult per dimension.

        Accepts a cached embedding directly — no internal encoding.
        """
        results: Dict[str, Dict[str, RetrievalResult]] = {}
        for dim in dimensions:
            results[dim] = {
                "positive": self.retrieve_best_with_meta(query_embedding, dim, "positive"),
                "negative": self.retrieve_best_with_meta(query_embedding, dim, "negative"),
            }
        return results

    # ------------------------------------------------------------------
    # Analysis Text Construction (ablation-aware)
    # ------------------------------------------------------------------

    def build_analysis_text(
        self,
        context: str,
        therapist_utterance: str,
        dimensions: List[str],
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[str, Dict[str, Dict[str, RetrievalResult]]]:
        """
        Constructs the full structured input text for the classifier,
        respecting all ablation flags from config.

        If query_embedding is provided it is used directly (cached path).
        Otherwise the utterance is encoded on the fly (inference convenience path).

        Returns:
            (text, retrieval_metadata)
        """
        cfg = self.config

        # Resolve embedding
        if query_embedding is None:
            query_embedding = self.encode_query(therapist_utterance)

        # Retrieve prototypes
        retrieved = self.retrieve_for_all_dimensions(query_embedding, dimensions)

        # Debug mode
        if cfg.debug_retrieval:
            self._debug_print(therapist_utterance, retrieved, dimensions)

        # Build text
        parts = []

        if cfg.use_context:
            parts.append(f"Context:\n{context}\n")

        parts.append(f"Therapist Utterance:\n\"{therapist_utterance}\"\n")

        if cfg.use_explanations and (cfg.use_positive_refs or cfg.use_negative_refs):
            parts.append("Prototype Analysis:\n")
            for dim in dimensions:
                dim_parts = [dim]
                if cfg.use_positive_refs:
                    pos_text = retrieved[dim]["positive"].explanation
                    dim_parts.append(f"Positive Reference:\n{pos_text}")
                if cfg.use_negative_refs:
                    neg_text = retrieved[dim]["negative"].explanation
                    dim_parts.append(f"Negative Reference:\n{neg_text}")
                parts.append("\n".join(dim_parts) + "\n")

        parts.append("\nTask:\nPredict therapeutic alignment scores.")
        return "\n".join(parts), retrieved

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def _debug_print(
        self,
        utterance: str,
        retrieved: Dict[str, Dict[str, RetrievalResult]],
        dimensions: List[str],
    ):
        print("\n" + "=" * 60)
        print(f"[DEBUG RETRIEVAL] Utterance: {utterance[:80]}...")
        print("=" * 60)
        for dim in dimensions:
            pos = retrieved[dim]["positive"]
            neg = retrieved[dim]["negative"]
            print(f"\n  {dim}")
            print(f"    + Cluster {pos.cluster_id:>3} | sim={pos.similarity:.4f} | {pos.explanation[:60]}...")
            print(f"    - Cluster {neg.cluster_id:>3} | sim={neg.similarity:.4f} | {neg.explanation[:60]}...")
        print("=" * 60 + "\n")
