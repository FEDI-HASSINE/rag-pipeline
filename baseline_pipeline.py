"""
baseline_pipeline.py
--------------------
Pipeline RAG simple pour comparaison avec le pipeline optimisé.

Différences intentionnelles (volontairement non-optimisé) :
  - Pas de chunking : chaque document entier = 1 vecteur
  - Embedding : TF-IDF classique (sklearn) projeté en 384d
  - Similarité : Cosine pure, pas de reranking hybride
  - Pas de BM25, pas de MMR, pas de cache
  - Top-K direct sans diversification
"""

from __future__ import annotations

import time
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class BaselineResult:
    rank: int
    doc_id: str
    doc_title: str
    text: str
    score: float


class BaselineRAGPipeline:
    """
    Pipeline de référence (simple) :
    - 1 document = 1 vecteur TF-IDF (pas de chunking)
    - Cosine similarity directe
    - Pas de reranking, pas de cache, pas de MMR
    """

    def __init__(self, top_k: int = 3):
        self.top_k    = top_k
        self.docs:    List[Dict]       = []
        self.vectors: Optional[np.ndarray] = None
        self.tfidf    = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self._dim       = 384
        self._proj      = None   # matrice de projection aléatoire → 384d
        self._indexed   = False

    # ── Indexation ────────────────────────────────────────────────────────────

    def index_documents(self, documents: List[Dict]) -> Dict:
        t0 = time.time()
        # Filter out documents with empty content to avoid TF-IDF empty vocabulary
        documents = [d for d in documents if d.get("content", "").strip()]
        if not documents:
            raise ValueError("Aucun document avec du contenu non-vide pour indexer.")
        self.docs = documents
        texts = [d["content"] for d in documents]

        # TF-IDF fit + transform
        tfidf_matrix = self.tfidf.fit_transform(texts).toarray().astype(np.float32)
        n_features = tfidf_matrix.shape[1]

        # Projection aléatoire stable → 384d (même interface que all-MiniLM)
        rng = np.random.RandomState(42)
        self._proj = rng.randn(n_features, self._dim).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0)

        projected = tfidf_matrix @ self._proj
        # Normalisation L2
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.vectors = (projected / norms).astype(np.float32)

        self._indexed = True
        elapsed = time.time() - t0
        return {
            "n_docs":          len(documents),
            "n_chunks":        len(documents),  # 1 chunk = 1 doc
            "indexing_time_s": round(elapsed, 3),
            "method":          "TF-IDF + cosine (baseline)",
        }

    # ── Recherche ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> Dict:
        if not self._indexed:
            raise RuntimeError("Appeler index_documents() d'abord.")

        k = top_k or self.top_k
        t_start = time.time()

        # Encoder la requête avec le même pipeline TF-IDF + projection
        q_tfidf  = self.tfidf.transform([query]).toarray().astype(np.float32)
        q_proj   = q_tfidf @ self._proj
        q_norm   = np.linalg.norm(q_proj)
        q_vec    = (q_proj / max(q_norm, 1e-9)).astype(np.float32)

        # Cosine similarity (produit scalaire sur vecteurs normalisés)
        scores = (self.vectors @ q_vec.T).flatten()
        top_idx = np.argsort(scores)[::-1][:k]

        results = [
            BaselineResult(
                rank=rank + 1,
                doc_id=self.docs[i]["id"],
                doc_title=self.docs[i]["title"],
                text=self.docs[i]["content"][:300],
                score=round(float(scores[i]), 4),
            )
            for rank, i in enumerate(top_idx)
        ]

        elapsed_ms = (time.time() - t_start) * 1000
        return {
            "query":          query,
            "results":        results,
            "search_time_ms": round(elapsed_ms, 2),
            "from_cache":     False,
        }
