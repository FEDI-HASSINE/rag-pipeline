"""
api.py
------
API REST FastAPI pour le moteur de recherche RAG BVZyme.

Lancement :
    uvicorn api:app --reload --port 8000

Endpoints :
    POST /search   → recherche sémantique
    GET  /health   → statut de l'API
    GET  /stats    → statistiques de l'index
    GET  /docs     → documentation interactive Swagger (auto-générée)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

# Offline mode : activé uniquement si le modèle est déjà en cache local
import pathlib as _pl
_cache = _pl.Path.home() / ".cache" / "huggingface" / "hub"
if _cache.exists() and any(_cache.iterdir()):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from data_loader import load_documents
from rag_pipeline import OptimizedRAGPipeline
from config import DATA_FOLDER, MODEL_NAME

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
QUERY_MAX_LEN = 500

# ─────────────────────────────────────────────────────────────────────────────
# État global du pipeline (initialisé au démarrage)
# ─────────────────────────────────────────────────────────────────────────────

class _State:
    pipeline: Optional[OptimizedRAGPipeline] = None
    n_docs: int = 0
    startup_time: float = 0.0

_state = _State()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — initialisation au démarrage du serveur
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le pipeline une seule fois au démarrage du serveur."""
    t0 = time.time()
    print("\n[RAG API] Démarrage du pipeline…")

    docs = load_documents(DATA_FOLDER)
    _state.n_docs = len(docs)

    _state.pipeline = OptimizedRAGPipeline(top_k=3)
    _state.pipeline.index_documents(docs)          # utilise le cache si dispo

    _state.startup_time = round(time.time() - t0, 3)
    print(f"[RAG API] Prêt en {_state.startup_time}s — "
          f"{_state.pipeline.get_stats()['n_chunks']} chunks indexés\n")

    yield  # l'application tourne ici

    print("[RAG API] Arrêt.")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BVZyme RAG Search API",
    description=(
        "Moteur de recherche sémantique sur les fiches techniques BVZyme.\n\n"
        "**Modèle** : `all-MiniLM-L6-v2` (sentence-transformers, dim=384)\n"
        "**Similarité** : Cosine Similarity + reranking hybride BM25 + MMR"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question en langage naturel",
        examples=["Quelle est la dose recommandée de xylanase ?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de résultats à retourner (1–10)",
    )

    @field_validator("query")
    @classmethod
    def query_not_too_long(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("La requête ne peut pas être vide.")
        if len(v) > QUERY_MAX_LEN:
            raise ValueError(
                f"La requête dépasse {QUERY_MAX_LEN} caractères "
                f"({len(v)} reçus)."
            )
        return v


class SearchResultItem(BaseModel):
    rank: int
    doc_title: str
    text: str
    cosine_score: float
    bm25_score: float
    final_score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    search_time_ms: float
    from_cache: bool


class HealthResponse(BaseModel):
    status: str
    n_chunks: int
    n_docs: int
    model: str
    startup_time_s: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_pipeline() -> OptimizedRAGPipeline:
    """Lève une 503 si le pipeline n'est pas encore prêt."""
    if _state.pipeline is None or not _state.pipeline._indexed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le pipeline n'est pas encore initialisé. Réessayez dans quelques secondes.",
        )
    return _state.pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Recherche sémantique",
    tags=["Search"],
)
def search(req: SearchRequest):
    """
    Effectue une recherche sémantique sur les fiches techniques BVZyme.

    - **query** : question en langage naturel (max 500 caractères)
    - **top_k** : nombre de résultats souhaités (défaut = 3, max = 10)
    """
    pipeline = _require_pipeline()

    try:
        response = pipeline.search(req.query, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la recherche : {exc}",
        )

    items = [
        SearchResultItem(
            rank=r.rank,
            doc_title=r.chunk.doc_title,
            text=r.chunk.text.replace("\n", " ").strip(),
            cosine_score=r.cosine_score,
            bm25_score=r.bm25_score,
            final_score=r.final_score,
        )
        for r in response["results"]
    ]

    return SearchResponse(
        query=response["query"],
        results=items,
        search_time_ms=response["search_time_ms"],
        from_cache=response.get("from_cache", False),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Statut de l'API",
    tags=["System"],
)
def health():
    """Vérifie que l'API et le pipeline sont opérationnels."""
    pipeline = _require_pipeline()
    stats = pipeline.get_stats()

    return HealthResponse(
        status="ok",
        n_chunks=stats["n_chunks"],
        n_docs=stats["n_docs"],
        model=MODEL_NAME,
        startup_time_s=_state.startup_time,
    )


@app.get(
    "/stats",
    summary="Statistiques de l'index",
    tags=["System"],
)
def stats():
    """Retourne les statistiques complètes du pipeline RAG."""
    pipeline = _require_pipeline()
    raw = pipeline.get_stats()

    return JSONResponse(content={
        "n_chunks":        raw["n_chunks"],
        "n_docs":          raw["n_docs"],
        "embedding_dim":   raw["embedding_dim"],
        "index_type":      raw["index_type"],
        "model":           MODEL_NAME if raw["real_embeddings"] else "TF-IDF fallback",
        "real_embeddings": raw["real_embeddings"],
        "cache_lru_size":  raw["cache_size"],
        "reranking":       raw["reranking"],
        "diversification": raw["diversification"],
        "startup_time_s":  _state.startup_time,
    })


@app.get("/", include_in_schema=False)
def root():
    return {"message": "BVZyme RAG API — voir /docs pour la documentation."}
