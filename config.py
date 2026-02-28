"""
config.py
---------
Configuration centralisée du pipeline RAG BVZyme.

Toutes les constantes du projet sont définies ici.
Les valeurs sensibles (chemins locaux) sont surchargées
via la variable d'environnement RAG_DATA_FOLDER ou le fichier .env.

Usage :
    from config import DATA_FOLDER, TOP_K, MODEL_NAME, ...
"""

from __future__ import annotations

import os
from pathlib import Path

# Charge .env si présent (python-dotenv optionnel)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Données
# ─────────────────────────────────────────────────────────────────────────────
DATA_FOLDER = Path(os.environ.get(
    "RAG_DATA_FOLDER",
    Path(__file__).parent.parent / "data" / "enzymes"   # chemin relatif par défaut
))

# ─────────────────────────────────────────────────────────────────────────────
# Contraintes imposées — NE PAS MODIFIER
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME    = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SIMILARITY    = "cosine"
TOP_K         = 3

# ─────────────────────────────────────────────────────────────────────────────
# Chunking & retrieval
# ─────────────────────────────────────────────────────────────────────────────
CHUNKER_TYPE      = "late"    # "late" | "semantic" | "recursive" | "adaptive"
MAX_CHUNK_TOKENS  = 300
MMR_LAMBDA        = 1.0       # 1.0 = désactivé
ALPHA_COSINE      = 0.80
ALPHA_BM25        = 0.10
ALPHA_QUALITY     = 0.10
QUERY_EXPANSIONS  = 3

# Poids spécialisés par type de requête
ALPHA_COSINE_PRODUCT  = 0.75   # requête produit nommé
ALPHA_BM25_PRODUCT    = 0.15
ALPHA_COSINE_NUMERIC  = 0.65   # requête avec chiffres/unités — BM25 modérément renforcé
ALPHA_BM25_NUMERIC    = 0.25
ALPHA_COSINE_LONG     = 0.85   # requête longue (>8 tokens)
ALPHA_BM25_LONG       = 0.05
