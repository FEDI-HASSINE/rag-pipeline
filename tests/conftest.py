# tests/conftest.py
# Fixtures pytest partagées entre tous les fichiers de test.

import logging
import os
import sys
import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s %(name)s: %(message)s",
)

# Rendre le dossier racine importable depuis tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_loader import load_documents
from rag_pipeline import OptimizedRAGPipeline, AdaptiveChunker, EmbeddingEngine
from config import DATA_FOLDER


@pytest.fixture(scope="session")
def sample_doc():
    """Document minimal pour tester le chunking sans charger les PDFs."""
    return {
        "id": "test_doc",
        "title": "Test Document",
        "content": (
            "La xylanase est une enzyme produite par fermentation d'Aspergillus niger. "
            "Elle est utilisée en boulangerie pour améliorer le volume du pain. "
            "La dose recommandée est de 5 à 30 ppm selon le type de farine. "
            "Le pH optimal se situe entre 4.5 et 6.0. "
            "La température d'utilisation recommandée est de 20 à 40 degrés Celsius. "
            "Le produit doit être conservé au frais, à l'abri de l'humidité. "
            "L'activité enzymatique est de 2040 XylH/g à réception. "
            "Ce produit est conforme aux normes alimentaires européennes."
        ),
    }


@pytest.fixture(scope="session")
def chunker():
    """Instance AdaptiveChunker avec paramètres par défaut."""
    return AdaptiveChunker()


@pytest.fixture(scope="session")
def embedding_engine():
    """Instance EmbeddingEngine (charge all-MiniLM-L6-v2 une seule fois)."""
    return EmbeddingEngine()


@pytest.fixture(scope="session")
def indexed_pipeline():
    """
    Pipeline complet, indexé sur les vrais PDFs.
    Scope=session : construit une seule fois pour toute la session de tests.
    Utilise le cache FAISS si disponible.
    """
    docs = load_documents(DATA_FOLDER)
    pipeline = OptimizedRAGPipeline(top_k=3)
    pipeline.index_documents(docs)
    return pipeline
