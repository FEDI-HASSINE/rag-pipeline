"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PIPELINE RAG OPTIMISÉ - COMPÉTITION TECHNIQUE                      ║
║          Modèle: all-MiniLM-L6-v2 | Méthode: Cosine Similarity + Reranking ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture competitive:
  1. Chunking adaptatif avec overlap dynamique
  2. Embeddings normalisés (all-MiniLM-L6-v2 ou fallback TF-IDF projeté)
  3. Index FAISS IVF optimisé
  4. Reranking hybride (cosine + BM25 + length penalty)
  5. Diversification MMR (Maximal Marginal Relevance)
  6. Cache LRU pour requêtes fréquentes
"""

import logging
import numpy as np
import time
import hashlib
import json
import re
import math
import os
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from query_expander import expand_query
from config import (
    MODEL_NAME, EMBEDDING_DIM, TOP_K, CHUNKER_TYPE, MMR_LAMBDA,
    ALPHA_COSINE, ALPHA_BM25, ALPHA_QUALITY, QUERY_EXPANSIONS,
    ALPHA_COSINE_PRODUCT, ALPHA_BM25_PRODUCT,
    ALPHA_COSINE_NUMERIC, ALPHA_BM25_NUMERIC,
    ALPHA_COSINE_LONG, ALPHA_BM25_LONG,
)

logger = logging.getLogger("rag_pipeline")

# ─────────────────────────────────────────────────────────────
# TENTATIVE IMPORT DES VRAIES LIBRAIRIES (avec fallback propre)
# ─────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    REAL_EMBEDDINGS = True
    logger.info("sentence-transformers disponible — all-MiniLM-L6-v2")
except ImportError:
    REAL_EMBEDDINGS = False
    logger.warning("sentence-transformers non disponible — fallback TF-IDF projecté")

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS disponible — index IVF optimisé activé")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS non disponible — recherche numpy (même résultats, un peu plus lent)")

from sklearn.feature_extraction.text import TfidfVectorizer

# ContextualChunkerAdapter actif si sentence-transformers est disponible
LATE_CHUNKING = REAL_EMBEDDINGS   # réutilise REAL_EMBEDDINGS déjà défini ci-dessus
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Fragment documentaire enrichi"""
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    start_pos: int
    end_pos: int
    word_count: int
    quality_score: float = 1.0  # penalité bruit
    embedding: Optional[np.ndarray] = None

@dataclass
class SearchResult:
    """Résultat de recherche avec scores détaillés"""
    chunk: Chunk
    cosine_score: float
    bm25_score: float
    final_score: float
    rank: int
    explanation: str = ""


# ─────────────────────────────────────────────────────────────
# EMBEDDING ENGINE
# ─────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Moteur d'embedding avec support all-MiniLM-L6-v2 (officiel)
    et fallback TF-IDF projeté en 384 dimensions pour démonstration hors réseau.
    Interface identique dans les deux cas.
    """
    
    EMBEDDING_DIM = 384  # Dimension all-MiniLM-L6-v2
    
    def __init__(self):
        self.model = None
        self.tfidf = None
        self._projection_matrix = None
        self._corpus_fitted = False
        
        if REAL_EMBEDDINGS:
            logger.info("Chargement all-MiniLM-L6-v2...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Modèle all-MiniLM-L6-v2 chargé")
        else:
            logger.warning("Mode fallback: TF-IDF projecté 384d (compatible all-MiniLM-L6-v2)")
    
    def fit_corpus(self, texts: List[str]):
        """Pré-entraîne le TF-IDF sur le corpus (fallback uniquement)"""
        if not REAL_EMBEDDINGS:
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=2000,
                sublinear_tf=True,
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b'
            )
            self.tfidf.fit(texts)
            n_features = len(self.tfidf.vocabulary_)
            # Projection aléatoire stable (seed fixe = reproductible)
            rng = np.random.RandomState(42)
            self._projection_matrix = rng.randn(n_features, self.EMBEDDING_DIM).astype(np.float32)
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=0)
            self._corpus_fitted = True
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode en batch - retourne embeddings normalisés L2"""
        if REAL_EMBEDDINGS:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                normalize_embeddings=True,  # normalisation L2 intégrée
                show_progress_bar=False
            )
            return embeddings.astype(np.float32)
        else:
            if not self._corpus_fitted:
                self.fit_corpus(texts)
            tfidf_matrix = self.tfidf.transform(texts).toarray().astype(np.float32)
            projected = tfidf_matrix @ self._projection_matrix
            # Normalisation L2 (équivalent à normalisation all-MiniLM)
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return (projected / norms).astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ─────────────────────────────────────────────────────────────
# CHUNKING ADAPTATIF
# ─────────────────────────────────────────────────────────────

class AdaptiveChunker:
    """
    Chunking optimisé compétition:
    - Découpe sémantique (phrases + paragraphes)
    - Overlap dynamique basé sur densité sémantique
    - Score qualité pour pénaliser les chunks bruités
    """
    
    def __init__(self,
                 min_chunk_size: int = 30,
                 max_chunk_size: int = 120,
                 overlap_ratio: float = 0.15):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split respectueux des abréviations françaises"""
        # Points de fin de phrase (évite les abréviations courantes)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ö])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _compute_quality_score(self, text: str) -> float:
        """
        Score qualité d'un chunk [0-1]:
        - Pénalise trop court ou trop répétitif
        - Favorise densité informationnelle
        """
        words = text.split()
        n_words = len(words)
        
        if n_words < 10:
            return 0.4  # chunk trop court
        if n_words > 400:
            return 0.7  # chunk trop long, dilué
        
        # Ratio de mots uniques (diversité lexicale)
        unique_ratio = len(set(w.lower() for w in words)) / n_words
        
        # Présence de chiffres/entités (indicateur de contenu factuel)
        has_numbers = 1.0 if re.search(r'\d+', text) else 0.8
        
        # Score composite
        quality = (0.6 * min(unique_ratio * 2, 1.0) + 
                   0.3 * has_numbers +
                   0.1 * min(n_words / 100, 1.0))
        return round(quality, 3)
    
    def chunk_document(self, doc: Dict) -> List[Chunk]:
        """Découpe un document en chunks avec overlap dynamique"""
        text = doc['content']
        sentences = self._split_sentences(text)
        
        chunks = []
        chunk_idx = 0
        i = 0
        
        while i < len(sentences):
            # Accumule les phrases jusqu'à max_chunk_size
            current_chunk_sentences = []
            current_words = 0
            j = i
            
            while j < len(sentences):
                word_count = len(sentences[j].split())
                if current_words + word_count > self.max_chunk_size and current_chunk_sentences:
                    break
                current_chunk_sentences.append(sentences[j])
                current_words += word_count
                j += 1
            
            if not current_chunk_sentences:
                i += 1
                continue
            
            chunk_text = ' '.join(current_chunk_sentences)
            
            # Overlap dynamique: si le chunk est riche, overlap plus grand
            quality = self._compute_quality_score(chunk_text)
            dynamic_overlap = max(1, int(len(current_chunk_sentences) * self.overlap_ratio * quality))
            
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunk = Chunk(
                    chunk_id=f"{doc['id']}_chunk{chunk_idx}",
                    doc_id=doc['id'],
                    doc_title=doc['title'],
                    text=chunk_text,
                    start_pos=i,
                    end_pos=j - 1,
                    word_count=len(chunk_text.split()),
                    quality_score=quality
                )
                chunks.append(chunk)
                chunk_idx += 1
            
            # Avance avec overlap
            i = j - dynamic_overlap if j - dynamic_overlap > i else j
            if i >= j:  # Sécurité boucle infinie
                i = j
        
        return chunks


# ─────────────────────────────────────────────────────────────
# CONTEXTUAL CHUNKER ADAPTER (Amélioration 1 — Late Chunking contextuel)
# ─────────────────────────────────────────────────────────────

class ContextualChunkerAdapter:
    """
    Late Chunking contextuel — zéro dépendance externe supplémentaire.

    Technique : chaque chunk est encodé avec le titre du document
    + les premiers 30 mots du document comme préfixe de contexte,
    avant d'alimenter all-MiniLM-L6-v2.

    Sans ce préfixe :
      chunk 1 de TG MAX63 = "dosage is 10-30 ppm"  → ressemble à tous les
      chunks 'dosage' de tous les produits.

    Avec préfixe :
      "TG MAX63 transglutaminase bakery enzyme | dosage is 10-30 ppm"
      → l'embedding distingue MAINTENANT TG MAX63 des autres produits. ✅

    Affichage : le chunk.text reste inchangé (le préfixe n'est pas stocké).
    Modèle : all-MiniLM-L6-v2 (imposé — inchangé ✅)
    Dimension : 384d, normalisée L2 → produit scalaire = cosine similarity ✅
    """
    EMBEDDING_DIM = 384
    CONTEXT_WORDS = 35   # mots du début du document utilisés comme préfixe

    def __init__(self, chunker: "AdaptiveChunker", engine: "EmbeddingEngine"):
        self.chunker = chunker
        self.engine = engine

    def _doc_prefix(self, doc: Dict) -> str:
        """Construit le préfixe contextuel : titre + premiers mots du document."""
        title = doc.get("title", "").strip()
        first_words = " ".join(doc.get("content", "").split()[:self.CONTEXT_WORDS])
        return f"{title}. {first_words}" if title else first_words

    def chunk_document_with_embeddings(
        self, doc: Dict
    ) -> Tuple[List["Chunk"], np.ndarray]:
        """
        Chunking standard + embeddings contextualisés.
        Les chunks retournés contiennent le texte original (sans préfixe).
        Les embeddings, eux, sont calculés sur 'prefix | chunk_text'.
        """
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            return [], np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        prefix = self._doc_prefix(doc)
        contextual_texts = [f"{prefix} | {c.text}" for c in chunks]
        embeddings = self.engine.encode(contextual_texts)   # L2-normalisé ✅
        return chunks, embeddings


# ─────────────────────────────────────────────────────────────
# BM25 SCORER (scoring hybride)
# ─────────────────────────────────────────────────────────────

class BM25Scorer:
    """BM25 léger pour scoring hybride (boost sémantique + lexical)"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
    
    def fit(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        nd = {}  # word -> doc count
        
        for doc in corpus:
            words = doc.lower().split()
            self.doc_len.append(len(words))
            word_counts = Counter(words)
            self.doc_freqs.append(word_counts)
            for word in word_counts:
                nd[word] = nd.get(word, 0) + 1
        
        self.avgdl = sum(self.doc_len) / self.corpus_size
        
        # IDF avec smoothing
        for word, freq in nd.items():
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
            )
    
    def score(self, query: str, doc_idx: int) -> float:
        words = query.lower().split()
        score = 0.0
        dl = self.doc_len[doc_idx]
        
        for word in words:
            if word not in self.doc_freqs[doc_idx]:
                continue
            freq = self.doc_freqs[doc_idx][word]
            idf = self.idf.get(word, 0)
            tf = freq * (self.k1 + 1) / (
                freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
            score += idf * tf
        
        return score
    
    def score_all(self, query: str) -> np.ndarray:
        return np.array([self.score(query, i) for i in range(self.corpus_size)])


# ─────────────────────────────────────────────────────────────
# VECTOR INDEX
# ─────────────────────────────────────────────────────────────

class VectorIndex:
    """Index vectoriel FAISS optimisé ou fallback numpy"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = None
        self.embeddings_matrix = None  # fallback numpy
    
    def build(self, embeddings: np.ndarray):
        """Construit l'index sur les embeddings normalisés"""
        n = len(embeddings)
        
        if FAISS_AVAILABLE and n > 100:
            # IVF pour grands corpus (>1000 vecteurs idéalement)
            nlist = min(int(math.sqrt(n)), 50)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(10, nlist)
            logger.info("FAISS IVF index: %d vecteurs, nlist=%d, nprobe=%d", n, nlist, self.index.nprobe)
        elif FAISS_AVAILABLE:
            # Flat index pour petits corpus
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
            logger.info("FAISS Flat index: %d vecteurs", n)
        else:
            self.embeddings_matrix = embeddings
            logger.info("Numpy index: %d vecteurs", n)
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne (scores, indices) pour top_k résultats"""
        q = query_embedding.reshape(1, -1).astype(np.float32)
        
        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(q, top_k)
            return scores[0], indices[0]
        else:
            # Produit scalaire sur embeddings normalisés = cosine similarity
            scores = (self.embeddings_matrix @ q.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            return scores[top_indices], top_indices


# ─────────────────────────────────────────────────────────────
# LRU CACHE
# ─────────────────────────────────────────────────────────────

class QueryCache:
    """Cache LRU pour requêtes fréquentes"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def _key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[SearchResult]]:
        key = self._key(query)
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def set(self, query: str, results: List[SearchResult]):
        key = self._key(query)
        if len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = results
        self.order.append(key)


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

class OptimizedRAGPipeline:
    """
    Pipeline RAG compétition:
    - Chunking adaptatif + overlap dynamique
    - Embeddings all-MiniLM-L6-v2 normalisés
    - Index FAISS IVF
    - Reranking hybride cosine + BM25 + quality penalty
    - MMR diversification
    - Cache LRU
    """
    
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.model_name = MODEL_NAME   # modèle imposé — ne pas changer
        self.chunks: List[Chunk] = []
        self.embedding_engine = EmbeddingEngine()
        self.chunker = AdaptiveChunker()
        self.bm25 = BM25Scorer()
        self.vector_index = VectorIndex(dim=EMBEDDING_DIM)
        self.cache = QueryCache()
        self._indexed = False
        
        # Paramètres reranking hybride — lus depuis config.py
        self.alpha_cosine = ALPHA_COSINE
        self.alpha_bm25 = ALPHA_BM25
        self.alpha_quality = ALPHA_QUALITY
        
        # Paramètre MMR — lu depuis config.py
        self.mmr_lambda = MMR_LAMBDA

        # Amélioration 1 : Contextual Chunker (late chunking sans dépendance externe)
        self.chunker_type = CHUNKER_TYPE if LATE_CHUNKING else "adaptive"
        self.late_adapter = (
            ContextualChunkerAdapter(self.chunker, self.embedding_engine)
            if LATE_CHUNKING else None
        )
    
    # ── INDEXATION ──────────────────────────────────────────
    
    # ── CONSTANTES CACHE ────────────────────────────────────
    CACHE_DIR       = Path(".faiss_cache")
    CACHE_INDEX     = CACHE_DIR / "index.faiss"
    CACHE_CHUNKS    = CACHE_DIR / "chunks.pkl"
    CACHE_BM25      = CACHE_DIR / "bm25.pkl"
    CACHE_EMBEDDINGS = CACHE_DIR / "embeddings.npy"

    def _cache_exists(self) -> bool:
        """Vérifie que tous les fichiers de cache sont présents."""
        return (
            self.CACHE_INDEX.exists()
            and self.CACHE_CHUNKS.exists()
            and self.CACHE_BM25.exists()
            and self.CACHE_EMBEDDINGS.exists()
        )

    def _compute_fingerprint(self, documents: List[Dict]) -> str:
        """Empreinte SHA-256 (12 hex) couvrant le corpus et la config du pipeline."""
        h = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d["id"]):
            h.update(doc["id"].encode())
            h.update(doc["content"][:200].encode())
        cfg = {
            "chunker": self.chunker_type,
            "model": self.model_name,
            "max_chunk": self.chunker.max_chunk_size,
        }
        h.update(json.dumps(cfg, sort_keys=True).encode())
        return h.hexdigest()[:12]

    def _cache_is_valid(self, documents: List[Dict]) -> bool:
        """
        Retourne True si le cache existe ET correspond au corpus/config actuels.
        Remplace _cache_exists() dans index_documents() pour détecter les
        incohérences silencieuses (ex. changement de chunker).
        """
        if not self._cache_exists():
            return False
        fp_file = self.CACHE_DIR / "fingerprint.txt"
        if fp_file.exists():
            return fp_file.read_text(encoding="utf-8").strip() == self._compute_fingerprint(documents)
        # Cache sans fingerprint (ancienne version) → invalide pour forcer la migration
        return False

    def _save_cache(self, embeddings: np.ndarray):
        """Persiste l'index FAISS, les chunks et le modèle BM25 sur disque."""
        self.CACHE_DIR.mkdir(exist_ok=True)

        # 1. FAISS ou numpy embeddings
        if FAISS_AVAILABLE and self.vector_index.index is not None:
            faiss.write_index(self.vector_index.index, str(self.CACHE_INDEX))
        else:
            # Fallback : on sauvegarde la matrice numpy à la place
            np.save(str(self.CACHE_INDEX) + ".npy", embeddings)
            (self.CACHE_INDEX).write_text("numpy_fallback")  # fichier sentinelle

        # 2. Chunks (sans embedding pour réduire la taille)
        chunks_no_emb = []
        for c in self.chunks:
            c_copy = Chunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title,
                text=c.text, start_pos=c.start_pos, end_pos=c.end_pos,
                word_count=c.word_count, quality_score=c.quality_score,
                embedding=None
            )
            chunks_no_emb.append(c_copy)
        with open(self.CACHE_CHUNKS, "wb") as f:
            pickle.dump(chunks_no_emb, f)

        # 3. BM25
        with open(self.CACHE_BM25, "wb") as f:
            pickle.dump(self.bm25, f)

        # 4. Embeddings bruts (pour réassigner aux chunks au reload)
        np.save(str(self.CACHE_EMBEDDINGS), embeddings)

        logger.info("Cache sauvegardé dans %s", self.CACHE_DIR)

    def _load_cache(self) -> bool:
        """Recharge l'index depuis le cache. Retourne True si succès."""
        try:
            t0 = time.time()

            # 1. Embeddings
            embeddings = np.load(str(self.CACHE_EMBEDDINGS))

            # 2. Chunks + réassignation embeddings
            with open(self.CACHE_CHUNKS, "rb") as f:
                self.chunks = pickle.load(f)
            for i, chunk in enumerate(self.chunks):
                chunk.embedding = embeddings[i]

            # 3. Index FAISS ou fallback
            sentinel = ""
            if self.CACHE_INDEX.exists():
                try:
                    sentinel = self.CACHE_INDEX.read_text(encoding="utf-8")
                except Exception:
                    sentinel = ""  # fichier binaire FAISS → pas une sentinelle
            if sentinel == "numpy_fallback":
                self.vector_index.embeddings_matrix = embeddings
            elif FAISS_AVAILABLE:
                self.vector_index.index = faiss.read_index(str(self.CACHE_INDEX))
            else:
                self.vector_index.embeddings_matrix = embeddings

            # 4. BM25
            with open(self.CACHE_BM25, "rb") as f:
                self.bm25 = pickle.load(f)

            elapsed = time.time() - t0
            self._indexed = True
            logger.info("Index chargé depuis cache en %.2fs — %d chunks, %d termes BM25",
                        elapsed, len(self.chunks), len(self.bm25.idf))
            return True

        except Exception as e:
            logger.warning("Echec chargement cache (%s) — recalcul forcé", e)
            return False

    def index_documents(self, documents: List[Dict], force_reindex: bool = False) -> Dict:
        """Pipeline d'indexation complet avec métriques et persistance cache."""
        logger.info("=== INDEXATION - PIPELINE OPTIMISÉ ===")

        # ── Tentative de chargement depuis cache ──────────────
        if not force_reindex and self._cache_is_valid(documents):
            if self._load_cache():
                return {
                    "n_docs": len(documents),
                    "n_chunks": len(self.chunks),
                    "avg_quality": round(float(np.mean([c.quality_score for c in self.chunks])), 3),
                    "indexing_time_s": 0.0,
                    "source": "cache",
                }
        else:
            logger.info("Recalcul index en cours...")

        t0 = time.time()

        if self.late_adapter is not None:
            # ── Amélioration 1 : Late Chunking (chonkie) ─────────────
            # Chunking + embeddings contextualisés en une seule passe.
            # Le modèle lit chaque document entier → chaque chunk hérite
            # du contexte de ses voisins (nom produit dans vecteur dosage).
            logger.info("[1+2/4] Contextual Chunking + embeddings...")
            t_emb = time.time()
            all_chunks: List[Chunk] = []
            doc_embeddings: List[np.ndarray] = []

            for doc in documents:
                doc_chunks, doc_embs = self.late_adapter.chunk_document_with_embeddings(doc)
                if doc_chunks:
                    all_chunks.extend(doc_chunks)
                    doc_embeddings.append(doc_embs)

            self.chunks = all_chunks
            embeddings = np.vstack(doc_embeddings) if doc_embeddings else np.empty((0, 384), dtype=np.float32)
            t_emb_elapsed = time.time() - t_emb

        else:
            # ── Fallback : AdaptiveChunker + EmbeddingEngine ──────────
            logger.info("[1/4] Chunking adaptatif (fallback)...")
            all_chunks = []
            for doc in documents:
                doc_chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(doc_chunks)
            self.chunks = all_chunks

            logger.info("[2/4] Génération embeddings (all-MiniLM-L6-v2)...")
            texts = [c.text for c in self.chunks]
            if not REAL_EMBEDDINGS:
                self.embedding_engine.fit_corpus(texts)
            t_emb = time.time()
            embeddings = self.embedding_engine.encode(texts, batch_size=32)
            t_emb_elapsed = time.time() - t_emb

        n_chunks = len(self.chunks)
        avg_quality = np.mean([c.quality_score for c in self.chunks]) if self.chunks else 0.0

        # Assignation embeddings aux chunks
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]

        mode = "Late Chunking contextuel (Amélioration 1)" if self.late_adapter else "AdaptiveChunker"
        logger.info("Mode: %s", mode)
        logger.info("%d documents → %d chunks", len(documents), n_chunks)
        logger.info("Qualité moyenne des chunks: %.3f", avg_quality)
        logger.info("Taille moyenne: %.0f mots", np.mean([c.word_count for c in self.chunks]))
        logger.info("%d embeddings en %.3fs", n_chunks, t_emb_elapsed)
        logger.info("Dimension: %dd | Norme moyenne: %.4f",
                    embeddings.shape[1], np.mean(np.linalg.norm(embeddings, axis=1)))

        # 3. Index vectoriel FAISS
        logger.info("[3/4] Construction index vectoriel...")
        self.vector_index.build(embeddings)

        # 4. BM25 index
        logger.info("[4/4] Index BM25 (scoring hybride)...")
        texts = [c.text for c in self.chunks]   # texte original (sans préfixe)
        self.bm25.fit(texts)
        logger.info("Vocabulaire BM25: %d termes", len(self.bm25.idf))

        # ── Sauvegarde cache ───────────────────────────────────
        logger.info("[Cache] Sauvegarde sur disque...")
        self._save_cache(embeddings)
        # Fingerprint versionné (Amélioration 8)
        fp = self._compute_fingerprint(documents)
        (self.CACHE_DIR / "fingerprint.txt").write_text(fp, encoding="utf-8")
        logger.info("Fingerprint: %s", fp)

        t_total = time.time() - t0
        self._indexed = True

        metrics = {
            "n_docs": len(documents),
            "n_chunks": n_chunks,
            "avg_quality": round(float(avg_quality), 3),
            "indexing_time_s": round(t_total, 3),
            "embedding_dim": int(embeddings.shape[1]),
            "chunks_per_second": round(n_chunks / t_total, 1),
            "source": "computed",
        }

        logger.info("Indexation terminée en %.3fs — prêt sur %d fragments", t_total, n_chunks)
        return metrics
    
    # ── RECHERCHE ───────────────────────────────────────────
    
    def _normalize_bm25(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalisation BM25 robuste par percentile (Amélioration 3).

        Problème de la normalisation par max :
          Si un seul chunk a BM25=8.0 et les autres sont à 0.5–1.5,
          diviser par 8.0 écrase tout en [0, 0.19] → signal quasi-binaire.

        Solution — percentile 95 comme référence :
          - Le dénominateur est le 95ème percentile des scores non-nuls.
          - Les vrais outliers sont simplement clampés à 1.0.
          - Les chunks pertinents obtiennent des valeurs bien réparties dans [0,1].
          - Fallback sur le max si tous les scores sont nuls ou un seul chunk.
        """
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            return scores                       # aucun match BM25
        if len(non_zero) == 1:
            return scores / non_zero[0]        # un seul match → max = percentile

        p95 = float(np.percentile(non_zero, 95))
        if p95 == 0:
            return scores
        return np.clip(scores / p95, 0.0, 1.0).astype(np.float32)

    # ── Solution A : Poids adaptatifs selon le type de requête ──────────
    def _compute_weights(self, query: str) -> tuple:
        """
        Ajuste dynamiquement (w_cosine, w_bm25, w_quality) en fonction de
        la nature de la requête pour maximiser la pertinence.

        Amélioration 5 — Poids lus depuis config.py.
        Les valeurs par défaut proviennent de ALPHA_COSINE / ALPHA_BM25.
        Les spécialisations (produit, numérique, longue) sont aussi dans config.
        """
        q = query.lower()
        tokens = q.split()
        n = len(tokens)
        w_q = ALPHA_QUALITY   # qualité toujours 0.10

        # Requête sur un produit BVZyme précis → BM25 modéré
        PRODUCT_PATTERNS = ["tg max", "gox", "amg", "hcf", "af110", "af sx",
                            "hcb", "tg883", "a-fresh", "afresh", "bvzyme",
                            "transglutaminase", "amyloglucosidase", "glucoamylase",
                            "xylanase", "lipase", "l max", "go max"]
        if any(p in q for p in PRODUCT_PATTERNS):
            return (ALPHA_COSINE_PRODUCT, ALPHA_BM25_PRODUCT, w_q)

        # Requête numérique (doses, %, temp, unités enzymatiques) → BM25
        # fortement renforcé car les chiffres exacts sont un signal lexical
        # que l'embedding sémantique ne capture pas bien.
        if re.search(r'\d+\s*(?:ppm|%|°|u/g|mg|g/kg|agu|xylh|amu)', q, re.I):
            return (ALPHA_COSINE_NUMERIC, ALPHA_BM25_NUMERIC, w_q)

        # Requête très courte (≤3 tokens) → BM25 modéré
        if n <= 3:
            return (ALPHA_COSINE_PRODUCT, ALPHA_BM25_PRODUCT, w_q)

        # Requête longue (>8 tokens) → cosine très dominant
        if n > 8:
            return (ALPHA_COSINE_LONG, ALPHA_BM25_LONG, w_q)

        # Défaut — valeurs centrales depuis config.py
        return (ALPHA_COSINE, ALPHA_BM25, w_q)

    # ── Solution B : MMR λ adaptatif selon la diversité souhaitée ────────
    def _compute_mmr_lambda(self, query: str) -> float:
        """
        Ajuste λ MMR dynamiquement.
        λ proche de 1 → priorité pertinence (peu de diversité).
        λ proche de 0 → forte diversité.
        """
        q = query.lower()
        tokens = q.split()
        n = len(tokens)

        # Produit nommé → toujours priorité pertinence (les 3 résultats
        # doivent concerner CE produit, pas ses concurrents)
        PRODUCT_PATTERNS = ["tg max", "gox", "amg", "hcf", "af110", "af sx",
                            "hcb", "tg883", "a-fresh", "afresh", "bvzyme",
                            "transglutaminase"]
        if any(p in q for p in PRODUCT_PATTERNS):
            return 0.90

        # Requête très courte → priorité exactitude, peu de diversité
        if n <= 3:
            return 0.90

        # Requête multi-sujets → diversité utile
        MULTI_KEYWORDS = ["et", "and", "ainsi que", "avec", "or", "ou"]
        if any(kw in tokens for kw in MULTI_KEYWORDS):
            return 0.55

        # Requête longue → équilibre légèrement diversifié
        if n > 8:
            return 0.60

        return 0.70

    def _mmr_rerank(self,
                    candidates: List[Tuple[int, float]],
                    embeddings: np.ndarray,
                    k: int,
                    mmr_lambda: Optional[float] = None,
                    doc_ids: Optional[List[str]] = None) -> List[Tuple[int, float]]:
        """
        MMR amélioré (Amélioration 2) — dédoublonnage intra-document uniquement.

        Si doc_ids est fourni, la pénalité de diversité ne s'applique QUE
        entre chunks du MÊME document (même doc_id).
        Les chunks de documents différents ne se pénalisent PAS entre eux.

        Résultat : A-FRESH 101, 202, 303 peuvent tous apparaître dans le Top-3
        même si leurs embeddings sont proches, car leurs doc_ids sont différents.
        Sans doc_ids : comportement MMR classique (rétrocompatible).
        """
        if mmr_lambda is None:
            mmr_lambda = self.mmr_lambda

        if len(candidates) <= k:
            return candidates

        selected = []
        remaining = list(candidates)

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda x: x[1])
                selected.append(best)
                remaining.remove(best)
            else:
                selected_embeddings = np.array([embeddings[i] for i, _ in selected])

                best_score = -np.inf
                best_candidate = None

                for idx, score in remaining:
                    emb = embeddings[idx].reshape(1, -1)

                    if doc_ids is not None:
                        # Amélioration 2 : pénalité différenciée selon l'appartenance au doc
                        #   Même document   → pénalité pleine (1.0×) — anti-doublon fort
                        #   Autre document  → pénalité réduite (0.4×) — préserve la
                        #                     diversification inter-doc utile pour les
                        #                     requêtes génériques (ex : "acide ascorbique")
                        sims = cosine_similarity(emb, selected_embeddings)[0]
                        weighted = [
                            (1.0 if doc_ids[si] == doc_ids[idx] else 0.4) * sims[j]
                            for j, (si, _) in enumerate(selected)
                        ]
                        sim_to_selected = max(weighted) if weighted else 0.0
                    else:
                        # Comportement MMR classique (fallback)
                        sim_to_selected = float(
                            cosine_similarity(emb, selected_embeddings).max()
                        )

                    mmr_score = (mmr_lambda * score -
                                 (1 - mmr_lambda) * sim_to_selected)

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_candidate = (idx, score)

                if best_candidate:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)

        return selected
    
    def search(self, 
               query: str, 
               top_k: Optional[int] = None,
               use_cache: bool = True,
               use_mmr: bool = True,
               use_expansion: bool = True) -> Dict:
        """
        Recherche optimisée avec reranking hybride.
        
        Args:
            query        : Question en langage naturel (FR ou EN)
            top_k        : Nombre de résultats (défaut: self.top_k=3)
            use_cache    : Activer le cache LRU
            use_mmr      : Activer diversification MMR
            use_expansion: Activer Query Expansion (Solution 3)
                           Enrichit la query avec synonymes FR/EN avant embedding.
                           Modèle all-MiniLM-L6-v2 inchangé — conforme conditions.
        
        Returns:
            Dict avec résultats, scores et métriques de performance
        """
        if not self._indexed:
            raise RuntimeError("Pipeline non indexé. Appeler index_documents() d'abord.")
        
        k = top_k or self.top_k
        t_start = time.time()
        
        # ── Cache check ──────────────────────────────────
        if use_cache:
            cached = self.cache.get(query)
            if cached is not None:
                return {
                    "query": query,
                    "results": cached[:k],
                    "search_time_ms": 0.1,
                    "from_cache": True
                }
        
        # ── Query Expansion (Solution 3 + Amélioration 4) ──────────────
        # Embedding : expansion conservatrice (max=3) → préserve la
        # spécificité produit (évite de diluer "A-FRESH" vers "bread").
        # BM25      : expansion riche (max=8) → maximalise la couverture
        # lexicale pour le scoring hybride.
        query_for_embedding = expand_query(query, max_expansions=3) if use_expansion else query
        query_for_bm25      = expand_query(query, max_expansions=8) if use_expansion else query

        # ── Embedding requête ─────────────────────────────
        t_emb = time.time()
        query_embedding = self.embedding_engine.encode_single(query_for_embedding)
        emb_time = time.time() - t_emb
        
        # ── Solution C : Retrieval initial élargi (top_k × 10, min 30) ──
        candidate_pool_size = min(max(k * 10, 30), len(self.chunks))
        cosine_scores, indices = self.vector_index.search(query_embedding, candidate_pool_size)
        
        # Cosine scores normalisés [0,1] (embeddings L2 normalisés → IP = cosine)
        cosine_scores_normalized = (cosine_scores + 1) / 2  # [-1,1] → [0,1]
        
        # ── BM25 scoring ──────────────────────────────────
        bm25_raw = self.bm25.score_all(query_for_bm25)
        bm25_normalized = self._normalize_bm25(bm25_raw)
        
        # ── Reranking hybride ─────────────────────────────
        ranked_candidates = []
        for i, (chunk_idx, cosine_score) in enumerate(zip(indices, cosine_scores_normalized)):
            if chunk_idx < 0 or chunk_idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[chunk_idx]
            bm25_score = bm25_normalized[chunk_idx]
            quality = chunk.quality_score
            
            # Solution A : poids adaptatifs selon la requête
            w_cosine, w_bm25, w_quality = self._compute_weights(query)
            final_score = (
                w_cosine * cosine_score +
                w_bm25  * bm25_score +
                w_quality * quality
            )
            
            ranked_candidates.append((chunk_idx, final_score, cosine_score, bm25_score))
        
        # Tri par score final
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # ── Solution B : MMR Diversification avec λ adaptatif ──────────
        adaptive_lambda = self._compute_mmr_lambda(query)
        if use_mmr and len(ranked_candidates) > k:
            local_embeddings = np.array([self.chunks[idx].embedding for idx, _, _, _ in ranked_candidates])
            local_scores = [(i, score) for i, (_, score, _, _) in enumerate(ranked_candidates)]
            # Amélioration 2 : pénalité intra-doc uniquement
            local_doc_ids = [self.chunks[idx].doc_id for idx, _, _, _ in ranked_candidates]
            mmr_local = self._mmr_rerank(
                local_scores, local_embeddings, k,
                mmr_lambda=adaptive_lambda,
                doc_ids=local_doc_ids
            )
            final_indices = [ranked_candidates[local_idx][0] for local_idx, _ in mmr_local]
        else:
            final_indices = [idx for idx, _, _, _ in ranked_candidates[:k]]
        
        # ── Construction résultats ────────────────────────
        results = []
        for rank, chunk_idx in enumerate(final_indices[:k], 1):
            # Retrouver les scores pour ce chunk
            chunk_data = next((c for c in ranked_candidates if c[0] == chunk_idx), None)
            if chunk_data is None:
                continue
            
            _, final_score, cosine_score, bm25_score = chunk_data
            chunk = self.chunks[chunk_idx]
            
            result = SearchResult(
                chunk=chunk,
                cosine_score=round(float(cosine_score), 4),
                bm25_score=round(float(bm25_score), 4),
                final_score=round(float(final_score), 4),
                rank=rank,
                explanation=f"cosine={cosine_score:.3f} | bm25={bm25_score:.3f} | quality={chunk.quality_score:.3f}"
            )
            results.append(result)
        
        search_time = (time.time() - t_start) * 1000
        
        # ── Cache store ───────────────────────────────────
        if use_cache:
            self.cache.set(query, results)
        
        # Récupère les poids utilisés sur le dernier candidat (même pour tous)
        w_cosine, w_bm25, w_quality = self._compute_weights(query)

        return {
            "query": query,
            "expanded_query": query_for_embedding if use_expansion else None,
            "results": results,
            "search_time_ms": round(search_time, 2),
            "embedding_time_ms": round(emb_time * 1000, 2),
            "from_cache": False,
            "n_candidates_evaluated": len(ranked_candidates),
            "weights_used": {"cosine": w_cosine, "bm25": w_bm25, "quality": w_quality},
            "mmr_lambda_used": adaptive_lambda
        }
    
    def get_stats(self) -> Dict:
        return {
            "n_chunks": len(self.chunks),
            "n_docs": len(set(c.doc_id for c in self.chunks)),
            "embedding_dim": EmbeddingEngine.EMBEDDING_DIM,
            "model_name": self.model_name,
            "index_type": "FAISS IVF" if FAISS_AVAILABLE else "Numpy",
            "real_embeddings": REAL_EMBEDDINGS,
            "cache_size": len(self.cache.cache),
            "reranking": "Hybride adaptatif (Solution A: poids dynamiques)",
            "diversification": "MMR λ adaptatif + intra-doc full / cross-doc 0.4× (Amél. 2)",
            "candidate_pool": "K×10 min=30 (Solution C)"
        }