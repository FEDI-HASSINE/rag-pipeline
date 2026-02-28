"""
tests/test_pipeline.py
----------------------
Suite de tests pytest pour le pipeline RAG BVZyme.

Lancement :
    pytest tests/ -v
    pytest tests/ -v --tb=short      # traces courtes
    pytest tests/test_pipeline.py::test_chunking -v   # test unique
"""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")

from rag_pipeline import AdaptiveChunker, EmbeddingEngine, OptimizedRAGPipeline


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

class TestChunking:
    """Tests du module AdaptiveChunker."""

    def test_chunks_not_empty(self, chunker, sample_doc):
        """chunk_document() doit retourner au moins un chunk non vide."""
        chunks = chunker.chunk_document(sample_doc)
        assert len(chunks) > 0, "Aucun chunk généré pour un document valide."

    def test_chunk_text_not_empty(self, chunker, sample_doc):
        """Chaque chunk doit avoir un texte non vide."""
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert c.text.strip(), f"Chunk vide détecté : {c.chunk_id}"

    def test_word_count_positive(self, chunker, sample_doc):
        """word_count doit être > 0 pour chaque chunk."""
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert c.word_count > 0, f"word_count=0 pour chunk {c.chunk_id}"

    def test_word_count_consistent(self, chunker, sample_doc):
        """word_count doit correspondre au nombre réel de mots dans le texte."""
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            real_count = len(c.text.split())
            assert c.word_count == real_count, (
                f"Incohérence word_count : déclaré={c.word_count}, réel={real_count}"
            )

    def test_quality_score_range(self, chunker, sample_doc):
        """quality_score doit être compris entre 0 et 1."""
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert 0.0 <= c.quality_score <= 1.0, (
                f"quality_score hors plage [0,1] : {c.quality_score} pour {c.chunk_id}"
            )

    def test_chunk_max_size_respected(self, chunker, sample_doc):
        """Aucun chunk ne doit dépasser max_chunk_size (avec tolérance d'une phrase)."""
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            # On accepte un léger dépassement dû à la dernière phrase
            assert c.word_count <= chunker.max_chunk_size * 1.5, (
                f"Chunk trop grand : {c.word_count} mots (max={chunker.max_chunk_size})"
            )

    def test_chunk_ids_unique(self, chunker, sample_doc):
        """Les chunk_id doivent être uniques pour un même document."""
        chunks = chunker.chunk_document(sample_doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Des chunk_id dupliqués ont été détectés."

    def test_chunk_inherits_doc_metadata(self, chunker, sample_doc):
        """Chaque chunk doit hériter de l'id et du titre du document parent."""
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert c.doc_id    == sample_doc["id"],    "doc_id incorrect."
            assert c.doc_title == sample_doc["title"], "doc_title incorrect."


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddings:
    """Tests du module EmbeddingEngine."""

    def test_embedding_dimension(self, embedding_engine):
        """L'embedding d'une phrase doit avoir exactement 384 dimensions."""
        emb = embedding_engine.encode_single("test de dimension")
        assert emb.shape == (384,), (
            f"Dimension inattendue : {emb.shape}, attendu (384,)"
        )

    def test_embedding_l2_norm(self, embedding_engine):
        """L'embedding doit être normalisé L2 (norme ≈ 1.0)."""
        emb = embedding_engine.encode_single("enzyme boulangerie xylanase")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-4, (
            f"Norme L2 inattendue : {norm:.6f} (attendu ≈ 1.0)"
        )

    def test_embedding_batch_shape(self, embedding_engine):
        """encode() en batch doit retourner (n, 384)."""
        texts = ["phrase un", "phrase deux", "phrase trois"]
        embs = embedding_engine.encode(texts)
        assert embs.shape == (3, 384), (
            f"Shape batch inattendue : {embs.shape}"
        )

    def test_embedding_batch_normalized(self, embedding_engine):
        """Tous les embeddings batch doivent être normalisés."""
        texts = ["alpha-amylase", "xylanase", "glucose oxidase", "lipase"]
        embs = embedding_engine.encode(texts)
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), (
            f"Embeddings batch non normalisés : {norms}"
        )

    def test_embedding_dtype(self, embedding_engine):
        """Les embeddings doivent être en float32."""
        emb = embedding_engine.encode_single("test dtype")
        assert emb.dtype == np.float32, (
            f"dtype inattendu : {emb.dtype} (attendu float32)"
        )

    def test_embedding_different_texts(self, embedding_engine):
        """Deux textes différents doivent produire des embeddings différents."""
        e1 = embedding_engine.encode_single("xylanase boulangerie")
        e2 = embedding_engine.encode_single("conservation produit frais sec")
        assert not np.allclose(e1, e2), "Deux textes différents ont le même embedding."


# ─────────────────────────────────────────────────────────────────────────────
# 3. COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineSimilarity:
    """Tests de la qualité sémantique des embeddings."""

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity entre deux vecteurs normalisés."""
        return float(np.dot(a, b))  # déjà normalisés L2

    def test_similar_sentences_high_score(self, embedding_engine):
        """Une question et sa réponse directe doivent scorer > 0.8."""
        q   = embedding_engine.encode_single("quelle est la dose de xylanase ?")
        ans = embedding_engine.encode_single("la dose recommandée de xylanase est de 5 à 30 ppm")
        score = self._cosine(q, ans)
        assert score > 0.8, (
            f"Score trop faible pour phrases similaires : {score:.4f} (attendu > 0.8)"
        )

    def test_unrelated_sentences_low_score(self, embedding_engine):
        """Deux phrases sans rapport sémantique doivent scorer < 0.5."""
        e1 = embedding_engine.encode_single("la xylanase améliore le volume du pain")
        e2 = embedding_engine.encode_single("zéro un zéro un bit octets bits binaire")
        score = self._cosine(e1, e2)
        assert score < 0.5, (
            f"Score trop élevé pour phrases sans rapport : {score:.4f} (attendu < 0.5)"
        )

    def test_symmetry(self, embedding_engine):
        """La similarité cosine doit être symétrique."""
        e1 = embedding_engine.encode_single("enzyme fongique")
        e2 = embedding_engine.encode_single("champignon enzymes")
        assert abs(self._cosine(e1, e2) - self._cosine(e2, e1)) < 1e-6

    def test_self_similarity_is_one(self, embedding_engine):
        """La similarité d'un texte avec lui-même doit être ≈ 1.0."""
        e = embedding_engine.encode_single("alpha-amylase fongique dose ppm")
        score = self._cosine(e, e)
        assert abs(score - 1.0) < 1e-4, f"Auto-similarité != 1 : {score}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. RECHERCHE — Top-K et tri
# ─────────────────────────────────────────────────────────────────────────────

class TestSearch:
    """Tests de la méthode search() du pipeline complet."""

    def test_returns_top3(self, indexed_pipeline):
        """search() doit retourner exactement 3 résultats par défaut."""
        resp = indexed_pipeline.search("dose xylanase boulangerie", use_cache=False)
        assert len(resp["results"]) == 3, (
            f"Nombre de résultats : {len(resp['results'])} (attendu 3)"
        )

    def test_results_sorted_descending(self, indexed_pipeline):
        """Les résultats doivent être triés par final_score décroissant."""
        resp = indexed_pipeline.search("température optimale enzyme", use_cache=False)
        scores = [r.final_score for r in resp["results"]]
        assert scores == sorted(scores, reverse=True), (
            f"Résultats non triés par score : {scores}"
        )

    def test_ranks_sequential(self, indexed_pipeline):
        """Les rangs doivent être 1, 2, 3 sans saut."""
        resp = indexed_pipeline.search("pH optimal lipase", use_cache=False)
        ranks = [r.rank for r in resp["results"]]
        assert ranks == list(range(1, len(ranks) + 1)), (
            f"Rangs non séquentiels : {ranks}"
        )

    def test_scores_in_valid_range(self, indexed_pipeline):
        """Tous les final_score doivent être dans [0, 1]."""
        resp = indexed_pipeline.search("conservation stockage produit", use_cache=False)
        for r in resp["results"]:
            assert 0.0 <= r.final_score <= 1.0, (
                f"final_score hors plage : {r.final_score}"
            )

    def test_result_has_required_fields(self, indexed_pipeline):
        """Chaque résultat doit avoir tous les champs attendus."""
        resp = indexed_pipeline.search("transglutaminase", use_cache=False)
        for r in resp["results"]:
            assert r.chunk.doc_title, "doc_title vide"
            assert r.chunk.text.strip(),  "text vide"
            assert 0.0 <= r.cosine_score <= 1.0
            assert 0.0 <= r.bm25_score   <= 1.0
            assert 0.0 <= r.final_score  <= 1.0

    def test_custom_top_k(self, indexed_pipeline):
        """top_k=1 doit retourner exactement 1 résultat."""
        resp = indexed_pipeline.search("glucose oxidase", top_k=1, use_cache=False)
        assert len(resp["results"]) == 1, (
            f"Attendu 1 résultat, obtenu {len(resp['results'])}"
        )

    def test_response_has_timing(self, indexed_pipeline):
        """La réponse doit contenir search_time_ms."""
        resp = indexed_pipeline.search("amylase", use_cache=False)
        assert "search_time_ms" in resp
        assert resp["search_time_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. CACHE LRU
# ─────────────────────────────────────────────────────────────────────────────

class TestCache:
    """Tests du cache LRU de requêtes."""

    QUERY = "test cache xylanase dose ppm"   # requête unique pour ces tests

    def test_second_call_from_cache(self, indexed_pipeline):
        """Le 2ème appel identique doit retourner from_cache=True."""
        # Vider le cache pour avoir un état propre
        indexed_pipeline.cache.cache.clear()
        indexed_pipeline.cache.order.clear()

        # 1er appel → calcul réel
        r1 = indexed_pipeline.search(self.QUERY, use_cache=True)
        assert not r1["from_cache"], "Le 1er appel ne devrait pas venir du cache."

        # 2ème appel → doit venir du cache
        r2 = indexed_pipeline.search(self.QUERY, use_cache=True)
        assert r2["from_cache"], "Le 2ème appel identique devrait venir du cache."

    def test_cache_faster_than_compute(self, indexed_pipeline):
        """Le temps du 2ème appel (cache) doit être < temps du 1er appel (calcul)."""
        indexed_pipeline.cache.cache.clear()
        indexed_pipeline.cache.order.clear()

        t0 = time.perf_counter()
        indexed_pipeline.search(self.QUERY, use_cache=True)
        t_compute = time.perf_counter() - t0

        t0 = time.perf_counter()
        indexed_pipeline.search(self.QUERY, use_cache=True)
        t_cache = time.perf_counter() - t0

        assert t_cache < t_compute, (
            f"Le cache ({t_cache*1000:.2f}ms) n'est pas plus rapide "
            f"que le calcul ({t_compute*1000:.2f}ms)."
        )

    def test_cache_disabled(self, indexed_pipeline):
        """use_cache=False ne doit pas consulter ni mettre en cache."""
        indexed_pipeline.cache.cache.clear()
        indexed_pipeline.cache.order.clear()

        r = indexed_pipeline.search(self.QUERY, use_cache=False)
        assert not r["from_cache"]
        # Le cache ne doit pas avoir été rempli
        assert len(indexed_pipeline.cache.cache) == 0, (
            "Le cache a été rempli alors que use_cache=False."
        )

    def test_cache_same_results(self, indexed_pipeline):
        """Les résultats depuis le cache doivent être identiques au calcul direct."""
        indexed_pipeline.cache.cache.clear()
        indexed_pipeline.cache.order.clear()

        r1 = indexed_pipeline.search(self.QUERY, use_cache=True)
        r2 = indexed_pipeline.search(self.QUERY, use_cache=True)

        scores_1 = [r.final_score for r in r1["results"]]
        scores_2 = [r.final_score for r in r2["results"]]
        assert scores_1 == scores_2, (
            f"Résultats différents entre calcul et cache :\n{scores_1}\n{scores_2}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUALITÉ DE RETRIEVAL — Precision@3 & scoring
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_ids", [
    (
        "What is the dosage of TG MAX63?",
        ["bvzyme_tg_max63_tds"],
    ),
    (
        "Which product for glucose oxidase gluten?",
        ["bvzyme_gox_110_tds_1"],
    ),
    (
        "A-FRESH anti-staling enzyme?",
        ["bvzyme_tds_a_fresh101", "bvzyme_tds_a_fresh202", "bvzyme_tds_a_fresh303"],
    ),
])
def test_precision_at_3(indexed_pipeline, question, expected_ids):
    """Le document attendu doit figurer dans le top-3."""
    resp = indexed_pipeline.search(question, use_cache=False)
    returned = [r.chunk.doc_id for r in resp["results"]]
    assert any(d in expected_ids for d in returned), (
        f"Top-3 retourné : {returned}\nAttendu parmi : {expected_ids}"
    )


def test_bm25_normalization_range(indexed_pipeline):
    """_normalize_bm25() doit produire des scores dans [0.0, 1.0]."""
    scores = np.array([0.5, 2.0, 10.0, 0.1, 3.0])
    norm = indexed_pipeline._normalize_bm25(scores)
    assert norm.min() >= 0.0, f"Score normalisé < 0 : {norm.min()}"
    assert norm.max() <= 1.0, f"Score normalisé > 1 : {norm.max()}"


def test_search_scores_ordered(indexed_pipeline):
    """Les final_score retournés doivent être triés par ordre décroissant."""
    resp = indexed_pipeline.search("xylanase activity", use_cache=False)
    scores = [r.final_score for r in resp["results"]]
    assert scores == sorted(scores, reverse=True), (
        f"Scores non triés décroissant : {scores}"
    )
