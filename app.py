"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   STEP 8 — Interface Streamlit                                               ║
║   RAG Semantic Search Engine — all-MiniLM-L6-v2                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Lancement :
    streamlit run app.py
"""

import os
import time
import sys
from pathlib import Path

# ── Offline mode (avoid re-downloading model) ─────────────────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import streamlit as st
import pandas as pd

# ── Page config (doit être la PREMIÈRE commande Streamlit) ────────────────────
st.set_page_config(
    page_title="RAG Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# IMPORTS LOCAUX
# ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_documents
from rag_pipeline import OptimizedRAGPipeline
from baseline_pipeline import BaselineRAGPipeline
from config import DATA_FOLDER as DATA_PATH

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
MAX_HISTORY = 10

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[str] = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = ""

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DU PIPELINE (cache_resource = 1 seule fois)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Chargement du pipeline RAG...")
def load_pipeline(force_reindex: bool = False):
    """Initialise et indexe le pipeline optimisé (chargé une seule fois)."""
    t0 = time.time()
    docs = load_documents(DATA_PATH)
    pipeline = OptimizedRAGPipeline(top_k=3)
    pipeline.index_documents(docs, force_reindex=force_reindex)
    elapsed = round(time.time() - t0, 2)
    return pipeline, docs, elapsed


@st.cache_resource(show_spinner="⏳ Chargement du pipeline Baseline...")
def load_baseline_pipeline(_docs: list):
    """Initialise le pipeline baseline TF-IDF (réutilise les docs déjà chargés)."""
    # Keep only docs that have non-empty content after boilerplate removal
    valid_docs = [d for d in _docs if d.get("content", "").strip()]
    if not valid_docs:
        raise RuntimeError("Aucun document valide pour le pipeline baseline (contenu vide).")
    pipeline = BaselineRAGPipeline(top_k=3)
    pipeline.index_documents(valid_docs)
    return pipeline


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    """Retourne un emoji couleur selon le score."""
    if score >= 0.70:
        return "🟢"
    elif score >= 0.50:
        return "🟡"
    else:
        return "🔴"


def extract_excerpt(text: str, query: str, max_len: int = 300) -> str:
    """Extrait la phrase la plus pertinente vis-à-vis de la requête."""
    query_words = set(query.lower().split())
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text[:max_len]
    best, best_score = sentences[0], -1
    for sent in sentences:
        overlap = sum(1 for w in sent.lower().split() if w in query_words)
        if overlap > best_score:
            best_score, best = overlap, sent
    return best[:max_len] + ("..." if len(best) > max_len else "")


def render_result_card(result, query: str, rank: int):
    """Affiche une carte de résultat pour le pipeline optimisé."""
    score = result.final_score
    color = score_color(score)
    stars = "⭐" * rank if rank <= 3 else f"#{rank}"

    with st.container():
        st.markdown(
            f"### {color} #{rank} {stars} &nbsp; `{result.chunk.doc_title}`"
        )
        col_score, col_meta = st.columns([2, 3])
        with col_score:
            st.metric("Score final", f"{score:.4f}")
            st.progress(min(score, 1.0), text=f"{score*100:.1f}%")
        with col_meta:
            st.markdown(
                f"**Cosine:** `{result.cosine_score:.4f}` &nbsp;|&nbsp; "
                f"**BM25:** `{result.bm25_score:.4f}`"
            )
            st.markdown(
                f"**Qualité chunk:** `{result.chunk.quality_score:.3f}` &nbsp;|&nbsp; "
                f"**Mots:** `{result.chunk.word_count}`"
            )

        excerpt = extract_excerpt(result.chunk.text, query)
        st.info(f'📄 *"{excerpt}"*')

        with st.expander("📖 Texte complet du fragment"):
            st.text(result.chunk.text)

        st.divider()


def render_baseline_card(result, rank: int):
    """Affiche une carte de résultat pour le pipeline baseline."""
    score = result.score
    color = score_color(score)
    with st.container():
        st.markdown(f"**{color} #{rank} `{result.doc_title}`**")
        st.progress(min(score, 1.0), text=f"{score*100:.1f}%")
        st.caption(f"Score cosine : {score:.4f}")
        st.text(result.text[:200] + "...")
        st.divider()


def add_to_history(query: str):
    """Ajoute la requête à l'historique (dédupliqué, max 10)."""
    if query and query not in st.session_state.history:
        st.session_state.history.insert(0, query)
        st.session_state.history = st.session_state.history[:MAX_HISTORY]


# ─────────────────────────────────────────────────────────────
# CHARGEMENT INITIAL
# ─────────────────────────────────────────────────────────────
pipeline, documents, startup_time = load_pipeline()
stats = pipeline.get_stats()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("🔍 RAG Semantic Search Engine")
    st.caption("Powered by **all-MiniLM-L6-v2** · Hybrid Cosine + BM25 + MMR · FAISS Index")
with col_badge:
    if pipeline._indexed:
        st.success("Index chargé ✅")
    else:
        st.error("❌ Non initialisé")

st.divider()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres de recherche")

    top_k = st.slider("Top-K résultats", min_value=1, max_value=10, value=3, step=1)

    mmr_lambda = st.slider(
        "MMR Lambda",
        min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        help="1.0 = pur relevance · 0.0 = pur diversité"
    )
    alpha_cosine = st.slider(
        "Poids Cosine",
        min_value=0.0, max_value=1.0, value=0.65, step=0.05,
        help="Poids de la similarité cosine dans le score hybride"
    )
    alpha_bm25 = st.slider(
        "Poids BM25",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Poids du score BM25 lexical dans le score hybride"
    )

    # Appliquer les paramètres live au pipeline
    pipeline.top_k = top_k
    pipeline.mmr_lambda = mmr_lambda
    pipeline.alpha_cosine = alpha_cosine
    pipeline.alpha_bm25 = alpha_bm25
    pipeline.alpha_quality = max(0.0, round(1.0 - alpha_cosine - alpha_bm25, 4))

    st.divider()

    col_reload, col_clear = st.columns(2)
    with col_reload:
        if st.button("🔄 Recharger", use_container_width=True, help="Réindexe depuis les PDFs"):
            st.cache_resource.clear()
            st.rerun()
    with col_clear:
        if st.button("🗑️ Cache", use_container_width=True, help="Vide le cache LRU des requêtes"):
            pipeline.cache.cache.clear()
            pipeline.cache.order.clear()
            st.toast("Cache LRU vidé ✅")

    st.divider()
    st.subheader("📊 Stats index")
    st.metric("Documents", stats.get("n_docs", "—"))
    st.metric("Chunks", stats.get("n_chunks", "—"))
    st.metric("Modèle", "all-MiniLM-L6-v2" if stats.get("real_embeddings") else "TF-IDF (fallback)")
    st.metric("Index", stats.get("index_type", "—"))
    st.metric("Démarrage", f"{startup_time} s")

# ─────────────────────────────────────────────────────────────
# ZONE PRINCIPALE — ONGLETS
# ─────────────────────────────────────────────────────────────
tab_search, tab_metrics, tab_history = st.tabs([
    "🔍 Recherche",
    "📊 Métriques",
    "🕓 Historique",
])

# ═══════════════════════════════════════════════════════════════
# ONGLET 1 — RECHERCHE
# ═══════════════════════════════════════════════════════════════
with tab_search:

    # Champ de saisie
    query_input = st.text_input(
        "Posez votre question...",
        value=st.session_state.trigger_query,
        placeholder="Ex: Quelle est la dose recommandée pour la transglutaminase TG MAX63 ?",
        key="query_input",
    )
    # Reset du trigger après affectation
    if st.session_state.trigger_query:
        st.session_state.trigger_query = ""

    col_search, col_compare = st.columns([1, 2])
    with col_search:
        search_clicked = st.button("🔍 Rechercher", type="primary", use_container_width=True)
    with col_compare:
        compare_baseline = st.checkbox(
            "Comparer avec Baseline (TF-IDF)",
            value=False,
            help="Affiche les résultats du pipeline simple TF-IDF en parallèle"
        )

    # ── Lancement de la recherche ─────────────────────────
    if search_clicked:
        query = query_input.strip()
        if not query:
            st.warning("⚠️ Veuillez saisir une question avant de lancer la recherche.")
        else:
            with st.spinner("Recherche en cours..."):
                response = pipeline.search(
                    query,
                    top_k=top_k,
                    use_mmr=True,
                    use_cache=True,
                )

            st.session_state.last_results = response
            st.session_state.last_query = query
            add_to_history(query)

    # ── Affichage résultats ───────────────────────────────
    response = st.session_state.last_results
    query = st.session_state.last_query

    if response is not None and query:
        results = response["results"]
        search_time = response["search_time_ms"]
        from_cache = response.get("from_cache", False)

        if not compare_baseline:
            # ── Mode simple : résultats optimisés ────────────────────────────
            st.subheader(f"Résultats pour : *{query}*")
            cache_badge = " ⚡ (depuis cache)" if from_cache else ""
            st.caption(
                f"⏱️ {search_time:.1f} ms{cache_badge} · "
                f"{len(results)} résultat(s) · "
                f"Index: {stats.get('index_type', '?')}"
            )

            if not results:
                st.error("Aucun résultat trouvé.")
            else:
                for r in results:
                    render_result_card(r, query, r.rank)

        else:
            # ── Mode comparaison ─────────────────────────────────────────────
            st.subheader(f"Comparaison pour : *{query}*")

            baseline_pipeline = load_baseline_pipeline(documents)
            with st.spinner("Recherche baseline..."):
                t_b = time.time()
                b_response = baseline_pipeline.search(query, top_k=top_k)
                b_time = (time.time() - t_b) * 1000

            col_base, col_opt = st.columns(2)
            with col_base:
                st.markdown("### 🔵 Baseline (TF-IDF)")
                st.caption(f"⏱️ {b_response['search_time_ms']:.1f} ms")
                for r in b_response["results"]:
                    render_baseline_card(r, r.rank)

            with col_opt:
                st.markdown("### 🟢 Optimisé (all-MiniLM-L6-v2)")
                st.caption(f"⏱️ {search_time:.1f} ms")
                for r in results:
                    render_result_card(r, query, r.rank)

            st.divider()
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Temps Baseline", f"{b_response['search_time_ms']:.1f} ms")
            with col_t2:
                st.metric("Temps Optimisé", f"{search_time:.1f} ms")
            with col_t3:
                ratio = search_time / max(b_response["search_time_ms"], 0.01)
                st.metric("Ratio", f"x{ratio:.1f}", delta=None)

# ═══════════════════════════════════════════════════════════════
# ONGLET 2 — MÉTRIQUES
# ═══════════════════════════════════════════════════════════════
with tab_metrics:
    response = st.session_state.last_results
    query = st.session_state.last_query

    if response is None or not query:
        st.info("💡 Lancez une recherche dans l'onglet **🔍 Recherche** pour voir les métriques.")
    else:
        results = response["results"]
        search_time = response["search_time_ms"]

        # ── Temps de recherche en grand ───────────────────────────────────────
        st.markdown(
            f"<h2 style='text-align:center; color:#4CAF50;'>⚡ {search_time:.1f} ms</h2>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"<p style='text-align:center;'>Requête: <em>{query}</em></p>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Graphique bar chart des scores ────────────────────────────────────
        st.subheader("📈 Scores des résultats Top-K")
        if results:
            chart_data = pd.DataFrame(
                {
                    "Document": [r.chunk.doc_title[:30] for r in results],
                    "Cosine": [r.cosine_score for r in results],
                    "BM25": [r.bm25_score for r in results],
                    "Score final": [r.final_score for r in results],
                }
            ).set_index("Document")
            st.bar_chart(chart_data)

            st.divider()

            # ── Tableau pandas ─────────────────────────────────────────────────
            st.subheader("📋 Tableau de résultats")
            table_data = pd.DataFrame(
                [
                    {
                        "Rang": r.rank,
                        "Document": r.chunk.doc_title,
                        "Cosine": r.cosine_score,
                        "BM25": r.bm25_score,
                        "Score final": r.final_score,
                        "Qualité": r.chunk.quality_score,
                        "Mots": r.chunk.word_count,
                    }
                    for r in results
                ]
            )
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            st.divider()

            # ── Détail reranking ───────────────────────────────────────────────
            st.subheader("⚙️ Paramètres de reranking actifs")
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                st.metric("α Cosine", f"{pipeline.alpha_cosine:.2f}")
            with param_col2:
                st.metric("α BM25", f"{pipeline.alpha_bm25:.2f}")
            with param_col3:
                st.metric("α Qualité", f"{pipeline.alpha_quality:.2f}")

            col_mmr, col_k, col_cache = st.columns(3)
            with col_mmr:
                st.metric("MMR λ", f"{pipeline.mmr_lambda:.1f}")
            with col_k:
                st.metric("Top-K", pipeline.top_k)
            with col_cache:
                st.metric("Cache LRU", f"{len(pipeline.cache.cache)} entrées")

        else:
            st.warning("Aucun résultat disponible.")

# ═══════════════════════════════════════════════════════════════
# ONGLET 3 — HISTORIQUE
# ═══════════════════════════════════════════════════════════════
with tab_history:
    history = st.session_state.history

    if not history:
        st.info("💡 L'historique des requêtes apparaîtra ici après votre première recherche.")
    else:
        st.subheader(f"🕓 Dernières {len(history)} requêtes")
        st.caption("Cliquez sur une requête pour la relancer.")

        for i, past_query in enumerate(history):
            col_q, col_btn = st.columns([5, 1])
            with col_q:
                st.markdown(f"**{i+1}.** {past_query}")
            with col_btn:
                if st.button("▶️", key=f"replay_{i}", help="Relancer cette requête"):
                    st.session_state.trigger_query = past_query
                    st.rerun()

        st.divider()
        if st.button("🗑️ Vider l'historique"):
            st.session_state.history = []
            st.rerun()
