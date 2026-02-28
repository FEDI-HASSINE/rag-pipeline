"""
optimize_weights.py — Amélioration 5
--------------------------------------
Grid search sur les poids hybrides (cosine, bm25, quality) du pipeline RAG.

Principe :
  - Remplace temporairement _compute_weights() par une version retournant des
    poids fixes pendant la recherche → mesure l'impact pur de chaque combinaison.
  - Utilise compute_metrics() de evaluate.py (Precision@1, P@3, MRR).
  - Imprime le classement des 15 meilleures combinaisons.
  - La fonction composite score = 0.5*MRR + 0.3*P@1 + 0.2*P@3 équilibre rang
    et couverture.

Usage :
    .venv\\Scripts\\python.exe optimize_weights.py
"""

from __future__ import annotations

import numpy as np

from evaluate import compute_metrics, EVAL_SET, DATA_FOLDER
from data_loader import load_documents
from rag_pipeline import OptimizedRAGPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données et indexation (une seule fois)
# ─────────────────────────────────────────────────────────────────────────────

print("Chargement des documents...")
docs = load_documents(DATA_FOLDER)

print("Indexation (depuis cache si dispo)...")
pipeline = OptimizedRAGPipeline()
pipeline.index_documents(docs)
print(f"  → {len(pipeline.chunks)} chunks indexés\n")


# ─────────────────────────────────────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────────────────────────────────────

STEP = 0.05   # granularité de la grille

results: list[dict] = []
n_tested = 0

print("Grid search en cours...\n")

for w_cos in np.arange(0.40, 1.01, STEP):
    for w_bm25 in np.arange(0.00, 1.01 - w_cos, STEP):
        w_q = round(1.0 - round(w_cos, 2) - round(w_bm25, 2), 2)
        if w_q < 0 or w_q > 0.30:   # qualité bornée à 30% max
            continue

        w_cos_r  = round(float(w_cos),  2)
        w_bm25_r = round(float(w_bm25), 2)
        w_q_r    = round(float(w_q),    2)

        # Patch _compute_weights pour retourner des poids fixes pendant ce run
        fixed = (w_cos_r, w_bm25_r, w_q_r)
        pipeline._compute_weights = lambda q, _f=fixed: _f   # type: ignore[method-assign]

        m = compute_metrics(pipeline, "optimized", EVAL_SET)
        n_tested += 1

        composite = 0.5 * m["mrr"] + 0.3 * m["precision_at_1"] + 0.2 * m["precision_at_3"]
        results.append({
            "cosine":  w_cos_r,
            "bm25":    w_bm25_r,
            "quality": w_q_r,
            "p1":      m["precision_at_1"],
            "p3":      m["precision_at_3"],
            "mrr":     m["mrr"],
            "score":   round(composite, 4),
        })

# Restaure _compute_weights original
del pipeline._compute_weights   # type: ignore[misc]

# ─────────────────────────────────────────────────────────────────────────────
# Affichage des résultats
# ─────────────────────────────────────────────────────────────────────────────

results.sort(key=lambda x: x["score"], reverse=True)

print(f"{'cos':>6} {'bm25':>6} {'qual':>6} │ {'P@1':>5} {'P@3':>5} {'MRR':>6} │ {'score':>7}")
print("─" * 60)
for r in results[:20]:
    print(
        f"{r['cosine']:>6.2f} {r['bm25']:>6.2f} {r['quality']:>6.2f} │ "
        f"{r['p1']:>5.3f} {r['p3']:>5.3f} {r['mrr']:>6.3f} │ {r['score']:>7.4f}"
    )

best = results[0]
print(f"\n{'='*60}")
print(f"MEILLEURS POIDS : cosine={best['cosine']}  bm25={best['bm25']}  quality={best['quality']}")
print(f"  P@1={best['p1']}  P@3={best['p3']}  MRR={best['mrr']}  score={best['score']}")
print(f"Combinaisons testées : {n_tested}")
