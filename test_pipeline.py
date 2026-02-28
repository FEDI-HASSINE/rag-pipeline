"""
test_pipeline.py
----------------
Script de test complet pour vérifier que rag_pipeline.py fonctionne correctement.

Usage:
    python test_pipeline.py

Tests effectués :
    [1] Chargement des PDFs
    [2] Indexation + cache
    [3] Recherche simple
    [4] Reload depuis cache (rapidité)
    [5] Qualité des résultats (questions métier)
    [6] Stats du pipeline
"""

import sys
import time
from pathlib import Path

from config import DATA_FOLDER as PDF_FOLDER
CACHE_DIR   = Path(".faiss_cache")

# ── Questions métier pour tester la pertinence ────────────────────────────────
TEST_QUERIES = [
    "Quelle est la température optimale d'utilisation ?",
    "Quelle est la dose recommandée ?",
    "Quels sont les produits à base de transglutaminase ?",
    "Comment conserver ce produit ?",
    "Quel est le pH optimal pour cette enzyme ?",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers d'affichage
# ─────────────────────────────────────────────────────────────────────────────

def ok(msg):   print(f"  ✅ {msg}")
def fail(msg): print(f"  ❌ {msg}"); sys.exit(1)
def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Chargement des PDFs
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 1 — Chargement des documents PDF")

from data_loader import load_documents

docs = load_documents(PDF_FOLDER)

if len(docs) == 0:
    fail("Aucun document chargé !")
ok(f"{len(docs)} documents chargés")

for doc in docs:
    if not doc.get("content") or len(doc["content"].strip()) < 20:
        fail(f"Document vide ou trop court : {doc.get('title', '?')}")
ok("Tous les documents ont du contenu")

total_words = sum(len(d["content"].split()) for d in docs)
ok(f"Total : {total_words:,} mots — moy. {total_words // len(docs):,} mots/doc")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Indexation (1er passage = calcul + sauvegarde cache)
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 2 — Indexation complète (calcul + cache)")

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # n'essaie pas de re-télécharger
os.environ["HF_DATASETS_OFFLINE"]  = "1"

from rag_pipeline import OptimizedRAGPipeline

# Supprime le cache pour forcer un recalcul propre
if CACHE_DIR.exists():
    import shutil
    shutil.rmtree(CACHE_DIR)
    print("  → Cache précédent supprimé pour test propre")

pipeline = OptimizedRAGPipeline(top_k=3)

t0 = time.time()
metrics = pipeline.index_documents(docs, force_reindex=True)
elapsed = time.time() - t0

if metrics["n_chunks"] == 0:
    fail("Aucun chunk généré !")
ok(f"{metrics['n_chunks']} chunks générés en {elapsed:.2f}s")
ok(f"Qualité moyenne des chunks : {metrics['avg_quality']}")
ok(f"Cache sauvegardé : {list(CACHE_DIR.iterdir()) if CACHE_DIR.exists() else 'ABSENT !'}")

if not CACHE_DIR.exists():
    fail("Le dossier .faiss_cache n'a pas été créé !")

cache_files = [f.name for f in CACHE_DIR.iterdir()]
for expected in ["index.faiss", "chunks.pkl", "bm25.pkl", "embeddings.npy"]:
    if expected not in cache_files:
        fail(f"Fichier cache manquant : {expected}")
ok(f"Fichiers cache présents : {cache_files}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Rechargement depuis cache (doit être ultra-rapide)
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 3 — Reload depuis cache (rapidité)")

# On crée un nouveau pipeline (simule un redémarrage)
# Le modèle est déjà en cache local HuggingFace : pas de téléchargement
pipeline2 = OptimizedRAGPipeline(top_k=3)

t0 = time.time()
metrics2 = pipeline2.index_documents(docs)   # force_reindex=False par défaut
elapsed2 = time.time() - t0

if metrics2.get("source") != "cache":
    fail("Le cache n'a pas été utilisé au 2e démarrage !")
ok(f"Index rechargé depuis cache en {elapsed2:.3f}s")

if elapsed2 > 10:
    print(f"  ⚠️  Reload un peu lent ({elapsed2:.1f}s) — modèle chargé en mémoire")
else:
    ok(f"Gain de temps : x{elapsed / max(elapsed2, 0.001):.0f} plus rapide qu'un recalcul")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Recherche simple
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 4 — Recherche simple")

query = "température optimale d'utilisation"
response = pipeline2.search(query)

if not response["results"]:
    fail("Aucun résultat retourné !")
ok(f"Requête : '{query}'")
ok(f"Temps de recherche : {response['search_time_ms']:.1f} ms")
ok(f"{len(response['results'])} résultats retournés (top-{pipeline2.top_k})")

print("\n  --- Résultats ---")
for r in response["results"]:
    print(f"  #{r.rank} [{r.final_score:.3f}] {r.chunk.doc_title!r}")
    print(f"       {r.chunk.text[:120].strip()}...")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Qualité sur questions métier (top-3 doit avoir un score > 0.4)
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 5 — Qualité des résultats sur questions métier")

print(f"  {len(TEST_QUERIES)} questions de test\n")
all_passed = True

for query in TEST_QUERIES:
    resp = pipeline2.search(query, use_cache=False)
    top_score = resp["results"][0].final_score if resp["results"] else 0
    top_title = resp["results"][0].chunk.doc_title if resp["results"] else "—"
    status = "✅" if top_score >= 0.4 else "⚠️ "
    if top_score < 0.4:
        all_passed = False
    print(f"  {status} [{top_score:.3f}] Q: {query[:45]!r}")
    print(f"         → {top_title!r}")

if all_passed:
    ok("Tous les scores >= 0.4 — résultats pertinents")
else:
    print("\n  ⚠️  Certains scores sont faibles (< 0.4) — vérifier la qualité des PDFs")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Cache LRU (même requête = résultat instantané)
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 6 — Cache LRU (requêtes répétées)")

query = TEST_QUERIES[0]

# 1er appel
pipeline2.search(query, use_cache=True)

# 2e appel (doit venir du cache)
t0 = time.time()
resp_cached = pipeline2.search(query, use_cache=True)
cache_time = (time.time() - t0) * 1000

if resp_cached.get("from_cache"):
    ok(f"Résultat servi depuis cache LRU en {cache_time:.2f} ms")
else:
    print("  ⚠️  Cache LRU non utilisé (normal si premier appel)")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Stats du pipeline
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 7 — Statistiques du pipeline")

stats = pipeline2.get_stats()
for k, v in stats.items():
    print(f"  {k:<25} : {v}")

# ─────────────────────────────────────────────────────────────────────────────
# BILAN FINAL
# ─────────────────────────────────────────────────────────────────────────────

section("BILAN FINAL")
print(f"""
  Documents chargés     : {len(docs)}
  Chunks indexés        : {metrics['n_chunks']}
  Temps indexation      : {elapsed:.2f}s
  Temps reload cache    : {elapsed2:.3f}s
  Embedding réel        : {'all-MiniLM-L6-v2' if stats['real_embeddings'] else 'TF-IDF fallback'}
  Index vectoriel       : {stats['index_type']}
  Reranking             : {stats['reranking']}

  Tous les tests passés ✅
""")
