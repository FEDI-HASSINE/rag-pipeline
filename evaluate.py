"""
evaluate.py
-----------
Évaluation objective de la qualité du pipeline RAG BVZyme.
Compare le pipeline optimisé (rag_pipeline.py) vs baseline (baseline_pipeline.py).

Métriques calculées :
  - Precision@1  : le bon document est-il en position #1 ?
  - Precision@3  : le bon document est-il dans le Top-3 ?
  - MRR          : Mean Reciprocal Rank (rang moyen du bon résultat)  - NDCG@3       : Normalized Discounted Cumulative Gain (Amélioration 6)  - Temps moyen  : latence de recherche en ms

Usage :
    python evaluate.py
    python evaluate.py --verbose    # affiche le détail de chaque requête
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from data_loader import load_documents
from rag_pipeline import OptimizedRAGPipeline
from baseline_pipeline import BaselineRAGPipeline
from config import DATA_FOLDER

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# JEU D'ÉVALUATION — 30 questions sur les vrais PDFs BVZyme  (Amélioration 6)
# expected_doc_ids : liste de doc_id acceptables (plusieurs produits similaires)
# Couverture : EN, FR, numérique, multi-attribut, hôrs corpus (test négatif)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SET = [
    # ──────────────────────────────────────
    # Q01–Q10 : jeu original (EN)
    # ──────────────────────────────────────
    {
        "id": "Q01",
        "query": "What is the dosage of transglutaminase TG MAX63 for bakery?",
        "expected_doc_ids": ["bvzyme_tg_max63_tds"],
        "note": "TG MAX63 — 5-25 ppm transglutaminase",
    },
    {
        "id": "Q02",
        "query": "Which product is glucose oxidase used to strengthen gluten network?",
        "expected_doc_ids": ["bvzyme_gox_110_tds_1", "bvzyme_go_max_63_tds", "bvzyme_go_max_65"],
        "note": "GOX 110 / GO MAX — glucose oxidase",
    },
    {
        "id": "Q03",
        "query": "What is the recommended dosage of ascorbic acid E300 for direct breadmaking?",
        "expected_doc_ids": ["acide_ascorbique"],
        "note": "Acide ascorbique — 20-60 ppm panification directe",
    },
    {
        "id": "Q04",
        "query": "Which bacterial xylanase enzyme is produced from Bacillus subtilis?",
        "expected_doc_ids": ["tds_bvzyme_hcb708", "tds_bvzyme_hcb709", "tds_bvzyme_hcb710"],
        "note": "HCB708/709/710 — xylanase bactérienne Bacillus subtilis",
    },
    {
        "id": "Q05",
        "query": "What enzyme product improves bread softness and shelf life A-SOFT?",
        "expected_doc_ids": [
            "bvzyme_tds_a_soft205", "bvzyme_tds_a_soft305_1", "bvzyme_tds_a_soft405"
        ],
        "note": "A-SOFT 205/305/405 — softener enzyme",
    },
    {
        "id": "Q06",
        "query": "What is the activity of AMG 880 amyloglucosidase enzyme?",
        "expected_doc_ids": ["bvzyme_tds_amg880", "bvzyme_tds_amg1400"],
        "note": "AMG 880 / AMG 1400 — amyloglucosidase",
    },
    {
        "id": "Q07",
        "query": "Which lipase product L MAX63 is used to improve dough extensibility?",
        "expected_doc_ids": ["bvzyme_tds_l_max63", "bvzyme_tds_l_max64", "bvzyme_tds_l_max65"],
        "note": "L MAX63/64/65 — lipase",
    },
    {
        "id": "Q08",
        "query": "What fungal alpha-amylase AF SX enzyme is used in flour treatment?",
        "expected_doc_ids": ["bvzymetdsaf_sx", "bvzyme_tds_af110", "bvzyme_tds_af220"],
        "note": "AF SX / AF 110 — amylase fongique",
    },
    {
        "id": "Q09",
        "query": "What is the xylanase activity of HCF MAX X fungal product?",
        "expected_doc_ids": [
            "tds_bvzyme_hcf_max_x", "tds_bvzyme_hcf_max63", "tds_bvzyme_hcf_max64"
        ],
        "note": "HCF MAX X — xylanase fongique haute activité 23500 XylH/g",
    },
    {
        "id": "Q10",
        "query": "Which enzyme product A-FRESH maintains bread freshness and anti-staling?",
        "expected_doc_ids": [
            "bvzyme_tds_a_fresh101", "bvzyme_tds_a_fresh202", "bvzyme_tds_a_fresh303"
        ],
        "note": "A-FRESH 101/202/303 — anti-staling enzyme",
    },

    # ──────────────────────────────────────
    # Q11–Q20 : requêtes en français (code-switching)
    # ──────────────────────────────────────
    {
        "id": "Q11",
        "query": "Quelle est la dose recommandée de xylanase HCF MAX X ?",
        "expected_doc_ids": ["tds_bvzyme_hcf_max_x", "tds_bvzyme_hcf_max63", "tds_bvzyme_hcf_max64"],
        "note": "FR — HCF MAX X dosage",
    },
    {
        "id": "Q12",
        "query": "Quelle est l'activité enzymatique de l'amylase fongique AF SX ?",
        "expected_doc_ids": ["bvzymetdsaf_sx", "bvzyme_tds_af110", "bvzyme_tds_af220"],
        "note": "FR — AF SX activité",
    },
    {
        "id": "Q13",
        "query": "Quelle transglutaminase améliore la rétention de l'eau dans la pâte ?",
        "expected_doc_ids": ["bvzyme_tg_max63_tds"],
        "note": "FR — TG MAX63 rétention d'eau",
    },
    {
        "id": "Q14",
        "query": "Quel produit enzymatique améliore la souplesse et la durabilité de la mie ?",
        "expected_doc_ids": [
            "bvzyme_tds_a_soft205", "bvzyme_tds_a_soft305_1", "bvzyme_tds_a_soft405",
            "bvzyme_tds_a_fresh101", "bvzyme_tds_a_fresh202", "bvzyme_tds_a_fresh303"
        ],
        "note": "FR — A-SOFT / A-FRESH souplesse mie",
    },
    {
        "id": "Q15",
        "query": "Quelle est la dose d'acide ascorbique E300 pour la panification directe ?",
        "expected_doc_ids": ["acide_ascorbique"],
        "note": "FR — acide ascorbique dosage",
    },
    {
        "id": "Q16",
        "query": "Quelles xylanases bactériennes sont produites par Bacillus subtilis ?",
        "expected_doc_ids": ["tds_bvzyme_hcb708", "tds_bvzyme_hcb709", "tds_bvzyme_hcb710"],
        "note": "FR — HCB xylanase bactérienne",
    },
    {
        "id": "Q17",
        "query": "Quel est le dosage recommandé de lipase L MAX63 pour la boulangerie ?",
        "expected_doc_ids": ["bvzyme_tds_l_max63", "bvzyme_tds_l_max64", "bvzyme_tds_l_max65"],
        "note": "FR — L MAX63 lipase dosage",
    },
    {
        "id": "Q18",
        "query": "Quel produit glucose oxydase renforce le réseau gluten ?",
        "expected_doc_ids": ["bvzyme_gox_110_tds_1", "bvzyme_go_max_63_tds", "bvzyme_go_max_65"],
        "note": "FR — GOX glucose oxydase gluten",
    },
    {
        "id": "Q19",
        "query": "Quelles sont les propriétés anti-rétrogradation de l'enzyme A-FRESH ?",
        "expected_doc_ids": [
            "bvzyme_tds_a_fresh101", "bvzyme_tds_a_fresh202", "bvzyme_tds_a_fresh303"
        ],
        "note": "FR — A-FRESH anti-staling",
    },
    {
        "id": "Q20",
        "query": "Quel est le niveau d'activité de l'amyloglucosidase AMG 880 ?",
        "expected_doc_ids": ["bvzyme_tds_amg880", "bvzyme_tds_amg1400"],
        "note": "FR — AMG 880 activité amyloglucosidase",
    },

    # ──────────────────────────────────────
    # Q21–Q27 : requêtes numériques et multi-attributs
    # ──────────────────────────────────────
    {
        "id": "Q21",
        "query": "Which enzyme has activity 880 AGU/g amyloglucosidase?",
        "expected_doc_ids": ["bvzyme_tds_amg880"],
        "note": "NUM — 880 AGU/g identifier",
    },
    {
        "id": "Q22",
        "query": "What is the pH range and temperature optimum for lipase L MAX63?",
        "expected_doc_ids": ["bvzyme_tds_l_max63", "bvzyme_tds_l_max64", "bvzyme_tds_l_max65"],
        "note": "MULTI — L MAX pH + temperature",
    },
    {
        "id": "Q23",
        "query": "What is the storage temperature and shelf life of BVZyme enzyme products?",
        "expected_doc_ids": [
            "bvzyme_tg_max63_tds", "bvzyme_gox_110_tds_1", "bvzymetdsaf_sx",
            "bvzyme_tds_amg880", "bvzyme_tds_l_max63"
        ],
        "note": "MULTI — storage / shelf life générique",
    },
    {
        "id": "Q24",
        "query": "What is the recommended dosage in ppm for transglutaminase in bread?",
        "expected_doc_ids": ["bvzyme_tg_max63_tds"],
        "note": "NUM — TG MAX63 ppm dosage",
    },
    {
        "id": "Q25",
        "query": "Which enzyme product has activity 23500 XylH/g xylanase?",
        "expected_doc_ids": ["tds_bvzyme_hcf_max_x"],
        "note": "NUM — 23500 XylH/g HCF MAX X",
    },
    {
        "id": "Q26",
        "query": "What is the application dosage range of glucose oxidase GOX 110?",
        "expected_doc_ids": ["bvzyme_gox_110_tds_1", "bvzyme_go_max_63_tds", "bvzyme_go_max_65"],
        "note": "NUM — GOX 110 dosage range",
    },
    {
        "id": "Q27",
        "query": "What enzyme is used for dough conditioning and improves extensibility?",
        "expected_doc_ids": [
            "bvzyme_tds_l_max63", "bvzyme_tds_l_max64", "bvzyme_tds_l_max65",
            "bvzymetdsaf_sx", "bvzyme_tds_af110"
        ],
        "note": "MULTI — dough conditioning lipase/amylase",
    },

    # ──────────────────────────────────────
    # Q28–Q30 : tests négatifs / hôrs-corpus
    # expected_doc_ids vide → aucun résultat "correct" attendu
    # (utile pour mesurer le taux de faux positifs du pipeline)
    # ──────────────────────────────────────
    {
        "id": "Q28",
        "query": "What is the price per kilogram of enzyme?",
        "expected_doc_ids": [],
        "note": "NEG — prix non mentionné dans le corpus",
    },
    {
        "id": "Q29",
        "query": "How to apply for a distributor contract with BVZyme?",
        "expected_doc_ids": [],
        "note": "NEG — hors-corpus (commercial)",
    },
    {
        "id": "Q30",
        "query": "What are the allergen declaration requirements for enzyme products?",
        "expected_doc_ids": [],
        "note": "NEG — hors-corpus (réglementation allérgènes)",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(returned_ids: list[str], expected_ids: set[str], k: int = 3) -> float:
    """
    Normalized Discounted Cumulative Gain @ k  (Amélioration 6).

    Plus nuancé que P@1 : récompense davantage un résultat correct en position 1
    qu'en position 3. Utile pour les questions à réponses multiples (Q28–Q30 sont
    des tests négatifs : expected vide → NDCG=1.0 car aucune réponse attendue).
    """
    if not expected_ids:                   # test négatif → score parfait si pipeline ne fabrique pas de réponse
        return 1.0                         # (on ne peut pas évaluer ce cas en DCG — on ignore)
    dcg  = sum(1.0 / math.log2(i + 2)
               for i, d in enumerate(returned_ids[:k])
               if d in expected_ids)
    idcg = sum(1.0 / math.log2(i + 2)
               for i in range(min(len(expected_ids), k)))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def _get_doc_ids_from_response(response: dict, pipeline_type: str) -> list[str]:
    """Extrait les doc_id depuis une réponse (optimisé ou baseline)."""
    if pipeline_type == "optimized":
        return [r.chunk.doc_id for r in response["results"]]
    else:
        return [r.doc_id for r in response["results"]]


def compute_metrics(
    pipeline: Any,
    pipeline_type: str,
    eval_set: list[dict],
    top_k: int = 3,
) -> dict:
    """
    Calcule Precision@1, Precision@3, MRR et temps moyen.

    Paramètres
    ----------
    pipeline       : OptimizedRAGPipeline ou BaselineRAGPipeline
    pipeline_type  : "optimized" | "baseline"
    eval_set       : liste de questions avec expected_doc_ids
    top_k          : nombre de résultats à considérer

    Retourne
    --------
    dict avec les métriques et le détail par question
    """
    p1_hits  = 0     # Precision@1
    p3_hits  = 0     # Precision@3
    mrr_sum  = 0.0   # Somme des 1/rang pour MRR
    ndcg_sum = 0.0   # Somme NDCG@3
    times    = []    # Temps de recherche en ms
    details  = []    # Détail par question

    for item in eval_set:
        query    = item["query"]
        expected = set(item["expected_doc_ids"])

        # Désactiver le cache pour mesure équitable
        if pipeline_type == "optimized":
            resp = pipeline.search(query, top_k=top_k, use_cache=False)
        else:
            resp = pipeline.search(query, top_k=top_k)

        times.append(resp["search_time_ms"])
        returned_ids = _get_doc_ids_from_response(resp, pipeline_type)

        # Precision@1
        hit_p1 = len(returned_ids) > 0 and returned_ids[0] in expected
        if hit_p1:
            p1_hits += 1

        # Precision@3 + MRR
        reciprocal_rank = 0.0
        hit_p3 = False
        for rank, doc_id in enumerate(returned_ids[:top_k], start=1):
            if doc_id in expected:
                hit_p3 = True
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
                break
        if hit_p3:
            p3_hits += 1
        mrr_sum += reciprocal_rank

        # NDCG@3
        ndcg = ndcg_at_k(returned_ids, expected, k=top_k)
        ndcg_sum += ndcg

        details.append({
            "id":         item["id"],
            "query":      query[:55],
            "expected":   list(expected)[:2],
            "returned":   returned_ids[:3],
            "hit_p1":     hit_p1,
            "hit_p3":     hit_p3,
            "rr":         reciprocal_rank,
            "ndcg":       ndcg,
            "time_ms":    resp["search_time_ms"],
        })

    n = len(eval_set)

    # NDCG corrigé : exclure les tests négatifs (expected_doc_ids vide)
    # pour éviter que NDCG=1.0 par défaut ne gonfle la moyenne.
    n_positive = sum(1 for item in eval_set if item["expected_doc_ids"])
    ndcg_positive_sum = sum(
        d["ndcg"] for d, item in zip(details, eval_set)
        if item["expected_doc_ids"]
    )

    return {
        "precision_at_1": round(p1_hits / n, 3),
        "precision_at_3": round(p3_hits / n, 3),
        "mrr":            round(mrr_sum / n, 3),
        "ndcg_at_3":      round(ndcg_positive_sum / max(n_positive, 1), 3),
        "avg_time_ms":    round(sum(times) / n, 2),
        "n_positive":     n_positive,
        "n_negative":     n - n_positive,
        "details":        details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────────────────────

def _print_detail_table(details_opt: list, details_base: list):
    """Affiche le détail question par question côte à côte."""
    table = Table(
        title="Détail par question",
        box=box.SIMPLE_HEAD,
        show_lines=True,
        border_style="dim",
    )
    table.add_column("ID",    style="dim",   width=4)
    table.add_column("Question",             width=32, no_wrap=True)
    table.add_column("P@1 base", justify="center", width=8)
    table.add_column("P@1 opt",  justify="center", width=8)
    table.add_column("RR base",  justify="center", width=8)
    table.add_column("RR opt",   justify="center", width=8)

    for d_opt, d_base in zip(details_opt, details_base):
        def p1_fmt(hit: bool) -> str:
            return "[green]✓[/green]" if hit else "[red]✗[/red]"

        def rr_fmt(rr: float) -> str:
            color = "green" if rr >= 0.5 else ("yellow" if rr > 0 else "red")
            return f"[{color}]{rr:.2f}[/{color}]"

        table.add_row(
            d_opt["id"],
            d_opt["query"][:32],
            p1_fmt(d_base["hit_p1"]),
            p1_fmt(d_opt["hit_p1"]),
            rr_fmt(d_base["rr"]),
            rr_fmt(d_opt["rr"]),
        )

    console.print(table)


def _print_summary_table(metrics_opt: dict, metrics_base: dict):
    """Affiche le tableau récapitulatif comparatif."""
    table = Table(
        title="Résultats de l'évaluation — Optimisé vs Baseline",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
        min_width=52,
    )
    table.add_column("Métrique",     style="cyan",  width=18)
    table.add_column("Baseline",     style="yellow", justify="center", width=14)
    table.add_column("Optimisé",     style="green",  justify="center", width=14)
    table.add_column("Δ",            style="bold",   justify="center", width=8)

    def delta(opt: float, base: float, higher_is_better: bool = True) -> str:
        diff = opt - base
        if abs(diff) < 0.001:
            return "[dim]=[/dim]"
        sign  = "+" if diff > 0 else ""
        color = "green" if (diff > 0) == higher_is_better else "red"
        return f"[{color}]{sign}{diff:.3f}[/{color}]"

    rows = [
        ("Precision@1",  metrics_base["precision_at_1"], metrics_opt["precision_at_1"], True,  "{:.1%}"),
        ("Precision@3",  metrics_base["precision_at_3"], metrics_opt["precision_at_3"], True,  "{:.1%}"),
        ("MRR",          metrics_base["mrr"],              metrics_opt["mrr"],            True,  "{:.3f}"),
        ("NDCG@3",       metrics_base["ndcg_at_3"],        metrics_opt["ndcg_at_3"],      True,  "{:.3f}"),
        ("Temps moyen",  metrics_base["avg_time_ms"],      metrics_opt["avg_time_ms"],    False, "{:.1f} ms"),
    ]

    for name, base_val, opt_val, higher_is_better, fmt in rows:
        table.add_row(
            name,
            fmt.format(base_val),
            fmt.format(opt_val),
            delta(opt_val, base_val, higher_is_better),
        )

    console.print()
    console.print(table)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────────────────────────────────────

def main(verbose: bool = False):
    console.print(Rule("[bold white]Évaluation RAG — BVZyme[/bold white]", style="white"))

    # ── Chargement des documents ──────────────────────────────────────────────
    with console.status("[cyan]Chargement des documents…[/cyan]", spinner="dots"):
        docs = load_documents(DATA_FOLDER)
    console.print(f"  [green]✓[/green] {len(docs)} documents chargés\n")

    # ── Pipeline Optimisé ─────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Pipeline OPTIMISÉ[/bold cyan]", style="cyan"))
    pipeline_opt = OptimizedRAGPipeline(top_k=3)
    pipeline_opt.index_documents(docs)

    with console.status("[cyan]Évaluation pipeline optimisé…[/cyan]", spinner="dots"):
        metrics_opt = compute_metrics(pipeline_opt, "optimized", EVAL_SET)
    console.print(f"  [green]✓[/green] Évaluation terminée ({len(EVAL_SET)} questions)\n")

    # ── Pipeline Baseline ─────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]Pipeline BASELINE[/bold yellow]", style="yellow"))
    pipeline_base = BaselineRAGPipeline(top_k=3)
    with console.status("[yellow]Indexation baseline…[/yellow]", spinner="dots"):
        pipeline_base.index_documents(docs)

    with console.status("[yellow]Évaluation pipeline baseline…[/yellow]", spinner="dots"):
        metrics_base = compute_metrics(pipeline_base, "baseline", EVAL_SET)
    console.print(f"  [green]✓[/green] Évaluation terminée ({len(EVAL_SET)} questions)\n")

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    _print_summary_table(metrics_opt, metrics_base)

    # ── Détail par question (optionnel) ───────────────────────────────────────
    if verbose:
        _print_detail_table(metrics_opt["details"], metrics_base["details"])

    # ── Verdict ───────────────────────────────────────────────────────────────
    p1_gain  = metrics_opt["precision_at_1"] - metrics_base["precision_at_1"]
    mrr_gain = metrics_opt["mrr"]            - metrics_base["mrr"]
    time_ratio = metrics_base["avg_time_ms"] / max(metrics_opt["avg_time_ms"], 0.01)

    verdict_lines = []
    if p1_gain > 0:
        verdict_lines.append(
            f"[green]+{p1_gain:.0%}[/green] de Precision@1 "
            f"({metrics_base['precision_at_1']:.0%} → {metrics_opt['precision_at_1']:.0%})"
        )
    elif p1_gain == 0:
        verdict_lines.append("[dim]Precision@1 identique[/dim]")
    else:
        verdict_lines.append(f"[red]{p1_gain:.0%}[/red] de Precision@1 (baseline meilleur)")

    if mrr_gain > 0:
        verdict_lines.append(
            f"[green]+{mrr_gain:.3f}[/green] de MRR "
            f"({metrics_base['mrr']:.3f} → {metrics_opt['mrr']:.3f})"
        )

    if time_ratio < 1:
        verdict_lines.append(
            f"[yellow]⚠ Pipeline optimisé x{1/time_ratio:.1f} plus lent[/yellow] "
            f"(chunking + reranking + MMR)"
        )
    else:
        verdict_lines.append(
            f"[cyan]Temps :[/cyan] baseline={metrics_base['avg_time_ms']:.1f}ms "
            f"| optimisé={metrics_opt['avg_time_ms']:.1f}ms"
        )

    console.print(Panel(
        "\n".join(verdict_lines),
        title="[bold]Verdict[/bold]",
        border_style="green" if p1_gain >= 0 else "red",
        padding=(0, 2),
    ))
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation pipeline RAG BVZyme")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Affiche le détail question par question",
    )
    args = parser.parse_args()
    main(verbose=args.verbose)
