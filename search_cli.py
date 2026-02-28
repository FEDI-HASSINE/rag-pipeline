"""
search_cli.py
-------------
Interface CLI interactive pour le moteur de recherche RAG.

Usage:
    python search_cli.py
    python search_cli.py --data "C:/chemin/vers/pdfs"
    python search_cli.py --reindex          # force recalcul index
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Rich ──────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.prompt import Prompt
    from rich import box
except ImportError:
    print("rich non installé — pip install rich")
    sys.exit(1)

from data_loader import load_documents
from rag_pipeline import OptimizedRAGPipeline
from config import DATA_FOLDER as DEFAULT_DATA_FOLDER, TOP_K

# ── Constantes ────────────────────────────────────────────────────────────────
EXCERPT_MAX_CHARS   = 200

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers d'affichage
# ─────────────────────────────────────────────────────────────────────────────

def _banner():
    """Affiche la bannière de démarrage."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔍 RAG Search Engine[/bold cyan]\n"
        "[dim]Moteur: all-MiniLM-L6-v2 · Cosine Similarity · Top-3[/dim]\n\n"
        "[white]Tapez votre question puis [bold]Entrée[/bold][/white]\n"
        "[dim]Commandes : [bold]:stats[/bold]  [bold]:clear[/bold]  [bold]:quit[/bold][/dim]",
        title="[bold white]BVZyme Knowledge Base[/bold white]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()


def _show_results(response: dict):
    """Affiche les Top-K résultats de manière formatée."""
    results  = response["results"]
    query    = response["query"]
    time_ms  = response["search_time_ms"]
    from_cache = response.get("from_cache", False)

    # En-tête requête
    cache_label = " [dim](cache LRU)[/dim]" if from_cache else ""
    console.print(Rule(f"[bold yellow]{query}[/bold yellow]", style="yellow"))
    console.print(
        f"  [dim]⏱  {time_ms:.1f} ms{cache_label}  ·  "
        f"{len(results)} résultats[/dim]\n"
    )

    if not results:
        console.print("  [red]Aucun résultat trouvé.[/red]\n")
        return

    for r in results:
        # Couleur du score
        score = r.final_score
        if score >= 0.7:
            score_color = "green"
        elif score >= 0.5:
            score_color = "yellow"
        else:
            score_color = "red"

        # Extrait pertinent : phrase du chunk qui contient le plus de mots de la requête
        query_words = set(response["query"].lower().split())
        sentences = [s.strip() for s in r.chunk.text.replace("\n", " ").split(".")
                     if len(s.strip()) > 20]
        if sentences:
            best_sentence = max(
                sentences,
                key=lambda s: sum(1 for w in query_words if w in s.lower())
            )
            excerpt = best_sentence.strip()
        else:
            excerpt = r.chunk.text.replace("\n", " ").strip()
        if len(excerpt) > EXCERPT_MAX_CHARS:
            excerpt = excerpt[:EXCERPT_MAX_CHARS].rsplit(" ", 1)[0] + "…"

        # Détails du score
        score_detail = (
            f"[dim]cosine={r.cosine_score:.3f} · "
            f"bm25={r.bm25_score:.3f}[/dim]"
        )

        console.print(
            f"  [bold white]#{r.rank}[/bold white]  "
            f"[bold cyan]{r.chunk.doc_title}[/bold cyan]  "
            f"[{score_color}][score={score:.4f}][/{score_color}]  "
            + score_detail
        )
        console.print(f"     [white]{excerpt}[/white]")
        console.print()


def _show_stats(pipeline: OptimizedRAGPipeline):
    """Affiche les statistiques de l'index."""
    stats = pipeline.get_stats()

    table = Table(
        title="Statistiques du Pipeline",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=False,
    )
    table.add_column("Paramètre",  style="cyan",  no_wrap=True, min_width=28)
    table.add_column("Valeur",     style="green", min_width=30)

    labels = {
        "n_chunks":        "Chunks indexés",
        "n_docs":          "Documents",
        "embedding_dim":   "Dimension embedding",
        "index_type":      "Type d'index",
        "real_embeddings": "Modèle embedding",
        "cache_size":      "Entrées cache LRU",
        "reranking":       "Reranking",
        "diversification": "Diversification",
    }
    for key, label in labels.items():
        value = stats.get(key, "—")
        if key == "real_embeddings":
            value = "all-MiniLM-L6-v2" if value else "TF-IDF fallback"
        table.add_row(label, str(value))

    console.print()
    console.print(table)
    console.print()


def _show_help():
    """Rappelle les commandes disponibles."""
    console.print(Panel(
        "[bold]:stats[/bold]   → Statistiques de l'index\n"
        "[bold]:clear[/bold]   → Vider le cache des requêtes\n"
        "[bold]:quit[/bold]    → Quitter  [dim](ou q · exit · Ctrl+C)[/dim]",
        title="Commandes",
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Démarrage du pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _startup(data_folder: str, force_reindex: bool) -> OptimizedRAGPipeline:
    """Charge les documents et initialise le pipeline."""
    console.print(Rule("[bold white]Démarrage[/bold white]", style="white"))

    # 1. Documents
    with console.status("[cyan]Chargement des documents…[/cyan]", spinner="dots"):
        docs = load_documents(data_folder)
    console.print(f"  [green]✓[/green] {len(docs)} documents chargés")

    # 2. Pipeline
    pipeline = OptimizedRAGPipeline(top_k=TOP_K)

    # 3. Indexation / cache
    console.print()
    metrics = pipeline.index_documents(docs, force_reindex=force_reindex)

    source = metrics.get("source", "computed")
    if source == "cache":
        console.print(f"  [green]✓[/green] Index prêt [dim](depuis cache)[/dim]")
    else:
        console.print(
            f"  [green]✓[/green] Index construit en "
            f"[bold]{metrics['indexing_time_s']}s[/bold] — "
            f"{metrics['n_chunks']} chunks"
        )

    console.print()
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Boucle interactive principale
# ─────────────────────────────────────────────────────────────────────────────

def run_cli(pipeline: OptimizedRAGPipeline):
    """Boucle REPL principale."""
    _banner()

    QUIT_COMMANDS  = {"q", "quit", "exit", ":quit"}
    STATS_COMMANDS = {":stats", ":stat"}
    CLEAR_COMMANDS = {":clear", ":cache"}
    HELP_COMMANDS  = {":help", ":h", "?"}

    while True:
        try:
            query = Prompt.ask("[bold cyan]>[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Au revoir.[/dim]")
            break

        if not query:
            continue

        query_lower = query.lower()

        # ── Commandes spéciales ──────────────────────────
        if query_lower in QUIT_COMMANDS:
            console.print("\n[dim]Au revoir.[/dim]")
            break

        if query_lower in STATS_COMMANDS:
            _show_stats(pipeline)
            continue

        if query_lower in CLEAR_COMMANDS:
            pipeline.cache.cache.clear()
            pipeline.cache.order.clear()
            console.print("  [green]✓[/green] Cache LRU vidé.\n")
            continue

        if query_lower in HELP_COMMANDS:
            _show_help()
            continue

        # ── Recherche ────────────────────────────────────
        try:
            with console.status("[cyan]Recherche en cours…[/cyan]", spinner="dots"):
                response = pipeline.search(query)
            _show_results(response)
        except Exception as exc:
            console.print(f"  [red]Erreur : {exc}[/red]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG Search CLI — BVZyme Knowledge Base",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data", "-d",
        default=DEFAULT_DATA_FOLDER,
        help=f"Dossier contenant les PDF/TXT/JSON\n(défaut: {DEFAULT_DATA_FOLDER})",
    )
    parser.add_argument(
        "--reindex", "-r",
        action="store_true",
        help="Force le recalcul de l'index (ignore le cache)",
    )
    args = parser.parse_args()

    try:
        pipeline = _startup(data_folder=args.data, force_reindex=args.reindex)
        run_cli(pipeline)
    except FileNotFoundError as e:
        console.print(f"\n[red]Dossier introuvable : {e}[/red]")
        console.print(f"[dim]Utilisez --data pour spécifier le chemin.[/dim]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrompu.[/dim]")


if __name__ == "__main__":
    main()
