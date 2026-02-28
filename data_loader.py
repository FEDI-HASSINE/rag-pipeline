"""
data_loader.py
--------------
Charge des documents depuis un dossier (PDF, TXT, JSON) et les normalise
au format {"id": str, "title": str, "content": str}.

Usage:
    from data_loader import load_documents
    docs = load_documents("./data")
    print(f"{len(docs)} documents chargés")
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# PDF extraction (pdfminer.six)
# ---------------------------------------------------------------------------
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional pretty output
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
    _use_rich = True
except ImportError:
    _use_rich = False

# ---------------------------------------------------------------------------
# Text cleaner
# ---------------------------------------------------------------------------
from text_cleaner import clean_document_text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Génère un identifiant simple depuis un nom de fichier."""
    name = Path(name).stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _load_txt(path: Path) -> dict[str, str]:
    """Charge un fichier .txt."""
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "id": _slugify(path.name),
        "title": path.stem,
        "content": content,
    }


def _load_json(path: Path) -> list[dict[str, str]]:
    """
    Charge un fichier .json.

    Formats acceptés :
    - dict unique  : {"title": ..., "content": ...} ou {"name": ..., "text": ...}
    - liste de dict: [{...}, {...}]
    """
    raw: Any = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    def _normalize_item(item: dict, idx: int = 0) -> dict[str, str]:
        title = (
            item.get("title")
            or item.get("name")
            or item.get("titre")
            or f"{path.stem}_{idx}"
        )
        content = (
            item.get("content")
            or item.get("text")
            or item.get("texte")
            or item.get("body")
            or ""
        )
        doc_id = item.get("id") or f"{_slugify(path.name)}_{idx}"
        return {"id": str(doc_id), "title": str(title), "content": str(content)}

    if isinstance(raw, list):
        return [_normalize_item(item, i) for i, item in enumerate(raw) if isinstance(item, dict)]
    elif isinstance(raw, dict):
        return [_normalize_item(raw)]
    else:
        return []


def _load_pdf(path: Path) -> dict[str, str]:
    """Extrait le texte d'un fichier .pdf avec pdfminer.six."""
    if not PDF_AVAILABLE:
        raise ImportError(
            "pdfminer.six est requis pour lire les PDF.\n"
            "Installez-le avec : pip install pdfminer.six"
        )
    text = pdf_extract_text(str(path)) or ""
    text = clean_document_text(text)
    return {
        "id": _slugify(path.name),
        "title": path.stem,
        "content": text,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_documents(folder: str | Path) -> list[dict[str, str]]:
    """
    Parcourt *folder* et charge tous les fichiers .txt, .json et .pdf.

    Paramètres
    ----------
    folder : str | Path
        Chemin vers le dossier contenant les documents.

    Retourne
    --------
    list[dict]
        Liste de documents normalisés :
        [{"id": str, "title": str, "content": str}, ...]
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder.resolve()}")

    documents: list[dict[str, str]] = []
    errors: list[str] = []
    skipped: list[str] = []

    supported = {".txt", ".json", ".pdf"}
    all_files = sorted(folder.iterdir())

    for file_path in all_files:
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in supported:
            skipped.append(file_path.name)
            continue
        try:
            if ext == ".txt":
                documents.append(_load_txt(file_path))
            elif ext == ".json":
                docs = _load_json(file_path)
                documents.extend(docs)
            elif ext == ".pdf":
                documents.append(_load_pdf(file_path))
        except Exception as exc:
            errors.append(f"{file_path.name} → {exc}")

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    total_words = sum(len(d["content"].split()) for d in documents)

    if _use_rich:
        _print_rich_summary(documents, errors, skipped, total_words)
    else:
        _print_plain_summary(documents, errors, skipped, total_words)

    if errors:
        print("\n[AVERTISSEMENTS] Fichiers en erreur :")
        for e in errors:
            print(f"  ✗ {e}")

    return documents


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_rich_summary(
    docs: list[dict],
    errors: list[str],
    skipped: list[str],
    total_words: int,
) -> None:
    """Affiche un résumé formaté avec Rich."""
    table = Table(title="Résumé — Chargement des documents", show_lines=True)
    table.add_column("Métrique", style="cyan", no_wrap=True)
    table.add_column("Valeur", style="green")

    table.add_row("Fichiers chargés", str(len(docs)))
    table.add_row("Mots total", f"{total_words:,}")
    table.add_row("Moy. mots / doc", f"{total_words // max(len(docs), 1):,}")
    table.add_row("Fichiers ignorés (format non supporté)", str(len(skipped)))
    table.add_row("Erreurs de lecture", str(len(errors)))

    _console.print(table)


def _print_plain_summary(
    docs: list[dict],
    errors: list[str],
    skipped: list[str],
    total_words: int,
) -> None:
    """Affiche un résumé en texte simple."""
    print("\n" + "=" * 45)
    print("  Résumé — Chargement des documents")
    print("=" * 45)
    print(f"  Fichiers chargés           : {len(docs)}")
    print(f"  Mots total                 : {total_words:,}")
    print(f"  Moy. mots / doc            : {total_words // max(len(docs), 1):,}")
    print(f"  Fichiers ignorés           : {len(skipped)}")
    print(f"  Erreurs de lecture         : {len(errors)}")
    print("=" * 45 + "\n")


# ---------------------------------------------------------------------------
# CLI rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else "./data"
    docs = load_documents(folder_arg)
    print(f"\n{len(docs)} documents chargés depuis « {folder_arg} »")
    for doc in docs[:5]:
        preview = doc["content"][:120].replace("\n", " ")
        print(f"  [{doc['id']}] {doc['title']!r} — {preview}…")
    if len(docs) > 5:
        print(f"  ... et {len(docs) - 5} autres.")
