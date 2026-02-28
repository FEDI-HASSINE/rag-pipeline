"""
text_cleaner.py
---------------
Pipeline de nettoyage de texte extrait par PDF pour le corpus BVZyme.

Architecture en couches :
    Raw PDF text
        ↓ [Couche 1 — Structurelle]   page breaks, unicode, encodage
        ↓ [Couche 2 — Topographique]  coupures de mots, lignes isolées
        ↓ [Couche 3 — Sémantique]     boilerplate, headers répétés
        ↓ [Couche 4 — Finale]         espaces, normalisation unités
    Texte propre

Usage:
    from text_cleaner import clean_document_text

    raw = pdf_extract_text(path)
    clean = clean_document_text(raw)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


# ════════════════════════════════════════════════════════════
# COUCHE 1 — Structurelle (artefacts bas-niveau)
# ════════════════════════════════════════════════════════════

def remove_page_breaks(text: str) -> str:
    """Remplace les sauts de page pdfminer (\\x0c) par une séparation propre."""
    return text.replace("\x0c", "\n\n")


def fix_encoding_artifacts(text: str) -> str:
    """
    Corrige les artefacts d'encodage courants (latin-1 vers UTF-8 mal interprétés)
    et normalise les caractères Unicode en forme NFC.
    """
    replacements = {
        "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
        "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
        "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
        "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
        "\u2013": "-",   # EN DASH
        "\u2014": "-",   # EM DASH
        "\u2022": "-",   # BULLET (•)
        "\u00b0": "°",   # DEGREE SIGN (normalisé)
        "\u00b5": "μ",   # MICRO SIGN
        "\u00d7": "x",   # MULTIPLICATION SIGN
        "\u2264": "<=",  # LESS-THAN OR EQUAL
        "\u2265": ">=",  # GREATER-THAN OR EQUAL
        # Wingdings / Symbol bullets courants dans pdfminer
        "\uf0b7": "-",   # Wingdings bullet (●)
        "\uf0fc": "-",
        "\uf0d8": "-",
        "\uf0e0": "-",
        # Artefacts encodage CP1252 / Latin-1
        "\u00e2\u0080\u0099": "'",
        "\u00c3\u00a9": "é",
        "\u00c3\u00a8": "è",
        "\u00c3\u00aa": "ê",
        "\u00c3\u00a0": "à",
        "\u00c3\u00b4": "ô",
        "\u00c3\u00bc": "ü",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalisation NFC (recompose les caractères accentués fragmentés)
    text = unicodedata.normalize("NFC", text)
    return text


def replace_unicode_bullets(text: str) -> str:
    """
    Remplace tous les bullets/points Unicode non-ASCII (catégorie Po, So, Ps)
    par un tiret simple pour uniformité.
    """
    # Capture les bullets Unicode restants après fix_encoding_artifacts
    text = re.sub(r"[\u2023\u2043\u204c\u204d\u2219\u25aa\u25cf\u25e6\u2b9a-\u2b9d]",
                  "-", text)
    return text


# ════════════════════════════════════════════════════════════
# COUCHE 2 — Topographique (structure des lignes)
# ════════════════════════════════════════════════════════════

def fix_hyphenation(text: str) -> str:
    """
    Recole uniquement les mots coupés en fin de ligne par un tiret typographique.
    Ex: "applica-\\ntion" → "application"
        "amino-\\nacid"   → "amino-acid"  (tiret sémantique conservé)

    NOTE: On ne touche PAS aux doubles sauts de ligne (\\n\\n) car dans les TDS
    BVZyme ils séparent des champs structurés (titre de section / valeur).
    """
    # Tiret en fin de ligne suivi d'un retour + minuscule → recoller sans tiret
    text = re.sub(r"(\w)-\n([a-zà-öø-ÿ])", r"\1\2", text)
    return text


def fix_broken_values(text: str) -> str:
    """
    Recole les valeurs fragmentées sur plusieurs lignes (artefact pdfminer).
    Ex: "10000U/\\ng"  → "10000U/g"
        "50-55\\n°C"   → "50-55°C"
        "< 15\\n%"     → "< 15%"

    Limité à 1-2 sauts de ligne maximum pour ne pas fusionner des paragraphes.
    """
    # Unité/symbole isolé sur la ligne suivante après une valeur numérique
    # (max 2 newlines + espaces horizontaux uniquement entre les deux)
    text = re.sub(
        r"(\d[ \t]*)\n[ \t]*\n?[ \t]*([%°μmgkLlUupmMCFHNJVWKΩcm/]+\b)",
        r"\1\2",
        text
    )
    # Slash + saut de ligne + unité (ex: "U/\ng" ou "U/\n\ng" → "U/g")
    text = re.sub(r"([A-Za-zΩμ°0-9])/\n[ \t]*\n?[ \t]*([a-zA-Zμg])", r"\1/\2", text)
    return text


def remove_short_isolated_lines(text: str) -> str:
    """
    Supprime uniquement les lignes isolées (entourées de lignes vides) qui sont
    clairement des artefacts de mise en page :
    - Lignes purement numériques : numéros de page ("1", "12", "- 2 -")
    - Lignes ne contenant que de la ponctuation/symboles   (".", "-", "***")
    - Lignes vides (elles-mêmes)

    NE supprime PAS les labels de section significatifs ("Activity", "Dosage"…)
    ni les titres courts même isolés.
    """
    # Pattern : ligne vide ou purement ponctuation/chiffres
    _junk = re.compile(r"^[\s\d\-–—=_.•·*#|/\\()]*$")

    lines = text.split("\n")
    result = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Toujours garder les lignes avec du texte substantiel (≥1 lettre)
        if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", stripped):
            result.append(line)
            continue

        # Ligne vide ou sans lettres : conserver seulement si elle sépare du contenu
        # (i.e. ne pas créer de séquences de plus de 2 lignes vides)
        if not stripped:
            # Compter les vides consécutifs déjà en fin de result
            consecutive = sum(1 for l in reversed(result) if not l.strip())
            if consecutive < 2:
                result.append(line)
            continue

        # Ligne sans lettres mais non vide (ex: "---", "123", "§2") : supprimer
        # si isolée ou si elle correspond au pattern junk
        if _junk.match(stripped):
            prev_empty = (i == 0) or not lines[i - 1].strip()
            next_empty = (i == len(lines) - 1) or not lines[i + 1].strip()
            if prev_empty and next_empty:
                continue  # artefact isolé → supprimé

        result.append(line)

    return "\n".join(result)


# ════════════════════════════════════════════════════════════
# COUCHE 3 — Sémantique (boilerplate, headers répétés)
# ════════════════════════════════════════════════════════════

# Patterns boilerplate BVZyme TDS (adresse société, contacts)
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Bloc adresse VTR&beyond (répété dans chaque TDS, une fois par page)
    re.compile(
        r"VTR&beyond\s*\n"
        r"[\s\S]{0,600}?"
        r"(?:www\.vtrbeyond\.com|Website\s*:\s*www\.[^\n]+)",
        re.IGNORECASE,
    ),
    # Ligne "TECHNICAL DATA SHEET" — flexible car pdfminer peut couper le mot
    # sur un saut de page : "TECHN\x0c ICAL DATA SHEET" → "TECHN\n\nICAL DATA SHEET"
    re.compile(r"TECH\w*[\s\n]*(?:ICAL\s+)?DATA\s+SHEET[^\n]{0,30}",
               re.IGNORECASE),
    # Ligne "FOOD SAFETY DATA" / "FOOD SAFTY DATA" seule
    re.compile(r"^\s*FOOD\s+SAF(?:E)?TY\s+DATA\s*$", re.MULTILINE | re.IGNORECASE),
    # Ligne contenant uniquement un numéro de page "Page X of Y" ou "- X -"
    re.compile(r"^\s*(?:Page\s+\d+\s+of\s+\d+|-\s*\d+\s*-)\s*$",
               re.MULTILINE | re.IGNORECASE),
    # Lignes de séparation décoratives (tirets, égaux, underscores)
    re.compile(r"^\s*[-=_]{4,}\s*$", re.MULTILINE),
]


def remove_boilerplate(text: str) -> str:
    """Supprime le boilerplate répétitif (adresse, titre TDS, séparateurs)."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


def remove_repeated_headers(text: str, min_repeats: int = 3) -> str:
    """
    Détecte et supprime les lignes qui se répètent ≥ min_repeats fois dans le
    document — typiquement les en-têtes/pieds-de-page imprimés sur chaque page.

    Exemple : "BVZyme GOX 110®" répété sur chaque page.
    """
    lines = text.split("\n")
    # Compter les occurrences de chaque ligne non-vide
    line_counts = Counter(
        ln.strip() for ln in lines
        if ln.strip() and len(ln.strip()) < 80  # les longs textes ne sont pas des headers
    )
    # Lignes qui se répètent trop
    repeated = {line for line, count in line_counts.items() if count >= min_repeats}

    if not repeated:
        return text

    cleaned = [ln for ln in lines if ln.strip() not in repeated]
    return "\n".join(cleaned)


# ════════════════════════════════════════════════════════════
# COUCHE 4 — Finale (normalisation espaces, unités)
# ════════════════════════════════════════════════════════════

def normalize_units(text: str) -> str:
    """
    Normalise les unités de mesure fragmentées ou mal formatées.
    Ex: "50 - 55 ° C" → "50-55°C"
        "10 000 U / g" → "10000 U/g"
        "< 15 %"       → "<15%"
    """
    # "° C" ou "° F" → "°C" / "°F"
    text = re.sub(r"°\s+([CFKc])", r"°\1", text)
    # Opérateurs de plage "50 - 55" → "50-55"
    text = re.sub(r"(\d)\s+-\s+(\d)", r"\1-\2", text)
    # Espace avant % gênant
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    # Slash entouré d'espaces "U / g" → "U/g"
    text = re.sub(r"\s*/\s*", r"/", text)
    # Signe < ou > suivi d'un espace avant un nombre
    text = re.sub(r"([<>])\s+(\d)", r"\1\2", text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    - Remplace les tabulations par un espace
    - Réduit les espaces multiples sur une même ligne à un seul
    - Réduit les sauts de ligne multiples (> 2) à exactement 2
    - Supprime les lignes ne contenant que des espaces
    """
    # Tabs → espace
    text = text.replace("\t", " ")
    # Espaces multiples sur une ligne → un seul
    text = re.sub(r"[ ]{2,}", " ", text)
    # Lignes ne contenant que des espaces → ligne vide
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
    # Plus de 2 sauts de ligne consécutifs → exactement 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ════════════════════════════════════════════════════════════

def clean_document_text(text: str) -> str:
    """
    Pipeline de nettoyage complet pour texte extrait de PDF BVZyme.

    Ordre d'application :
        1. remove_page_breaks          — \\x0c → \\n\\n
        2. fix_encoding_artifacts      — artefacts Unicode/CP1252
        3. replace_unicode_bullets     — bullets Unicode restants → -
        4. remove_boilerplate          — adresse VTR&beyond
        5. remove_repeated_headers     — titres répétés par page
        6. fix_hyphenation             — mots coupés par tiret
        7. fix_broken_values           — valeurs/unités fragmentées
        8. remove_short_isolated_lines — numéros de page, artefacts
        9. normalize_units             — °C, %, plages
       10. normalize_whitespace        — espaces finaux

    Returns
    -------
    str
        Texte nettoyé, prêt pour le chunking.
    """
    text = remove_page_breaks(text)
    text = fix_encoding_artifacts(text)
    text = replace_unicode_bullets(text)
    text = remove_boilerplate(text)
    text = remove_repeated_headers(text)
    text = fix_hyphenation(text)
    text = fix_broken_values(text)
    text = remove_short_isolated_lines(text)
    text = normalize_units(text)
    text = normalize_whitespace(text)
    return text


# ════════════════════════════════════════════════════════════
# CLI de diagnostic (usage direct : python text_cleaner.py fichier.pdf)
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python text_cleaner.py <fichier.pdf> [--diff]")
        sys.exit(0)

    pdf_path = Path(sys.argv[1])
    show_diff = "--diff" in sys.argv

    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print("Erreur: pdfminer.six requis (pip install pdfminer.six)")
        sys.exit(1)

    raw = extract_text(str(pdf_path)) or ""
    clean = clean_document_text(raw)

    print(f"{'='*60}")
    print(f"  Fichier  : {pdf_path.name}")
    print(f"  Brut     : {len(raw):>6} chars | {len(raw.split()):>5} mots")
    print(f"  Nettoyé  : {len(clean):>6} chars | {len(clean.split()):>5} mots")
    reduction = (1 - len(clean) / max(len(raw), 1)) * 100
    print(f"  Réduction: {reduction:.1f}%")
    print(f"{'='*60}\n")

    if show_diff:
        print("── BRUT (500 premiers chars) ──")
        print(repr(raw[:500]))
        print("\n── NETTOYÉ (500 premiers chars) ──")
        print(repr(clean[:500]))
    else:
        print(clean[:1500])
