"""
query_expander.py
-----------------
Solution 3 — Query Expansion pour le corpus BVZyme.

Principe : enrichit la requête originale avec des termes équivalents
(traduction FR→EN, synonymes métier, variantes orthographiques) AVANT
l'embedding all-MiniLM-L6-v2.

Le modèle d'embedding reste identique : all-MiniLM-L6-v2, dim=384.
L'embedding est calculé sur le texte étendu → meilleure couverture
sémantique avec le corpus en anglais.

Conformité conditions imposées :
    ✅ Modèle    : all-MiniLM-L6-v2 (inchangé)
    ✅ Librairie : sentence-transformers (inchangée)
    ✅ Dimension : 384 (inchangée)
    ✅ Similarité : Cosine Similarity (inchangée)
    ✅ Top-K    : 3 (inchangé)
"""

from __future__ import annotations

import re
from typing import Optional


# ════════════════════════════════════════════════════════════
# DICTIONNAIRE D'EXPANSION — Domaine enzymologie boulangère
# ════════════════════════════════════════════════════════════
#
# Structure : mot/expression → liste de termes à ajouter
# Règles :
#   - Clé  : regex insensible à la casse (cherché dans la query)
#   - Valeur : mots/phrases ajoutés EN FIN de la query étendue
#   - On ajoute, on ne remplace pas → query originale toujours préservée

_EXPANSION_MAP: dict[str, list[str]] = {

    # ── Terminologie technique FR→EN ─────────────────────────────────────────
    r"temp[eé]rature\s+optimale?":  ["optimal temperature", "optimum temperature",
                                     "operating temperature", "temperature range"],
    r"temp[eé]rature":              ["temperature"],
    r"dosage":                      ["dosage", "recommended dose", "usage level",
                                     "application rate"],
    r"dose\s+recommand[eé]e?":      ["recommended dosage", "suggested dosage",
                                     "application rate", "usage level ppm"],
    r"quantit[eé]\s+recommand[eé]e?": ["recommended amount", "dosage", "ppm",
                                        "suggested optimum"],
    r"conservation":                ["storage", "shelf life", "store",
                                     "date of minimum durability"],
    r"stockage":                    ["storage", "store", "shelf life",
                                     "cool dry place"],
    r"dur[eé]e\s+(?:de\s+)?(?:conservation|vie)": ["shelf life", "durability",
                                                    "expiry", "months storage"],
    r"activit[eé]\s+enzymatique?":  ["enzyme activity", "enzymatic activity",
                                     "activity U/g"],
    r"activit[eé]":                 ["activity", "enzymatic"],
    r"efficacit[eé]":               ["efficacy", "effectiveness", "performance"],
    r"propri[eé]t[eé]":             ["properties", "characteristics"],
    r"application":                 ["application", "bakery", "bread", "flour"],
    r"fonction":                    ["function", "role", "effect"],
    r"description":                 ["product description", "description"],
    r"caract[eé]ristiques?":        ["characteristics", "properties", "specs"],

    # ── Unités & valeurs ─────────────────────────────────────────────────────
    r"\bppm\b":                     ["ppm", "parts per million", "g per tonne"],
    r"\bU/g\b":                     ["U/g", "units per gram", "activity"],
    r"\bpH\b":                      ["pH", "acidity", "optimal pH"],
    r"\b°C\b|\bdegr[eé]":          ["°C", "degrees celsius", "temperature"],

    # ── Enzymes spécifiques ──────────────────────────────────────────────────
    r"transglutaminase":            ["transglutaminase", "TG", "protein cross-linking",
                                     "gluten network", "cross-linking"],
    r"glucose\s*oxydase?|gox":      ["glucose oxidase", "GOX", "gluten strengthening",
                                     "oxidase"],
    r"amylase\s+fongique?|fungal\s+amylase": ["fungal amylase", "alpha-amylase",
                                               "amylolytic", "starch degradation"],
    r"alpha.?amylase":              ["alpha-amylase", "amylase", "amylolytic activity",
                                     "starch", "dextrin"],
    r"xylanase":                    ["xylanase", "hemicellulase", "pentosanase",
                                     "arabinoxylan", "xylan degradation"],
    r"lipase":                      ["lipase", "lipolytic", "fat degradation",
                                     "emulsification"],
    r"prot[eé]ase":                 ["protease", "proteolytic", "gluten modification",
                                     "protein breakdown"],
    r"hemicellulase":               ["hemicellulase", "xylanase", "arabinoxylan",
                                     "fiber"],
    r"cellulase":                   ["cellulase", "cellulose degradation",
                                     "fiber breakdown"],
    r"lactase":                     ["lactase", "lactose", "dairy"],
    r"glucoamylase|amyloglucosidase": ["glucoamylase", "amyloglucosidase",
                                       "starch hydrolysis", "glucose"],
    r"invertase":                   ["invertase", "sucrase", "sucrose splitting"],
    r"oxydase|oxidase":             ["oxidase", "oxidation", "gluten strengthening"],

    # ── Gammes produits BVZyme ────────────────────────────────────────────────
    r"hcf\s*(?:max|400|500|63|64|65|sx)?": ["HCF", "hemicellulase", "xylanase",
                                             "fungal amylase"],
    r"hcb\s*(?:708|709|710)?":      ["HCB", "hemicellulase", "bakery complex"],
    r"tg\s*(?:max\s*63|max\s*64|881|883)?": ["TG", "transglutaminase",
                                              "cross-linking enzyme"],
    r"gox\s*(?:110)?|go\s*max":     ["GOX", "glucose oxidase", "oxidase"],
    r"a.?fresh\s*(?:101|202|303)?": ["A-FRESH", "anti-staling", "freshness",
                                     "maltogenic amylase"],
    r"a.?soft\s*(?:205|305|405)?":  ["A-SOFT", "softness", "crumb softness",
                                     "amylase"],
    r"af\s*(?:110|220|330|sx)?":    ["AF", "fungal amylase", "alpha-amylase"],
    r"amg\s*(?:880|1400)?":         ["AMG", "glucoamylase", "amyloglucosidase"],
    r"l\s*max\s*(?:63|64|65|x)?|l55": ["lipase", "L MAX", "lipolytic"],

    # ── Propriétés physico-chimiques ─────────────────────────────────────────
    r"humidit[eé]|moisture":        ["moisture", "humidity", "water content"],
    r"aspect":                      ["aspect", "appearance", "free flowing powder"],
    r"couleur|color":               ["color", "colour", "white cream"],
    r"microbiologie|microbiology":  ["microbiology", "total plate count",
                                     "salmonella", "coliforms"],
    r"m[eé]taux\s+lourds?":        ["heavy metals", "cadmium", "mercury",
                                     "arsenic", "lead"],
    r"arsenic":                     ["arsenic", "heavy metals", "mg/kg"],
    r"plomb|lead":                  ["lead", "heavy metals", "mg/kg"],
    r"cadmium":                     ["cadmium", "heavy metals", "mg/kg"],
    r"mercure|mercury":             ["mercury", "heavy metals", "mg/kg"],
    r"allerg[eè]ne?s?":             ["allergens", "gluten", "regulation 1169/2011"],
    r"ogm|gmo":                     ["GMO", "genetically modified", "regulation 1829"],
    r"ionisation":                  ["ionization", "irradiation", "without irradiation"],

    # ── Ingrédients & matières premières ─────────────────────────────────────
    r"farine":                      ["flour", "wheat flour", "bakers flour"],
    r"gluten":                      ["gluten", "gluten network", "protein network"],
    r"amidon":                      ["starch", "amylose", "amylopectin"],
    r"sucre":                       ["sugar", "sucrose", "saccharide"],
    r"levure":                      ["yeast", "fermentation"],
    r"pain|bread":                  ["bread", "bakery", "loaf", "baking"],
    r"biscuit":                     ["biscuit", "cookie", "bakery product"],
    r"pâte|dough":                  ["dough", "batter", "gluten network"],
    r"boulangerie|bakery":          ["bakery", "baking", "bread making"],
    r"panification":                ["bread making", "baking process",
                                     "panification", "bakery"],
    r"fermentation":                ["fermentation", "yeast", "rising"],
    r"élasticité|elasticity":       ["elasticity", "extensibility", "dough strength"],
    r"volume":                      ["volume", "loaf volume", "oven spring"],
    r"texture":                     ["texture", "crumb", "softness", "firmness"],
    r"croûte|crust":                ["crust", "crust color", "browning"],
    r"fraîcheur|freshness":         ["freshness", "anti-staling", "shelf life"],
    r"ramollissement|staling":      ["staling", "anti-staling", "firmness over time"],
    r"moelleux|softness":           ["softness", "crumb softness", "tender crumb"],

    # ── Conditions opératoires ────────────────────────────────────────────────
    r"ph\s+optimale?|optimal\s+pH": ["optimal pH", "optimum pH", "pH range",
                                     "acidity"],
    r"stabilit[eé]":                ["stability", "thermal stability", "pH stability"],
    r"inactivation":                ["inactivation", "heat inactivation",
                                     "temperature inactivation"],
    r"substrat":                    ["substrate", "substrate specificity"],
    r"inhibiteur|inhibitor":        ["inhibitor", "inhibition"],
    r"cofacteur|cofactor":          ["cofactor", "coenzyme"],
    r"cinétique|kinetics":          ["kinetics", "reaction rate", "Km", "Vmax"],
}


# ════════════════════════════════════════════════════════════
# FONCTIONS PRINCIPALES
# ════════════════════════════════════════════════════════════

def expand_query(query: str, max_expansions: int = 3) -> str:
    """
    Enrichit la requête avec des termes sémantiquement équivalents.

    Le résultat est une chaîne qui contient :
      1. La requête originale (toujours en premier)
      2. Les termes d'expansion pertinents (dédupliqués, limités)

    Le modèle all-MiniLM-L6-v2 encode ensuite ce texte étendu,
    ce qui produit un vecteur qui couvre mieux le vocabulaire anglais
    du corpus BVZyme.

    Amélioration 4 : max_expansions réduit de 8 → 3 par défaut.
    Utiliser max_expansions=8 pour BM25 (enrichissement lexical).
    Utiliser max_expansions=3 (défaut) pour l'embedding (préserve la
    spécificité produit — ex. "A-FRESH" ne se dilue pas vers "bread").

    Args:
        query        : Requête originale (FR, EN, ou mixte)
        max_expansions: Nombre max de termes ajoutés (défaut 3)

    Returns:
        Requête étendue prête pour l'embedding ou BM25.

    Exemple:
        >>> expand_query("température optimale transglutaminase")
        "température optimale transglutaminase optimal temperature optimum
         temperature transglutaminase TG protein cross-linking gluten network"
    """
    query_lower = query.lower()
    expansions: list[str] = []
    seen: set[str] = set(query_lower.split())  # évite de répéter les mots déjà présents

    for pattern, terms in _EXPANSION_MAP.items():
        if re.search(pattern, query_lower, flags=re.IGNORECASE):
            for term in terms:
                # Déduplique : n'ajoute pas si le terme est déjà dans la query
                term_words = term.lower().split()
                if not any(w in seen for w in term_words):
                    expansions.append(term)
                    seen.update(term_words)

                if len(expansions) >= max_expansions:
                    break
        if len(expansions) >= max_expansions:
            break

    if not expansions:
        return query  # aucune expansion trouvée → query inchangée

    expanded = query + " " + " ".join(expansions)
    return expanded.strip()


def get_expansion_terms(query: str) -> list[str]:
    """
    Retourne uniquement la liste des termes ajoutés (sans la query originale).
    Utile pour affichage/debug.
    """
    original_words = set(query.lower().split())
    expanded = expand_query(query)
    added = expanded[len(query):].strip().split()
    # Reconstruit les termes multi-mots
    extra = expanded[len(query):].strip()
    return [t for t in extra.split("  ") if t] if extra else []


def explain_expansion(query: str) -> str:
    """Affiche les règles déclenchées pour une requête (debug/test)."""
    query_lower = query.lower()
    triggered = []
    for pattern, terms in _EXPANSION_MAP.items():
        if re.search(pattern, query_lower, flags=re.IGNORECASE):
            triggered.append(f"  [{pattern}] → {terms[:3]}")
    if not triggered:
        return f"  Aucune règle déclenchée pour : {query!r}"
    return "\n".join(triggered)


# ════════════════════════════════════════════════════════════
# CLI de debug
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    queries = sys.argv[1:] or [
        "température optimale transglutaminase",
        "dosage recommandé glucose oxidase ppm",
        "conservation stockage enzyme boulangerie",
        "métaux lourds arsenic plomb cadmium",
        "alpha-amylase xylanase quantité recommandée",
        "what is the optimal pH for GOX 110",
        "TG MAX63 dosage bread",
    ]

    for q in queries:
        expanded = expand_query(q)
        added = expanded[len(q):].strip()
        print(f"\nOriginal : {q!r}")
        print(f"Étendue  : {expanded!r}")
        print(f"Ajoutés  : {added!r}" if added else "Ajoutés  : (aucun)")
        print(f"Règles   :\n{explain_expansion(q)}")
