import logging
import unicodedata
from collections import defaultdict

# --- Quick Similarity -------------------------------------------------------
try:
    from rapidfuzz import fuzz, utils as rf_utils
    def fast_similarity(a: str, b: str, *, score_cutoff: float = 0.0) -> float:
        # Quicker with RapidFuzz (C++)
        s = fuzz.token_sort_ratio(a, b, score_cutoff=int(score_cutoff * 100))
        return s / 100.0
    RAPIDFUZZ = True
except Exception:
    from difflib import SequenceMatcher
    def fast_similarity(a: str, b: str, *, score_cutoff: float = 0.0) -> float:
        # Fallback Python (slower)
        s = SequenceMatcher(None, a, b).ratio()
        return s if s >= score_cutoff else 0.0
    RAPIDFUZZ = False

# --- Normalization & trigrams ----------------------------------------------
def normalize(txt: str) -> str:
    # NFKC + lower + strip + collapse spaces (to be adjusted per use case)
    if not txt:
        return ""
    t = unicodedata.normalize("NFKC", txt).lower().strip()
    t = " ".join(t.split())
    return t

def trigrams(s: str):
    if not s:
        return set()
    s = f"  {s}  "   # simple padding
    return {s[i:i+3] for i in range(len(s)-2)}

def jaccard(a_set, b_set) -> float:
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    if inter == 0:
        return 0.0
    return inter / (len(a_set) + len(b_set) - inter)

# --- Main function ------------------------------------------------------
def dedup_by_similarity(
    dict_list,
    *,
    similarity_threshold: float = 0.85,     # final threshold [0..1]
    min_trigram_jaccard: float = 0.10,      # very cheap pre-filter
    log_matches: bool = True
):
    """
    Returns (unique_dicts, cnt) where:
      - unique_dicts: the retained sublist
      - cnt: number of matches found (as in your code)
    Key optimizations:
      - Inverted trigram index -> only compare a small set of candidates
      - Fast similarity + cutoff (very efficient with RapidFuzz)
      - Normalization to reduce noise
    """
    unique_dicts = []
    norm_cache = []          # normalized content for unique_dicts
    tri_cache = []           # trigram signatures for unique_dicts
    inv = defaultdict(set)   # trigram -> {index inside unique_dicts}
    cnt = 0

    for current_dict in dict_list:
        content = current_dict.get("content")
        if not content:
            continue

        c_norm = normalize(content)
        if not c_norm:
            continue
        c_tris = trigrams(c_norm)

        # Retrieve candidates by union on shared trigrams
        if c_tris:
            candidate_idx = set().union(*(inv[t] for t in c_tris if t in inv))
        else:
            candidate_idx = set()

        is_similar = False

        # Additional heuristic: if too few trigrams in common,
        # we can cut short. We filter on a minimum Jaccard before the costly measure.
        # NB: we can tighten min_trigram_jaccard if texts are long.
        for j in candidate_idx:
            uj_tris = tri_cache[j]
            if jaccard(c_tris, uj_tris) < min_trigram_jaccard:
                continue

            # Optional: mini length filter (useful for Levenshtein/Jaro; neutral for token_ratio)
            # if not (0.5 <= len(c_norm)/len(norm_cache[j]) <= 2.0):
            #     continue

            score = fast_similarity(c_norm, norm_cache[j], score_cutoff=similarity_threshold)
            if score > similarity_threshold:
                is_similar = True
                if log_matches:
                    logging.info(
                        f"{content[-100:]}\n is similar to: \n{unique_dicts[j].get('content','')[-100:]}\n____________________"
                    )
                cnt += 1
                break

        if not is_similar:
            # new unique: index it
            j = len(unique_dicts)
            unique_dicts.append(current_dict)
            norm_cache.append(c_norm)
            tri_cache.append(c_tris)
            for t in c_tris:
                inv[t].add(j)

    return unique_dicts, cnt
