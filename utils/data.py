"""
Shared data utilities for Talmudic similarity pipelines.

Expected DataFrame columns:
  ref   : str  — citation (e.g. "Berakhot 2a")
  str   : str  — raw text
  morph : list of lists — [[raw_token, bitmap], ...]
  lex   : str  — lemmatized string (space-separated lemmas)
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import Optional


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_passages(path: str) -> pd.DataFrame:
    """Load passage DataFrame from pickle or parquet."""
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {path}. Use .pkl or .parquet")


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_pairs(
    df: pd.DataFrame,
    known_positives: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a pairs DataFrame for similarity modeling.

    Parameters
    ----------
    df : passages DataFrame (ref, str, morph, lex)
    known_positives : DataFrame with columns [ref_a, ref_b]
                      representing your 4,500 Levenshtein-matched pairs.
                      If None, all pairs are treated as unlabeled.

    Returns
    -------
    DataFrame with columns:
      ref_a, ref_b, str_a, str_b, lex_a, lex_b, morph_a, morph_b,
      is_positive (1 / 0 / -1 for unknown)
    """
    df = df.set_index("ref")

    rows = []
    if known_positives is not None:
        for _, row in known_positives.iterrows():
            a, b = row["ref_a"], row["ref_b"]
            if a in df.index and b in df.index:
                rows.append(_make_pair(df, a, b, label=1))

    pairs = pd.DataFrame(rows)
    return pairs


def _make_pair(df, ref_a, ref_b, label):
    return {
        "ref_a":   ref_a,
        "ref_b":   ref_b,
        "str_a":   df.loc[ref_a, "str"],
        "str_b":   df.loc[ref_b, "str"],
        "lex_a":   df.loc[ref_a, "lex"],
        "lex_b":   df.loc[ref_b, "lex"],
        "morph_a": df.loc[ref_a, "morph"],
        "morph_b": df.loc[ref_b, "morph"],
        "label":   label,   # 1=positive, 0=negative, -1=unknown
    }


# ---------------------------------------------------------------------------
# Morphological feature extraction
# ---------------------------------------------------------------------------

def extract_lemmas(lex_string: str) -> list[str]:
    """Split lemmatized string into individual lemma tokens."""
    return lex_string.split()


def extract_morph_tokens(morph: list) -> list[str]:
    """Return just the raw tokens from morph list-of-lists."""
    return [item[0] for item in morph]


def extract_morph_bitmaps(morph: list) -> list:
    """Return just the bitmaps from morph list-of-lists."""
    return [item[1] for item in morph]


def bitmap_to_features(bitmap) -> dict:
    """
    Parse a morphological bitmap into a named feature dict.

    Adjust bit positions to match your Dicta bitmap spec.
    This is a placeholder — replace with your actual bitmap schema.
    """
    if isinstance(bitmap, int):
        bits = bitmap
    elif isinstance(bitmap, (list, np.ndarray)):
        # If bitmap is already a list of bits, convert to int
        bits = int("".join(str(int(b)) for b in bitmap), 2)
    else:
        return {}

    return {
        "pos":    (bits >> 0) & 0xF,   # part of speech (4 bits)
        "gender": (bits >> 4) & 0x3,   # gender (2 bits)
        "number": (bits >> 6) & 0x3,   # number (2 bits)
        "person": (bits >> 8) & 0x3,   # person (2 bits)
        "binyan": (bits >> 10) & 0xF,  # verbal binyan (4 bits)
        "tense":  (bits >> 14) & 0x7,  # tense/aspect (3 bits)
    }


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_hard_negatives(
    pairs_df: pd.DataFrame,
    topic_labels: dict,           # {ref: topic_id}
    n_negatives: int = 4500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample hard negatives: passage pairs from *different* topic clusters.
    Requires topic labels dict mapping ref -> topic_id.

    These are hard negatives because they're within the same corpus
    but covering different legal/thematic domains.
    """
    rng = np.random.default_rng(seed)

    refs = list(topic_labels.keys())
    sampled = []
    attempts = 0
    max_attempts = n_negatives * 20

    while len(sampled) < n_negatives and attempts < max_attempts:
        a, b = rng.choice(refs, size=2, replace=False)
        if topic_labels[a] != topic_labels[b]:
            # Check not already a known positive
            existing = pairs_df[
                ((pairs_df.ref_a == a) & (pairs_df.ref_b == b)) |
                ((pairs_df.ref_a == b) & (pairs_df.ref_b == a))
            ]
            if existing.empty:
                sampled.append((a, b))
        attempts += 1

    neg_rows = []
    # Need full df to reconstruct pairs — caller passes it separately
    return sampled   # list of (ref_a, ref_b) tuples; caller builds rows


def add_negatives_to_pairs(
    pairs_df: pd.DataFrame,
    passages_df: pd.DataFrame,
    topic_labels: dict,
    n_negatives: int = 4500,
) -> pd.DataFrame:
    """Combine existing positive pairs with sampled hard negatives."""
    passages_df = passages_df.set_index("ref")
    neg_refs = sample_hard_negatives(pairs_df, topic_labels, n_negatives)

    neg_rows = []
    for ref_a, ref_b in neg_refs:
        if ref_a in passages_df.index and ref_b in passages_df.index:
            neg_rows.append(_make_pair(passages_df, ref_a, ref_b, label=0))

    return pd.concat([pairs_df, pd.DataFrame(neg_rows)], ignore_index=True)
