"""
Evaluation: compare Pipeline 1 (Haiku) vs Pipeline 2 (topic model) outputs.

Metrics:
  - Label agreement between pipelines
  - Baseline cosine similarity distributions (positive vs negative)
  - Hard pair analysis: where pipelines disagree
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def load_pipeline_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved outputs from both pipelines."""
    p1 = pd.read_parquet("outputs/pipeline1_labeled_pairs.parquet")

    # Try to find a pipeline 2 output (any method)
    for method in ("lda", "nmf", "bertopic"):
        path = f"outputs/pipeline2_{method}_labeled_pairs.parquet"
        if Path(path).exists():
            p2 = pd.read_parquet(path)
            print(f"Loaded pipeline 2 ({method})")
            return p1, p2

    raise FileNotFoundError("No Pipeline 2 output found in outputs/")


def compare_topic_labels(
    p1_pairs: pd.DataFrame,
    p2_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    For the shared positive pairs, compare topic labels assigned
    by Haiku (Pipeline 1) vs topic model (Pipeline 2).

    High disagreement pairs are the most interesting for Sonnet evaluation.
    """
    # Merge on ref_a, ref_b
    merged = p1_pairs[p1_pairs.label == 1].merge(
        p2_pairs[p2_pairs.label == 1][["ref_a", "ref_b", "topic_a", "topic_b",
                                        "lemma_cosine_sim"]],
        on=["ref_a", "ref_b"],
        suffixes=("_haiku", "_topicmodel"),
        how="inner"
    )

    print(f"Shared positive pairs: {len(merged)}")

    # Agreement: both pipelines assign same topic to ref_a
    # (Pipeline 1 topic is string, Pipeline 2 is int — compare separately)
    print("\nPipeline 1 topic distribution (Haiku):")
    print(merged["topic_a_haiku"].value_counts().head(10).to_string())

    print("\nPipeline 2 topic distribution (topic model):")
    print(merged["topic_a_topicmodel"].value_counts().head(10).to_string())

    return merged


def similarity_distribution(pairs_df: pd.DataFrame) -> dict:
    """
    Report cosine similarity stats by label (positive vs negative).
    Only meaningful for Pipeline 2 output which has lemma_cosine_sim.
    """
    if "lemma_cosine_sim" not in pairs_df.columns:
        print("lemma_cosine_sim not available (Pipeline 1 output)")
        return {}

    stats = pairs_df.groupby("label")["lemma_cosine_sim"].describe()
    print("\nCosine similarity by label:")
    print(stats.to_string())

    # Separation score: how well does cosine sim distinguish pos from neg?
    pos_mean = pairs_df[pairs_df.label == 1]["lemma_cosine_sim"].mean()
    neg_mean = pairs_df[pairs_df.label == 0]["lemma_cosine_sim"].mean()
    pos_std  = pairs_df[pairs_df.label == 1]["lemma_cosine_sim"].std()

    separation = (pos_mean - neg_mean) / (pos_std + 1e-9)
    print(f"\nBaseline separation (Cohen's d proxy): {separation:.2f}")
    print("  > 0.8: good baseline | 0.5–0.8: moderate | < 0.5: poor")

    return {"pos_mean": pos_mean, "neg_mean": neg_mean, "separation": separation}


def find_hard_pairs(
    pairs_df: pd.DataFrame,
    n: int = 100,
) -> pd.DataFrame:
    """
    Identify the hardest pairs for a similarity model:
    - Positives with LOW cosine similarity (same topic, different vocabulary)
    - Negatives with HIGH cosine similarity (different topic, similar surface)

    These are the pairs most worth reviewing with Claude Sonnet.
    """
    if "lemma_cosine_sim" not in pairs_df.columns:
        print("Need lemma_cosine_sim — run Pipeline 2 first")
        return pairs_df.head(0)

    hard_positives = (
        pairs_df[pairs_df.label == 1]
        .nsmallest(n // 2, "lemma_cosine_sim")
        [["ref_a", "ref_b", "str_a", "str_b", "lemma_cosine_sim", "label"]]
    )

    hard_negatives = (
        pairs_df[pairs_df.label == 0]
        .nlargest(n // 2, "lemma_cosine_sim")
        [["ref_a", "ref_b", "str_a", "str_b", "lemma_cosine_sim", "label"]]
    )

    hard_pairs = pd.concat([hard_positives, hard_negatives], ignore_index=True)
    hard_pairs.to_parquet("outputs/hard_pairs_for_sonnet.parquet")
    print(f"Saved {len(hard_pairs)} hard pairs to "
          f"outputs/hard_pairs_for_sonnet.parquet")
    print("Submit these to Sonnet evaluation in Pipeline 1 for ground truth.")
    return hard_pairs


def run_evaluation():
    """Run full comparison report."""
    print("=" * 60)
    print("PIPELINE COMPARISON REPORT")
    print("=" * 60)

    try:
        p1, p2 = load_pipeline_outputs()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run both pipelines first.")
        return

    print(f"\nPipeline 1 pairs: {len(p1)} "
          f"({p1.label.value_counts().to_dict()})")
    print(f"Pipeline 2 pairs: {len(p2)} "
          f"({p2.label.value_counts().to_dict()})")

    print("\n--- Topic Label Comparison ---")
    merged = compare_topic_labels(p1, p2)

    print("\n--- Baseline Similarity (Pipeline 2) ---")
    stats = similarity_distribution(p2)

    print("\n--- Hard Pairs ---")
    hard = find_hard_pairs(p2, n=200)

    print("\n--- Next Steps ---")
    if stats.get("separation", 0) > 0.8:
        print("✓ Strong baseline separation — lemma TF-IDF alone may be "
              "sufficient for coarse topic similarity.")
        print("  Proceed to fine-tune sentence-transformers on labeled pairs.")
    elif stats.get("separation", 0) > 0.5:
        print("~ Moderate separation — augment with Sonnet evaluation of "
              "hard pairs, then fine-tune.")
    else:
        print("✗ Weak separation — consider:")
        print("  1. Tune n_topics (run evaluate_topic_range=True)")
        print("  2. Filter low-confidence Haiku labels (confidence < 0.7)")
        print("  3. Move to English translation bridge for hard pairs")


if __name__ == "__main__":
    run_evaluation()
