"""
Pipeline 1: Claude Haiku (Batch API) topic labeling + hard negative mining.

Steps:
  1. Send each passage to Claude Haiku (Batch API) for topic tagging
  2. Use topic tags to mine hard negatives
  3. Produce a labeled pairs DataFrame ready for similarity model training
  4. Optionally evaluate hardest pairs with Claude Sonnet

Cost estimate (4,500 passages, ~100 tokens each):
  - Haiku input:  4,500 * 100 * $1/1M   = ~$0.45
  - Haiku output: 4,500 *  50 * $5/1M   = ~$1.13
  - Batch discount (50%):                 ~$0.80 total
"""

from __future__ import annotations

import os
import json
import time
import pandas as pd
from pathlib import Path
import anthropic

from utils.data import add_negatives_to_pairs, build_pairs


# ---------------------------------------------------------------------------
# Prompt design
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in Talmudic literature and Jewish law.
Your task is to classify a Talmudic passage into legal/thematic topics.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Required fields:
{
  "topic_primary": "<one of: tort_law | purity | sabbath | prayer | 
                    marriage_family | commerce_contracts | temple_ritual |
                    agricultural_law | criminal_law | other>",
  "topic_secondary": "<optional more specific label, or null>",
  "key_concepts": ["<lemma1>", "<lemma2>", "<lemma3>"],
  "confidence": <0.0-1.0>
}"""


def _make_user_message(ref: str, raw_text: str, lex: str) -> str:
    return (
        f"Reference: {ref}\n\n"
        f"Raw text:\n{raw_text[:500]}\n\n"   # cap at 500 chars to control cost
        f"Lemmatized form:\n{lex[:500]}"
    )


# ---------------------------------------------------------------------------
# Batch submission
# ---------------------------------------------------------------------------

def submit_batch(
    passages_df: pd.DataFrame,
    batch_size: int = 10_000,
) -> list[str]:
    """
    Submit passages to Claude Haiku via the Message Batches API.
    Returns a list of batch IDs (one per batch of up to 10,000 requests).

    Batches are submitted but NOT polled here — call poll_batches() separately.
    This allows async overnight processing.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    batch_ids = []
    chunks = [
        passages_df.iloc[i:i+batch_size]
        for i in range(0, len(passages_df), batch_size)
    ]

    for chunk_idx, chunk in enumerate(chunks):
        requests = []
        for _, row in chunk.iterrows():
            requests.append({
                "custom_id": str(row["ref"]),
                "params": {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {
                            "role": "user",
                            "content": _make_user_message(
                                row["ref"], row["str"], row["lex"]
                            )
                        }
                    ]
                }
            })

        batch = client.beta.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        print(f"Submitted batch {chunk_idx + 1}/{len(chunks)}: {batch.id} "
              f"({len(requests)} requests)")

    # Persist batch IDs so you can poll later
    Path("outputs/batch_ids.json").write_text(json.dumps(batch_ids))
    print(f"\nBatch IDs saved to outputs/batch_ids.json")
    print("Poll status with: poll_batches(batch_ids)")
    return batch_ids


# ---------------------------------------------------------------------------
# Batch polling + result parsing
# ---------------------------------------------------------------------------

def poll_batches(
    batch_ids: list[str],
    poll_interval_seconds: int = 60,
    max_wait_hours: int = 24,
) -> dict[str, dict]:
    """
    Poll batch status until all complete. Returns dict: {ref -> topic_data}.
    Blocks until done — run in a background process or notebook cell.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = {}
    pending = set(batch_ids)
    max_polls = (max_wait_hours * 3600) // poll_interval_seconds

    for poll_num in range(int(max_polls)):
        still_pending = set()
        for batch_id in pending:
            batch = client.beta.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                # Retrieve results
                for result in client.beta.messages.batches.results(batch_id):
                    ref = result.custom_id
                    if result.result.type == "succeeded":
                        try:
                            raw = result.result.message.content[0].text
                            parsed = json.loads(raw)
                            results[ref] = parsed
                        except (json.JSONDecodeError, IndexError, KeyError) as e:
                            print(f"Parse error for {ref}: {e}")
                            results[ref] = {"topic_primary": "other",
                                           "confidence": 0.0, "error": str(e)}
                    else:
                        print(f"Failed: {ref} — {result.result.type}")
                        results[ref] = {"topic_primary": "other", "confidence": 0.0}
                print(f"Batch {batch_id} complete. "
                      f"Total results so far: {len(results)}")
            else:
                still_pending.add(batch_id)
                counts = batch.request_counts
                print(f"Batch {batch_id}: {batch.processing_status} | "
                      f"processing={counts.processing} "
                      f"succeeded={counts.succeeded} "
                      f"errored={counts.errored}")

        pending = still_pending
        if not pending:
            break

        print(f"Waiting {poll_interval_seconds}s... "
              f"({len(pending)} batches pending)")
        time.sleep(poll_interval_seconds)

    # Persist results
    Path("outputs/haiku_topic_labels.json").write_text(
        json.dumps(results, ensure_ascii=False)
    )
    print(f"\nResults saved to outputs/haiku_topic_labels.json")
    return results


# ---------------------------------------------------------------------------
# Sonnet evaluation of hard pairs (optional, targeted)
# ---------------------------------------------------------------------------

SONNET_SYSTEM = """You are an expert in Talmudic law.
Given two Talmudic passages, judge whether they discuss the same legal or 
thematic topic, even if the terminology differs.

Respond ONLY with JSON:
{
  "same_topic": <true|false>,
  "similarity_type": "<parallel_tradition|legal_analogy|shared_source|unrelated>",
  "reasoning": "<one sentence>",
  "confidence": <0.0-1.0>
}"""


def evaluate_hard_pairs_sonnet(
    hard_pairs: pd.DataFrame,
    max_pairs: int = 200,
) -> pd.DataFrame:
    """
    Use Claude Sonnet to evaluate the hardest pairs:
    low lemma overlap but same topic cluster per Haiku.
    Limited to max_pairs to control cost.

    Cost estimate: 200 pairs * ~300 tokens * $3/1M ≈ $0.18
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sample = hard_pairs.head(max_pairs)
    evaluations = []

    for _, row in sample.iterrows():
        prompt = (
            f"Passage A ({row['ref_a']}):\n{row['str_a'][:400]}\n\n"
            f"Passage B ({row['ref_b']}):\n{row['str_b'][:400]}"
        )
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=SONNET_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(response.content[0].text)
        except Exception as e:
            result = {"same_topic": None, "error": str(e)}

        evaluations.append({
            "ref_a": row["ref_a"],
            "ref_b": row["ref_b"],
            **result
        })

    return pd.DataFrame(evaluations)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline_1(
    passages_df: pd.DataFrame,
    known_positives_df: pd.DataFrame,
    mode: str = "submit",          # "submit" | "poll" | "build"
    batch_ids: list[str] = None,
    topic_labels: dict = None,
) -> "pd.DataFrame | list[str]":
    """
    Full Pipeline 1 entrypoint.

    mode="submit" : submit batches, return batch_ids (run first)
    mode="poll"   : poll existing batches, return topic_labels dict
    mode="build"  : given topic_labels, build final labeled pairs DataFrame
    """
    os.makedirs("outputs", exist_ok=True)

    if mode == "submit":
        return submit_batch(passages_df)

    elif mode == "poll":
        assert batch_ids, "Provide batch_ids for poll mode"
        return poll_batches(batch_ids)

    elif mode == "build":
        assert topic_labels, "Provide topic_labels dict for build mode"

        # Attach topic labels to passages
        passages_df = passages_df.copy()
        passages_df["topic"] = passages_df["ref"].map(
            lambda r: topic_labels.get(r, {}).get("topic_primary", "other")
        )
        passages_df["topic_confidence"] = passages_df["ref"].map(
            lambda r: topic_labels.get(r, {}).get("confidence", 0.0)
        )

        # Build positive pairs from known Levenshtein matches
        pairs_df = build_pairs(passages_df.reset_index(drop=True)
                               if "ref" not in passages_df.columns
                               else passages_df,
                               known_positives_df)

        # Mine hard negatives using topic labels
        topic_map = dict(zip(passages_df["ref"], passages_df["topic"]))
        labeled_pairs = add_negatives_to_pairs(
            pairs_df, passages_df, topic_map, n_negatives=len(pairs_df)
        )

        # Attach topic labels to pairs for analysis
        labeled_pairs["topic_a"] = labeled_pairs["ref_a"].map(topic_map)
        labeled_pairs["topic_b"] = labeled_pairs["ref_b"].map(topic_map)

        labeled_pairs.to_parquet("outputs/pipeline1_labeled_pairs.parquet")
        print(f"Saved {len(labeled_pairs)} pairs to "
              f"outputs/pipeline1_labeled_pairs.parquet")
        print(labeled_pairs["label"].value_counts().to_string())
        return labeled_pairs

    else:
        raise ValueError(f"Unknown mode: {mode}")
