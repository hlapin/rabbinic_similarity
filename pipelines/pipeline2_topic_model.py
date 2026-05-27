"""
Pipeline 2: Topic model trained on Dicta lemmas → hard negative mining.

Steps:
  1. Vectorize passages using the `lex` column (Dicta lemmas):
       - CountVectorizer (raw counts) → LDA input
       - TfidfVectorizer (log-normalized) → NMF, BERTopic, cosine similarity
  2. Fit BERTopic (or LDA/NMF fallback) on the appropriate matrix
  3. Assign topic labels to all passages
  4. Mine hard negatives: cross-topic pairs
  5. Produce labeled pairs DataFrame identical in schema to Pipeline 1 output

Why lemmas, not raw text:
  - Sidesteps tokenization/vocabulary gaps in Hebrew/Aramaic LLM models
  - Morphological normalization means אוֹנָאָה and אונאה are the same feature
  - Dicta lemmatization handles prefix stripping, so ובאונאה → אונאה

GPU acceleration (BERTopic only):
  - UMAP → cuML UMAP          (cuml package, requires CUDA)
  - HDBSCAN → cuML HDBSCAN    (cuml package, requires CUDA)
  - Falls back to CPU versions if CUDA / cuML unavailable

Dependencies:
  pip install bertopic scikit-learn umap-learn hdbscan
  GPU extras (optional):
    pip install cudf-cu12 cuml-cu12 --extra-index-url https://pypi.nvidia.com
  (LDA/NMF path: scikit-learn only — no extra deps)
"""

from __future__ import annotations

import subprocess
import sys
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

from utils.data import add_negatives_to_pairs, build_pairs


# ---------------------------------------------------------------------------
# GPU / dependency detection
# ---------------------------------------------------------------------------

def _cuda_is_available() -> bool:
    """Check for a working CUDA device via torch (lightweight probe)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    # Fallback: probe nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _install_package(pip_spec: str) -> bool:
    """Install a package at runtime. Returns True on success."""
    print(f"  Installing {pip_spec} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_spec,
         "--quiet", "--disable-pip-version-check"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ✗ Failed: {result.stderr.strip()[:200]}")
        return False
    print(f"  ✓ Installed {pip_spec}")
    return True


def _ensure_cpu_bertopic_deps() -> bool:
    """
    Ensure bertopic, umap-learn, hdbscan are importable.
    Installs them if missing. Returns True if all available after check.
    """
    deps = {
        "bertopic": "bertopic",
        "umap":     "umap-learn",
        "hdbscan":  "hdbscan",
    }
    all_ok = True
    for import_name, pip_name in deps.items():
        if importlib.util.find_spec(import_name) is None:
            print(f"  '{import_name}' not found.")
            ok = _install_package(pip_name)
            all_ok = all_ok and ok
    return all_ok


def _cuda_version() -> tuple[int, int] | None:
    """
    Return (major, minor) CUDA version from nvidia-smi, or None if unavailable.
    Used to select the correct cuML wheel (cu11 vs cu12).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        # nvidia-smi CUDA version via nvcc is more reliable
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if nvcc.returncode == 0:
            import re
            m = re.search(r"release (\d+)\.(\d+)", nvcc.stdout)
            if m:
                return int(m.group(1)), int(m.group(2))
        # Fallback: read from /usr/local/cuda/version.txt
        cuda_ver_file = Path("/usr/local/cuda/version.txt")
        if cuda_ver_file.exists():
            text = cuda_ver_file.read_text()
            import re
            m = re.search(r"(\d+)\.(\d+)", text)
            if m:
                return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def _ensure_cuml() -> bool:
    """
    Ensure cuML is importable (provides GPU UMAP + HDBSCAN).
    Only attempted when CUDA is present.

    Selects the correct wheel based on CUDA version:
      CUDA 11.x → cuml-cu11
      CUDA 12.x → cuml-cu12  (Colab T4/A100, Kaggle T4/P100 post-2024)

    Uses the RAPIDS pip index rather than the full conda install,
    which avoids the 2-5GB conda environment download.
    """
    if importlib.util.find_spec("cuml") is not None:
        return True

    cuda_ver = _cuda_version()
    if cuda_ver is None:
        print("  Could not determine CUDA version — skipping cuML install")
        return False

    major = cuda_ver[0]
    print(f"  CUDA {major}.{cuda_ver[1]} detected")

    if major >= 12:
        cuml_pkg = "cuml-cu12"
    elif major == 11:
        cuml_pkg = "cuml-cu11"
    else:
        print(f"  CUDA {major} too old for cuML — minimum is CUDA 11")
        return False

    print(f"  Installing {cuml_pkg} from NVIDIA PyPI (this may take a few minutes) ...")
    ok = _install_package(
        f"{cuml_pkg} --extra-index-url https://pypi.nvidia.com"
    )
    return ok and importlib.util.find_spec("cuml") is not None


# Probe once at module load; results cached as module-level flags
print("Detecting compute environment ...")
CUDA_AVAILABLE = _cuda_is_available()
print(f"  CUDA: {'✓ available' if CUDA_AVAILABLE else '✗ not found'}")

# Check / install CPU BERTopic stack
print("Checking BERTopic dependencies ...")
BERTOPIC_AVAILABLE = _ensure_cpu_bertopic_deps()
if BERTOPIC_AVAILABLE:
    from bertopic import BERTopic
    from umap import UMAP as CpuUMAP
    from hdbscan import HDBSCAN as CpuHDBSCAN
    print("  ✓ BERTopic stack ready (CPU)")
else:
    print("  ✗ BERTopic unavailable — only LDA/NMF will work")

# Check / install cuML if CUDA present
CUML_AVAILABLE = False
if CUDA_AVAILABLE and BERTOPIC_AVAILABLE:
    print("CUDA detected — checking cuML for GPU acceleration ...")
    CUML_AVAILABLE = _ensure_cuml()
    if CUML_AVAILABLE:
        from cuml.manifold import UMAP as GpuUMAP
        from cuml.cluster import HDBSCAN as GpuHDBSCAN
        print("  ✓ cuML ready — BERTopic will run on GPU")
    else:
        print("  ✗ cuML install failed — BERTopic will use CPU UMAP/HDBSCAN")

print()  # blank line after environment report


# ---------------------------------------------------------------------------
# Vectorization — two matrices, one vocabulary
# ---------------------------------------------------------------------------

def _base_vocab_params(
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 15_000,
) -> dict:
    """Shared vocabulary parameters for both vectorizers."""
    return dict(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        analyzer="word",
        token_pattern=r"[^\s]+",  # any non-whitespace token (handles Hebrew)
    )


def build_count_matrix(
    passages_df: pd.DataFrame,
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 15_000,
) -> tuple:
    """
    Fit a raw term-count matrix on lemmatized texts.

    This is the correct input for LDA. LDA is a generative model whose
    likelihood is defined over integer word counts — feeding it TF-IDF
    floats breaks the multinomial assumption and produces incoherent topics.

    Returns (vectorizer, count_matrix, feature_names)
    """
    vectorizer = CountVectorizer(**_base_vocab_params(min_df, max_df, max_features))
    matrix = vectorizer.fit_transform(passages_df["lex"])
    feature_names = vectorizer.get_feature_names_out()

    print(f"Count matrix:  {matrix.shape[0]} passages × "
          f"{matrix.shape[1]} lemma features  (for LDA)")
    return vectorizer, matrix, feature_names


def build_tfidf_matrix(
    passages_df: pd.DataFrame,
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 15_000,
) -> tuple:
    """
    Fit a TF-IDF matrix on lemmatized texts.

    Used for:
      - NMF  (non-negativity constraint is compatible with TF-IDF floats)
      - BERTopic (feeds TF-IDF representations into UMAP)
      - Cosine similarity baseline (IDF downweighting + L2 norm needed for
        fair comparison across passages of different lengths)

    sublinear_tf=True: replaces raw tf with 1+log(tf), dampening the effect
    of Talmudic repetition of formulaic phrases (e.g. תא שמע, הכי קאמר).

    Returns (vectorizer, tfidf_matrix, feature_names)
    """
    vectorizer = TfidfVectorizer(
        **_base_vocab_params(min_df, max_df, max_features),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(passages_df["lex"])
    feature_names = vectorizer.get_feature_names_out()

    print(f"TF-IDF matrix: {matrix.shape[0]} passages × "
          f"{matrix.shape[1]} lemma features  (for NMF, BERTopic, cosine sim)")
    return vectorizer, matrix, feature_names


# ---------------------------------------------------------------------------
# LDA topic model (baseline — scikit-learn only)
# ---------------------------------------------------------------------------

def fit_lda(
    count_matrix,
    n_topics: int = 20,
    seed: int = 42,
) -> tuple:
    """
    Fit LDA on a raw term-COUNT matrix (not TF-IDF).

    LDA is a generative model with a multinomial likelihood over word counts.
    Passing TF-IDF floats violates this assumption — the model will run but
    optimizes the wrong objective. Always use CountVectorizer output here.

    n_topics=20: reasonable starting point for Talmud's tractate structure.
    Tune using evaluate_lda_range() before committing.

    Returns (model, topic_assignments, topic_distributions)
    """
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=seed,
        learning_method="batch",
        max_iter=50,
        n_jobs=-1,
    )
    topic_distributions = model.fit_transform(count_matrix)
    topic_assignments = topic_distributions.argmax(axis=1)

    print(f"LDA fitted: {n_topics} topics")
    return model, topic_assignments, topic_distributions


def evaluate_lda_range(
    count_matrix,
    n_range: range = range(10, 41, 5),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fit LDA for multiple topic counts on a COUNT matrix, return perplexity.
    Use to choose n_topics before final fit.
    Lower perplexity = better fit, but watch for overfitting past ~30 topics.
    """
    results = []
    for n in n_range:
        model = LatentDirichletAllocation(
            n_components=n, random_state=seed,
            learning_method="batch", max_iter=30, n_jobs=-1
        )
        model.fit(count_matrix)
        perplexity = model.perplexity(count_matrix)
        results.append({"n_topics": n, "perplexity": perplexity})
        print(f"  n_topics={n}: perplexity={perplexity:.1f}")
    return pd.DataFrame(results)


def print_lda_topics(
    model,
    feature_names: np.ndarray,
    n_top_lemmas: int = 10,
) -> None:
    """Print top lemmas per topic for manual inspection and labeling."""
    for topic_idx, topic in enumerate(model.components_):
        top_lemmas = [feature_names[i]
                      for i in topic.argsort()[:-n_top_lemmas - 1:-1]]
        print(f"Topic {topic_idx:02d}: {' | '.join(top_lemmas)}")


# ---------------------------------------------------------------------------
# NMF alternative (often sharper topics than LDA on sparse matrices)
# ---------------------------------------------------------------------------

def fit_nmf(
    tfidf_matrix,
    n_topics: int = 20,
    seed: int = 42,
) -> tuple:
    """
    NMF as an alternative to LDA.
    Often produces more coherent topics on short texts due to non-negativity
    constraint — words either belong to a topic or they don't.
    Same return signature as fit_lda.
    """
    model = NMF(
        n_components=n_topics,
        random_state=seed,
        init="nndsvda",
        max_iter=300,
    )
    topic_distributions = model.fit_transform(tfidf_matrix)
    topic_assignments = topic_distributions.argmax(axis=1)

    print(f"NMF fitted: {n_topics} topics")
    return model, topic_assignments, topic_distributions


# ---------------------------------------------------------------------------
# BERTopic — CPU or GPU depending on environment
# ---------------------------------------------------------------------------

def _build_umap(seed: int, use_gpu: bool):
    """
    Construct UMAP model on GPU (cuML) or CPU (umap-learn).

    cuML UMAP differences vs CPU:
      - Does not support random_state via constructor; set cupy seed instead
      - metric must be 'euclidean' or 'cosine' (subset of CPU support)
      - Returns cupy arrays; BERTopic handles conversion automatically
    """
    if use_gpu:
        import cupy as cp
        cp.random.seed(seed)
        return GpuUMAP(
            n_neighbors=15,
            n_components=5,
            metric="euclidean",   # cuML UMAP is faster with euclidean
        )
    else:
        return CpuUMAP(
            n_neighbors=15,
            n_components=5,
            metric="cosine",
            random_state=seed,
            low_memory=False,
        )


def _build_hdbscan(use_gpu: bool):
    """
    Construct HDBSCAN model on GPU (cuML) or CPU (hdbscan).

    cuML HDBSCAN differences vs CPU:
      - prediction_data not supported (no soft clustering)
      - cluster_selection_method: only 'eom' supported
      - gen_min_span_tree not supported
    """
    if use_gpu:
        return GpuHDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
        )
    else:
        return CpuHDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            gen_min_span_tree=True,
        )


def fit_bertopic(
    passages_df: pd.DataFrame,
    n_topics: int = 20,
    seed: int = 42,
    force_cpu: bool = False,
) -> tuple:
    """
    BERTopic over lemmatized texts, using TF-IDF matrix as embeddings.

    Why we pass embeddings explicitly:
      BERTopic's default pipeline runs a sentence-transformer to produce
      embeddings, then feeds them into UMAP. For Hebrew/Aramaic there is no
      suitable sentence-transformer, so we skip that step entirely by
      pre-computing a TF-IDF matrix and passing it directly as `embeddings`
      to fit_transform(). BERTopic then goes straight to UMAP → HDBSCAN
      → topic representation, bypassing the neural embedding step.

      The internal CountVectorizer (vectorizer_model) is used only for the
      final topic-word representation step (c-TF-IDF), not for embeddings.
      It must use token_pattern=r"[^\s]+" to handle Hebrew/Aramaic tokens;
      the default r"(?u)\b\w\w+\b" pattern matches only ASCII word characters
      and produces an empty vocabulary on Hebrew text.

    GPU path (when CUDA + cuML available):
      UMAP and HDBSCAN run on GPU via cuML — typically 5-20x faster than CPU
      for corpora of this size.

    CPU path (fallback):
      Standard umap-learn + hdbscan; slower but identical results.

    Parameters
    ----------
    force_cpu : set True to skip GPU even when available (for debugging)
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic dependencies missing. Run:\n"
            "  pip install bertopic umap-learn hdbscan"
        )

    use_gpu = CUML_AVAILABLE and not force_cpu
    device_label = "GPU (cuML)" if use_gpu else "CPU"
    print(f"BERTopic: using {device_label} for UMAP + HDBSCAN")

    umap_model    = _build_umap(seed, use_gpu)
    hdbscan_model = _build_hdbscan(use_gpu)

    # This vectorizer is used only for c-TF-IDF topic-word representations,
    # NOT for producing embeddings. token_pattern must handle Hebrew tokens.
    vectorizer_model = CountVectorizer(
        token_pattern=r"[^\s]+",
        min_df=2,         # lowered from 3: small corpus, avoid empty vocab
        stop_words=None,  # Dicta lemmas already normalized; keep all
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=n_topics,
        calculate_probabilities=not use_gpu,  # cuML HDBSCAN: no soft probs
        verbose=True,
    )

    docs = passages_df["lex"].tolist()

    # Pre-compute TF-IDF embeddings and pass them directly to fit_transform.
    # This bypasses BERTopic's default sentence-transformer embedding step,
    # which has no Hebrew/Aramaic model available.
    _, tfidf_matrix, _ = build_tfidf_matrix(passages_df)
    # Convert sparse matrix to dense numpy array for UMAP compatibility
    embeddings = tfidf_matrix.toarray().astype(np.float32)

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    n_found = len(set(topics))
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"BERTopic ({device_label}): {n_found} topics found, "
          f"{n_outliers} outliers (topic -1)")

    return topic_model, np.array(topics), np.array(probs) if probs is not None else np.zeros(len(topics))


# ---------------------------------------------------------------------------
# Lemma-based cosine similarity (zero-cost baseline)
# ---------------------------------------------------------------------------

def compute_pair_similarities(
    pairs_df: pd.DataFrame,
    passages_df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
) -> pd.DataFrame:
    """
    Add a `lemma_cosine_sim` column to pairs_df.
    This is your free baseline — if separation between positive and negative
    labels is already strong here, a simple retrieval system may suffice.
    """
    ref_to_idx = {ref: i for i, ref in enumerate(passages_df["ref"])}

    sims = []
    for _, row in pairs_df.iterrows():
        idx_a = ref_to_idx.get(row["ref_a"])
        idx_b = ref_to_idx.get(row["ref_b"])
        if idx_a is not None and idx_b is not None:
            sim = cosine_similarity(
                tfidf_matrix[idx_a], tfidf_matrix[idx_b]
            )[0][0]
        else:
            sim = np.nan
        sims.append(sim)

    pairs_df = pairs_df.copy()
    pairs_df["lemma_cosine_sim"] = sims
    return pairs_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline_2(
    passages_df: pd.DataFrame,
    known_positives_df: pd.DataFrame,
    method: str = "lda",
    n_topics: int = 20,
    evaluate_topic_range: bool = False,
    force_cpu: bool = False,
) -> pd.DataFrame:
    """
    Full Pipeline 2 entrypoint.

    Parameters
    ----------
    passages_df          : passage DataFrame (ref, str, morph, lex)
    known_positives_df   : DataFrame with [ref_a, ref_b] (4,500 pairs)
    method               : "lda" | "nmf" | "bertopic"
    n_topics             : number of topics (tune with evaluate_topic_range=True)
    evaluate_topic_range : print perplexity for n in 10..40 then exit (LDA/NMF)
    force_cpu            : disable GPU even when available (BERTopic only)

    Returns
    -------
    Labeled pairs DataFrame matching Pipeline 1 output schema,
    plus `lemma_cosine_sim` baseline score and `device` metadata column.
    """
    Path("outputs").mkdir(exist_ok=True)

    # Step 1: Build both matrices from the same vocabulary parameters.
    # LDA requires raw counts; NMF/BERTopic/cosine sim use TF-IDF.
    count_vec,  count_matrix,  count_features  = build_count_matrix(passages_df)
    tfidf_vec,  tfidf_matrix,  tfidf_features  = build_tfidf_matrix(passages_df)

    # Step 2 (optional): tune topic count
    if evaluate_topic_range and method in ("lda", "nmf"):
        print("Evaluating topic range ...")
        # LDA range evaluation uses count matrix; NMF uses tfidf
        eval_matrix = count_matrix if method == "lda" else tfidf_matrix
        scores = evaluate_lda_range(eval_matrix)
        scores.to_csv("outputs/lda_perplexity_scores.csv", index=False)
        print("Saved to outputs/lda_perplexity_scores.csv")
        return scores

    # Step 3: Fit topic model with the methodologically correct matrix
    if method == "lda":
        # COUNT matrix -- respects LDA's multinomial generative assumption
        model, topic_assignments, topic_dists = fit_lda(count_matrix, n_topics)
        print("\nTop lemmas per topic:")
        print_lda_topics(model, count_features)
        primary_vectorizer = count_vec
        primary_features   = count_features
        device = "cpu"

    elif method == "nmf":
        # TF-IDF matrix -- NMF's non-negativity constraint is compatible with
        # TF-IDF floats; IDF downweighting improves coherence on formulaic text
        model, topic_assignments, topic_dists = fit_nmf(tfidf_matrix, n_topics)
        print("\nTop lemmas per topic (NMF):")
        print_lda_topics(model, tfidf_features)
        primary_vectorizer = tfidf_vec
        primary_features   = tfidf_features
        device = "cpu"

    elif method == "bertopic":
        # TF-IDF representations fed into UMAP dimensionality reduction
        model, topic_assignments, topic_dists = fit_bertopic(
            passages_df, n_topics, force_cpu=force_cpu
        )
        primary_vectorizer = tfidf_vec
        primary_features   = tfidf_features
        device = "gpu" if (CUML_AVAILABLE and not force_cpu) else "cpu"

    else:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: lda, nmf, bertopic")

    # Step 4: Attach topic labels to passages
    passages_df = passages_df.copy()
    passages_df["topic"] = topic_assignments

    # Step 5: Build positive pairs from known Levenshtein matches
    pairs_df = build_pairs(passages_df, known_positives_df)

    # Step 6: Baseline cosine similarity -- always uses TF-IDF matrix
    pairs_df = compute_pair_similarities(
        pairs_df, passages_df, tfidf_vec, tfidf_matrix
    )

    # Step 7: Mine hard negatives (cross-topic pairs)
    topic_map = dict(zip(passages_df["ref"], passages_df["topic"]))
    labeled_pairs = add_negatives_to_pairs(
        pairs_df, passages_df, topic_map, n_negatives=len(pairs_df)
    )

    # Step 8: Annotate for downstream analysis
    labeled_pairs["topic_a"] = labeled_pairs["ref_a"].map(topic_map)
    labeled_pairs["topic_b"] = labeled_pairs["ref_b"].map(topic_map)
    labeled_pairs["device"]  = device

    # Save pairs
    out_path = f"outputs/pipeline2_{method}_labeled_pairs.parquet"
    labeled_pairs.to_parquet(out_path)
    print(f"\nSaved {len(labeled_pairs)} pairs → {out_path}")
    print(labeled_pairs["label"].value_counts().to_string())

    # Save model + both vectorizers
    import pickle
    model_path = f"outputs/pipeline2_{method}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":              model,
            "primary_vectorizer": primary_vectorizer,   # count (LDA) or tfidf (NMF/BERTopic)
            "primary_features":   primary_features,
            "tfidf_vectorizer":   tfidf_vec,            # always saved; used for cosine sim
            "tfidf_features":     tfidf_features,
            "device":             device,
            "n_topics":           n_topics,
            "method":             method,
        }, f)
    print(f"Model saved → {model_path}")

    return labeled_pairs