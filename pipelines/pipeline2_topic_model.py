"""
Pipeline 2: Topic model trained on Dicta lemmas → hard negative mining.

Steps:
  1. Vectorize passages using the `lex` column (Dicta lemmas):
       - CountVectorizer (raw counts) → LDA input
       - TfidfVectorizer (log-normalized) → NMF, cosine similarity
  2. Fit BERTopic (or LDA/NMF fallback) on the appropriate matrix
  3. Assign topic labels to all passages
  4. Mine hard negatives: cross-topic pairs
  5. Produce labeled pairs DataFrame identical in schema to Pipeline 1 output

Why lemmas, not raw text:
  - Sidesteps tokenization/vocabulary gaps in Hebrew/Aramaic LLM models
  - Morphological normalization means אוֹנָאָה and אונאה are the same feature
  - Dicta lemmatization handles prefix stripping, so ובאונאה → אונאה

BERTopic — multilingual sentence-transformer:
  - Uses language="multilingual" (paraphrase-multilingual-MiniLM-L12-v2)
  - Handles Hebrew/Aramaic natively; no workaround needed
  - Downloads ~400MB model on first run
  - vectorizer_model uses token_pattern=r"[^\s]+" for Hebrew c-TF-IDF step

GPU acceleration (BERTopic only):
  - UMAP → cuML UMAP          (cuml package, requires CUDA)
  - HDBSCAN → cuML HDBSCAN    (cuml package, requires CUDA)
  - Falls back to CPU versions if CUDA / cuML unavailable

Dependencies:
  pip install bertopic scikit-learn umap-learn hdbscan sentence-transformers
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
    Ensure bertopic, umap-learn, hdbscan, sentence-transformers are importable.
    Installs them if missing. Returns True if all available after check.
    """
    deps = {
        "bertopic":              "bertopic",
        "umap":                  "umap-learn",
        "hdbscan":               "hdbscan",
        "sentence_transformers": "sentence-transformers",
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
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if nvcc.returncode == 0:
            import re
            m = re.search(r"release (\d+)\.(\d+)", nvcc.stdout)
            if m:
                return int(m.group(1)), int(m.group(2))
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

print("Checking BERTopic dependencies ...")
BERTOPIC_AVAILABLE = _ensure_cpu_bertopic_deps()
if BERTOPIC_AVAILABLE:
    from bertopic import BERTopic
    from umap import UMAP as CpuUMAP
    from hdbscan import HDBSCAN as CpuHDBSCAN
    print("  ✓ BERTopic stack ready (CPU)")
else:
    print("  ✗ BERTopic unavailable — only LDA/NMF will work")

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

print()


# ---------------------------------------------------------------------------
# Vectorization — two matrices, one vocabulary
# ---------------------------------------------------------------------------

def _base_vocab_params(
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 15_000,
    ngram_range: tuple[int, int] = (1, 1),
) -> dict:
    """
    Shared vocabulary parameters for both vectorizers.

    Parameters
    ----------
    min_df       : ignore lemmas appearing in fewer than min_df passages.
                   Lower (e.g. 2) retains rare legal terms; higher (e.g. 5)
                   keeps only well-attested vocabulary.
    max_df       : ignore lemmas appearing in more than this fraction of
                   passages. Removes formulaic phrases common to all topics
                   (e.g. אמר, כדי) that add noise rather than signal.
    max_features : vocabulary size cap. 15,000 covers virtually all
                   attested Talmudic lemmas; reduce for faster iteration.
    ngram_range  : (1,1) = unigrams only; (1,2) = unigrams + bigrams.
                   Bigrams capture collocations like שבת מלאכה or
                   נזיקין ממון that are more topic-specific than either
                   lemma alone. Note: bigrams increase matrix size ~10x.
    """
    return dict(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer="word",
        token_pattern=r"[^\s]+",  # any non-whitespace token (handles Hebrew)
    )


def build_count_matrix(
    passages_df: pd.DataFrame,
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 15_000,
    ngram_range: tuple[int, int] = (1, 1),
) -> tuple:
    """
    Fit a raw term-count matrix on lemmatized texts.

    This is the correct input for LDA. LDA is a generative model whose
    likelihood is defined over integer word counts — feeding it TF-IDF
    floats breaks the multinomial assumption and produces incoherent topics.

    Returns (vectorizer, count_matrix, feature_names)
    """
    vectorizer = CountVectorizer(
        **_base_vocab_params(min_df, max_df, max_features, ngram_range)
    )
    matrix = vectorizer.fit_transform(passages_df["lex"])
    feature_names = vectorizer.get_feature_names_out()

    print(f"Count matrix:  {matrix.shape[0]} passages × "
          f"{matrix.shape[1]} lemma features  "
          f"(ngram={ngram_range}, min_df={min_df}, max_df={max_df})  [LDA]")
    return vectorizer, matrix, feature_names


def build_tfidf_matrix(
    passages_df: pd.DataFrame,
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 15_000,
    ngram_range: tuple[int, int] = (1, 1),
) -> tuple:
    """
    Fit a TF-IDF matrix on lemmatized texts.

    Used for:
      - NMF  (non-negativity constraint is compatible with TF-IDF floats)
      - Cosine similarity baseline (IDF downweighting + L2 norm needed for
        fair comparison across passages of different lengths)

    sublinear_tf=True: replaces raw tf with 1+log(tf), dampening the effect
    of Talmudic repetition of formulaic phrases (e.g. תא שמע, הכי קאמר).

    Returns (vectorizer, tfidf_matrix, feature_names)
    """
    vectorizer = TfidfVectorizer(
        **_base_vocab_params(min_df, max_df, max_features, ngram_range),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(passages_df["lex"])
    feature_names = vectorizer.get_feature_names_out()

    print(f"TF-IDF matrix: {matrix.shape[0]} passages × "
          f"{matrix.shape[1]} lemma features  "
          f"(ngram={ngram_range}, min_df={min_df}, max_df={max_df})  [NMF, cosine sim]")
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

    n_range: e.g. range(10, 61, 5) to test 10, 15, 20 ... 60 topics.
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
# BERTopic — multilingual sentence-transformer + CPU/GPU UMAP/HDBSCAN
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
            metric="euclidean",
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
    n_topics: int = 60,
    min_df: int = 1,
    seed: int = 42,
    force_cpu: bool = False,
) -> tuple:
    """
    BERTopic over lemmatized texts using a multilingual sentence-transformer.

    Why language="multilingual":
      BERTopic defaults to language="english", which loads an English-only
      sentence-transformer and uses an ASCII-only internal tokenizer. Hebrew
      and Aramaic tokens are silently dropped, producing an empty vocabulary
      and a ValueError during the c-TF-IDF topic representation step.
      Setting language="multilingual" loads paraphrase-multilingual-MiniLM-L12-v2
      (~400MB, downloaded once), which encodes Hebrew/Aramaic natively.

    Why vectorizer_model uses token_pattern=r"[^\s]+":
      BERTopic's c-TF-IDF step (topic-word representation) uses a separate
      CountVectorizer internally. We override it with a Hebrew-compatible
      token pattern. min_df=1 is used here (not the corpus-level min_df)
      because c-TF-IDF operates on per-topic document groups which are
      much smaller than the full corpus.

    nr_topics:
      BERTopic's HDBSCAN typically finds more topics than desired (135 on
      this corpus). nr_topics merges them down to the target count using
      cosine similarity of topic vectors. Default 60 reflects empirical
      results on this corpus.

    GPU path (when CUDA + cuML available):
      UMAP and HDBSCAN run on GPU via cuML — typically 5-20x faster than CPU.

    Parameters
    ----------
    n_topics  : target number of topics after BERTopic reduction (nr_topics)
    min_df    : min document frequency for the internal c-TF-IDF vectorizer.
                Keep at 1 — topic document groups are small after reduction.
    force_cpu : set True to skip GPU even when available (for debugging)
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic dependencies missing. Run:\n"
            "  pip install bertopic umap-learn hdbscan sentence-transformers"
        )

    use_gpu = CUML_AVAILABLE and not force_cpu
    device_label = "GPU (cuML)" if use_gpu else "CPU"
    print(f"BERTopic: using {device_label} for UMAP + HDBSCAN")
    print(f"BERTopic: language=multilingual, nr_topics={n_topics}")

    umap_model    = _build_umap(seed, use_gpu)
    hdbscan_model = _build_hdbscan(use_gpu)

    # Override internal c-TF-IDF vectorizer with Hebrew-compatible token pattern.
    # min_df=1 because this vectorizer runs on per-topic subsets, not the full corpus.
    vectorizer_model = CountVectorizer(
        token_pattern=r"[^\s]+",
        min_df=min_df,
        stop_words=None,
    )

    topic_model = BERTopic(
        language="multilingual",          # loads paraphrase-multilingual-MiniLM-L12-v2
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=n_topics,
        calculate_probabilities=not use_gpu,
        verbose=True,
    )

    # Train on lemmatized text (lex) for topic quality.
    # str_docs (original text) is returned so callers can pass it to
    # get_representative_docs() and the topic browser for readable output.
    lex_docs = passages_df["lex"].tolist()
    str_docs = passages_df["str"].tolist()
    topics, probs = topic_model.fit_transform(lex_docs)

    n_found    = len(set(t for t in topics if t != -1))
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"BERTopic ({device_label}): {n_found} topics after reduction, "
          f"{n_outliers} outliers ({100*n_outliers/len(topics):.1f}%)")
    print("To inspect topics with original text call: "
          "topic_model.get_representative_docs() after updating docs with str_docs")

    return (
        topic_model,
        np.array(topics),
        np.array(probs) if probs is not None else np.zeros(len(topics)),
        str_docs,
    )


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
    n_range: range = range(10, 41, 5),
    evaluate_topic_range: bool = False,
    min_df: int = 3,
    max_df: float = 0.85,
    ngram_range: tuple[int, int] = (1, 1),
    force_cpu: bool = False,
) -> pd.DataFrame:
    """
    Full Pipeline 2 entrypoint.

    Parameters
    ----------
    passages_df          : passage DataFrame with columns: ref, str, morph, lex
    known_positives_df   : DataFrame with columns: ref_a, ref_b
    method               : "lda" | "nmf" | "bertopic"
    n_topics             : number of topics for LDA/NMF; nr_topics reduction
                           target for BERTopic (default 60 for BERTopic,
                           20 for LDA/NMF — override as needed)
    n_range              : range of topic counts to evaluate when
                           evaluate_topic_range=True, e.g. range(10, 61, 5).
                           Only used for LDA/NMF.
    evaluate_topic_range : fit model for each n in n_range, print perplexity,
                           then return without building pairs. LDA/NMF only.
    min_df               : minimum document frequency for vectorizers.
                           Lower = more vocabulary, more noise.
                           Higher = cleaner features, smaller matrix.
                           Not applied to BERTopic's internal c-TF-IDF
                           vectorizer (always min_df=1 there).
    max_df               : maximum document frequency fraction for vectorizers.
                           Removes formulaic phrases shared across all topics.
    ngram_range          : (1,1) unigrams only; (1,2) adds bigrams.
                           Bigrams improve topic coherence for collocations
                           but increase matrix size ~10x. Not used for
                           BERTopic (sentence-transformer handles context).
    force_cpu            : BERTopic only — disable GPU even when available.

    Returns
    -------
    Labeled pairs DataFrame matching Pipeline 1 output schema,
    plus `lemma_cosine_sim` baseline score and `device` metadata column.
    """
    Path("outputs").mkdir(exist_ok=True)

    # Step 1: Build vectorized matrices.
    # LDA requires raw counts; NMF/cosine sim use TF-IDF.
    # BERTopic uses its own sentence-transformer embeddings — matrices are
    # still built here for the cosine similarity baseline step.
    count_vec,  count_matrix,  count_features  = build_count_matrix(
        passages_df, min_df=min_df, max_df=max_df, ngram_range=ngram_range
    )
    tfidf_vec,  tfidf_matrix,  tfidf_features  = build_tfidf_matrix(
        passages_df, min_df=min_df, max_df=max_df, ngram_range=ngram_range
    )

    # Step 2 (optional): tune topic count — LDA/NMF only
    if evaluate_topic_range and method in ("lda", "nmf"):
        print(f"Evaluating topic range {list(n_range)} ...")
        eval_matrix = count_matrix if method == "lda" else tfidf_matrix
        scores = evaluate_lda_range(eval_matrix, n_range=n_range)
        scores.to_csv("outputs/lda_perplexity_scores.csv", index=False)
        print("Saved to outputs/lda_perplexity_scores.csv")
        return scores

    # Step 3: Fit topic model
    if method == "lda":
        model, topic_assignments, topic_dists = fit_lda(count_matrix, n_topics)
        print("\nTop lemmas per topic:")
        print_lda_topics(model, count_features)
        primary_vectorizer = count_vec
        primary_features   = count_features
        device = "cpu"

    elif method == "nmf":
        model, topic_assignments, topic_dists = fit_nmf(tfidf_matrix, n_topics)
        print("\nTop lemmas per topic (NMF):")
        print_lda_topics(model, tfidf_features)
        primary_vectorizer = tfidf_vec
        primary_features   = tfidf_features
        device = "cpu"

    elif method == "bertopic":
        # BERTopic uses its own multilingual sentence-transformer for embeddings.
        # n_topics here is the nr_topics reduction target (default 60).
        # min_df/max_df/ngram_range are not passed — sentence-transformer
        # handles tokenization; only the cosine sim baseline uses the matrices.
        model, topic_assignments, topic_dists, bertopic_str_docs = fit_bertopic(
            passages_df, n_topics=n_topics, force_cpu=force_cpu
        )
        # Store original-text docs on the model for use in evaluation /
        # topic browser. Callers access via: topic_model.get_document_info(str_docs)
        model._str_docs = bertopic_str_docs
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

    # Step 6: Baseline cosine similarity — always uses TF-IDF matrix
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

    # Save model + vectorizers
    import pickle
    model_path = f"outputs/pipeline2_{method}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":              model,
            "primary_vectorizer": primary_vectorizer,
            "primary_features":   primary_features,
            "tfidf_vectorizer":   tfidf_vec,
            "tfidf_features":     tfidf_features,
            "device":             device,
            "n_topics":           n_topics,
            "method":             method,
            "min_df":             min_df,
            "max_df":             max_df,
            "ngram_range":        ngram_range,
        }, f)
    print(f"Model saved → {model_path}")

    return labeled_pairs
