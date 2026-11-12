"""MFA transcript reading, contextual embeddings and time-alignment for the SLEP pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import List, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("slep_thermo.embeddings")


def read_mfa_words(words_path: Path, times_path: Path) -> Tuple[List[str], np.ndarray]:
    """Read the dataset's word transcript and matching word-onset file.

    Supports one token per line or whitespace-separated tokens. The numeric
    onset file is parsed as a sequence of floating-point values.
    """
    words_path, times_path = Path(words_path), Path(times_path)
    if not words_path.is_file():
        raise FileNotFoundError(f"MFA word transcript not found: {words_path}")
    if not times_path.is_file():
        raise FileNotFoundError(f"MFA word onset file not found: {times_path}")

    raw = words_path.read_text(encoding="utf-8-sig").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    # Dataset files normally contain one word per line; fall back to whitespace
    # tokenisation if the number of lines does not match the onset count.
    time_text = times_path.read_text(encoding="utf-8-sig")
    time_values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", time_text)
    if not time_values:
        raise ValueError(f"No numeric word onsets found in {times_path}")
    word_times = np.asarray([float(value) for value in time_values], dtype=float)
    words = lines if len(lines) == len(word_times) else raw.split()
    if len(words) != len(word_times):
        raise ValueError(
            f"Word/onset count mismatch: {len(words)} words in {words_path}, "
            f"but {len(word_times)} onsets in {times_path}. Inspect the MFA files."
        )
    # Remove pause tokens and corresponding times together.
    keep = np.asarray([w.lower() not in {"<sp>", "sp"} for w in words], dtype=bool)
    words = [w for w, keep_word in zip(words, keep) if keep_word]
    word_times = word_times[keep]
    if np.any(~np.isfinite(word_times)) or np.any(np.diff(word_times) < 0):
        raise ValueError("Word onset times must be finite and non-decreasing.")
    return words, word_times


def make_context_texts(words: Sequence[str], context_words: int) -> List[str]:
    """Causal context ending at each word (current and preceding words only)."""
    return [" ".join(words[max(0, i - context_words + 1): i + 1]) for i in range(len(words))]


def embedding_cache_key(words: Sequence[str], model_name: str, context_words: int) -> str:
    payload = "\n".join(words) + f"\nMODEL={model_name}\nCONTEXT={context_words}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_create_embeddings(
    words: Sequence[str],
    model_name: str,
    context_words: int,
    cache_path: Path,
    batch_size: int = 128,
) -> np.ndarray:
    """Load a validated cache or create contextual embeddings with SentenceTransformers."""
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    key = embedding_cache_key(words, model_name, context_words)
    if cache_path.is_file() and metadata_path.is_file():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("cache_key") == key:
            emb = np.load(cache_path)
            if emb.ndim == 2 and emb.shape[0] == len(words):
                LOG.info("Loaded cached semantic embeddings: %s", cache_path)
                return np.asarray(emb, dtype=np.float32)
        LOG.warning("Embedding cache exists but metadata does not match; regenerating it.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Run `poetry install --with semantic` "
            "or provide precomputed --word-embeddings."
        ) from exc

    texts = make_context_texts(words, context_words)
    LOG.info("Encoding %d contextual word sequences with %s", len(texts), model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=True,
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    metadata_path.write_text(json.dumps({
        "cache_key": key,
        "model_name": model_name,
        "context_words": context_words,
        "n_words": len(words),
        "embedding_dim": int(embeddings.shape[1]),
    }, indent=2), encoding="utf-8")
    return embeddings


def align_embeddings_to_time(
    words: Sequence[str],
    word_onsets_s: np.ndarray,
    embeddings: np.ndarray,
    time_from_podcast_onset_s: np.ndarray,
    neural_lag_s: float,
) -> np.ndarray:
    """Assign the latest causal context vector to each neural time point."""
    if len(words) != len(word_onsets_s) or len(words) != embeddings.shape[0]:
        raise ValueError("Words, word onsets and embeddings must have matching first dimensions.")
    effective_onsets = np.asarray(word_onsets_s, dtype=float) + neural_lag_s
    idx = np.searchsorted(effective_onsets, time_from_podcast_onset_s, side="right") - 1
    aligned = np.full((len(time_from_podcast_onset_s), embeddings.shape[1]), np.nan, dtype=np.float32)
    valid = idx >= 0
    aligned[valid] = embeddings[idx[valid]]
    return aligned
