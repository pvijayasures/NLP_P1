import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from src.config import (
    TEXT_COLUMN, CLEAN_TEXT_COLUMN, LOWERCASE, REMOVE_URLS, REMOVE_HTML,
    REMOVE_NUMBERS, REMOVE_PUNCTUATION, NORMALIZE_WHITESPACE, NORMALIZE_PROFANITY,
    NORMALIZE_REPEATED_CHARACTERS, REMOVE_STOPWORDS, KEEP_NEGATIONS,
    REMOVE_SHORT_TOKENS, MIN_TOKEN_LENGTH, STEM_WORDS, DROP_DUPLICATES,
    DROP_EMPTY_TEXTS, URL_PATTERN, HTML_PATTERN, NUMBER_PATTERN,
    WHITESPACE_PATTERN, TOKEN_PATTERN, PUNCTUATION, NEGATION_WORDS,
    PROFANITY_PATTERNS, INTERIM_PATH, PROCESSED_PATH, TRAIN_INTERIM_FILE,
    TRAIN_PROCESSED_FILE
)


@dataclass(frozen=True)
class PreprocessConfig:
    text_column: str = TEXT_COLUMN
    clean_text_column: str = CLEAN_TEXT_COLUMN
    lowercase: bool = LOWERCASE
    remove_urls: bool = REMOVE_URLS
    remove_html: bool = REMOVE_HTML
    remove_numbers: bool = REMOVE_NUMBERS
    remove_punctuation: bool = REMOVE_PUNCTUATION
    normalize_whitespace: bool = NORMALIZE_WHITESPACE
    normalize_profanity: bool = NORMALIZE_PROFANITY
    normalize_repeated_characters: bool = NORMALIZE_REPEATED_CHARACTERS
    remove_stopwords: bool = REMOVE_STOPWORDS
    keep_negations: bool = KEEP_NEGATIONS
    remove_short_tokens: bool = REMOVE_SHORT_TOKENS
    min_token_length: int = MIN_TOKEN_LENGTH
    stem_words: bool = STEM_WORDS
    drop_duplicates: bool = DROP_DUPLICATES
    drop_empty_texts: bool = DROP_EMPTY_TEXTS


DEFAULT_CONFIG = PreprocessConfig()
STEMMER = SnowballStemmer("english")
STOPWORDS = set(stopwords.words("english"))


# -------------------------------------------------------------------
# Core Preprocessing Helpers
# -------------------------------------------------------------------

def _get_stopword_set(cfg: PreprocessConfig) -> set[str]:
    return STOPWORDS - NEGATION_WORDS if cfg.keep_negations else STOPWORDS


def clean_text(text: str, cfg: PreprocessConfig, active_stops: set[str] | None = None) -> str:
    if not isinstance(text, str): return ""

    # Normalization
    if cfg.lowercase: text = text.lower()
    if cfg.remove_urls: text = re.sub(URL_PATTERN, " ", text)
    if cfg.remove_html: text = re.sub(HTML_PATTERN, " ", text)
    if cfg.normalize_profanity:
        for p, r in PROFANITY_PATTERNS.items(): text = re.sub(p, r, text)
    if cfg.normalize_repeated_characters:
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if cfg.remove_numbers: text = re.sub(NUMBER_PATTERN, " ", text)
    if cfg.remove_punctuation:
        text = text.translate(str.maketrans("", "", PUNCTUATION))
    if cfg.normalize_whitespace:
        text = re.sub(WHITESPACE_PATTERN, " ", text).strip()

    # Token-based processing
    tokens = re.findall(TOKEN_PATTERN, text)
    if cfg.remove_stopwords:
        stops = active_stops if active_stops is not None else _get_stopword_set(cfg)
        tokens = [t for t in tokens if t not in stops]
    if cfg.remove_short_tokens:
        tokens = [t for t in tokens if len(t) >= cfg.min_token_length]
    if cfg.stem_words:
        tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def _preprocess_logic(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Internal shared logic for processing a dataframe based on a config."""
    df_copy = df.copy()
    if cfg.drop_duplicates:
        df_copy = df_copy.drop_duplicates(subset=[cfg.text_column])

    active_stops = _get_stopword_set(cfg) if cfg.remove_stopwords else None

    df_copy[cfg.clean_text_column] = df_copy[cfg.text_column].fillna("").apply(
        lambda x: clean_text(x, cfg, active_stops)
    )

    if cfg.drop_empty_texts:
        df_copy = df_copy[df_copy[cfg.clean_text_column].str.strip() != ""]

    return df_copy


# -------------------------------------------------------------------
# Option A: Ablation Experiment (Many Files)
# -------------------------------------------------------------------

def run_ablation_experiment(input_csv: str | Path = TRAIN_INTERIM_FILE):
    """Runs all ablation variants and saves them with 'exp_' prefix."""

    # Define ablation variants relative to DEFAULT_CONFIG
    variants: dict[str, PreprocessConfig] = {
        "baseline_full": DEFAULT_CONFIG,  # Everything is ON
        "no_stemming": replace(DEFAULT_CONFIG, stem_words=False),
        "keep_stopwords": replace(DEFAULT_CONFIG, remove_stopwords=False),
        "keep_punctuation": replace(DEFAULT_CONFIG, remove_punctuation=False),
        "no_profanity_norm": replace(DEFAULT_CONFIG, normalize_profanity=False),
        "keep_short_tokens": replace(DEFAULT_CONFIG, remove_short_tokens=False),
        "minimal_cleaning": PreprocessConfig(
            lowercase=True,
            remove_urls=True,
            remove_html=True,
            remove_stopwords=False,
            remove_punctuation=False,
            stem_words=False,
            remove_short_tokens=False
        ),
    }

    input_path = INTERIM_PATH / input_csv if not Path(input_csv).is_absolute() else Path(input_csv)
    raw_df = pd.read_csv(input_path)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for name, cfg in variants.items():
        print(f"Experimenting: {name}...")
        processed_df = _preprocess_logic(raw_df, cfg)
        output_file = PROCESSED_PATH / f"exp_{name}.csv"
        processed_df.to_csv(output_file, index=False)
        summary_data.append({"variant": name, "output": str(output_file), **asdict(cfg)})

    pd.DataFrame(summary_data).to_csv(PROCESSED_PATH / "experiment_summary.csv", index=False)
    print(f"Ablation complete. Files saved in {PROCESSED_PATH}")


# -------------------------------------------------------------------
# Option B: Standard Production Pipeline (One File)
# -------------------------------------------------------------------

def run_standard_pipeline(input_csv: str | Path = TRAIN_INTERIM_FILE, output_csv: str | Path = TRAIN_PROCESSED_FILE):
    """Runs one cleaning pass using ONLY the settings in src.config."""

    input_path = INTERIM_PATH / input_csv if not Path(input_csv).is_absolute() else Path(input_csv)
    output_path = PROCESSED_PATH / output_csv if not Path(output_csv).is_absolute() else Path(output_csv)

    print(f"Running production pipeline using settings from config.py...")
    df = pd.read_csv(input_path)

    processed_df = _preprocess_logic(df, DEFAULT_CONFIG)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Success! Processed data saved to: {output_path}")


# -------------------------------------------------------------------
# Main Switch
# -------------------------------------------------------------------

if __name__ == "__main__":
    # CHOOSE ONE:

    # 1. Run this to test different settings (creates exp_*.csv files)
    # run_ablation_experiment()

    # 2. Run this once you've picked your best settings in config.py
    run_standard_pipeline()