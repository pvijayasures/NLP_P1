from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from src.config import (
    TEXT_COLUMN,
    CLEAN_TEXT_COLUMN,
    LOWERCASE,
    REMOVE_URLS,
    REMOVE_HTML,
    REMOVE_NUMBERS,
    REMOVE_PUNCTUATION,
    NORMALIZE_WHITESPACE,
    NORMALIZE_PROFANITY,
    NORMALIZE_REPEATED_CHARACTERS,
    REMOVE_STOPWORDS,
    KEEP_NEGATIONS,
    REMOVE_SHORT_TOKENS,
    MIN_TOKEN_LENGTH,
    STEM_WORDS,
    LEMMATIZE_WORDS,
    DROP_DUPLICATES,
    DROP_EMPTY_TEXTS,
    URL_PATTERN,
    HTML_PATTERN,
    NUMBER_PATTERN,
    WHITESPACE_PATTERN,
    TOKEN_PATTERN,
    PUNCTUATION,
    NEGATION_WORDS,
    PROFANITY_PATTERNS,
    INTERIM_PATH,
    PROCESSED_PATH,
    TRAIN_INTERIM_FILE,
    TRAIN_PROCESSED_FILE,
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
    lemmatize_words: bool = LEMMATIZE_WORDS

    drop_duplicates: bool = DROP_DUPLICATES
    drop_empty_texts: bool = DROP_EMPTY_TEXTS


DEFAULT_CONFIG = PreprocessConfig()
STEMMER = SnowballStemmer("english")
STOPWORDS = set(stopwords.words("english"))


def _resolve_paths(input_file: str | Path, output_file: str | Path) -> tuple[Path, Path]:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.is_absolute():
        input_path = INTERIM_PATH / input_path
    if not output_path.is_absolute():
        output_path = PROCESSED_PATH / output_path

    return input_path, output_path


def _stopword_set(cfg: PreprocessConfig) -> set[str]:
    return STOPWORDS - NEGATION_WORDS if cfg.keep_negations else STOPWORDS


def normalize_profanity(text: str) -> str:
    for pattern, replacement in PROFANITY_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    return text


def normalize_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_text(text: str, cfg: PreprocessConfig = DEFAULT_CONFIG) -> str:
    if not isinstance(text, str):
        return ""

    if cfg.lowercase:
        text = text.lower()
    if cfg.remove_urls:
        text = re.sub(URL_PATTERN, " ", text)
    if cfg.remove_html:
        text = re.sub(HTML_PATTERN, " ", text)
    if cfg.normalize_profanity:
        text = normalize_profanity(text)
    if cfg.normalize_repeated_characters:
        text = normalize_repeated_characters(text)
    if cfg.remove_numbers:
        text = re.sub(NUMBER_PATTERN, " ", text)
    if cfg.remove_punctuation:
        text = text.translate(str.maketrans("", "", PUNCTUATION))
    if cfg.normalize_whitespace:
        text = re.sub(WHITESPACE_PATTERN, " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, text)


def remove_stopwords_tokens(tokens: list[str], cfg: PreprocessConfig = DEFAULT_CONFIG) -> list[str]:
    active_stopwords = _stopword_set(cfg)
    return [token for token in tokens if token not in active_stopwords]


def remove_short_tokens(tokens: list[str], cfg: PreprocessConfig = DEFAULT_CONFIG) -> list[str]:
    return [token for token in tokens if len(token) >= cfg.min_token_length]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(token) for token in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return tokens


def clean_text(text: str, cfg: PreprocessConfig = DEFAULT_CONFIG) -> str:
    normalized_text = normalize_text(text, cfg=cfg)
    tokens = tokenize(normalized_text)

    if cfg.remove_stopwords:
        tokens = remove_stopwords_tokens(tokens, cfg=cfg)
    if cfg.remove_short_tokens:
        tokens = remove_short_tokens(tokens, cfg=cfg)
    if cfg.stem_words:
        tokens = stem_tokens(tokens)
    if cfg.lemmatize_words:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, cfg: PreprocessConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    if cfg.text_column not in df.columns:
        raise ValueError(f"Column '{cfg.text_column}' not found in dataframe.")

    processed_df = df.copy()

    if cfg.drop_duplicates:
        processed_df = processed_df.drop_duplicates(subset=[cfg.text_column])

    processed_df[cfg.clean_text_column] = processed_df[cfg.text_column].fillna("").apply(
        lambda value: clean_text(value, cfg=cfg)
    )

    if cfg.drop_empty_texts:
        processed_df = processed_df[processed_df[cfg.clean_text_column].str.strip() != ""]

    return processed_df


def preprocess_file(
    input_file: str | Path = TRAIN_INTERIM_FILE,
    output_file: str | Path = TRAIN_PROCESSED_FILE,
    cfg: PreprocessConfig = DEFAULT_CONFIG,
) -> Path:
    input_path, output_path = _resolve_paths(input_file=input_file, output_file=output_file)

    df = pd.read_csv(input_path)
    processed_df = preprocess_dataframe(df, cfg=cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular text preprocessing pipeline.")
    parser.add_argument("--input-file", type=str, default=TRAIN_INTERIM_FILE)
    parser.add_argument("--output-file", type=str, default=TRAIN_PROCESSED_FILE)
    parser.add_argument("--text-column", type=str, default=TEXT_COLUMN)
    parser.add_argument("--clean-text-column", type=str, default=CLEAN_TEXT_COLUMN)

    parser.add_argument("--remove-stopwords", action=argparse.BooleanOptionalAction, default=REMOVE_STOPWORDS)
    parser.add_argument("--keep-negations", action=argparse.BooleanOptionalAction, default=KEEP_NEGATIONS)
    parser.add_argument("--stem-words", action=argparse.BooleanOptionalAction, default=STEM_WORDS)
    parser.add_argument("--remove-short-tokens", action=argparse.BooleanOptionalAction, default=REMOVE_SHORT_TOKENS)
    parser.add_argument("--min-token-length", type=int, default=MIN_TOKEN_LENGTH)

    parser.add_argument("--drop-duplicates", action=argparse.BooleanOptionalAction, default=DROP_DUPLICATES)
    parser.add_argument("--drop-empty-texts", action=argparse.BooleanOptionalAction, default=DROP_EMPTY_TEXTS)

    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return replace(
        DEFAULT_CONFIG,
        text_column=args.text_column,
        clean_text_column=args.clean_text_column,
        remove_stopwords=args.remove_stopwords,
        keep_negations=args.keep_negations,
        stem_words=args.stem_words,
        remove_short_tokens=args.remove_short_tokens,
        min_token_length=args.min_token_length,
        drop_duplicates=args.drop_duplicates,
        drop_empty_texts=args.drop_empty_texts,
    )


def main() -> None:
    args = _parse_args()
    cfg = _config_from_args(args)
    output_path = preprocess_file(input_file=args.input_file, output_file=args.output_file, cfg=cfg)
    print(f"Preprocessed data saved to: {output_path}")


if __name__ == "__main__":
    main()

