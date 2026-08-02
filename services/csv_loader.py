"""Loaders for the MMV master data and synonym dictionary.

Reads the two data files shipped in ``data/`` into pandas DataFrames / dicts.
No graph or LLM logic lives here — this is pure I/O so it can be unit-tested
and reused by any node.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Resolve the project's data directory relative to this file so the loader
# works regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

MMV_MASTER_PATH = _DATA_DIR / "mmv_master.csv"
SYNONYM_DICT_PATH = _DATA_DIR / "synonym_dictionary.json"

# Columns expected in mmv_master.csv, with their target dtypes.
_MMV_DTYPES = {
    "mmv_id": "string",
    "make": "string",
    "model": "string",
    "variant": "string",
    "fuel_type": "string",
    "transmission": "string",
    "cc": "Int64",
    "seating_capacity": "Int64",
    "year_start": "Int64",
    "year_end": "Int64",
}


def load_mmv_master(path: str | Path = MMV_MASTER_PATH) -> pd.DataFrame:
    """Load the MMV master catalogue into a DataFrame.

    String columns are stripped of surrounding whitespace; numeric columns are
    coerced to nullable integer types so missing values survive as ``<NA>``
    rather than turning the column into a float.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MMV master CSV not found at: {path}")

    df = pd.read_csv(path, dtype="string")

    missing = set(_MMV_DTYPES) - set(df.columns)
    if missing:
        raise ValueError(
            f"MMV master is missing expected columns: {sorted(missing)}"
        )

    for col, dtype in _MMV_DTYPES.items():
        if dtype == "string":
            df[col] = df[col].str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    return df


def load_synonym_dictionary(path: str | Path = SYNONYM_DICT_PATH) -> dict:
    """Load the synonym dictionary JSON into a nested dict.

    Returns the parsed structure as-is (top-level keys such as
    ``make_synonyms``, ``model_synonyms``, etc.), so callers can look up the
    category they need.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Synonym dictionary not found at: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Synonym dictionary must be a JSON object at the top level")

    return data


def load_all(
    mmv_path: str | Path = MMV_MASTER_PATH,
    synonym_path: str | Path = SYNONYM_DICT_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Convenience helper that loads both data sources at once."""
    return load_mmv_master(mmv_path), load_synonym_dictionary(synonym_path)


if __name__ == "__main__":
    mmv_df, synonyms = load_all()
    print(f"Loaded {len(mmv_df)} MMV records:")
    print(mmv_df.head(10).to_string(index=False))
    print()
    print("Synonym categories:", list(synonyms.keys()))
