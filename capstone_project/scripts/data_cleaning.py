"""
data_cleaning.py
----------------
Cleans and preprocesses the raw job market dataset.
Steps:
  1. Load raw CSV
  2. Normalize column names
  3. Remove duplicates
  4. Handle missing values
  5. Type conversions
  6. Save processed CSV
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "job_market_data.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_jobs.csv")


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records, {len(df.columns)} columns")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info(f"Normalized columns: {list(df.columns)}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["job_id"], keep="first")
    removed = before - len(df)
    logger.info(f"Removed {removed} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing salary with median per experience level
    if "salary_inr" in df.columns:
        df["salary_inr"] = df.groupby("experience_level")["salary_inr"].transform(
            lambda x: x.fillna(x.median())
        )
        df["salary_inr"] = df["salary_inr"].fillna(df["salary_inr"].median())

    # Fill categorical columns
    for col in ["location", "remote", "industry", "company"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Fill skills column
    if "skills_required" in df.columns:
        df["skills_required"] = df["skills_required"].fillna("")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after handling: {missing_after}")
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
        df["year"] = df["date_posted"].dt.year
        df["month"] = df["date_posted"].dt.month
        df["year_month"] = df["date_posted"].dt.to_period("M").astype(str)

    if "salary_inr" in df.columns:
        df["salary_inr"] = pd.to_numeric(df["salary_inr"], errors="coerce").astype(float)
        df["salary_lpa"] = (df["salary_inr"] / 100000).round(2)

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived columns."""
    if "experience_level" in df.columns:
        level_map = {"Entry": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Principal": 5}
        df["level_numeric"] = df["experience_level"].map(level_map).fillna(0)

    if "skills_required" in df.columns:
        df["skill_count"] = df["skills_required"].apply(
            lambda x: len([s.strip() for s in str(x).split(",") if s.strip()])
        )

    return df


def clean_pipeline(raw_path: str = RAW_PATH, save_path: str = PROCESSED_PATH) -> pd.DataFrame:
    df = load_data(raw_path)
    df = normalize_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_types(df)
    df = add_derived_columns(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Cleaned data saved to {save_path}")
    logger.info(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = clean_pipeline()
    print("\n=== Cleaning Summary ===")
    print(df.dtypes)
    print(f"\nSample:\n{df.head(3)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
