"""
eda_analysis.py
---------------
Performs Exploratory Data Analysis on the cleaned job market dataset.
Generates:
  - Summary statistics
  - Distribution plots
  - Skill demand bar chart
  - Salary analysis
  - Job level breakdown
  - Correlation analysis
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_jobs.csv")
EDA_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "eda_summary.json")


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    return df


def basic_statistics(df: pd.DataFrame) -> dict:
    stats = {
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "missing_values": int(df.isnull().sum().sum()),
        "unique_job_titles": int(df["job_title"].nunique()) if "job_title" in df.columns else 0,
        "unique_companies": int(df["company"].nunique()) if "company" in df.columns else 0,
        "unique_locations": int(df["location"].nunique()) if "location" in df.columns else 0,
        "date_range": {
            "start": str(df["date_posted"].min()) if "date_posted" in df.columns else "N/A",
            "end": str(df["date_posted"].max()) if "date_posted" in df.columns else "N/A"
        }
    }
    if "salary_inr" in df.columns:
        stats["salary_stats"] = {
            "mean": round(float(df["salary_inr"].mean()), 2),
            "median": round(float(df["salary_inr"].median()), 2),
            "min": round(float(df["salary_inr"].min()), 2),
            "max": round(float(df["salary_inr"].max()), 2),
            "std": round(float(df["salary_inr"].std()), 2)
        }
    return stats


def top_skills_analysis(df: pd.DataFrame, top_n: int = 20) -> dict:
    all_skills = []
    for s in df["skills"].dropna():
        all_skills.extend([x.strip() for x in str(s).split(",") if x.strip()])
    counter = Counter(all_skills)
    top = counter.most_common(top_n)
    return {
        "labels": [t[0] for t in top],
        "values": [t[1] for t in top],
        "total_unique_skills": len(counter)
    }


def job_level_distribution(df: pd.DataFrame) -> dict:
    if "experience_level" not in df.columns:
        return {}
    dist = df["experience_level"].value_counts()
    return {"labels": dist.index.tolist(), "values": dist.values.tolist()}


def salary_by_level(df: pd.DataFrame) -> dict:
    if "salary_lpa" not in df.columns or "experience_level" not in df.columns:
        return {}
    grouped = df.groupby("experience_level")["salary_lpa"].mean().round(2)
    order = ["Entry", "Mid", "Senior", "Lead", "Principal"]
    ordered = {k: float(grouped.get(k, 0)) for k in order if k in grouped}
    return {
        "labels": list(ordered.keys()),
        "values": list(ordered.values())
    }


def jobs_by_location(df: pd.DataFrame) -> dict:
    if "location" not in df.columns:
        return {}
    dist = df["location"].value_counts().head(10)
    return {"labels": dist.index.tolist(), "values": dist.values.tolist()}


def jobs_over_time(df: pd.DataFrame) -> dict:
    if "year_month" not in df.columns:
        return {}
    monthly = df.groupby("year_month").size().reset_index(name="count")
    monthly = monthly.sort_values("year_month")
    return {
        "labels": monthly["year_month"].tolist(),
        "values": monthly["count"].tolist()
    }


def skill_by_job_title(df: pd.DataFrame) -> dict:
    if "job_title" not in df.columns or "skills" not in df.columns:
        return {}
    result = {}
    for title in df["job_title"].unique():
        subset = df[df["job_title"] == title]["skills"].dropna()
        all_skills = []
        for s in subset:
            all_skills.extend([x.strip() for x in str(s).split(",") if x.strip()])
        top5 = Counter(all_skills).most_common(5)
        result[title] = [t[0] for t in top5]
    return result


def remote_distribution(df: pd.DataFrame) -> dict:
    if "remote" not in df.columns:
        return {}
    dist = df["remote"].value_counts()
    return {"labels": dist.index.tolist(), "values": dist.values.tolist()}


def industry_distribution(df: pd.DataFrame) -> dict:
    if "industry" not in df.columns:
        return {}
    dist = df["industry"].value_counts()
    return {"labels": dist.index.tolist(), "values": dist.values.tolist()}


def correlation_analysis(df: pd.DataFrame) -> dict:
    num_cols = ["salary_inr", "skill_count", "level_numeric"]
    available = [c for c in num_cols if c in df.columns]
    if len(available) < 2:
        return {}
    corr = df[available].corr().round(3)
    return {col: corr[col].to_dict() for col in corr.columns}


def run_eda():
    df = load_data()
    logger.info("Running EDA...")

    summary = {
        "basic_stats": basic_statistics(df),
        "top_skills": top_skills_analysis(df),
        "job_level_distribution": job_level_distribution(df),
        "salary_by_level": salary_by_level(df),
        "jobs_by_location": jobs_by_location(df),
        "jobs_over_time": jobs_over_time(df),
        "skill_by_job_title": skill_by_job_title(df),
        "remote_distribution": remote_distribution(df),
        "industry_distribution": industry_distribution(df),
        "correlation": correlation_analysis(df)
    }

    with open(EDA_OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"EDA summary saved to {EDA_OUTPUT_PATH}")
    return summary


if __name__ == "__main__":
    results = run_eda()
    print("\n=== EDA Summary ===")
    print(f"Total records: {results['basic_stats']['total_records']}")
    print(f"Unique skills: {results['top_skills']['total_unique_skills']}")
    print(f"Top skill: {results['top_skills']['labels'][0]}")
    print(f"\nSalary stats: {results['basic_stats'].get('salary_stats', {})}")
