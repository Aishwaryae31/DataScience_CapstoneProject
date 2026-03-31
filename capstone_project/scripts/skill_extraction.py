"""
skill_extraction.py
-------------------
Extracts and normalizes individual skills from the skills_required column.
Produces:
  - A cleaned skills column (list as comma-separated string)
  - A skill frequency dataframe
  - A skill co-occurrence matrix
"""

import pandas as pd
import numpy as np
import os
import re
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_jobs.csv")
SKILLS_FREQ_PATH = os.path.join(BASE_DIR, "data", "processed", "skill_frequency.csv")
SKILLS_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "skill_frequency.json")

# Normalization map: alias → canonical name
SKILL_ALIASES = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "NLP",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "sci-kit learn": "Scikit-learn",
    "sql server": "SQL",
    "ms sql": "SQL",
    "postgresql": "SQL",
    "mysql": "SQL",
    "gcp": "GCP",
    "google cloud": "GCP",
    "azure cloud": "Azure",
    "amazon web services": "AWS",
    "js": "JavaScript",
    "reactjs": "React",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "vuejs": "Vue.js",
    "angularjs": "Angular",
}


def normalize_skill(skill: str) -> str:
    """Clean and normalize a single skill string."""
    skill = skill.strip().lower()
    skill = re.sub(r"[^a-z0-9\s\.\+#]", "", skill)
    skill = re.sub(r"\s+", " ", skill).strip()
    # Check alias map
    canonical = SKILL_ALIASES.get(skill)
    if canonical:
        return canonical
    # Title-case if not already
    return skill.title()


def extract_skills_from_row(skills_str) -> list:
    """Parse a comma-separated skills string into a clean list."""
    if pd.isna(skills_str) or str(skills_str).strip() == "":
        return []
    raw_skills = str(skills_str).split(",")
    cleaned = []
    for s in raw_skills:
        norm = normalize_skill(s)
        if len(norm) > 1:
            cleaned.append(norm)
    return cleaned


def compute_skill_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each skill appears across all job listings."""
    all_skills = []
    for skills_list in df["skills"]:
        if isinstance(skills_list, list):
            all_skills.extend(skills_list)
        elif isinstance(skills_list, str) and skills_list:
            all_skills.extend([s.strip() for s in skills_list.split(",") if s.strip()])

    counter = Counter(all_skills)
    freq_df = pd.DataFrame(counter.items(), columns=["skill", "frequency"])
    freq_df = freq_df.sort_values("frequency", ascending=False).reset_index(drop=True)
    freq_df["percentage"] = (freq_df["frequency"] / len(df) * 100).round(2)
    return freq_df


def compute_skill_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate skill demand per year-month (for forecasting)."""
    rows = []
    for _, row in df.iterrows():
        skills = row.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        ym = row.get("year_month", "Unknown")
        for skill in skills:
            rows.append({"year_month": ym, "skill": skill})

    if not rows:
        return pd.DataFrame(columns=["year_month", "skill", "count"])

    skill_time_df = pd.DataFrame(rows)
    grouped = skill_time_df.groupby(["year_month", "skill"]).size().reset_index(name="count")
    return grouped


def run_extraction(processed_path: str = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(processed_path)
    logger.info(f"Loaded {len(df)} records for skill extraction")

    # Detect correct source column
    if "skills_required" in df.columns:
        source_col = "skills_required"
    elif "skills" in df.columns:
        source_col = "skills"
    else:
        raise Exception("No skills column found in dataset")

    df["skills"] = df[source_col].apply(extract_skills_from_row)

    # Save skills as comma-separated string
    df["skills_clean"] = df["skills"].apply(lambda x: ", ".join(x))

    # Compute frequency
    freq_df = compute_skill_frequency(df)
    freq_df.to_csv(SKILLS_FREQ_PATH, index=False)
    logger.info(f"Skill frequency saved to {SKILLS_FREQ_PATH}")

    # Save top 30 skills as JSON for dashboard
    top30 = freq_df.head(30)
    json_data = {
        "labels": top30["skill"].tolist(),
        "values": top30["frequency"].tolist()
    }
    with open(SKILLS_JSON_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Skill JSON saved to {SKILLS_JSON_PATH}")

    # Compute skill by time
    skill_time = compute_skill_by_time(df)
    skill_time_path = os.path.join(BASE_DIR, "data", "processed", "skill_by_time.csv")
    skill_time.to_csv(skill_time_path, index=False)
    logger.info(f"Skill-by-time saved to {skill_time_path}")

    # Save updated df
    df_save = df.drop(columns=["skills"], errors="ignore")
    df_save["skills"] = df["skills_clean"]
    df_save.to_csv(processed_path, index=False)
    logger.info("Updated processed CSV with clean skills column")

    return freq_df


if __name__ == "__main__":
    freq = run_extraction()
    print("\n=== Top 20 Skills ===")
    print(freq.head(20).to_string(index=False))
