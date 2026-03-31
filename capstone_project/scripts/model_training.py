"""
model_training.py
-----------------
Trains machine learning models on the processed job market dataset.

Models:
  1. Salary Prediction (Regression) — Ridge Regression
  2. High-Demand Skill Classification — Random Forest

Outputs performance metrics and saves trained models.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_jobs.csv")
MODELS_DIR = os.path.join(BASE_DIR, "data", "processed")
METRICS_PATH = os.path.join(BASE_DIR, "data", "processed", "model_metrics.json")


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    return df


def prepare_salary_features(df: pd.DataFrame):
    """Encode categorical features for salary regression."""
    df = df.copy()
    level_map = {"Entry": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Principal": 5}
    df["level_enc"] = df["experience_level"].map(level_map).fillna(2)

    le_loc = LabelEncoder()
    le_title = LabelEncoder()
    le_industry = LabelEncoder()

    df["loc_enc"] = le_loc.fit_transform(df["location"].fillna("Unknown"))
    df["title_enc"] = le_title.fit_transform(df["job_title"].fillna("Unknown"))
    df["industry_enc"] = le_industry.fit_transform(df["industry"].fillna("Unknown"))

    features = ["level_enc", "loc_enc", "title_enc", "industry_enc", "skill_count"]
    X = df[features].fillna(0)
    y = df["salary_lpa"].fillna(df["salary_lpa"].median())
    return X, y


def train_salary_model(df: pd.DataFrame) -> dict:
    logger.info("Training salary regression model...")
    X, y = prepare_salary_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Cross-val
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    metrics = {
        "model": "Gradient Boosting Regressor",
        "task": "Salary Prediction (LPA)",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "CV_R2_mean": round(float(cv_scores.mean()), 4),
        "CV_R2_std": round(float(cv_scores.std()), 4),
        "features": ["experience_level", "location", "job_title", "industry", "skill_count"]
    }

    # Save model
    with open(os.path.join(MODELS_DIR, "salary_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Salary model: RMSE={rmse:.2f}, R2={r2:.4f}")
    return metrics


def prepare_skill_classification(df: pd.DataFrame):
    """Label top skills as high-demand (1) vs low-demand (0)."""
    all_skills = []
    for s in df["skills"].dropna():
        all_skills.extend([x.strip() for x in str(s).split(",") if x.strip()])
    counter = Counter(all_skills)
    median_freq = np.median(list(counter.values()))

    # High demand if skill appears > median times
    high_demand = {skill for skill, cnt in counter.items() if cnt >= median_freq}

    rows = []
    for _, row in df.iterrows():
        skills = [x.strip() for x in str(row.get("skills", "")).split(",") if x.strip()]
        level = row.get("experience_level", "Mid")
        for skill in skills:
            rows.append({
                "skill": skill,
                "level": level,
                "location": row.get("location", "Unknown"),
                "industry": row.get("industry", "Unknown"),
                "is_high_demand": 1 if skill in high_demand else 0
            })

    skill_df = pd.DataFrame(rows)

    le_skill = LabelEncoder()
    le_level = LabelEncoder()
    le_loc = LabelEncoder()
    le_ind = LabelEncoder()

    skill_df["skill_enc"] = le_skill.fit_transform(skill_df["skill"])
    skill_df["level_enc"] = le_level.fit_transform(skill_df["level"])
    skill_df["loc_enc"] = le_loc.fit_transform(skill_df["location"])
    skill_df["ind_enc"] = le_ind.fit_transform(skill_df["industry"])

    X = skill_df[["skill_enc", "level_enc", "loc_enc", "ind_enc"]]
    y = skill_df["is_high_demand"]
    return X, y, le_skill


def train_classification_model(df: pd.DataFrame) -> dict:
    logger.info("Training skill demand classification model...")
    X, y, le = prepare_skill_classification(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)

    feature_importances = dict(zip(
        ["skill", "experience_level", "location", "industry"],
        [round(float(v), 4) for v in model.feature_importances_]
    ))

    metrics = {
        "model": "Random Forest Classifier",
        "task": "High-Demand Skill Classification",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(acc, 4),
        "precision_macro": round(float(report["macro avg"]["precision"]), 4),
        "recall_macro": round(float(report["macro avg"]["recall"]), 4),
        "f1_macro": round(float(report["macro avg"]["f1-score"]), 4),
        "feature_importances": feature_importances
    }

    with open(os.path.join(MODELS_DIR, "classification_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Classification model: Accuracy={acc:.4f}")
    return metrics


def run_model_training():
    df = load_data()

    if "salary_lpa" not in df.columns:
        df["salary_lpa"] = df.get("salary_inr", 1000000) / 100000

    if "skill_count" not in df.columns:
        df["skill_count"] = df["skills"].apply(
            lambda x: len([s.strip() for s in str(x).split(",") if s.strip()])
        )

    salary_metrics = train_salary_model(df)
    classification_metrics = train_classification_model(df)

    all_metrics = {
        "salary_regression": salary_metrics,
        "skill_classification": classification_metrics
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"All metrics saved to {METRICS_PATH}")
    return all_metrics


if __name__ == "__main__":
    metrics = run_model_training()
    print("\n=== Model Performance ===")
    sm = metrics["salary_regression"]
    print(f"Salary Model: RMSE={sm['RMSE']}, MAE={sm['MAE']}, R2={sm['R2']}")
    cm = metrics["skill_classification"]
    print(f"Classification Model: Accuracy={cm['accuracy']}, F1={cm['f1_macro']}")
