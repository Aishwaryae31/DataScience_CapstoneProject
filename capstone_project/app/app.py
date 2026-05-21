"""
app.py
------
Flask backend for the Skill Demand Dashboard.
Serves API endpoints consumed by the frontend dashboard.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from collections import Counter
from flask import Flask, render_template, jsonify, request

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_jobs.csv")
EDA_PATH       = os.path.join(BASE_DIR, "data", "processed", "eda_summary.json")
METRICS_PATH   = os.path.join(BASE_DIR, "data", "processed", "model_metrics.json")
FORECAST_PATH  = os.path.join(BASE_DIR, "data", "processed", "forecast_results.json")

# ─── Data Cache ───────────────────────────────────────────────────────────────
_df_cache = None
_eda_cache = None
_metrics_cache = None
_forecast_cache = None


def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(PROCESSED_PATH)
    return _df_cache


def get_eda():
    global _eda_cache
    if _eda_cache is None:
        with open(EDA_PATH) as f:
            _eda_cache = json.load(f)
    return _eda_cache


def get_metrics():
    global _metrics_cache
    if _metrics_cache is None:
        with open(METRICS_PATH) as f:
            _metrics_cache = json.load(f)
    return _metrics_cache


def get_forecast():
    global _forecast_cache
    if _forecast_cache is None:
        with open(FORECAST_PATH) as f:
            _forecast_cache = json.load(f)
    return _forecast_cache


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stats")
def stats():
    try:
        df = get_df()
        eda = get_eda()

        top_skills = eda.get("top_skills", {})
        top_skill = top_skills.get("labels", ["Python"])[0] if top_skills.get("labels") else "Python"
        total_skills = int(top_skills.get("total_unique_skills", 0))

        avg_salary = round(float(df["salary_lpa"].mean()), 2) if "salary_lpa" in df.columns else 0

        # Growth rate: compare first vs last quarter job counts
        if "year_month" in df.columns:
            monthly = df.groupby("year_month").size().sort_index()
            if len(monthly) >= 2:
                first_half = monthly.iloc[:len(monthly)//2].mean()
                second_half = monthly.iloc[len(monthly)//2:].mean()
                growth_rate = round(((second_half - first_half) / max(first_half, 1)) * 100, 1)
            else:
                growth_rate = 12.5
        else:
            growth_rate = 12.5

        return jsonify({
            "total_jobs": int(len(df)),
            "avg_salary": avg_salary,
            "growth_rate": float(growth_rate),
            "skills_tracked": total_skills,
            "top_skill": top_skill,
            "unique_companies": int(df["company"].nunique()) if "company" in df.columns else 0,
            "unique_locations": int(df["location"].nunique()) if "location" in df.columns else 0
        })
    except Exception as e:
        logger.error(f"/stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/skills")
def skills():
    try:
        df = get_df()
        location = request.args.get("location", "All")
        job_level = request.args.get("level", "All")

        # Filter
        filtered = df.copy()
        if location != "All" and "location" in filtered.columns:
            filtered = filtered[filtered["location"] == location]
        if job_level != "All" and "experience_level" in filtered.columns:
            filtered = filtered[filtered["experience_level"] == job_level]

        # Count skills safely
        all_skills = []
        for s in filtered["skills"].dropna():
            all_skills.extend([x.strip() for x in str(s).split(",") if x.strip()])

        if not all_skills:
            return jsonify({"labels": [], "values": [], "top_skill": "N/A", "total_skills": 0})

        counter = Counter(all_skills)
        top30 = counter.most_common(30)
        labels = [t[0] for t in top30]
        values = [t[1] for t in top30]

        return jsonify({
            "labels": labels,
            "values": values,
            "top_skill": labels[0] if labels else "N/A",
            "total_skills": len(counter)
        })
    except Exception as e:
        logger.error(f"/skills error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/forecast")
def forecast():
    try:
        data = get_forecast()
        skill = request.args.get("skill", None)

        if skill and skill in data:
            return jsonify(data[skill])

        # Return summary of all forecasts
        summary = {}
        for sk, info in data.items():
            if sk == "_overall":
                continue
            summary[sk] = {
                "trend": info.get("trend", "unknown"),
                "forecast_values": info.get("forecast_values", []),
                "forecast_labels": info.get("forecast_labels", []),
                "historical_labels": info.get("historical_labels", []),
                "historical_values": info.get("historical_values", []),
                "metrics": info.get("metrics", {})
            }
        return jsonify(summary)
    except Exception as e:
        logger.error(f"/forecast error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/eda")
def eda():
    try:
        data = get_eda()
        return jsonify(data)
    except Exception as e:
        logger.error(f"/eda error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/models")
def models():
    try:
        data = get_metrics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"/models error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/filters")
def filters():
    try:
        df = get_df()
        locations = ["All"] + sorted(df["location"].dropna().unique().tolist()) if "location" in df.columns else ["All"]
        levels = ["All"] + sorted(df["experience_level"].dropna().unique().tolist()) if "experience_level" in df.columns else ["All"]
        forecast_data = get_forecast()
        forecast_skills = [k for k in forecast_data.keys() if k != "_overall"]
        return jsonify({
            "locations": locations,
            "levels": levels,
            "forecast_skills": forecast_skills
        })
    except Exception as e:
        logger.error(f"/filters error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Skill Demand Dashboard...")
    app.run(debug=True, port=5000)
