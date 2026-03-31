# Data-Driven Skill Demand Analysis & Trend Forecasting

> Final Year Capstone Project — Interactive Job Market Intelligence Dashboard

---

## Quick Start

```bash
# 1. Install dependencies
pip install flask pandas numpy scikit-learn

# 2. Run the full data pipeline (one command)
cd capstone_project
python scripts/data_cleaning.py
python scripts/skill_extraction.py
python scripts/eda_analysis.py
python scripts/model_training.py
python scripts/forecasting.py

# 3. Launch the dashboard
cd app
python app.py
# Open http://localhost:5000
```

---

## Project Structure

```
capstone_project/
├── data/
│   ├── raw/
│   │   ├── generate_dataset.py      # Generates 1,000 synthetic job listings
│   │   └── job_market_data.csv      # Raw dataset (auto-generated)
│   └── processed/
│       ├── cleaned_jobs.csv          # Cleaned & feature-engineered dataset
│       ├── skill_frequency.csv       # Skill demand counts
│       ├── skill_by_time.csv         # Monthly skill demand (for forecasting)
│       ├── eda_summary.json          # EDA results
│       ├── model_metrics.json        # ML model performance
│       ├── forecast_results.json     # 6-month skill demand forecasts
│       ├── salary_model.pkl          # Trained salary regression model
│       └── classification_model.pkl  # Trained skill classifier
│
├── scripts/
│   ├── data_cleaning.py             # Clean raw data, handle nulls, derive features
│   ├── skill_extraction.py          # Parse & normalize skills, compute frequency
│   ├── eda_analysis.py              # Statistical analysis & EDA JSON output
│   ├── model_training.py            # Train ML models (Gradient Boosting + Random Forest)
│   └── forecasting.py               # Time-series forecasting (Linear Trend + Exp. Smoothing)
│
├── app/
│   ├── app.py                       # Flask backend with REST API
│   └── templates/
│       └── index.html               # Interactive dashboard (Chart.js)
│
├── reports/
│   ├── proposal.md                  # Project proposal
│   ├── eda_report.md                # EDA findings & insights
│   ├── model_report.md              # ML model explanation & metrics
│   └── final_report.md              # Full project documentation
│
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/stats` | GET | KPI summary (total jobs, avg salary, growth rate) |
| `/skills?location=X&level=Y` | GET | Top skills with optional filters |
| `/forecast?skill=Python` | GET | 6-month forecast for a skill |
| `/eda` | GET | Full EDA JSON |
| `/models` | GET | ML model performance metrics |
| `/filters` | GET | Available filter options |

---

## Results Summary

| Model | Metric | Score |
|-------|--------|-------|
| Salary Regression (Gradient Boosting) | R² | **0.9327** |
| Salary Regression | RMSE | 4.05 LPA |
| Skill Classification (Random Forest) | Accuracy | **87.7%** |
| Skill Classification | F1 (macro) | 0.83 |
| Demand Forecast | Top Trend | All 10 skills **↑ Increasing** |

---

## Top 5 In-Demand Skills

| Rank | Skill | % of Jobs |
|------|-------|-----------|
| 1 | Python | 41.3% |
| 2 | SQL | 29.1% |
| 3 | Docker | 19.1% |
| 4 | AWS | 18.9% |
| 5 | TensorFlow | 13.6% |

---

## Dependencies

```
flask>=2.0
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
```

No statsmodels required — forecasting uses pure NumPy.

---

## Academic Deliverables Checklist

- [x] Project Proposal (`reports/proposal.md`)
- [x] Cleaned Dataset (`data/processed/cleaned_jobs.csv`)
- [x] Data Processing Scripts (`scripts/`)
- [x] EDA Report with insights (`reports/eda_report.md`)
- [x] Interactive Dashboard (`app/app.py` + `app/templates/index.html`)
- [x] ML Model Report (`reports/model_report.md`)
- [x] Forecasting Results (`data/processed/forecast_results.json`)
- [x] Final Report (`reports/final_report.md`)

---

*Built with Python, Flask, scikit-learn, and Chart.js*
