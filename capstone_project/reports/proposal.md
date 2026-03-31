# Project Proposal

**Title:** Data-Driven Skill Demand Analysis and Trend Forecasting from Job Market Data  
**Student:** [Your Name]  
**Supervisor:** [Supervisor Name]  
**Institution:** [Institution Name]  
**Date:** 2024

---

## 1. Problem Statement

The rapid evolution of technology has created a widening skills gap between what employers demand and what job seekers offer. According to industry reports, over 60% of employers struggle to find candidates with the right technical skill sets. Current job seekers and educators lack real-time, data-driven visibility into which skills are rising, which are declining, and what future demand will look like.

Traditional career guidance relies on outdated surveys and anecdotal evidence. There is a critical need for an automated, data-driven system that:

- Continuously analyzes job posting data to identify skill trends
- Quantifies the demand for specific technical skills across industries and locations
- Provides predictive forecasts of future skill demand
- Surfaces actionable insights through an interactive dashboard

This project addresses this gap by building an end-to-end data science pipeline that ingests job market data, extracts skill signals, trains predictive models, and visualizes trends in an interactive web dashboard.

---

## 2. Objectives

### Primary Objectives

1. **Data Collection & Processing:** Build a robust pipeline to collect, clean, and structure job market data including job titles, required skills, salary, location, and experience level.

2. **Skill Extraction & Analysis:** Implement NLP-based skill extraction from unstructured job description text; compute skill frequency, co-occurrence, and temporal trends.

3. **Exploratory Data Analysis (EDA):** Conduct comprehensive EDA to uncover patterns in skill demand, salary distribution, geographic variation, and industry concentration.

4. **Predictive Modeling:** Train machine learning models to:
   - Predict salary based on job features (regression)
   - Classify skills as high-demand vs. low-demand (classification)

5. **Time-Series Forecasting:** Apply linear trend regression and exponential smoothing to forecast future demand for top skills over a 6-month horizon.

6. **Interactive Dashboard:** Deploy a web-based Flask dashboard that visualizes all analytical outputs with real-time filtering by location and job level.

### Secondary Objectives

- Identify the top 20 most in-demand technical skills across the Indian job market
- Analyze salary disparities across experience levels and locations
- Provide actionable recommendations for skill development prioritization

---

## 3. Data Source

### Dataset Description

For this project, a synthetic dataset of **1,000 job listings** has been generated to simulate the Indian IT job market. The dataset is modeled on realistic distributions observed in platforms such as Naukri.com, LinkedIn Jobs, and Indeed India.

### Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | String | Unique job identifier |
| `job_title` | String | Position title (e.g., Data Scientist, ML Engineer) |
| `company` | String | Hiring company name |
| `location` | String | City (Bangalore, Mumbai, etc.) |
| `experience_level` | String | Entry / Mid / Senior / Lead / Principal |
| `salary_inr` | Float | Annual salary in INR |
| `skills_required` | String | Comma-separated required skills |
| `date_posted` | Date | Job posting date (Jan 2022 – Dec 2024) |
| `remote` | String | Yes / No / Hybrid |
| `industry` | String | IT, Finance, Healthcare, E-commerce, etc. |

### Scope

- **Time period:** January 2022 – December 2024
- **Geography:** Major Indian tech cities
- **Job functions:** Data Science, Software Engineering, DevOps, Analytics, AI/ML
- **Skills tracked:** 92 unique technical skills

### Future Extension

This pipeline is designed to be extended with live data scraped from job portals using APIs or web scraping tools (BeautifulSoup, Selenium), enabling real-time analysis.

---

## 4. Methodology Overview

```
Raw Data → Cleaning → Skill Extraction → EDA → Modeling → Forecasting → Dashboard
```

1. **Data Cleaning** (`data_cleaning.py`): Remove duplicates, handle nulls, normalize columns, type conversion
2. **Skill Extraction** (`skill_extraction.py`): Parse skills strings, normalize aliases, compute frequency
3. **EDA** (`eda_analysis.py`): Statistical summaries, distribution analysis, correlation matrices
4. **Model Training** (`model_training.py`): Gradient Boosting (salary), Random Forest (skill classification)
5. **Forecasting** (`forecasting.py`): Linear trend regression + exponential smoothing per skill
6. **Dashboard** (`app/app.py` + `index.html`): Flask REST API + Chart.js frontend

---

## 5. Expected Outcomes

- A fully working, interactive web dashboard accessible at `http://localhost:5000`
- Cleaned dataset of 1,000+ job records
- Trained models with R² > 0.85 for salary prediction
- 6-month demand forecasts for top 10 skills
- Comprehensive EDA and model reports

---

## 6. Timeline

| Week | Activity |
|------|----------|
| 1–2 | Data collection and cleaning |
| 3–4 | Skill extraction and EDA |
| 5–6 | Model training and evaluation |
| 7–8 | Forecasting and validation |
| 9–10 | Dashboard development |
| 11–12 | Testing, documentation, and submission |
