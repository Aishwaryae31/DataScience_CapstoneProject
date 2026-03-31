"""
Dataset Generator for Job Market Data
Generates a realistic synthetic dataset for the capstone project.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

JOB_TITLES = [
    "Data Scientist", "Machine Learning Engineer", "Data Analyst",
    "Software Engineer", "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "DevOps Engineer", "Cloud Architect",
    "Data Engineer", "AI Engineer", "Business Intelligence Analyst",
    "Product Manager", "Cybersecurity Analyst", "NLP Engineer"
]

COMPANIES = [
    "TechCorp India", "Infosys", "TCS", "Wipro", "HCL Technologies",
    "Cognizant", "Capgemini", "Accenture", "IBM India", "Oracle India",
    "Amazon India", "Microsoft India", "Google India", "Flipkart", "Swiggy",
    "Zomato", "Paytm", "Byju's", "Freshworks", "Zoho"
]

LOCATIONS = [
    "Bangalore", "Mumbai", "Hyderabad", "Chennai", "Pune",
    "Delhi NCR", "Kolkata", "Ahmedabad", "Noida", "Gurgaon"
]

EXPERIENCE_LEVELS = ["Entry", "Mid", "Senior", "Lead", "Principal"]

SKILLS_POOL = {
    "Data Scientist": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
                       "SQL", "Statistics", "R", "Pandas", "NumPy", "Scikit-learn", "NLP",
                       "Data Visualization", "Big Data", "Spark"],
    "Machine Learning Engineer": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "Docker",
                                   "Kubernetes", "MLOps", "AWS", "Azure", "Deep Learning",
                                   "Model Deployment", "CI/CD", "FastAPI", "REST APIs"],
    "Data Analyst": ["SQL", "Excel", "Python", "Tableau", "Power BI", "Statistics",
                     "Data Visualization", "Pandas", "R", "Business Intelligence"],
    "Software Engineer": ["Java", "Python", "C++", "JavaScript", "Spring Boot", "REST APIs",
                           "Microservices", "SQL", "Git", "Agile", "Docker"],
    "Backend Developer": ["Python", "Java", "Node.js", "Django", "FastAPI", "Flask",
                           "REST APIs", "SQL", "MongoDB", "Redis", "Docker", "AWS"],
    "Frontend Developer": ["JavaScript", "React", "Vue.js", "Angular", "HTML", "CSS",
                            "TypeScript", "Redux", "GraphQL", "Webpack"],
    "Full Stack Developer": ["JavaScript", "Python", "React", "Node.js", "SQL", "MongoDB",
                              "Docker", "AWS", "REST APIs", "HTML", "CSS"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Jenkins",
                         "Terraform", "Ansible", "Linux", "Bash Scripting", "Monitoring"],
    "Cloud Architect": ["AWS", "Azure", "GCP", "Kubernetes", "Terraform", "Microservices",
                         "Docker", "Security", "Networking", "Cost Optimization"],
    "Data Engineer": ["Python", "Spark", "Hadoop", "SQL", "Airflow", "Kafka", "AWS",
                       "Snowflake", "ETL", "Big Data", "Databricks", "dbt"],
    "AI Engineer": ["Python", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "LLMs",
                     "Prompt Engineering", "OpenAI API", "LangChain", "Vector Databases"],
    "Business Intelligence Analyst": ["SQL", "Tableau", "Power BI", "Excel", "Data Modeling",
                                        "ETL", "Business Analysis", "Python", "Statistics"],
    "Product Manager": ["Product Strategy", "Agile", "Scrum", "Data Analysis", "SQL",
                         "User Research", "Roadmapping", "A/B Testing", "Communication"],
    "Cybersecurity Analyst": ["Network Security", "Python", "SIEM", "Penetration Testing",
                               "Ethical Hacking", "Firewalls", "Incident Response", "OWASP"],
    "NLP Engineer": ["Python", "NLP", "TensorFlow", "BERT", "Transformers", "LLMs",
                      "SpaCy", "NLTK", "Deep Learning", "Text Processing"]
}

SALARY_RANGES = {
    "Entry": (300000, 700000),
    "Mid": (700000, 1500000),
    "Senior": (1500000, 2800000),
    "Lead": (2500000, 4000000),
    "Principal": (3500000, 6000000)
}

def generate_dataset(n_records=1000):
    records = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)

    for i in range(n_records):
        job_title = random.choice(JOB_TITLES)
        level = random.choice(EXPERIENCE_LEVELS)
        salary_min, salary_max = SALARY_RANGES[level]
        salary = random.randint(salary_min, salary_max)

        # Skills: pick 4–8 from the job's pool
        skill_pool = SKILLS_POOL.get(job_title, ["Python", "SQL"])
        num_skills = random.randint(4, min(8, len(skill_pool)))
        skills = random.sample(skill_pool, num_skills)

        date_posted = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )

        records.append({
            "job_id": f"JOB{str(i+1).zfill(5)}",
            "job_title": job_title,
            "company": random.choice(COMPANIES),
            "location": random.choice(LOCATIONS),
            "experience_level": level,
            "salary_inr": salary,
            "skills_required": ", ".join(skills),
            "date_posted": date_posted.strftime("%Y-%m-%d"),
            "remote": random.choice(["Yes", "No", "Hybrid"]),
            "industry": random.choice(["IT", "Finance", "Healthcare", "E-commerce", "EdTech", "Fintech"])
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset(1000)
    out_path = os.path.join(os.path.dirname(__file__), "job_market_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {out_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
