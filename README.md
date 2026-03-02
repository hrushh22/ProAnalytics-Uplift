# Promotion Uplift Modeling Using Causal Machine Learning
**Team ProAnalytics**

## Overview
This project estimates the true causal impact of retail promotions using 
uplift modeling — identifying which customers buy *because* of a promotion 
vs those who would have bought anyway.

## Team
- Adith Kadam Ramesh (Project Lead)
- Sudesna Das Rochi
- Sahitya Kotla
- Dhrumil Panchal
- Hrushik Mehta
- Aneesha Prasad

## Dataset
Dunnhumby Complete Journey Dataset

## Models Used
- Propensity Score Modeling (Logistic Regression)
- Baseline: Random Forest, Gradient Boosting, XGBoost
- T-Learner (Causal)
- X-Learner (Causal — Primary)
- K-Means Clustering + PCA

## Setup
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
python uplift_pipeline.py
```

## Output
Open `outputs/dashboard.html` in your browser for the interactive dashboard.

## Results
- **5,000 customers analyzed**
- **2,366 persuadable customers (47.3%)**
- **Qini coefficient: 107.10**
- **Interactive dashboard with uplift curves, customer segmentation, and ROI analysis**
