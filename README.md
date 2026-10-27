# 🏦 Loan Approval Prediction System

Machine Learning system that predicts whether a loan application will be **Approved** or **Rejected**.

## 📊 Dataset
`loan_approval_dataset.csv` — 4,269 records with 13 features:
- Applicant demographics (dependents, education, self-employed)
- Financial info (income, loan amount, loan term, CIBIL score)
- Asset values (residential, commercial, luxury, bank)

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/data_cleaning.py
python src/model_training.py
python src/evaluation.py
streamlit run app/app.py