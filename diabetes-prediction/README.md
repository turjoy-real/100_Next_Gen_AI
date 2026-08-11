# Diabetes Prediction using Machine Learning

End-to-end classification pipeline on the Pima Indians Diabetes dataset (`diabetes.csv`).

## What’s included

| Path | Description |
|------|-------------|
| `data/diabetes.csv` | Dataset |
| `notebooks/diabetes_prediction.ipynb` | Full assignment notebook (Parts 1–5) |
| `src/generate_notebook.py` | Regenerates the notebook |
| `src/train_and_evaluate.py` | Runs the pipeline + writes report/figures |
| `reports/diabetes_prediction_report.md` | Written results summary |

## Pipeline overview

1. **Data prep** — EDA, treat medical zeros as missing, median impute, StandardScaler, stratified 80/20 split  
2. **Models** — Logistic Regression, Decision Tree, Random Forest  
3. **Training** — fit on train; compare train vs test for overfitting  
4. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix  
5. **Analysis** — prioritize **Recall** for disease screening; limitations & improvements  

## Setup

From the repo root (`100_Next_Gen_AI`):

```bash
# if using the project venv
source .venv/bin/activate
pip install scikit-learn pandas matplotlib seaborn jupyter nbconvert
```

## Run

```bash
cd diabetes-prediction

# Generate notebook (if needed)
python src/generate_notebook.py

# Train models + write report
python src/train_and_evaluate.py

# Or open / execute the notebook
jupyter notebook notebooks/diabetes_prediction.ipynb
```

## Assignment mapping

- **Part 1** — load, explore, X/y split, train-test split, preprocess  
- **Part 2** — three classifiers + why / parameters / how they predict  
- **Part 3** — fit, train metrics, overfitting check  
- **Part 4** — test metrics + confusion matrix (TP/TN/FP/FN)  
- **Part 5** — metric choice (Recall), limitations, ≥2 improvements  
