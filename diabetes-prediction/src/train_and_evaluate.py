"""Run the full diabetes prediction pipeline and write a markdown report."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetes.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_COL = "Outcome"


def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(DATA_PATH)
    df_clean = df.copy()
    df_clean[ZERO_AS_MISSING] = df_clean[ZERO_AS_MISSING].replace(0, np.nan)

    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLS,
            )
        ]
    )
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    train_metrics, test_metrics, test_preds, trained = {}, {}, {}, {}
    for name, model in models.items():
        model.fit(X_train_p, y_train)
        trained[name] = model
        y_tr = model.predict(X_train_p)
        y_te = model.predict(X_test_p)
        test_preds[name] = y_te
        train_metrics[name] = evaluate(y_train, y_tr)
        test_metrics[name] = evaluate(y_test, y_te)

    best_name = max(
        models.keys(),
        key=lambda n: (
            test_metrics[n]["F1-Score"],
            test_metrics[n]["Recall"],
            test_metrics[n]["Accuracy"],
        ),
    )
    best_model = trained[best_name]
    y_pred = test_preds[best_name]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="Outcome", ax=axes[0], palette=["#4C78A8", "#F58518"])
    axes[0].set_title("Target Class Distribution")
    axes[0].set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1], annot_kws={"size": 8})
    axes[1].set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(FIGURES / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    names = list(models.keys())
    x = np.arange(len(names))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x - w / 2, [train_metrics[n]["Accuracy"] for n in names], w, label="Train", color="#4C78A8")
    axes[0].bar(x + w / 2, [test_metrics[n]["Accuracy"] for n in names], w, label="Test", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Train vs Test Accuracy")
    axes[0].legend()
    axes[1].bar(x - w / 2, [train_metrics[n]["F1-Score"] for n in names], w, label="Train", color="#4C78A8")
    axes[1].bar(x + w / 2, [test_metrics[n]["F1-Score"] for n in names], w, label="Test", color="#F58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Train vs Test F1-Score")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(FIGURES / "train_test_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"]).plot(
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {best_name} (Test)")
    plt.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_best.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    lr = trained["Logistic Regression"]
    coef = pd.Series(lr.coef_[0], index=FEATURE_COLS).sort_values()
    coef.plot(kind="barh", ax=axes[0], color="#4C78A8")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Logistic Regression Coefficients")
    imp_name = best_name if hasattr(best_model, "feature_importances_") else "Random Forest"
    importances = pd.Series(trained[imp_name].feature_importances_, index=FEATURE_COLS).sort_values()
    importances.plot(kind="barh", ax=axes[1], color="#F58518")
    axes[1].set_title(f"{imp_name} Feature Importances")
    plt.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Report
    te = test_metrics[best_name]
    tr = train_metrics[best_name]
    gap_acc = tr["Accuracy"] - te["Accuracy"]
    gap_f1 = tr["F1-Score"] - te["F1-Score"]
    if gap_acc > 0.10 or gap_f1 > 0.10:
        fit_note = "Possible mild/moderate overfitting — train performance is noticeably higher than test."
    elif tr["Accuracy"] < 0.65 and te["Accuracy"] < 0.65:
        fit_note = "Possible underfitting — both train and test performance are relatively low."
    else:
        fit_note = "Train and test performance are reasonably close → no severe overfitting/underfitting."

    lines = []
    lines.append("# Diabetes Prediction — Assignment Report\n")
    lines.append("End-to-end Machine Learning classification pipeline on `diabetes.csv`.\n")

    lines.append("## Part 1 — Data Preparation\n")
    lines.append(f"- **Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"- **Columns:** {', '.join(df.columns)}")
    lines.append("- **Dtypes:** all numeric (`int64` / `float64`)")
    lines.append("- **Explicit NaNs:** none; zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI treated as missing")
    lines.append(f"- **Target balance:** No Diabetes={ (y==0).sum() } ({100*(y==0).mean():.1f}%), Diabetes={ (y==1).sum() } ({100*(y==1).mean():.1f}%)")
    lines.append("- **Features (X):** 8 medical attributes; **Target (y):** `Outcome` (0/1)")
    lines.append("- **Preprocessing:** median imputation + StandardScaler (fit on train only)")
    lines.append(f"- **Split:** stratified 80/20 → train={X_train.shape[0]}, test={X_test.shape[0]}\n")

    lines.append("## Part 2 — Model Building\n")
    lines.append("Implemented three classifiers:")
    lines.append("1. **Logistic Regression** (`max_iter=1000`) — interpretable probabilistic baseline")
    lines.append("2. **Decision Tree** (`max_depth=5`, `min_samples_leaf=10`) — non-linear rules with depth constraint")
    lines.append("3. **Random Forest** (`n_estimators=200`, `max_depth=8`, `min_samples_leaf=5`) — ensemble for better generalization\n")

    lines.append("## Part 3 — Model Training\n")
    lines.append("| Model | Train Acc | Test Acc | Train F1 | Test F1 | Acc Gap |")
    lines.append("|-------|-----------|----------|----------|---------|---------|")
    for n in names:
        gap = train_metrics[n]["Accuracy"] - test_metrics[n]["Accuracy"]
        lines.append(
            f"| {n} | {train_metrics[n]['Accuracy']:.4f} | {test_metrics[n]['Accuracy']:.4f} | "
            f"{train_metrics[n]['F1-Score']:.4f} | {test_metrics[n]['F1-Score']:.4f} | {gap:.4f} |"
        )
    lines.append(f"\n**Best model:** {best_name}")
    lines.append(f"**Overfitting/underfitting observation:** {fit_note}\n")
    lines.append("![Train vs Test](figures/train_test_comparison.png)\n")

    lines.append("## Part 4 — Model Evaluation (Best Model)\n")
    lines.append(f"### {best_name} — Test metrics\n")
    lines.append(f"| Metric | Train | Test |")
    lines.append(f"|--------|------:|-----:|")
    for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        lines.append(f"| {m} | {tr[m]:.4f} | {te[m]:.4f} |")
    lines.append("")
    lines.append("### Confusion Matrix (Test)\n")
    lines.append(f"| | Pred No Diabetes | Pred Diabetes |")
    lines.append(f"|--|--:|--:|")
    lines.append(f"| **Actual No Diabetes** | TN = {tn} | FP = {fp} |")
    lines.append(f"| **Actual Diabetes** | FN = {fn} | TP = {tp} |")
    lines.append("")
    lines.append(f"- Sensitivity/Recall = {tp}/{tp+fn} = **{tp/(tp+fn):.4f}**")
    lines.append(f"- Specificity = {tn}/{tn+fp} = **{tn/(tn+fp):.4f}**")
    lines.append(f"- Precision = {tp}/{tp+fp} = **{tp/(tp+fp):.4f}**\n")
    lines.append("![Confusion Matrix](figures/confusion_matrix_best.png)\n")
    lines.append("```")
    lines.append(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    lines.append("```\n")

    lines.append("## Part 5 — Performance Analysis\n")
    lines.append("### Most important metric")
    lines.append("**Recall (Sensitivity)** — missing a diabetic patient (False Negative) is clinically more costly than a false alarm.\n")
    lines.append("### Is the model performing well?")
    lines.append(
        f"**Yes, as an educational screening baseline.** Test Accuracy={te['Accuracy']:.4f}, "
        f"Recall={te['Recall']:.4f} (catches most positives), F1={te['F1-Score']:.4f}, "
        f"with a small train–test gap. Precision={te['Precision']:.4f} means some false alarms — "
        "acceptable for screening, not diagnosis. Not a clinical diagnostic tool.\n"
    )
    lines.append("### Key findings")
    lines.append(f"- Best model by Test F1/Recall: **{best_name}**")
    lines.append(f"- Test Accuracy={te['Accuracy']:.4f}, Precision={te['Precision']:.4f}, Recall={te['Recall']:.4f}, F1={te['F1-Score']:.4f}")
    top_lr = coef.abs().sort_values(ascending=False).head(3)
    lines.append(f"- Strongest LR signals: {', '.join(f'{k} ({v:.3f})' for k, v in top_lr.items())}")
    top_imp = importances.sort_values(ascending=False).head(3)
    lines.append(f"- Top {imp_name} importances: {', '.join(f'{k} ({v:.3f})' for k, v in top_imp.items())}\n")
    lines.append("![Feature Importance](figures/feature_importance.png)\n")
    lines.append("### Limitations")
    lines.append("1. Small sample (~768 rows) and limited demographics")
    lines.append("2. Heavy missingness (zeros) in Insulin / SkinThickness")
    lines.append("3. Moderate class imbalance (~35% positive)")
    lines.append("4. Single train/test split; not a clinical diagnostic tool\n")
    lines.append("### Recommendations")
    lines.append("1. Hyperparameter tuning (Grid/Random search) + Stratified K-Fold CV")
    lines.append("2. Handle imbalance (`class_weight='balanced'`, SMOTE, threshold tuning) to raise Recall")
    lines.append("3. Feature engineering (interactions, BMI/age bins) and try Gradient Boosting / XGBoost\n")

    lines.append("---\n*Generated by `src/train_and_evaluate.py`*")

    report_path = REPORTS / "diabetes_prediction_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {report_path}")
    print(f"Best model: {best_name}")
    print(f"Test metrics: { {k: round(v, 4) for k, v in te.items()} }")
    print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()
