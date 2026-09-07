# 100 Next Gen AI — Assignments

Machine Learning / Data Analysis / Prompt Engineering assignment workspace.

| Project | Description |
|---------|-------------|
| [`zomato-data-analysis/`](zomato-data-analysis/) | Zomato restaurant dataset — Exploratory Data Analysis (EDA) |
| [`diabetes-prediction/`](diabetes-prediction/) | Diabetes prediction — end-to-end classification with Scikit-learn |
| [`prompt-engineering/`](prompt-engineering/) | Assignment 3 — Prompt engineering on real-world scenarios (Q1–Q13) |

## Prerequisites

- **Python 3.9+** (3.10+ recommended)
- `pip` and `venv` (usually bundled with Python)
- Git

Check your Python version:

```bash
python3 --version
```

## Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd 100_Next_Gen_AI
```

## One-time setup (recommended)

Create a virtual environment at the repo root and install dependencies for **both** projects:

```bash
# 1) Create & activate venv
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install all project dependencies
pip install -r requirements.txt
```

`.venv/` is gitignored — every new clone needs these steps once.

### Verify install

```bash
python -c "import pandas, sklearn, matplotlib, seaborn, numpy; print('OK')"
```

## Datasets

Root CSVs are also copied into each project’s `data/` folder:

| File | Used by |
|------|---------|
| `test_zomato.csv` | `zomato-data-analysis/data/zomato.csv` |
| `diabetes.csv` | `diabetes-prediction/data/diabetes.csv` |

If a project `data/` file is missing after clone, copy from the repo root:

```bash
cp test_zomato.csv zomato-data-analysis/data/zomato.csv
cp diabetes.csv diabetes-prediction/data/diabetes.csv
```

---

## Run — Zomato EDA

```bash
source .venv/bin/activate   # if not already active
cd zomato-data-analysis

# Full analysis + charts + reports
python src/analysis.py

# Open / run the notebook
jupyter notebook notebooks/zomato_eda.ipynb
```

Details: [`zomato-data-analysis/README.md`](zomato-data-analysis/README.md)

---

## Run — Diabetes Prediction

```bash
source .venv/bin/activate   # if not already active
cd diabetes-prediction

# Train models + write report/figures
python src/train_and_evaluate.py

# Open / run the notebook
jupyter notebook notebooks/diabetes_prediction.ipynb
```

Optional — regenerate the notebook from the generator script:

```bash
python src/generate_notebook.py
```

Details: [`diabetes-prediction/README.md`](diabetes-prediction/README.md)

---

## Assignment 3 — Prompt Engineering

Prompts and write-up: [`prompt-engineering/reports/assignment_3_prompt_engineering.md`](prompt-engineering/reports/assignment_3_prompt_engineering.md)

---

## Project layout

```text
100_Next_Gen_AI/
├── README.md                 ← you are here
├── requirements.txt          ← combined deps for fresh setup
├── .gitignore
├── diabetes.csv
├── test_zomato.csv
├── diabetes-prediction/
│   ├── data/
│   ├── notebooks/
│   ├── reports/
│   ├── src/
│   └── requirements.txt
├── prompt-engineering/
│   ├── reports/
│   │   └── assignment_3_prompt_engineering.md
│   └── README.md
└── zomato-data-analysis/
    ├── data/
    ├── notebooks/
    ├── outputs/
    ├── reports/
    ├── src/
    └── requirements.txt
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `python3: command not found` | Install Python 3, or use `python` instead of `python3` |
| `ModuleNotFoundError: sklearn` / `pandas` | Activate `.venv` and re-run `pip install -r requirements.txt` |
| Jupyter kernel missing packages | Select the `.venv` kernel, or run: `python -m ipykernel install --user --name=100-next-gen-ai` |
| `diabetes.csv` / `zomato.csv` not found | Copy from repo root into the project `data/` folder (commands above) |
| macOS / Xcode Python is old | Prefer [python.org](https://www.python.org/downloads/) or Homebrew (`brew install python`) |

## Deactivate the virtual environment

```bash
deactivate
```
