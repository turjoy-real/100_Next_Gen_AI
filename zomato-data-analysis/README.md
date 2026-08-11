# Zomato Restaurant Dataset — Exploratory Data Analysis

## Project objective

Perform a complete, reproducible Exploratory Data Analysis (EDA) on the Zomato restaurant dataset, covering data understanding, cleaning, statistical analysis, required business questions, visualizations, and insights.

## Dataset

- **File:** `data/zomato.csv`
- **Source file provided:** `test_zomato.csv` (copied into `data/zomato.csv`)
- Contains restaurant metadata such as city, cuisines, average cost for two, online delivery, table booking, price range, aggregate rating, and votes.

## Technologies used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- ReportLab (PDF report generation)

## Project structure

```text
zomato-data-analysis/
├── data/
│   └── zomato.csv
├── notebooks/
│   └── zomato_eda.ipynb
├── reports/
│   ├── zomato_analysis_report.md
│   └── zomato_analysis_report.pdf
├── outputs/
│   ├── analysis_results.json
│   └── charts/
│       ├── top_10_cities.png
│       ├── rating_distribution.png
│       ├── online_delivery.png
│       ├── top_10_cuisines.png
│       └── rating_by_price_range.png
├── src/
│   ├── analysis.py
│   └── generate_notebook.py
├── requirements.txt
└── README.md
```

## How to install dependencies

From the project root (`zomato-data-analysis/`):

```bash
python3 -m pip install -r requirements.txt
```

## How to run the notebook

```bash
cd notebooks
python3 -m jupyter notebook zomato_eda.ipynb
```

Or execute end-to-end without opening the UI:

```bash
cd notebooks
python3 -m jupyter nbconvert --to notebook --execute zomato_eda.ipynb --inplace
```

## How to run the Python analysis

From the project root:

```bash
python3 src/analysis.py
```

This will:

1. Load and clean the dataset
2. Compute required metrics
3. Save all required charts to `outputs/charts/`
4. Write `outputs/analysis_results.json`
5. Generate Markdown and PDF reports in `reports/`

## Output files

| Output | Description |
|---|---|
| `notebooks/zomato_eda.ipynb` | Full EDA notebook |
| `outputs/charts/*.png` | Five required charts |
| `outputs/analysis_results.json` | Machine-readable calculated results |
| `reports/zomato_analysis_report.md` | Documentation report (Markdown) |
| `reports/zomato_analysis_report.pdf` | Documentation report (PDF) |

## Summary of analysis performed

1. Inspected shape, dtypes, missing values, and duplicates
2. Converted `Average Cost for two` to numeric
3. Removed a small number of corrupted shifted rows
4. Answered required city/cuisine questions (with cuisine splitting)
5. Compared ratings by online delivery and table booking
6. Produced statistical summaries and five required visualizations
7. Documented business insights with calculated values only

## Assumptions and limitations

- Average cost mixes multiple currencies, so global mean cost is not directly comparable across countries.
- Aggregate rating `0` often corresponds to “Not rated” rather than a true zero-quality score.
- Associations (for example, delivery vs rating) are observational and not causal.
