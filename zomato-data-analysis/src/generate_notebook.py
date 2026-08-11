"""Generate the submission-ready Zomato EDA Jupyter notebook."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "zomato_eda.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(text: str) -> list[str]:
    # Jupyter JSON stores sources as arrays of lines ending with \n (except possibly last)
    text = text.strip("\n")
    if not text:
        return []
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = []

cells.append(
    md(
        """# Zomato Restaurant Dataset — Exploratory Data Analysis (EDA)

**Student-style professional assignment notebook**

This notebook performs Exploratory Data Analysis on the provided Zomato restaurant CSV.
All numerical answers are calculated directly from the dataset (no hardcoded final results)."""
    )
)

cells.append(
    md(
        """## Objective

Perform EDA covering:

1. Data understanding and exploration
2. Data cleaning and preprocessing
3. Statistical analysis
4. Required business questions
5. Required visualizations
6. Business insights and conclusion"""
    )
)

cells.append(
    md(
        """## Import Libraries"""
    )
)

cells.append(
    code(
        """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Clean, readable plotting defaults
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Project paths (notebook is inside notebooks/)
PROJECT_ROOT = Path("..").resolve()
DATA_PATH = PROJECT_ROOT / "data" / "zomato.csv"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

print("Libraries imported successfully.")
print("Data path:", DATA_PATH)"""
    )
)

cells.append(
    md(
        """# 1. Data Loading and Understanding

## Load Dataset

We load the CSV with Pandas and keep the original object as `df` throughout the notebook."""
    )
)

cells.append(
    code(
        """df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")"""
    )
)

cells.append(
    md(
        """## Dataset Overview

Preview the first and last rows to understand structure and spot formatting issues early."""
    )
)

cells.append(
    code(
        """df.head()"""
    )
)

cells.append(
    code(
        """df.tail()"""
    )
)

cells.append(
    md(
        """## Data Exploration

Inspect shape, columns, dtypes, and a compact info summary."""
    )
)

cells.append(
    code(
        """print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\\nColumn names:")
print(list(df.columns))"""
    )
)

cells.append(
    code(
        """df.info()"""
    )
)

cells.append(
    code(
        """df.dtypes"""
    )
)

cells.append(
    md(
        """## Missing Value Analysis

We compute missing counts and percentages for every column before deciding on treatment."""
    )
)

cells.append(
    code(
        """missing_count = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)

missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct,
}).sort_values("missing_count", ascending=False)

missing_summary"""
    )
)

cells.append(
    md(
        """### Missing-value interpretation

- Missingness is generally low (most columns ≤ ~0.1–0.2%).
- `Cuisines` has the highest missing count and matters for cuisine-frequency analysis.
- A few rows appear structurally broken (fields shifted), which can create cascading nulls.
- We will **not** blindly drop all missing values. We preserve usable records and only remove clearly corrupted rows."""
    )
)

cells.append(
    md(
        """## Duplicate Analysis"""
    )
)

cells.append(
    code(
        """duplicate_count = int(df.duplicated().sum())
print("Number of fully duplicate rows:", duplicate_count)

if "Restaurant ID" in df.columns:
    dup_ids = int(pd.to_numeric(df["Restaurant ID"], errors="coerce").duplicated().sum())
    print("Duplicate Restaurant ID values (after numeric coercion check):", dup_ids)"""
    )
)

cells.append(
    md(
        """### Duplicate interpretation

If full-row duplicates are zero, there is no need to drop duplicates.
We still check `Restaurant ID` because that should uniquely identify restaurants.
Decision: **do not remove duplicates** unless duplicate rows are actually present."""
    )
)

cells.append(
    md(
        """## Basic Descriptive Statistics

`describe(include="all")` gives an overview across numeric and categorical columns.
Later we compute focused statistics only for meaningful numerical fields."""
    )
)

cells.append(
    code(
        """df.describe(include="all")"""
    )
)

cells.append(
    md(
        """# Data Cleaning and Preprocessing

## Cleaning plan (based on actual inspection)

Observed issues in this CSV:

1. An extra `Unnamed: 0` index-like column.
2. `Average Cost for two` is stored as **object/string** and must be converted to numeric.
3. Two adjacent rows are corrupted because an address contains an unquoted line break
   (`Super Loco` / Robertson Quay fields are shifted).
4. Some Yes/No columns can contain invalid shifted values on corrupted rows.
5. A few `Cuisines` values are missing.

We create `df_clean` and keep original `df` unchanged."""
    )
)

cells.append(
    code(
        """df_clean = df.copy()
cleaning_notes = []

# 1) Drop leftover index column if present
unnamed_cols = [c for c in df_clean.columns if str(c).startswith("Unnamed")]
if unnamed_cols:
    df_clean = df_clean.drop(columns=unnamed_cols)
    cleaning_notes.append(f"Dropped columns: {unnamed_cols}")

# 2) Convert Average Cost for two to numeric safely
print("Average Cost for two dtype BEFORE conversion:", df_clean["Average Cost for two"].dtype)
nulls_before = int(df_clean["Average Cost for two"].isna().sum())

df_clean["Average Cost for two"] = pd.to_numeric(
    df_clean["Average Cost for two"], errors="coerce"
)

nulls_after = int(df_clean["Average Cost for two"].isna().sum())
converted_to_nan = nulls_after - nulls_before

print("Average Cost for two dtype AFTER conversion:", df_clean["Average Cost for two"].dtype)
print("Nulls before conversion:", nulls_before)
print("Nulls after conversion:", nulls_after)
print("Values changed to NaN during conversion:", converted_to_nan)
cleaning_notes.append(
    f"Average Cost for two converted to numeric; {converted_to_nan} value(s) became NaN."
)"""
    )
)

cells.append(
    md(
        """### Why convert with `errors="coerce"`?

Non-numeric / shifted values become `NaN` instead of crashing the notebook.
We then inspect and remove clearly corrupted rows rather than inventing replacements."""
    )
)

cells.append(
    code(
        """# 3) Identify and remove corrupted / shifted rows
restaurant_id_num = pd.to_numeric(df_clean["Restaurant ID"], errors="coerce")
valid_flags = (
    df_clean["Has Online delivery"].isin(["Yes", "No"])
    & df_clean["Has Table booking"].isin(["Yes", "No"])
)
keep_mask = restaurant_id_num.notna() & valid_flags

print("Rows flagged as corrupted/unusable:", int((~keep_mask).sum()))
display_cols = [
    "Restaurant ID", "Restaurant Name", "City",
    "Average Cost for two", "Has Table booking", "Has Online delivery"
]
df_clean.loc[~keep_mask, display_cols]"""
    )
)

cells.append(
    code(
        """rows_before = len(df_clean)
df_clean = df_clean.loc[keep_mask].copy()
rows_after = len(df_clean)

cleaning_notes.append(
    f"Dropped {rows_before - rows_after} corrupted row(s) with shifted fields / invalid IDs."
)

# 4) Normalize Yes/No columns (keep only Yes/No)
yes_no_cols = [
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu",
]
for col in yes_no_cols:
    if col in df_clean.columns:
        normalized = df_clean[col].astype(str).str.strip().str.title()
        df_clean[col] = normalized.where(normalized.isin(["Yes", "No"]))

# 5) Ensure key numeric columns are numeric
for col in [
    "Restaurant ID", "Country Code", "Longitude", "Latitude",
    "Price range", "Aggregate rating", "Votes"
]:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 6) Strip whitespace on important text fields
for col in ["Restaurant Name", "City", "Cuisines", "Rating text", "Currency"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype("string").str.strip()

df_clean = df_clean.reset_index(drop=True)

print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)
print("Average Cost for two dtype in df_clean:", df_clean["Average Cost for two"].dtype)
print("\\nCleaning decisions:")
for note in cleaning_notes:
    print("-", note)"""
    )
)

cells.append(
    md(
        """### Cleaning decisions summary

| Issue | Decision | Reason |
|---|---|---|
| `Unnamed: 0` | Drop | Leftover index; not analytically useful |
| `Average Cost for two` object dtype | Convert with `to_numeric(errors="coerce")` | Required for averages/statistics |
| Corrupted multi-line address rows | Drop | Fields are shifted; values are unreliable |
| Missing `Cuisines` | Keep rows | Still useful for city/rating/cost analysis |
| Exact duplicate rows | None found / not dropped | No evidence they should be removed |
| Sparse missing ratings/votes | Keep | Avoid unnecessarily reducing sample size |"""
    )
)

cells.append(
    md(
        """# Statistical Summary

Focused descriptive statistics for important **numerical** columns only."""
    )
)

cells.append(
    code(
        """numeric_cols = [
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes",
]

stats = df_clean[numeric_cols].agg(["count", "mean", "median", "std", "min", "max"])
quartiles = df_clean[numeric_cols].quantile([0.25, 0.5, 0.75])
quartiles.index = ["25% (Q1)", "50% (Q2/Median)", "75% (Q3)"]

print("Core descriptive statistics")
display(stats.round(4))
print("\\nQuartiles")
display(quartiles.round(4))"""
    )
)

cells.append(
    md(
        """# Part 1 — Required Questions

## Q1. How many unique cities are present?"""
    )
)

cells.append(
    code(
        """unique_cities = int(df_clean["City"].nunique())
print("Number of unique cities:", unique_cities)"""
    )
)

cells.append(
    md(
        """**Interpretation:** After cleaning invalid/shifted rows, this is the number of distinct city labels present in the dataset."""
    )
)

cells.append(
    md(
        """## Q2. Which city has the highest number of restaurants?"""
    )
)

cells.append(
    code(
        """city_counts = df_clean["City"].value_counts()
top_city = city_counts.index[0]
top_city_count = int(city_counts.iloc[0])

print(f"City with the highest number of restaurants: {top_city}")
print(f"Restaurant count: {top_city_count}")
city_counts.head(5).to_frame("Restaurant Count")"""
    )
)

cells.append(
    md(
        """**Interpretation:** The leading city has substantially more restaurants than the next cities, indicating strong concentration in that market within this dataset."""
    )
)

cells.append(
    md(
        """## Q3. Which cuisine appears most frequently?

Restaurants can list multiple cuisines (e.g., `French, Japanese, Desserts`).
We split on commas, strip whitespace, and count each cuisine individually."""
    )
)

cells.append(
    code(
        """cuisine_series = (
    df_clean["Cuisines"]
    .dropna()
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
)
cuisine_series = cuisine_series[cuisine_series.ne("")]

cuisine_counts = cuisine_series.value_counts()
top_cuisine = cuisine_counts.index[0]
top_cuisine_count = int(cuisine_counts.iloc[0])

print(f"Most frequent cuisine: {top_cuisine}")
print(f"Occurrences: {top_cuisine_count}")
cuisine_counts.head(10).to_frame("Count")"""
    )
)

cells.append(
    md(
        """**Interpretation:** After properly splitting multi-cuisine strings, the top cuisine reflects individual cuisine tags rather than full cuisine combinations."""
    )
)

cells.append(
    md(
        """# 2. Data Analysis

## 1) Average restaurant rating"""
    )
)

cells.append(
    code(
        """avg_rating = float(df_clean["Aggregate rating"].mean())
print(f"Average Aggregate Rating: {avg_rating:.4f}")

# Helpful context: many restaurants are marked Not rated with rating 0
not_rated = int((df_clean["Aggregate rating"] == 0).sum())
rated_avg = float(df_clean.loc[df_clean["Aggregate rating"] > 0, "Aggregate rating"].mean())
print(f"Restaurants with rating 0 (often 'Not rated'): {not_rated}")
print(f"Average rating excluding zeros: {rated_avg:.4f}")"""
    )
)

cells.append(
    md(
        """## 2) Restaurant with the highest number of votes"""
    )
)

cells.append(
    code(
        """max_votes = float(df_clean["Votes"].max())
top_voted = df_clean.loc[
    df_clean["Votes"] == max_votes,
    ["Restaurant Name", "City", "Votes", "Aggregate rating", "Cuisines"]
].copy()

print(f"Highest vote count: {int(max_votes)}")
print(f"Number of restaurants tied at this vote count: {len(top_voted)}")
top_voted"""
    )
)

cells.append(
    md(
        """**Interpretation:** If only one row is returned, that restaurant uniquely has the highest vote count. If multiple rows appear, they are tied."""
    )
)

cells.append(
    md(
        """## 3) Average cost for two across all restaurants"""
    )
)

cells.append(
    code(
        """avg_cost = float(df_clean["Average Cost for two"].mean())
print(f"Average Cost for two: {avg_cost:.2f}")
print("Note: the dataset mixes multiple currencies, so this global average should be interpreted cautiously.")"""
    )
)

cells.append(
    md(
        """## 4) Average ratings: Online Delivery Yes vs No"""
    )
)

cells.append(
    code(
        """online_summary = (
    df_clean.groupby("Has Online delivery", dropna=True)["Aggregate rating"]
    .agg(Restaurant_Count="count", Average_Rating="mean")
    .reset_index()
    .rename(columns={"Has Online delivery": "Online_Delivery_Status"})
)

online_yes = float(
    online_summary.loc[
        online_summary["Online_Delivery_Status"] == "Yes", "Average_Rating"
    ].iloc[0]
)
online_no = float(
    online_summary.loc[
        online_summary["Online_Delivery_Status"] == "No", "Average_Rating"
    ].iloc[0]
)
online_diff = online_yes - online_no

print(online_summary.to_string(index=False))
print(f"\\nDifference (Yes - No): {online_diff:.4f}")
online_summary"""
    )
)

cells.append(
    md(
        """**Interpretation:** Compare the two average ratings above. A positive difference means restaurants with online delivery have a higher mean rating in this dataset (observation only, not causation)."""
    )
)

cells.append(
    md(
        """## 5) Average ratings: Table Booking Yes vs No"""
    )
)

cells.append(
    code(
        """table_summary = (
    df_clean.groupby("Has Table booking", dropna=True)["Aggregate rating"]
    .agg(Restaurant_Count="count", Average_Rating="mean")
    .reset_index()
    .rename(columns={"Has Table booking": "Table_Booking_Status"})
)

table_yes = float(
    table_summary.loc[
        table_summary["Table_Booking_Status"] == "Yes", "Average_Rating"
    ].iloc[0]
)
table_no = float(
    table_summary.loc[
        table_summary["Table_Booking_Status"] == "No", "Average_Rating"
    ].iloc[0]
)
table_diff = table_yes - table_no

print(table_summary.to_string(index=False))
print(f"\\nDifference (Yes - No): {table_diff:.4f}")
table_summary"""
    )
)

cells.append(
    md(
        """**Interpretation:** Compare average ratings for table-booking vs non-booking restaurants using the calculated difference above."""
    )
)

cells.append(
    md(
        """## 6) Top 10 cities with the highest number of restaurants"""
    )
)

cells.append(
    code(
        """top10_cities = (
    df_clean["City"].value_counts()
    .head(10)
    .rename_axis("City")
    .reset_index(name="Restaurant Count")
)
top10_cities.insert(0, "Rank", range(1, len(top10_cities) + 1))
top10_cities"""
    )
)

cells.append(
    md(
        """# 3. Required Visualizations

Each chart includes a title, axis labels, and a short interpretation based on what the chart shows."""
    )
)

cells.append(
    md(
        """## Visualization 1 — Top 10 Cities (Bar Chart)"""
    )
)

cells.append(
    code(
        """top_cities_plot = df_clean["City"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(x=top_cities_plot.index, y=top_cities_plot.values, ax=ax, color="#2a6f97")
ax.set_title("Top 10 Cities by Number of Restaurants")
ax.set_xlabel("City")
ax.set_ylabel("Number of Restaurants")
ax.tick_params(axis="x", rotation=35)
for label in ax.get_xticklabels():
    label.set_ha("right")
fig.tight_layout()
fig.savefig(CHART_DIR / "top_10_cities.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    md(
        """**Interpretation:** The chart shows the cities with the largest restaurant counts in descending order. The leading city has a clear gap over the rest, indicating concentrated restaurant presence in that city within this dataset."""
    )
)

cells.append(
    md(
        """## Visualization 2 — Aggregate Rating Distribution (Histogram)"""
    )
)

cells.append(
    code(
        """fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_clean["Aggregate rating"].dropna(), bins=20, color="#468faf", edgecolor="white")
ax.set_title("Distribution of Aggregate Ratings")
ax.set_xlabel("Aggregate Rating")
ax.set_ylabel("Number of Restaurants")
fig.tight_layout()
fig.savefig(CHART_DIR / "rating_distribution.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    md(
        """**Interpretation:** The histogram shows a large spike near 0 (commonly 'Not rated') and a denser cluster of ratings roughly in the mid-to-high range for rated restaurants."""
    )
)

cells.append(
    md(
        """## Visualization 3 — Online Delivery Availability (Bar Chart)"""
    )
)

cells.append(
    code(
        """delivery_counts = df_clean["Has Online delivery"].value_counts().reindex(["Yes", "No"])

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=delivery_counts.index, y=delivery_counts.values, ax=ax, color="#1b4965")
ax.set_title("Number of Restaurants Offering Online Delivery")
ax.set_xlabel("Has Online Delivery")
ax.set_ylabel("Number of Restaurants")
fig.tight_layout()
fig.savefig(CHART_DIR / "online_delivery.png", dpi=150, bbox_inches="tight")
plt.show()

delivery_counts.to_frame("Restaurant Count")"""
    )
)

cells.append(
    md(
        """**Interpretation:** The chart compares raw restaurant counts for online delivery Yes vs No. In this dataset, restaurants without online delivery form the larger group."""
    )
)

cells.append(
    md(
        """## Visualization 4 — Top 10 Most Common Cuisines (Bar Chart)

Multi-cuisine fields are split so each cuisine is counted individually."""
    )
)

cells.append(
    code(
        """top_cuisines_plot = cuisine_counts.head(10)

fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(x=top_cuisines_plot.index, y=top_cuisines_plot.values, ax=ax, color="#5fa8d3")
ax.set_title("Top 10 Most Common Cuisines")
ax.set_xlabel("Cuisine")
ax.set_ylabel("Number of Occurrences")
ax.tick_params(axis="x", rotation=35)
for label in ax.get_xticklabels():
    label.set_ha("right")
fig.tight_layout()
fig.savefig(CHART_DIR / "top_10_cuisines.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    md(
        """**Interpretation:** After splitting cuisine lists, the chart ranks individual cuisine tags by frequency. The top cuisine dominates the cuisine landscape in this dataset."""
    )
)

cells.append(
    md(
        """## Visualization 5 — Average Rating by Price Range (Line Chart)

Price range is treated as an ordered numeric scale (1 → 2 → 3 → 4)."""
    )
)

cells.append(
    code(
        """rating_by_price = (
    df_clean.groupby("Price range", dropna=True)["Aggregate rating"]
    .mean()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(
    rating_by_price.index.astype(int),
    rating_by_price.values,
    marker="o",
    linewidth=2,
    color="#013a63",
)
ax.set_title("Average Restaurant Rating by Price Range")
ax.set_xlabel("Price Range")
ax.set_ylabel("Average Aggregate Rating")
ax.set_xticks(list(rating_by_price.index.astype(int)))
fig.tight_layout()
fig.savefig(CHART_DIR / "rating_by_price_range.png", dpi=150, bbox_inches="tight")
plt.show()

rating_by_price.rename("Average Aggregate Rating").to_frame()"""
    )
)

cells.append(
    md(
        """**Interpretation:** The line chart shows how mean aggregate rating changes across ordered price ranges. In this dataset, higher price ranges tend to correspond to higher average ratings (association only)."""
    )
)

cells.append(
    md(
        """# 4. Insights and Conclusion

## Mandatory business questions"""
    )
)

cells.append(
    code(
        """print("1) Highest restaurant presence city:")
print(f"   {top_city} with {top_city_count} restaurants")

print("\\n2) Most popular cuisine:")
print(f"   {top_cuisine} with {top_cuisine_count} occurrences")

print("\\n3) Online delivery vs ratings:")
print(
    f"   Restaurants WITH online delivery have an average rating of {online_yes:.4f}, "
    f"compared with {online_no:.4f} for restaurants WITHOUT online delivery, "
    f"a difference of {online_diff:.4f}."
)
print("   Observation only — this does not prove that online delivery causes higher ratings.")

print("\\n4) Table booking vs ratings:")
print(
    f"   Restaurants WITH table booking have an average rating of {table_yes:.4f}, "
    f"compared with {table_no:.4f} for restaurants WITHOUT table booking, "
    f"a difference of {table_diff:.4f}."
)
print("   Observation only — this does not prove that table booking causes better ratings.")

print("\\n5) What I learned from this dataset:")
print("   - Restaurant coverage is highly concentrated in a few cities.")
print("   - Cuisine tags are dominated by a small set of popular categories after splitting.")
print("   - A large share of restaurants are unrated (rating 0), which pulls down the overall mean.")
print("   - Features like online delivery and table booking are associated with higher mean ratings,")
print("     but currency mixing and rating-zero conventions are important limitations.")"""
    )
)

cells.append(
    md(
        """## Additional business insights (data-supported)

The next cell derives extra insights only from calculated aggregates."""
    )
)

cells.append(
    code(
        """# Insight A: restaurant concentration
top3_share = city_counts.head(3).sum() / city_counts.sum() * 100
print(
    f"Insight A — City concentration: Top 3 cities account for {top3_share:.1f}% "
    f"of all restaurants in the cleaned dataset."
)

# Insight B: delivery availability
delivery_share = (df_clean["Has Online delivery"].value_counts(normalize=True) * 100).round(2)
print(
    f"Insight B — Delivery availability: Yes={delivery_share.get('Yes', 0)}%, "
    f"No={delivery_share.get('No', 0)}%."
)

# Insight C: table booking availability
booking_share = (df_clean["Has Table booking"].value_counts(normalize=True) * 100).round(2)
print(
    f"Insight C — Table booking availability: Yes={booking_share.get('Yes', 0)}%, "
    f"No={booking_share.get('No', 0)}%."
)

# Insight D: price range vs rating association
price_rating = (
    df_clean.groupby("Price range")["Aggregate rating"].mean().sort_index()
)
print("Insight D — Average rating by price range:")
for pr, val in price_rating.items():
    print(f"   Price range {int(pr)}: {val:.4f}")

# Insight E: voting pattern for highly rated restaurants
high_rated = df_clean[df_clean["Aggregate rating"] >= 4.5]
print(
    f"Insight E — Highly rated restaurants (rating >= 4.5): {len(high_rated)} restaurants, "
    f"median votes={high_rated['Votes'].median():.0f}, "
    f"compared with overall median votes={df_clean['Votes'].median():.0f}."
)"""
    )
)

cells.append(
    md(
        """## Conclusion

- The cleaned Zomato dataset is usable for city, cuisine, rating, and service-feature analysis after handling dtype issues and a small number of corrupted rows.
- Restaurant presence is heavily skewed toward the top city.
- Cuisine frequency is best measured after splitting multi-cuisine strings.
- Online delivery and table booking are both associated with higher average ratings in this sample.
- Analysts should treat rating-zero and multi-currency cost fields carefully before making business decisions.

**End of notebook.**"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Wrote notebook: {NB_PATH}")
print(f"Cells: {len(cells)}")
