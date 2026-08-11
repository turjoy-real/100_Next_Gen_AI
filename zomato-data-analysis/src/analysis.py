"""
Zomato Restaurant Dataset — Exploratory Data Analysis
Reusable analysis script that mirrors the notebook workflow.
Run from the project root:
    python src/analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "zomato.csv"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
REPORT_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "analysis_results.json"

YES_NO_COLS = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
NUMERIC_FOCUS = ["Average Cost for two", "Price range", "Aggregate rating", "Votes"]


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and keep the original dataframe intact for exploration."""
    return pd.read_csv(path)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "missing_count": df.isnull().sum(),
            "missing_pct": (df.isnull().mean() * 100).round(2),
        }
    )
    return summary.sort_values("missing_count", ascending=False)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Create a cleaned copy of the dataset.

    Cleaning decisions (documented for reproducibility):
    1. Drop the leftover index column `Unnamed: 0` if present.
    2. Convert `Average Cost for two` to numeric with coercion; report NaNs created.
    3. Remove two corrupted rows caused by an unquoted multi-line address
       (Restaurant ID / Yes-No fields shifted). These rows are not usable.
    4. Normalize Yes/No columns to Title-case Yes/No only (invalid -> NaN).
    5. Ensure key numeric columns are numeric.
    6. Strip whitespace from text columns used in analysis.
    7. Do NOT drop remaining missing Cuisines / ratings — preserve useful rows.
    """
    cleaning_log: dict = {}
    df_clean = df.copy()

    # 1. Drop unnamed index column
    unnamed_cols = [c for c in df_clean.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df_clean = df_clean.drop(columns=unnamed_cols)
        cleaning_log["dropped_unnamed_columns"] = unnamed_cols

    # 2. Average Cost for two -> numeric
    before_nulls = int(df_clean["Average Cost for two"].isna().sum())
    df_clean["Average Cost for two"] = pd.to_numeric(
        df_clean["Average Cost for two"], errors="coerce"
    )
    after_nulls = int(df_clean["Average Cost for two"].isna().sum())
    cleaning_log["avg_cost_nulls_before"] = before_nulls
    cleaning_log["avg_cost_nulls_after"] = after_nulls
    cleaning_log["avg_cost_converted_to_nan"] = after_nulls - before_nulls

    # 3. Remove corrupted / shifted rows
    restaurant_id_num = pd.to_numeric(df_clean["Restaurant ID"], errors="coerce")
    valid_yes_no = df_clean["Has Online delivery"].isin(["Yes", "No"]) & df_clean[
        "Has Table booking"
    ].isin(["Yes", "No"])
    keep_mask = restaurant_id_num.notna() & valid_yes_no
    dropped = int((~keep_mask).sum())
    cleaning_log["corrupted_rows_dropped"] = dropped
    if dropped:
        cleaning_log["dropped_restaurant_names"] = (
            df_clean.loc[~keep_mask, "Restaurant Name"].astype(str).tolist()
        )
    df_clean = df_clean.loc[keep_mask].copy()

    # 4. Normalize Yes/No columns
    for col in YES_NO_COLS:
        if col in df_clean.columns:
            normalized = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = normalized.where(normalized.isin(["Yes", "No"]))

    # 5. Numeric conversions for key columns
    for col in ["Restaurant ID", "Country Code", "Longitude", "Latitude", "Price range", "Aggregate rating", "Votes"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 6. Strip whitespace on important text fields
    for col in ["Restaurant Name", "City", "Cuisines", "Rating text", "Currency"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("string").str.strip()

    df_clean = df_clean.reset_index(drop=True)
    cleaning_log["rows_after_cleaning"] = int(len(df_clean))
    cleaning_log["columns_after_cleaning"] = int(df_clean.shape[1])
    return df_clean, cleaning_log


def explode_cuisines(df: pd.DataFrame) -> pd.Series:
    """Split multi-cuisine strings and return a Series of individual cuisine labels."""
    return (
        df["Cuisines"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .loc[lambda s: s.ne("")]
    )


def compute_results(df_clean: pd.DataFrame, cleaning_log: dict, df_raw: pd.DataFrame) -> dict:
    """Compute all assignment answers from the cleaned dataframe."""
    cuisine_counts = explode_cuisines(df_clean).value_counts()
    top_cuisine = cuisine_counts.index[0]
    top_cuisine_count = int(cuisine_counts.iloc[0])

    city_counts = df_clean["City"].value_counts()
    top_city = city_counts.index[0]
    top_city_count = int(city_counts.iloc[0])

    avg_rating = float(df_clean["Aggregate rating"].mean())
    max_votes = float(df_clean["Votes"].max())
    top_voted = df_clean.loc[
        df_clean["Votes"] == max_votes, ["Restaurant Name", "City", "Votes", "Aggregate rating"]
    ]

    avg_cost = float(df_clean["Average Cost for two"].mean())

    online_summary = (
        df_clean.groupby("Has Online delivery", dropna=True)["Aggregate rating"]
        .agg(restaurant_count="count", average_rating="mean")
        .reset_index()
        .rename(columns={"Has Online delivery": "online_delivery_status"})
    )
    online_yes = float(
        online_summary.loc[
            online_summary["online_delivery_status"] == "Yes", "average_rating"
        ].iloc[0]
    )
    online_no = float(
        online_summary.loc[
            online_summary["online_delivery_status"] == "No", "average_rating"
        ].iloc[0]
    )

    table_summary = (
        df_clean.groupby("Has Table booking", dropna=True)["Aggregate rating"]
        .agg(restaurant_count="count", average_rating="mean")
        .reset_index()
        .rename(columns={"Has Table booking": "table_booking_status"})
    )
    table_yes = float(
        table_summary.loc[
            table_summary["table_booking_status"] == "Yes", "average_rating"
        ].iloc[0]
    )
    table_no = float(
        table_summary.loc[
            table_summary["table_booking_status"] == "No", "average_rating"
        ].iloc[0]
    )

    top10_cities = (
        city_counts.head(10)
        .rename_axis("City")
        .reset_index(name="Restaurant Count")
    )
    top10_cities.insert(0, "Rank", range(1, len(top10_cities) + 1))

    top10_cuisines = (
        cuisine_counts.head(10)
        .rename_axis("Cuisine")
        .reset_index(name="Count")
    )

    rating_by_price = (
        df_clean.groupby("Price range", dropna=True)["Aggregate rating"]
        .mean()
        .sort_index()
    )

    numeric_stats = df_clean[NUMERIC_FOCUS].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    # Add quartiles
    quartiles = df_clean[NUMERIC_FOCUS].quantile([0.25, 0.5, 0.75])
    quartiles.index = ["25%", "50%", "75%"]

    missing_raw = missing_value_summary(df_raw)
    duplicates_raw = int(df_raw.duplicated().sum())

    # Additional insights support
    rated = df_clean[df_clean["Aggregate rating"] > 0]
    not_rated_count = int((df_clean["Aggregate rating"] == 0).sum())
    delivery_share = (
        df_clean["Has Online delivery"].value_counts(normalize=True) * 100
    ).round(2)
    booking_share = (
        df_clean["Has Table booking"].value_counts(normalize=True) * 100
    ).round(2)

    results = {
        "raw_shape": list(df_raw.shape),
        "clean_shape": list(df_clean.shape),
        "duplicates_raw": duplicates_raw,
        "missing_raw": missing_raw.to_dict(),
        "cleaning_log": cleaning_log,
        "unique_cities": int(df_clean["City"].nunique()),
        "top_city": top_city,
        "top_city_count": top_city_count,
        "top_cuisine": top_cuisine,
        "top_cuisine_count": top_cuisine_count,
        "avg_rating": avg_rating,
        "max_votes": max_votes,
        "top_voted_restaurants": top_voted.to_dict(orient="records"),
        "avg_cost_for_two": avg_cost,
        "online_summary": online_summary.to_dict(orient="records"),
        "online_yes_avg": online_yes,
        "online_no_avg": online_no,
        "online_diff": online_yes - online_no,
        "table_summary": table_summary.to_dict(orient="records"),
        "table_yes_avg": table_yes,
        "table_no_avg": table_no,
        "table_diff": table_yes - table_no,
        "top10_cities": top10_cities.to_dict(orient="records"),
        "top10_cuisines": top10_cuisines.to_dict(orient="records"),
        "rating_by_price": {str(k): float(v) for k, v in rating_by_price.items()},
        "numeric_stats": numeric_stats.round(4).to_dict(),
        "quartiles": quartiles.round(4).to_dict(),
        "not_rated_count": not_rated_count,
        "rated_avg": float(rated["Aggregate rating"].mean()) if len(rated) else None,
        "delivery_share": delivery_share.to_dict(),
        "booking_share": booking_share.to_dict(),
        "avg_cost_dtype_raw": str(df_raw["Average Cost for two"].dtype),
        "avg_cost_dtype_clean": str(df_clean["Average Cost for two"].dtype),
    }
    return results


def save_charts(df_clean: pd.DataFrame) -> dict[str, Path]:
    """Create and save all five required charts. Returns chart file paths."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Top 10 cities
    city_counts = df_clean["City"].value_counts().head(10).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax, color="#2a6f97")
    ax.set_title("Top 10 Cities by Number of Restaurants")
    ax.set_xlabel("City")
    ax.set_ylabel("Number of Restaurants")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    paths["top_10_cities"] = CHART_DIR / "top_10_cities.png"
    fig.savefig(paths["top_10_cities"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Rating distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_clean["Aggregate rating"].dropna(), bins=20, color="#468faf", edgecolor="white")
    ax.set_title("Distribution of Aggregate Ratings")
    ax.set_xlabel("Aggregate Rating")
    ax.set_ylabel("Number of Restaurants")
    fig.tight_layout()
    paths["rating_distribution"] = CHART_DIR / "rating_distribution.png"
    fig.savefig(paths["rating_distribution"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Online delivery counts
    delivery_counts = df_clean["Has Online delivery"].value_counts().reindex(["Yes", "No"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=delivery_counts.index, y=delivery_counts.values, ax=ax, color="#1b4965")
    ax.set_title("Number of Restaurants Offering Online Delivery")
    ax.set_xlabel("Has Online Delivery")
    ax.set_ylabel("Number of Restaurants")
    fig.tight_layout()
    paths["online_delivery"] = CHART_DIR / "online_delivery.png"
    fig.savefig(paths["online_delivery"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Top 10 cuisines
    cuisine_counts = explode_cuisines(df_clean).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=cuisine_counts.index, y=cuisine_counts.values, ax=ax, color="#5fa8d3")
    ax.set_title("Top 10 Most Common Cuisines")
    ax.set_xlabel("Cuisine")
    ax.set_ylabel("Number of Occurrences")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    paths["top_10_cuisines"] = CHART_DIR / "top_10_cuisines.png"
    fig.savefig(paths["top_10_cuisines"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Average rating by price range (ordered)
    rating_by_price = (
        df_clean.groupby("Price range", dropna=True)["Aggregate rating"].mean().sort_index()
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
    paths["rating_by_price_range"] = CHART_DIR / "rating_by_price_range.png"
    fig.savefig(paths["rating_by_price_range"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    return paths


def write_markdown_report(results: dict, chart_paths: dict[str, Path]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / "zomato_analysis_report.md"

    online_yes = results["online_yes_avg"]
    online_no = results["online_no_avg"]
    table_yes = results["table_yes_avg"]
    table_no = results["table_no_avg"]
    top_voted = results["top_voted_restaurants"][0]

    lines = [
        "# Zomato Restaurant Dataset — EDA Report",
        "",
        "## 1. Dataset Overview",
        "",
        "This report summarizes exploratory data analysis on the Zomato restaurant dataset.",
        f"- **Rows (raw):** {results['raw_shape'][0]}",
        f"- **Columns (raw):** {results['raw_shape'][1]}",
        f"- **Rows (cleaned):** {results['clean_shape'][0]}",
        f"- **Columns (cleaned):** {results['clean_shape'][1]}",
        "",
        "Key fields analyzed include City, Cuisines, Average Cost for two, Online Delivery,",
        "Table Booking, Price Range, Aggregate Rating, and Votes.",
        "",
        "## 2. Data Cleaning",
        "",
        f"- **Duplicate rows in raw data:** {results['duplicates_raw']}",
        f"- **Average Cost for two dtype (raw → clean):** "
        f"`{results['avg_cost_dtype_raw']}` → `{results['avg_cost_dtype_clean']}`",
        f"- **Values converted to NaN during Average Cost conversion:** "
        f"{results['cleaning_log']['avg_cost_converted_to_nan']}",
        f"- **Corrupted rows dropped:** {results['cleaning_log']['corrupted_rows_dropped']} "
        "(address-line break shifted columns for Super Loco / Robertson Quay).",
        "- Missing Cuisines (9 remaining after cleaning) were retained for non-cuisine analyses.",
        "- Remaining missing values are sparse (<0.2% for most columns) and were not force-dropped.",
        "",
        "## 3. Data Analysis",
        "",
        "### Required questions",
        f"1. **Unique cities:** {results['unique_cities']}",
        f"2. **City with most restaurants:** {results['top_city']} "
        f"({results['top_city_count']:,} restaurants)",
        f"3. **Most frequent cuisine (after splitting multi-cuisine fields):** "
        f"{results['top_cuisine']} ({results['top_cuisine_count']:,} occurrences)",
        "",
        "### Additional analysis",
        f"- **Average aggregate rating:** {results['avg_rating']:.4f}",
        f"- **Highest-voted restaurant:** {top_voted['Restaurant Name']} "
        f"({top_voted['City']}) with {int(top_voted['Votes']):,} votes",
        f"- **Average cost for two:** {results['avg_cost_for_two']:.2f} (mixed currencies; interpret carefully)",
        f"- **Online delivery ratings:** Yes = {online_yes:.4f}, No = {online_no:.4f}, "
        f"difference = {results['online_diff']:.4f}",
        f"- **Table booking ratings:** Yes = {table_yes:.4f}, No = {table_no:.4f}, "
        f"difference = {results['table_diff']:.4f}",
        "",
        "### Statistical summary (key numerical columns)",
        "",
        "| Metric | Average Cost for two | Price range | Aggregate rating | Votes |",
        "|---|---:|---:|---:|---:|",
    ]

    stats = results["numeric_stats"]
    for metric in ["count", "mean", "median", "std", "min", "max"]:
        row = [metric]
        for col in NUMERIC_FOCUS:
            row.append(f"{stats[col][metric]:.4f}" if metric != "count" else f"{int(stats[col][metric])}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## 4. Data Visualizations",
            "",
            "### Top 10 cities",
            f"![Top 10 cities]({_rel(chart_paths['top_10_cities'])})",
            f"{results['top_city']} has the largest restaurant presence in this dataset, "
            "followed by other NCR cities such as Gurgaon and Noida.",
            "",
            "### Aggregate rating distribution",
            f"![Rating distribution]({_rel(chart_paths['rating_distribution'])})",
            "The rating distribution shows a large spike at 0 (Not rated) and a cluster of "
            "rated restaurants roughly between 3.0 and 4.5.",
            "",
            "### Online delivery availability",
            f"![Online delivery]({_rel(chart_paths['online_delivery'])})",
            "Most restaurants in the dataset do not offer online delivery; the No category "
            "is substantially larger than Yes.",
            "",
            "### Top 10 cuisines",
            f"![Top 10 cuisines]({_rel(chart_paths['top_10_cuisines'])})",
            f"{results['top_cuisine']} is the most common cuisine tag after splitting "
            "multi-cuisine restaurant listings.",
            "",
            "### Average rating by price range",
            f"![Rating by price range]({_rel(chart_paths['rating_by_price_range'])})",
            "Average ratings generally increase with higher price ranges in this dataset, "
            "suggesting a positive association between price tier and reported rating "
            "(correlation/observation only, not causation).",
            "",
            "## 5. Conclusion",
            "",
            f"- Highest restaurant presence: **{results['top_city']}**.",
            f"- Most popular cuisine tag: **{results['top_cuisine']}**.",
            f"- Restaurants with online delivery have a higher average rating "
            f"({online_yes:.4f}) than those without ({online_no:.4f}).",
            f"- Restaurants with table booking have a higher average rating "
            f"({table_yes:.4f}) than those without ({table_no:.4f}).",
            "- Additional observations: restaurant concentration is heavily skewed toward "
            "New Delhi/NCR; many restaurants are unrated (rating = 0); price range and "
            "average rating move upward together in this sample.",
            "",
            "### Limitations",
            "- Average Cost for two mixes multiple currencies, so global mean cost is not "
            "directly comparable across countries.",
            "- Ratings of 0 often mean 'Not rated' rather than a true zero-quality score.",
            "- Associations (e.g., delivery vs rating) should not be interpreted as causal.",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _rel(path: Path) -> str:
    """Return a path relative to the reports/ folder for Markdown image links."""
    try:
        return str(Path("..") / path.relative_to(REPORT_DIR.parent))
    except ValueError:
        return str(path)


def write_pdf_report(results: dict, chart_paths: dict[str, Path]) -> Path:
    """Generate a short PDF report using reportlab from calculated results."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / "zomato_analysis_report.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Heading1"], spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body2", parent=styles["BodyText"], fontSize=10, leading=13)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    story = []

    online_yes = results["online_yes_avg"]
    online_no = results["online_no_avg"]
    table_yes = results["table_yes_avg"]
    table_no = results["table_no_avg"]
    top_voted = results["top_voted_restaurants"][0]

    story.append(Paragraph("Zomato Restaurant Dataset — EDA Report", title_style))
    story.append(Paragraph("1. Dataset Overview", h2))
    story.append(
        Paragraph(
            f"Raw dataset: {results['raw_shape'][0]} rows × {results['raw_shape'][1]} columns. "
            f"Cleaned dataset: {results['clean_shape'][0]} rows × {results['clean_shape'][1]} columns. "
            "The analysis covers cities, cuisines, cost, delivery, table booking, ratings, and votes.",
            body,
        )
    )

    story.append(Paragraph("2. Data Cleaning", h2))
    story.append(
        Paragraph(
            f"Duplicates found: {results['duplicates_raw']}. "
            f"Average Cost for two converted from {results['avg_cost_dtype_raw']} to "
            f"{results['avg_cost_dtype_clean']}; "
            f"{results['cleaning_log']['avg_cost_converted_to_nan']} value(s) became NaN during conversion. "
            f"{results['cleaning_log']['corrupted_rows_dropped']} corrupted row(s) dropped due to a "
            "broken multi-line address. Missing cuisine values were retained where possible.",
            body,
        )
    )

    story.append(Paragraph("3. Data Analysis", h2))
    story.append(
        Paragraph(
            f"Unique cities: {results['unique_cities']}. "
            f"Top city: {results['top_city']} ({results['top_city_count']:,} restaurants). "
            f"Most frequent cuisine: {results['top_cuisine']} ({results['top_cuisine_count']:,}). "
            f"Average rating: {results['avg_rating']:.4f}. "
            f"Highest votes: {top_voted['Restaurant Name']} with {int(top_voted['Votes']):,} votes. "
            f"Average cost for two: {results['avg_cost_for_two']:.2f}. "
            f"Online delivery avg rating Yes {online_yes:.4f} vs No {online_no:.4f} "
            f"(diff {results['online_diff']:.4f}). "
            f"Table booking avg rating Yes {table_yes:.4f} vs No {table_no:.4f} "
            f"(diff {results['table_diff']:.4f}).",
            body,
        )
    )

    story.append(Paragraph("4. Data Visualizations", h2))
    for key, caption in [
        ("top_10_cities", "Top 10 cities by restaurant count."),
        ("rating_distribution", "Distribution of aggregate ratings."),
        ("online_delivery", "Online delivery Yes/No counts."),
        ("top_10_cuisines", "Top 10 cuisines after splitting multi-cuisine fields."),
        ("rating_by_price_range", "Average rating by ordered price range."),
    ]:
        story.append(Paragraph(caption, body))
        img_path = chart_paths[key]
        story.append(Image(str(img_path), width=6.2 * inch, height=3.4 * inch))
        story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("5. Conclusion", h2))
    story.append(
        Paragraph(
            f"{results['top_city']} has the highest restaurant presence. "
            f"{results['top_cuisine']} is the most popular cuisine tag. "
            f"Restaurants with online delivery have higher average ratings "
            f"({online_yes:.4f} vs {online_no:.4f}). "
            f"Restaurants with table booking also show higher average ratings "
            f"({table_yes:.4f} vs {table_no:.4f}). "
            "These are observational differences in this dataset, not causal claims. "
            "Limitation: average cost mixes currencies; rating 0 often means Not rated.",
            body,
        )
    )

    doc.build(story)
    return pdf_path


def main() -> None:
    configure_plotting()
    df = load_raw_data()
    df_clean, cleaning_log = clean_data(df)
    results = compute_results(df_clean, cleaning_log, df)
    chart_paths = save_charts(df_clean)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        **results,
        "chart_paths": {k: str(v) for k, v in chart_paths.items()},
    }
    RESULTS_PATH.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    md_path = write_markdown_report(results, chart_paths)
    pdf_path = write_pdf_report(results, chart_paths)

    print("=== Analysis complete ===")
    print(f"Raw shape: {results['raw_shape']}")
    print(f"Clean shape: {results['clean_shape']}")
    print(f"Duplicates: {results['duplicates_raw']}")
    print(f"Unique cities: {results['unique_cities']}")
    print(f"Top city: {results['top_city']} ({results['top_city_count']})")
    print(f"Top cuisine: {results['top_cuisine']} ({results['top_cuisine_count']})")
    print(f"Avg rating: {results['avg_rating']:.4f}")
    print(f"Report MD: {md_path}")
    print(f"Report PDF: {pdf_path}")
    print(f"Results JSON: {RESULTS_PATH}")
    for name, path in chart_paths.items():
        print(f"Chart {name}: {path}")


if __name__ == "__main__":
    main()
