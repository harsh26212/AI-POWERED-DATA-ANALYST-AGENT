import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import zipfile
from agent import ask_agent

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stDecoration"] {
    display: none;
}

[data-testid="stStatusWidget"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.chat-small-card {
    background: rgba(240, 242, 246, 0.65);
    border: 1px solid rgba(120, 120, 120, 0.18);
    border-radius: 14px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.92rem;
}
.chat-user-card { background: rgba(230, 244, 255, 0.85); }
.chat-bot-card { background: rgba(245, 245, 245, 0.90); }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Data Analyst Agent")
st.caption("Clean, filter, visualize, and understand any CSV/Excel dataset using AI.")

uploaded_file = st.file_uploader("📁 Upload CSV or Excel File", type=["csv", "xlsx"])


def clear_outputs():
    os.makedirs("outputs", exist_ok=True)
    for file in os.listdir("outputs"):
        path = os.path.join("outputs", file)
        if os.path.isfile(path):
            os.remove(path)


def safe_label(value):
    if value is None:
        return "None"
    return str(value)


def download_file_button(label, file_path, file_name, mime):
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as file:
            st.download_button(label, file, file_name=file_name, mime=mime)


def create_charts_zip(chart_paths, zip_path="outputs/charts.zip"):
    os.makedirs("outputs", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for chart_path in chart_paths:
            if chart_path and os.path.exists(chart_path):
                zip_file.write(chart_path, arcname=os.path.basename(chart_path))
    return zip_path


def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return clean_columns(df)


def clean_data(
    df,
    remove_duplicates=True,
    handle_missing=True,
    clean_text=True,
    convert_numeric=True,
    remove_constant=True,
    remove_outliers=False,
):
    report = {
        "before_rows": df.shape[0],
        "before_cols": df.shape[1],
        "before_missing": int(df.isnull().sum().sum()),
        "before_duplicates": int(df.duplicated().sum()),
        "removed_columns": [],
        "outliers_removed": 0,
    }

    df = df.copy()

    if remove_duplicates:
        df = df.drop_duplicates()

    if clean_text:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    if convert_numeric:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            sample_values = df[col].dropna().astype(str).head(20)
            contains_letters = sample_values.str.contains(r"[A-Za-z]", regex=True).any()

            if contains_letters:
                df[col] = df[col].astype(str)
                continue

            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

            if converted.notna().mean() > 0.7:
                df[col] = converted
            else:
                df[col] = df[col].astype(str)

    if remove_constant:
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
        report["removed_columns"] = constant_cols

    if handle_missing:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                if pd.isna(median_value):
                    median_value = 0
                df[col] = df[col].fillna(median_value)
            else:
                df[col] = df[col].fillna("Unknown")

    if remove_outliers:
        before_outlier_rows = df.shape[0]
        for col in df.select_dtypes(include=["number"]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr != 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        report["outliers_removed"] = before_outlier_rows - df.shape[0]

    report["after_rows"] = df.shape[0]
    report["after_cols"] = df.shape[1]
    report["after_missing"] = int(df.isnull().sum().sum())
    report["after_duplicates"] = int(df.duplicated().sum())

    return df, report


def get_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def smart_chart_suggestion(df, x_col, y_col):
    if x_col and y_col:
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            return "Scatter Plot"
        if not pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            return "Bar Chart"
    return "Histogram"


def save_chart(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def create_chart(df, graph_type, x_col=None, y_col=None, path="outputs/chart.png", clear=True):
    if clear:
        clear_outputs()
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(15, 8))

    try:
        if graph_type == "Histogram":
            data = df[y_col].dropna()
            plt.hist(data, bins=25)
            plt.title(f"Distribution of {y_col}")
            plt.xlabel(y_col)
            plt.ylabel("Frequency")

        elif graph_type == "Box Plot":
            data = df[y_col].dropna()
            plt.boxplot(data)
            plt.title(f"Box Plot of {y_col}")
            plt.ylabel(y_col)

        elif graph_type == "Count Plot":
            data = df[x_col].dropna().astype(str)
            counts = data.value_counts().head(30)
            plt.bar(counts.index.astype(str), counts.values)
            plt.title(f"Count Plot of {x_col}")
            plt.xlabel(x_col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")

        else:
            data = df[[x_col, y_col]].dropna()
            if data.empty:
                st.error("No valid data.")
                return []

            if graph_type == "Line Chart" and not pd.api.types.is_numeric_dtype(data[x_col]):
                st.warning("X column is categorical. Showing Bar Chart instead.")
                graph_type = "Bar Chart"

            if graph_type == "Bar Chart":
                grouped = data.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(30)
                plt.bar(grouped.index.astype(str), grouped.values)
                plt.title(f"Average {y_col} by {x_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45, ha="right")

            elif graph_type == "Line Chart":
                grouped = data.groupby(x_col)[y_col].mean().sort_index()
                plt.plot(grouped.index, grouped.values, marker="o")
                plt.title(f"{y_col} Trend by {x_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45, ha="right")

            elif graph_type == "Scatter Plot":
                plot_data = data.head(500)
                plt.scatter(plot_data[x_col], plot_data[y_col])
                plt.title(f"{y_col} vs {x_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45, ha="right")

            elif graph_type == "Pie Chart":
                plt.close()
                grouped = data.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(10)
                plt.figure(figsize=(10, 10))
                plt.pie(grouped.values, labels=grouped.index.astype(str), autopct="%1.1f%%", startangle=90)
                plt.title(f"Top 10 {y_col} Share by {x_col}")

        save_chart(path)
        return [path]

    except Exception as e:
        st.error(f"Chart error: {e}")
        return []



def make_kpi_cards(df):
    """Show only useful dataset-specific KPI cards, not basic row/column/missing counters."""
    measure_cols = get_measure_numeric_columns(df)
    category_cols = get_good_categorical_columns(df)
    date_cols = get_date_or_year_columns(df)

    st.subheader("📌 Smart KPI Cards")

    if measure_cols:
        main_num = measure_cols[0]
        a, b, c, d = st.columns(4)
        a.metric(f"Average {main_num}", f"{df[main_num].mean():.2f}")
        b.metric(f"Maximum {main_num}", f"{df[main_num].max():.2f}")
        c.metric(f"Minimum {main_num}", f"{df[main_num].min():.2f}")
        d.metric(f"Total {main_num}", f"{df[main_num].sum():.2f}")

        if category_cols:
            cat_col = category_cols[0]
            grouped = df[[cat_col, main_num]].dropna().groupby(cat_col)[main_num].mean().sort_values(ascending=False)
            if not grouped.empty:
                e, f = st.columns(2)
                e.metric(f"Top {cat_col}", str(grouped.idxmax()))
                f.metric(f"Top Avg {main_num}", f"{grouped.max():.2f}")

        if date_cols:
            date_col = date_cols[0]
            trend = df[[date_col, main_num]].dropna().groupby(date_col)[main_num].mean().sort_index()
            if len(trend) >= 2:
                g, h = st.columns(2)
                g.metric(f"First {date_col} Avg", f"{trend.iloc[0]:.2f}")
                h.metric(f"Latest {date_col} Avg", f"{trend.iloc[-1]:.2f}")

    elif category_cols:
        cat_col = category_cols[0]
        counts = df[cat_col].dropna().astype(str).value_counts()
        if not counts.empty:
            a, b, c = st.columns(3)
            a.metric(f"Unique {cat_col}", int(counts.shape[0]))
            b.metric(f"Most Common {cat_col}", str(counts.idxmax()))
            c.metric("Highest Count", int(counts.max()))
    else:
        st.info("No suitable KPI columns found for this dataset.")


def _save_simple_chart(path):
    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _norm_col_name(col):
    return str(col).lower().replace("_", " ").replace("-", " ").strip()


def get_date_or_year_columns(df):
    """Detect year/date-like columns so they are used only as time axes, not measure values."""
    result = []
    for col in df.columns:
        name = _norm_col_name(col)
        if any(word in name for word in ["year", "date", "time"]):
            result.append(col)
            continue
        if name in ["month", "months", "day", "days"]:
            result.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if not s.empty:
                if s.between(1800, 2100).mean() > 0.90 and s.nunique() > 10:
                    result.append(col)
    return list(dict.fromkeys(result))


def get_measure_numeric_columns(df):
    """Pick real value/measure columns and avoid YEAR/ID/index/code columns.

    Dynamic priority examples:
    - Rainfall: ANNUAL, JAN, FEB, seasonal columns
    - Sales: sales, revenue, profit, amount
    - Taxi: fare, distance, duration
    """
    bad_keywords = ["id", "code", "serial", "index", "rank", "year", "date", "time", "latitude", "longitude", "lat", "lng", "zip", "pin"]
    priority_keywords = [
        "annual", "total", "sales", "revenue", "profit", "amount", "price", "cost", "fare",
        "rainfall", "rain", "value", "score", "marks", "quantity", "qty", "distance", "duration",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "jjas", "mam", "ond"
    ]
    date_cols = set(get_date_or_year_columns(df))
    candidates = []

    for col in df.select_dtypes(include="number").columns:
        name = _norm_col_name(col)
        if col in date_cols:
            continue
        if any(k in name for k in bad_keywords):
            continue

        s = df[col].dropna()
        if s.empty or s.nunique() <= 1:
            continue

        score = 0.0
        # Prefer meaningful business/science measure names.
        for i, key in enumerate(priority_keywords):
            if key in name:
                score += 1000 - i
                break
        # Prefer columns with variation and enough values.
        score += min(float(s.nunique()), 100.0)
        score += min(float(abs(s.mean())) if pd.notna(s.mean()) else 0.0, 200.0) / 10
        score += min(float(s.std()) if pd.notna(s.std()) else 0.0, 500.0) / 10
        candidates.append((score, col))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    return [col for _, col in candidates]


def get_good_categorical_columns(df):
    """Pick categorical columns that are useful for grouping across different datasets."""
    candidates = []
    total_rows = len(df)
    bad_keywords = ["id", "code", "address", "phone", "email", "name", "url", "link"]
    priority_keywords = ["subdivision", "state", "city", "country", "region", "category", "type", "class", "department", "segment", "product"]

    for col in df.select_dtypes(exclude="number").columns:
        name = _norm_col_name(col)
        if any(k in name for k in bad_keywords):
            continue
        nunique = df[col].dropna().astype(str).nunique()
        if nunique < 2:
            continue
        if total_rows > 0 and nunique / total_rows > 0.80:
            continue
        if nunique > 150:
            continue

        score = 100 - min(nunique, 100)
        for i, key in enumerate(priority_keywords):
            if key in name:
                score += 500 - i
                break
        candidates.append((score, col))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    return [col for _, col in candidates]


def _has_nearly_equal_counts(series):
    counts = series.dropna().astype(str).value_counts()
    if len(counts) < 2:
        return True
    return counts.max() - counts.min() <= 1


def generate_auto_dashboard_charts(df):
    """Generate accurate, dataset-aware dashboard charts.

    Stronger rules:
    - YEAR/date columns are used only for trend axis.
    - Category count chart is skipped when all categories have same count.
    - If a measure exists, dashboard focuses on useful charts like top category by average measure.
    - Correlation heatmap uses simple readable coolwarm colors.
    """
    os.makedirs("outputs", exist_ok=True)
    chart_paths = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    measure_cols = get_measure_numeric_columns(df)
    category_cols = get_good_categorical_columns(df)
    date_cols = get_date_or_year_columns(df)

    main_measure = measure_cols[0] if measure_cols else None
    main_category = category_cols[0] if category_cols else None
    main_date = date_cols[0] if date_cols else None

    # 1. Distribution of main real measure column.
    if main_measure:
        data = df[main_measure].dropna()
        if not data.empty:
            path = "outputs/dashboard_numeric_distribution.png"
            plt.figure(figsize=(12, 6))
            plt.hist(data, bins=25, edgecolor="black")
            plt.title(f"Distribution of {main_measure}")
            plt.xlabel(main_measure)
            plt.ylabel("Frequency")
            chart_paths.append((f"Distribution of {main_measure}", _save_simple_chart(path)))

    # 2. Top category by average measure. This replaces useless equal-count category graphs.
    if main_measure and main_category:
        grouped = (
            df[[main_category, main_measure]]
            .dropna()
            .groupby(main_category)[main_measure]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        if not grouped.empty:
            path = "outputs/dashboard_top_category_by_measure.png"
            plt.figure(figsize=(12, 6))
            plt.bar(grouped.index.astype(str), grouped.values)
            plt.title(f"Top {main_category} by Average {main_measure}")
            plt.xlabel(main_category)
            plt.ylabel(f"Average {main_measure}")
            plt.xticks(rotation=45, ha="right")
            chart_paths.append((f"Top {main_category} by Avg {main_measure}", _save_simple_chart(path)))

    # 3. Trend over year/date if available.
    if main_measure and main_date:
        trend_data = df[[main_date, main_measure]].dropna()
        if not trend_data.empty:
            grouped = trend_data.groupby(main_date)[main_measure].mean().sort_index()
            if len(grouped) >= 2:
                path = "outputs/dashboard_trend.png"
                plt.figure(figsize=(12, 6))
                plt.plot(grouped.index, grouped.values, marker="o")
                plt.title(f"Average {main_measure} Trend by {main_date}")
                plt.xlabel(main_date)
                plt.ylabel(f"Average {main_measure}")
                plt.xticks(rotation=45, ha="right")
                chart_paths.append((f"Trend: {main_measure} by {main_date}", _save_simple_chart(path)))

    # 4. Category count only if no measure chart exists OR counts are actually meaningful.
    if main_category and (not main_measure) and not _has_nearly_equal_counts(df[main_category]):
        counts = df[main_category].dropna().astype(str).value_counts().head(15)
        if not counts.empty:
            path = "outputs/dashboard_category_count.png"
            plt.figure(figsize=(12, 6))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title(f"Top Values in {main_category}")
            plt.xlabel(main_category)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            chart_paths.append((f"Top Values in {main_category}", _save_simple_chart(path)))

    # 5. Missing values chart only when missing values exist.
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(15)
    if not missing.empty:
        path = "outputs/dashboard_missing_values.png"
        plt.figure(figsize=(12, 6))
        plt.bar(missing.index.astype(str), missing.values)
        plt.title("Top Columns with Missing Values")
        plt.xlabel("Columns")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=45, ha="right")
        chart_paths.append(("Missing Values", _save_simple_chart(path)))

    # 6. Correlation heatmap: use real measure columns only.
    corr_candidates = measure_cols[:8]
    if len(corr_candidates) >= 2:
        corr = df[corr_candidates].corr(numeric_only=True)
        path = "outputs/dashboard_correlation_heatmap.png"
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns.astype(str), rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index.astype(str))
        plt.title("Correlation Heatmap")

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    text_color = "white" if abs(value) >= 0.65 else "black"
                    plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

        chart_paths.append(("Correlation Heatmap", _save_simple_chart(path)))

    return chart_paths


def render_auto_dashboard(df):
    st.header("⚡ Auto Dashboard Generator")
    st.caption("Automatically creates smart KPI cards and accurate charts based on your uploaded dataset.")

    make_kpi_cards(df)
    if st.button("⚡ Generate Auto Dashboard", width="stretch"):

    # if st.button("⚡ Generate Auto Dashboard", use_container_width=True):
        with st.spinner("Generating dashboard..."):
            charts = generate_auto_dashboard_charts(df)

        if not charts:
            st.warning("Not enough suitable columns to generate dashboard charts.")
        else:
            st.subheader("📊 Auto Generated Dashboard")
            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for j, col_box in enumerate(cols):
                    if i + j < len(charts):
                        title, path = charts[i + j]
                        with col_box:
                            st.markdown(f"### {title}")
                            st.image(path, use_container_width=True)

            zip_path = create_charts_zip([path for _, path in charts], zip_path="outputs/auto_dashboard_charts.zip")
            download_file_button(
                "⬇️ Download Auto Dashboard Charts ZIP",
                zip_path,
                "auto_dashboard_charts.zip",
                "application/zip",
            )

            dashboard_prompt = f"""
You are an expert data analyst.

Analyze this dataset dashboard summary:
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing values: {int(df.isnull().sum().sum())}
Duplicate rows: {int(df.duplicated().sum())}
Numeric columns: {df.select_dtypes(include='number').columns.tolist()}
Categorical columns: {df.select_dtypes(exclude='number').columns.tolist()}

Numeric summary:
{df.select_dtypes(include='number').describe().round(2).to_string() if not df.select_dtypes(include='number').empty else 'No numeric columns'}

Give short dashboard insights in simple language.
Mention important KPIs, data quality issues, and useful patterns.
Do not write code.
"""
            try:
                st.subheader("🧠 AI Dashboard Insights")
                st.success(ask_agent(dashboard_prompt))
            except Exception as e:
                st.info(f"Dashboard generated. AI insights failed: {e}")

def generate_chart_code(graph_type, x_col=None, y_col=None):
    return f'''
import pandas as pd
import matplotlib.pyplot as plt

graph_type = "{graph_type}"
x_col = "{x_col}"
y_col = "{y_col}"

plt.figure(figsize=(15, 8))
# Chart generated using matplotlib based on selected graph type.
plt.tight_layout()
plt.savefig("outputs/chart.png", dpi=180, bbox_inches="tight")
plt.close()
'''


def generate_graph_insights(df, x_col, y_col, graph_type):
    try:
        if graph_type in ["Histogram", "Box Plot"]:
            data = df[y_col].dropna()
            return f"""
Column analyzed: {y_col}
Mean: {data.mean():.2f}
Median: {data.median():.2f}
Minimum: {data.min():.2f}
Maximum: {data.max():.2f}
This chart shows the distribution or spread of {y_col}.
"""

        if graph_type == "Count Plot":
            counts = df[x_col].dropna().astype(str).value_counts()
            return f"""
Column analyzed: {x_col}
Most frequent value: {counts.idxmax()} with {counts.max()} records
Least frequent value: {counts.idxmin()} with {counts.min()} records
This chart shows frequency distribution of categories.
"""

        data = df[[x_col, y_col]].dropna()
        grouped = data.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        return f"""
Highest average {y_col}: {grouped.max():.2f} for {x_col} = {grouped.idxmax()}
Lowest average {y_col}: {grouped.min():.2f} for {x_col} = {grouped.idxmin()}
Overall average {y_col}: {grouped.mean():.2f}
This graph explains how {y_col} changes across {x_col}.
"""
    except Exception as e:
        return f"Could not generate graph insights: {e}"





def get_dataset_context(df, max_rows=8):
    """Create compact dataset context for AI prompts. Works with all datasets."""
    preview = df.head(max_rows).to_string(index=False)
    missing = df.isnull().sum().sort_values(ascending=False).head(15).to_dict()
    dtypes = df.dtypes.astype(str).to_dict()

    numeric_summary = "No numeric columns available."
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        numeric_summary = numeric_df.describe().round(2).to_string()

    categorical_summary = {}
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()[:10]
    for col in cat_cols:
        categorical_summary[col] = df[col].astype(str).value_counts().head(8).to_dict()

    return f"""
Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns
Columns: {df.columns.tolist()}
Data types: {dtypes}
Missing values top columns: {missing}
Numeric summary:
{numeric_summary}
Top categorical values:
{categorical_summary}
Sample rows:
{preview}
"""


def normalize_text(text):
    return str(text).lower().replace("_", " ").replace("-", " ").strip()


def find_matching_columns(df, question):
    """Find columns mentioned in a natural-language question."""
    q = normalize_text(question)
    matches = []
    for col in df.columns:
        col_norm = normalize_text(col)
        if col_norm in q or str(col).lower() in q:
            matches.append(col)
            continue
        # loose token match for names like "sales amount" / "amount"
        tokens = [t for t in col_norm.split() if len(t) >= 3]
        if tokens and all(t in q for t in tokens[:2]):
            matches.append(col)
    return list(dict.fromkeys(matches))


def build_local_answer(df, question):
    """Create deterministic statistics so Chat with Data is more accurate for every dataset."""
    q = normalize_text(question)
    matched_cols = find_matching_columns(df, question)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    lines = []
    lines.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    if "column" in q or "field" in q:
        lines.append(f"Columns: {', '.join(map(str, df.columns.tolist()))}")

    if "missing" in q or "null" in q or "blank" in q:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            lines.append("No missing values found in the current filtered dataset.")
        else:
            lines.append("Missing values by column: " + str(missing.head(10).to_dict()))

    if "duplicate" in q:
        lines.append(f"Duplicate rows: {int(df.duplicated().sum())}")

    if any(word in q for word in ["highest", "maximum", "max", "top"]):
        cols = [c for c in matched_cols if c in numeric_cols] or numeric_cols[:3]
        for col in cols:
            if df[col].notna().any():
                idx = df[col].idxmax()
                lines.append(f"Highest {col}: {df.loc[idx, col]} at row index {idx}.")

    if any(word in q for word in ["lowest", "minimum", "min", "bottom"]):
        cols = [c for c in matched_cols if c in numeric_cols] or numeric_cols[:3]
        for col in cols:
            if df[col].notna().any():
                idx = df[col].idxmin()
                lines.append(f"Lowest {col}: {df.loc[idx, col]} at row index {idx}.")

    if any(word in q for word in ["average", "mean", "median", "summary", "describe"]):
        cols = [c for c in matched_cols if c in numeric_cols] or numeric_cols[:5]
        for col in cols:
            if df[col].notna().any():
                lines.append(
                    f"{col} summary -> mean: {df[col].mean():.2f}, median: {df[col].median():.2f}, min: {df[col].min():.2f}, max: {df[col].max():.2f}."
                )

    if any(word in q for word in ["count", "frequency", "most common", "unique"]):
        cols = matched_cols or categorical_cols[:3]
        for col in cols[:3]:
            counts = df[col].dropna().astype(str).value_counts().head(5)
            if not counts.empty:
                lines.append(f"Top values in {col}: {counts.to_dict()}")

    if "correlation" in q or "relationship" in q or "related" in q:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True).abs()
            pairs = []
            for i, c1 in enumerate(numeric_cols):
                for c2 in numeric_cols[i+1:]:
                    val = corr.loc[c1, c2]
                    if pd.notna(val):
                        pairs.append((c1, c2, val))
            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
            lines.append("Strongest numeric relationships: " + str([(a, b, round(v, 3)) for a, b, v in pairs]))
        else:
            lines.append("Correlation needs at least two numeric columns.")

    if len(lines) == 1:
        lines.append("Relevant numeric columns: " + (", ".join(numeric_cols[:10]) if numeric_cols else "None"))
        lines.append("Relevant categorical columns: " + (", ".join(categorical_cols[:10]) if categorical_cols else "None"))
        if matched_cols:
            lines.append("Columns detected from your question: " + ", ".join(map(str, matched_cols)))

    return "\n".join(lines)


def chat_with_data(df, user_question):
    context = get_dataset_context(df)
    local_answer = build_local_answer(df, user_question)
    prompt = f"""
You are an accurate AI data analyst. Answer the user's question using ONLY the dataset context and computed statistics below.

Computed statistics from the current filtered dataset:
{local_answer}

Dataset context:
{context}

User question: {user_question}

Answer rules:
- Give a direct answer first.
- Use the computed statistics as the source of truth.
- Be specific to the columns that exist in this dataset.
- If the user asks for something not possible from available columns, say exactly which column/data is needed.
- Do not invent values.
- Keep the answer simple and practical.
- Do not write code unless the user asks for code.
"""
    return ask_agent(prompt)


# ================= AI COLUMN EXPLANATION HELPERS =================
def get_column_profile(df, columns):
    """Build accurate, dataset-aware column profiles for AI column explanation."""
    profiles = []
    total_rows = len(df)

    for col in columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = (missing_count / total_rows * 100) if total_rows else 0
        unique_count = int(s.nunique(dropna=True))
        dtype = str(s.dtype)

        profile = {
            "column": str(col),
            "data_type": dtype,
            "missing_count": missing_count,
            "missing_percent": round(missing_pct, 2),
            "unique_values": unique_count,
            "sample_values": s.dropna().astype(str).head(8).tolist(),
        }

        if pd.api.types.is_numeric_dtype(s):
            numeric_s = pd.to_numeric(s, errors="coerce").dropna()
            if not numeric_s.empty:
                profile.update({
                    "min": round(float(numeric_s.min()), 3),
                    "max": round(float(numeric_s.max()), 3),
                    "mean": round(float(numeric_s.mean()), 3),
                    "median": round(float(numeric_s.median()), 3),
                })
        else:
            top_values = s.dropna().astype(str).value_counts().head(5).to_dict()
            profile["top_values"] = top_values

        profiles.append(profile)

    return profiles


def local_column_guess(df, columns):
    """Fallback explanation when AI fails; also useful for better accuracy."""
    lines = []
    for col in columns:
        s = df[col]
        name = _norm_col_name(col) if '_norm_col_name' in globals() else str(col).lower()
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            kind = "numeric/measurable value"
            stats = f"Range: {s.min()} to {s.max()}, Average: {s.mean():.2f}" if s.notna().any() else "No valid numeric values"
        else:
            kind = "categorical/text information"
            top = s.dropna().astype(str).value_counts().head(3).to_dict()
            stats = f"Top values: {top}" if top else "No valid text values"

        if any(k in name for k in ["id", "code", "serial"]):
            possible = "This looks like an identifier/code column used to uniquely identify records."
        elif any(k in name for k in ["date", "time", "year", "month", "day"]):
            possible = "This looks like a time/date column used for trend or period-wise analysis."
        elif any(k in name for k in ["sales", "revenue", "profit", "amount", "price", "cost", "fare"]):
            possible = "This looks like a financial/business value column."
        elif any(k in name for k in ["city", "state", "country", "region", "category", "type", "class"]):
            possible = "This looks like a grouping/category column useful for comparison charts."
        elif any(k in name for k in ["rain", "rainfall", "temperature", "distance", "duration", "score", "marks"]):
            possible = "This looks like a measurable performance/science/quantity column."
        else:
            possible = "Meaning depends on dataset context; use sample values and datatype to understand it."

        lines.append(
            f"**{col}**\n"
            f"- Type: {kind} (`{dtype}`)\n"
            f"- Possible meaning: {possible}\n"
            f"- Data quality: {missing} missing values, {unique} unique values\n"
            f"- Quick detail: {stats}"
        )

    return "\n\n".join(lines)


def render_ai_column_explanation(df):
    """Light, clean, dynamic AI Column Explanation block near Dataset Overview."""
    st.markdown(
        """
        <style>
        .ai-col-box {
            background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 18px 20px;
            margin-top: 10px;
            margin-bottom: 8px;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08);
        }
        .ai-col-title {
            font-size: 20px;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 4px;
        }
        .ai-col-subtitle {
            font-size: 14px;
            color: #475569;
            margin-bottom: 2px;
        }
        </style>
        <div class="ai-col-box">
            <div class="ai-col-title">🤖 AI Column Explanation</div>
            <div class="ai-col-subtitle">Select columns from dropdown and get medium-detail, dataset-aware meaning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_cols = df.columns.tolist()
    default_cols = all_cols[:min(5, len(all_cols))]

    selected_columns = st.multiselect(
        "Select columns to explain",
        options=all_cols,
        default=default_cols,
        help="Choose only those columns whose meaning/details you want to know.",
    )
    detail_level = "Medium"

    if not selected_columns:
        st.info("Please select at least one column for explanation.")
        return

    if st.button("✨ Generate Column Explanation", use_container_width=True):
        profiles = get_column_profile(df, selected_columns)
        fallback = local_column_guess(df, selected_columns)

        column_prompt = f"""
You are an expert data analyst. Explain selected dataset columns accurately using ONLY the provided column profiles and sample values.

Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns
Selected columns: {selected_columns}
Detail level: {detail_level}

Column profiles:
{profiles}

Local rule-based hints:
{fallback}

Output rules:
- Explain only the selected columns.
- For each column, write:
  1. Column name
  2. What this column is probably about
  3. Why it is useful in analysis
  4. Data quality note: missing values, unique values, and any issue
  5. Best chart/use case for this column
- Use simple student-friendly English/Hinglish style.
- Do not invent exact meaning if it is unclear; say "likely" or "probably".
- Keep answer structured and readable.
- Do not write code.
"""
        with st.spinner("Generating accurate column explanation..."):
            try:
                explanation = ask_agent(column_prompt)
                st.success("Column explanation generated successfully.")
                st.markdown(explanation)
            except Exception as e:
                st.warning(f"AI failed, showing local explanation instead. Error: {e}")
                st.markdown(fallback)

def generate_pdf_report(df, x_col, y_col, graph_type, insights, chart_path):
    pdf_path = "outputs/analysis_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI DATA ANALYSIS REPORT", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("1. Dataset Overview", styles["Heading2"]))
    story.append(Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {df.shape[1]}", styles["Normal"]))
    story.append(Paragraph(f"Missing Values: {int(df.isnull().sum().sum())}", styles["Normal"]))
    story.append(Paragraph(f"Duplicate Rows: {int(df.duplicated().sum())}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Selected Analysis", styles["Heading2"]))
    story.append(Paragraph(f"Graph Type: {graph_type}", styles["Normal"]))
    if x_col:
        story.append(Paragraph(f"X Column: {x_col}", styles["Normal"]))
    if y_col:
        story.append(Paragraph(f"Y Column: {y_col}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Generated Graph", styles["Heading2"]))
    if chart_path and os.path.exists(chart_path):
        story.append(Image(chart_path, width=450, height=260))
    story.append(Spacer(1, 12))

    graph_findings = generate_graph_insights(df, x_col, y_col, graph_type)
    story.append(Paragraph("4. Graph-Based Findings", styles["Heading2"]))
    for line in graph_findings.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. AI Insights", styles["Heading2"]))
    for line in insights.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))
    story.append(Spacer(1, 12))


    story.append(Paragraph("7. Conclusion", styles["Heading2"]))
    story.append(Paragraph("This report summarizes the selected dataset, generated graph, graph-based findings, and AI insights.", styles["Normal"]))
    doc.build(story)
    return pdf_path


def generate_multi_pdf_report(
    df,
    chart_1_info,
    chart_2_info,
    compare_insights,
    chart_1_path,
    chart_2_path,
):
    pdf_path = "outputs/multi_chart_analysis_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI MULTI-CHART COMPARISON REPORT", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Dataset Overview", styles["Heading2"]))
    story.append(Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {df.shape[1]}", styles["Normal"]))
    story.append(Paragraph(f"Missing Values: {int(df.isnull().sum().sum())}", styles["Normal"]))
    story.append(Paragraph(f"Duplicate Rows: {int(df.duplicated().sum())}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Chart 1 Details", styles["Heading2"]))
    story.append(Paragraph(f"Graph Type: {chart_1_info['graph_type']}", styles["Normal"]))
    story.append(Paragraph(f"X Column: {safe_label(chart_1_info['x_col'])}", styles["Normal"]))
    story.append(Paragraph(f"Y Column: {safe_label(chart_1_info['y_col'])}", styles["Normal"]))
    if chart_1_path and os.path.exists(chart_1_path):
        story.append(Image(chart_1_path, width=450, height=250))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Chart 2 Details", styles["Heading2"]))
    story.append(Paragraph(f"Graph Type: {chart_2_info['graph_type']}", styles["Normal"]))
    story.append(Paragraph(f"X Column: {safe_label(chart_2_info['x_col'])}", styles["Normal"]))
    story.append(Paragraph(f"Y Column: {safe_label(chart_2_info['y_col'])}", styles["Normal"]))
    if chart_2_path and os.path.exists(chart_2_path):
        story.append(Image(chart_2_path, width=450, height=250))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Graph-Based Findings", styles["Heading2"]))
    graph_1_findings = generate_graph_insights(
        df,
        chart_1_info["x_col"],
        chart_1_info["y_col"],
        chart_1_info["graph_type"],
    )
    graph_2_findings = generate_graph_insights(
        df,
        chart_2_info["x_col"],
        chart_2_info["y_col"],
        chart_2_info["graph_type"],
    )
    story.append(Paragraph("Graph 1 Findings", styles["Heading3"]))
    for line in graph_1_findings.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Graph 2 Findings", styles["Heading3"]))
    for line in graph_2_findings.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. AI Comparison Insights", styles["Heading2"]))
    for line in compare_insights.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))
    story.append(Spacer(1, 12))


    story.append(Paragraph("7. Conclusion", styles["Heading2"]))
    story.append(Paragraph("This report compares two selected charts from the same dataset and summarizes key patterns, differences, and insights.", styles["Normal"]))

    doc.build(story)
    return pdf_path




# ================= RIGHT SIDE CIRCLE FLOATING CHATBOT =================
def make_chat_suggestions(df):
    """Create dataset-aware quick questions for any dataset."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    suggestions = [
        "Give me a short summary of this dataset",
        "Which columns have missing values?",
        "Are there duplicate rows?",
    ]
    if numeric_cols:
        suggestions.append(f"What is the average, minimum and maximum of {numeric_cols[0]}?")
        suggestions.append(f"Which row has the highest {numeric_cols[0]}?")
    if cat_cols:
        suggestions.append(f"What are the most common values in {cat_cols[0]}?")
    if len(numeric_cols) >= 2:
        suggestions.append(f"Is there any relationship between {numeric_cols[0]} and {numeric_cols[1]}?")
    return suggestions[:6]


def render_right_side_chatbot(df):
    """Compact chatbot placed at the extreme right at the end of the page.
    It stays small as a button and opens only when clicked.
    """

    if "dataset_chat_history" not in st.session_state:
        st.session_state.dataset_chat_history = []
    if "chatbot_question_box" not in st.session_state:
        st.session_state.chatbot_question_box = ""

    st.markdown(
        """
        <style>
        .mini-chat-note {
            font-size: 12px;
            opacity: .75;
            margin-bottom: 8px;
        }
        .mini-user-bubble {
            background: #dbeafe;
            color: #0f172a;
            padding: 8px 10px;
            border-radius: 13px 13px 4px 13px;
            margin: 6px 0 6px 26px;
            font-size: 13px;
        }
        .mini-bot-bubble {
            background: #f8fafc;
            color: #111827;
            padding: 8px 10px;
            border-radius: 13px 13px 13px 4px;
            margin: 6px 26px 6px 0;
            font-size: 13px;
            border: 1px solid rgba(0,0,0,.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Put the tiny chatbot button at the far right.
    left_space, chat_button_col = st.columns([8.8, 1.2])
    with chat_button_col:
        with st.popover("💬 Chat", use_container_width=True):
            st.markdown("### 🤖 AI Chatbot")
            st.markdown("<div class='mini-chat-note'>Ask about current cleaned + filtered data.</div>", unsafe_allow_html=True)

            with st.expander("💡 Smart Questions", expanded=False):
                for i, q in enumerate(make_chat_suggestions(df)):
                    if st.button(q, key=f"smart_question_{i}", use_container_width=True):
                        st.session_state.chatbot_question_box = q
                        st.rerun()

            if st.session_state.dataset_chat_history:
                for role, msg in st.session_state.dataset_chat_history[-6:]:
                    safe_msg = str(msg).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    if role == "user":
                        st.markdown(f"<div class='mini-user-bubble'><b>You</b><br>{safe_msg}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='mini-bot-bubble'><b>AI</b><br>{safe_msg}</div>", unsafe_allow_html=True)
            else:
                st.info("Ask a question or choose a smart question.")

            user_question = st.text_area(
                "Ask your question",
                value=st.session_state.chatbot_question_box,
                placeholder="Example: Give summary of dataset",
                height=80,
                key="chatbot_text_area",
            )

            send_col, clear_col = st.columns([2, 1])
            with send_col:
                send_clicked = st.button("Ask", key="ask_right_chatbot", use_container_width=True)
            with clear_col:
                clear_clicked = st.button("Clear", key="clear_right_chatbot", use_container_width=True)

            if clear_clicked:
                st.session_state.dataset_chat_history = []
                st.session_state.chatbot_question_box = ""
                st.rerun()

            if send_clicked:
                if not user_question.strip():
                    st.warning("Please type a question first.")
                else:
                    with st.spinner("Analyzing..."):
                        try:
                            answer = chat_with_data(df, user_question.strip())
                        except Exception as e:
                            answer = build_local_answer(df, user_question.strip()) + f"\n\nAI failed, local answer shown. Error: {e}"

                    st.session_state.dataset_chat_history.append(("user", user_question.strip()))
                    st.session_state.dataset_chat_history.append(("bot", answer))
                    st.session_state.chatbot_question_box = ""
                    st.rerun()


# ================= MAIN APP =================
if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to start.")
    st.stop()

# df is defined here, so NameError will not happen.
df = load_data(uploaded_file)

st.success(f"✅ Uploaded successfully: {uploaded_file.name}")
st.markdown("---")

st.header("📌 Dataset Overview")

col1, col2 = st.columns(2)
col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])

render_ai_column_explanation(df)

st.markdown("---")

st.header("🧹 Data Cleaning")
cleaning_on = st.toggle("Enable Cleaning", True)

if cleaning_on:
    st.subheader("⚙️ Cleaning Controls")
    c1, c2, c3 = st.columns(3)

    with c1:
        remove_duplicates = st.checkbox("Remove Duplicates", True)
        handle_missing = st.checkbox("Handle Missing Values", True)
    with c2:
        clean_text = st.checkbox("Clean Text Values", True)
        convert_numeric = st.checkbox("Convert Numeric Text", True)
    with c3:
        remove_constant = st.checkbox("Remove Constant Columns", True)
        remove_outliers = st.checkbox("Remove Outliers", False)

    df, report = clean_data(
        df,
        remove_duplicates=remove_duplicates,
        handle_missing=handle_missing,
        clean_text=clean_text,
        convert_numeric=convert_numeric,
        remove_constant=remove_constant,
        remove_outliers=remove_outliers,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Before", report["before_rows"])
    c2.metric("Rows After", report["after_rows"])
    c3.metric("Missing Before", report["before_missing"])
    c4.metric("Missing After", report["after_missing"])

    with st.expander("🔍 View Cleaning Details"):
        st.write(f"Rows: {report['before_rows']} → {report['after_rows']}")
        st.write(f"Columns: {report['before_cols']} → {report['after_cols']}")
        st.write(f"Missing Values: {report['before_missing']} → {report['after_missing']}")
        st.write(f"Duplicate Rows: {report['before_duplicates']} → {report['after_duplicates']}")
        st.write(f"Removed Constant Columns: {report['removed_columns']}")
        st.write(f"Outliers Removed: {report['outliers_removed']}")

    st.subheader("🤖 AI Cleaning Suggestions")
    if st.button("Get AI Cleaning Suggestions"):
        cleaning_prompt = f"""
You are a professional data cleaning expert.

Dataset information:
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Column names: {df.columns.tolist()}
Missing values per column:
{df.isnull().sum().to_dict()}

Data types:
{df.dtypes.astype(str).to_dict()}

Give practical cleaning suggestions for this dataset.
Include missing value handling, duplicate handling, data type conversion, outlier handling, and columns that may need attention.
Keep it clear and concise. Do not write code.
"""
        suggestions = ask_agent(cleaning_prompt)
        st.info(suggestions)
else:
    st.warning("Cleaning is OFF. Raw dataset is being used.")

st.markdown("---")

st.header("🔎 Filter Data")
filter_col = st.selectbox("Select column to filter", df.columns)

if pd.api.types.is_numeric_dtype(df[filter_col]):
    min_val = df[filter_col].min()
    max_val = df[filter_col].max()
    if not pd.isna(min_val) and not pd.isna(max_val) and min_val != max_val:
        selected_range = st.slider("Select range", float(min_val), float(max_val), (float(min_val), float(max_val)))
        df = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]
else:
    unique_values = df[filter_col].dropna().astype(str).unique().tolist()
    if len(unique_values) > 100:
        unique_values = unique_values[:100]
        st.info("Showing first 100 unique values for filtering.")
    values = st.multiselect("Select values", unique_values)
    if values:
        df = df[df[filter_col].astype(str).isin(values)]

if df.empty:
    st.error("No data left after filtering.")
    st.stop()

st.markdown("---")

st.header("📄 Dataset Details")
tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Info", "Describe", "Correlation"])


with tab1:
    st.dataframe(df.head().astype(str))
with tab2:
    st.text(get_info(df))
with tab3:
    st.dataframe(df.describe(include="all").astype(str))
with tab4:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        st.dataframe(numeric_df.corr())
    else:
        st.info("Not enough numeric columns for correlation matrix.")

st.markdown("---")
render_auto_dashboard(df)

# with tab1:
#     st.dataframe(df.head())
# with tab2:
#     st.text(get_info(df))
# with tab3:
#     st.dataframe(df.describe(include="all"))
# with tab4:
#     numeric_df = df.select_dtypes(include="number")
#     if numeric_df.shape[1] >= 2:
#         st.dataframe(numeric_df.corr())
#     else:
#         st.info("Not enough numeric columns for correlation matrix.")

st.download_button("⬇️ Download Cleaned Data", df.to_csv(index=False), "cleaned_data.csv", mime="text/csv")

st.markdown("---")

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found for graph.")
    st.stop()

st.header("📈 Visualization")
graph_options = ["Bar Chart", "Line Chart", "Histogram", "Scatter Plot", "Pie Chart", "Box Plot", "Count Plot"]
chart_mode = st.radio(
    "Chart Mode",
    ["Single Chart", "📈 Multi-chart Compare Mode (2 graphs side-by-side)"],
    horizontal=True
)


def choose_chart_columns(prefix, graph_type):
    x_value = None
    y_value = None

    if graph_type in ["Histogram", "Box Plot"]:
        y_value = st.selectbox(f"{prefix} Y Column (Numeric)", numeric_cols, key=f"{prefix}_y")
    elif graph_type == "Count Plot":
        x_value = st.selectbox(f"{prefix} X Column", df.columns, key=f"{prefix}_x")
    else:
        c_left, c_right = st.columns(2)
        with c_left:
            x_value = st.selectbox(f"{prefix} X Column", df.columns, key=f"{prefix}_x")
        with c_right:
            y_value = st.selectbox(f"{prefix} Y Column", numeric_cols, key=f"{prefix}_y")
        suggested = smart_chart_suggestion(df, x_value, y_value)
        st.info(f"💡 {prefix} Recommended Chart: {suggested}")

    return x_value, y_value

if chart_mode == "Single Chart":
    graph_type = st.selectbox("Graph Type", graph_options)
    x_col, y_col = choose_chart_columns("Single Chart", graph_type)
else:
    left_box, right_box = st.columns(2)

    with left_box:
        st.subheader("Graph 1")
        graph_type_1 = st.selectbox("Graph 1 Type", graph_options, key="graph_type_1")
        x_col_1, y_col_1 = choose_chart_columns("Graph 1", graph_type_1)

    with right_box:
        st.subheader("Graph 2")
        graph_type_2 = st.selectbox("Graph 2 Type", graph_options, key="graph_type_2")
        x_col_2, y_col_2 = choose_chart_columns("Graph 2", graph_type_2)

st.markdown("---")

st.header("🤖 AI Analysis")
predefined_queries = [
    "Show trend of data",
    "Compare values",
    "Show distribution",
    "Find highest values",
    "Find lowest values",
    "Give summary insights",
    "Explain the chart",
    "Find pattern in data",
    "Check relationship between selected columns",
]

selected_query = st.selectbox("Choose quick analysis", ["Custom"] + predefined_queries)
if selected_query != "Custom":
    user_query = st.text_input("Your Query", value=selected_query, disabled=True)
else:
    user_query = st.text_input("Ask your analysis question", value="show insights")

if st.button("🚀 Analyze Dataset"):
    if chart_mode == "Single Chart":
        charts = create_chart(df, graph_type, x_col, y_col)
        if charts:
            chart_path = charts[0]
            st.subheader("📊 Generated Chart")
            st.image(chart_path)
            download_file_button(
                "⬇️ Download Chart PNG",
                chart_path,
                "single_chart.png",
                "image/png",
            )

            st.subheader("💻 Generated Code Used")
            st.code(generate_chart_code(graph_type, x_col, y_col), language="python")

            st.subheader("🧠 AI Insights")
            insight_prompt = f"""
You are an expert data analyst.

Analyze the selected dataset and graph carefully.

Dataset Information:
- Total rows: {df.shape[0]}
- Total columns: {df.shape[1]}
- Selected graph type: {graph_type}
- X column: {x_col}
- Y column: {y_col}
- User query: {user_query}

Your task:
1. Explain what the selected graph shows.
2. Identify the highest and lowest values if applicable.
3. Explain any visible trend, comparison, pattern, distribution, or relationship.
4. Mention possible reasons behind the pattern if logical.
5. Mention any data quality concerns if visible.
6. Give a short conclusion.

Rules:
- Keep the answer clear and professional.
- Do not write code.
- Do not give generic answers.
- Base the answer only on selected columns and chart type.
- Use simple language.
- Format response with headings and bullet points.
"""
            insights = ask_agent(insight_prompt)
            st.success(insights)

            pdf_path = generate_pdf_report(df, x_col, y_col, graph_type, insights, chart_path)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("📄 Download PDF Report", pdf_file, file_name="analysis_report.pdf", mime="application/pdf")

    else:
        clear_outputs()
        chart_1 = create_chart(df, graph_type_1, x_col_1, y_col_1, path="outputs/chart_1.png", clear=False)
        chart_2 = create_chart(df, graph_type_2, x_col_2, y_col_2, path="outputs/chart_2.png", clear=False)

        if chart_1 and chart_2:
            st.subheader("📈 Multi-chart Comparison")
            show_left, show_right = st.columns(2)
            with show_left:
                st.markdown("### Graph 1")
                st.image(chart_1[0])
                st.caption(f"{graph_type_1} | X: {x_col_1} | Y: {y_col_1}")
            with show_right:
                st.markdown("### Graph 2")
                st.image(chart_2[0])
                st.caption(f"{graph_type_2} | X: {x_col_2} | Y: {y_col_2}")

            st.subheader("🧠 AI Comparison Insights")
            compare_prompt = f"""
You are an expert data analyst.

Compare these two charts from the same dataset.

Dataset Information:
- Total rows: {df.shape[0]}
- Total columns: {df.shape[1]}
- User query: {user_query}

Graph 1:
- Type: {graph_type_1}
- X column: {x_col_1}
- Y column: {y_col_1}

Graph 2:
- Type: {graph_type_2}
- X column: {x_col_2}
- Y column: {y_col_2}

Explain:
1. What Graph 1 shows.
2. What Graph 2 shows.
3. Similarities and differences between both graphs.
4. Important trend, pattern, highest/lowest values if applicable.
5. Short conclusion in simple language.

Do not write code. Use headings and bullet points.
"""
            compare_insights = ask_agent(compare_prompt)
            st.success(compare_insights)

            st.subheader("⬇️ Download Charts")
            d1, d2, d3 = st.columns(3)
            with d1:
                download_file_button("Download Graph 1 PNG", chart_1[0], "graph_1.png", "image/png")
            with d2:
                download_file_button("Download Graph 2 PNG", chart_2[0], "graph_2.png", "image/png")
            with d3:
                charts_zip = create_charts_zip([chart_1[0], chart_2[0]])
                download_file_button("Download Both Charts ZIP", charts_zip, "multi_charts.zip", "application/zip")

            chart_1_info = {"graph_type": graph_type_1, "x_col": x_col_1, "y_col": y_col_1}
            chart_2_info = {"graph_type": graph_type_2, "x_col": x_col_2, "y_col": y_col_2}
            multi_pdf_path = generate_multi_pdf_report(
                df,
                chart_1_info,
                chart_2_info,
                compare_insights,
                chart_1[0],
                chart_2[0],
            )
            download_file_button(
                "📄 Download Multi-chart PDF Report",
                multi_pdf_path,
                "multi_chart_analysis_report.pdf",
                "application/pdf",
            )


# ================= COMPACT RIGHT-END CHATBOT =================
st.markdown("---")
render_right_side_chatbot(df)
