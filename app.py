import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
from agent import ask_agent

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("📊 AI Data Analyst Agent")
st.caption("Clean, filter, visualize, and understand any CSV/Excel dataset using AI.")

uploaded_file = st.file_uploader("📁 Upload CSV or Excel File", type=["csv", "xlsx"])


def clear_outputs():
    os.makedirs("outputs", exist_ok=True)
    for file in os.listdir("outputs"):
        path = os.path.join("outputs", file)
        if os.path.isfile(path):
            os.remove(path)


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


def clean_data(df):
    report = {}

    report["before_rows"] = df.shape[0]
    report["before_cols"] = df.shape[1]
    report["before_missing"] = int(df.isnull().sum().sum())
    report["before_duplicates"] = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            converted = pd.to_numeric(
                df[col].str.replace(",", "", regex=False),
                errors="coerce"
            )
            if converted.notna().mean() > 0.7:
                df[col] = converted

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0
            df[col] = df[col].fillna(median_value)
        else:
            df[col] = df[col].fillna("Unknown")

    report["after_rows"] = df.shape[0]
    report["after_cols"] = df.shape[1]
    report["after_missing"] = int(df.isnull().sum().sum())
    report["after_duplicates"] = int(df.duplicated().sum())

    return df, report


def get_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def save_chart(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def smart_chart_suggestion(df, x_col, y_col):
    if x_col and y_col:
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            return "Scatter Plot"
        if not pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            return "Bar Chart"
    return "Histogram"


def create_chart(df, graph_type, x_col=None, y_col=None):
    clear_outputs()
    path = "outputs/chart.png"

    plt.figure(figsize=(15, 8))

    try:
        if graph_type == "Histogram":
            data = df[y_col].dropna()
            if data.empty:
                st.error("No valid numeric data for histogram.")
                return []
            plt.hist(data, bins=25)
            plt.title(f"Distribution of {y_col}")
            plt.xlabel(y_col)
            plt.ylabel("Frequency")

        elif graph_type == "Box Plot":
            data = df[y_col].dropna()
            if data.empty:
                st.error("No valid numeric data for box plot.")
                return []
            plt.boxplot(data)
            plt.title(f"Box Plot of {y_col}")
            plt.ylabel(y_col)

        elif graph_type == "Count Plot":
            data = df[x_col].dropna().astype(str)
            if data.empty:
                st.error("No valid data for count plot.")
                return []
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
                plt.pie(
                    grouped.values,
                    labels=grouped.index.astype(str),
                    autopct="%1.1f%%",
                    startangle=90
                )
                plt.title(f"Top 10 {y_col} Share by {x_col}")

        save_chart(path)
        return [path]

    except Exception as e:
        st.error(f"Chart error: {e}")
        return []


def generate_chart_code(graph_type, x_col=None, y_col=None):
    return f"""
import pandas as pd
import matplotlib.pyplot as plt

graph_type = "{graph_type}"
x_col = "{x_col}"
y_col = "{y_col}"

plt.figure(figsize=(15, 8))

# This code dynamically changes based on graph type.
# Histogram / Box Plot need only Y column.
# Count Plot needs only X column.
# Bar / Line / Scatter / Pie need X and Y columns.

plt.tight_layout()
plt.savefig("outputs/chart.png", dpi=180, bbox_inches="tight")
plt.close()
"""


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
This chart shows the distribution/spread of {y_col}.
"""

        if graph_type == "Count Plot":
            counts = df[x_col].dropna().astype(str).value_counts()
            return f"""
Column analyzed: {x_col}
Most frequent value: {counts.idxmax()} with {counts.max()} records
Least frequent value: {counts.idxmin()} with {counts.min()} records
This chart shows category frequency distribution.
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
    if os.path.exists(chart_path):
        story.append(Image(chart_path, width=450, height=260))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Graph-Based Findings", styles["Heading2"]))
    graph_findings = generate_graph_insights(df, x_col, y_col, graph_type)
    for line in graph_findings.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))

    story.append(Spacer(1, 12))

    story.append(Paragraph("5. AI Insights", styles["Heading2"]))
    for line in insights.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))

    story.append(Spacer(1, 12))

    story.append(Paragraph("6. Conclusion", styles["Heading2"]))
    story.append(Paragraph(
        "This report summarizes the selected dataset, generated graph, graph-based findings, and AI insights.",
        styles["Normal"]
    ))

    doc.build(story)
    return pdf_path


if uploaded_file:
    df = load_data(uploaded_file)

    st.success(f"✅ Uploaded successfully: {uploaded_file.name}")
    st.markdown("---")

    st.header("📌 Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])

    st.markdown("---")

    st.header("🧹 Data Cleaning")
    cleaning_on = st.toggle("Enable Cleaning", True)

    if cleaning_on:
        df, report = clean_data(df)

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
    else:
        st.warning("Cleaning is OFF. Raw dataset is being used.")

    st.markdown("---")

    st.header("🔎 Filter Data")
    filter_col = st.selectbox("Select column to filter", df.columns)

    if pd.api.types.is_numeric_dtype(df[filter_col]):
        min_val = df[filter_col].min()
        max_val = df[filter_col].max()

        if not pd.isna(min_val) and not pd.isna(max_val) and min_val != max_val:
            selected_range = st.slider(
                "Select range",
                float(min_val),
                float(max_val),
                (float(min_val), float(max_val))
            )
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
        st.dataframe(df.head())

    with tab2:
        st.text(get_info(df))

    with tab3:
        st.dataframe(df.describe(include="all"))

    with tab4:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] >= 2:
            st.dataframe(numeric_df.corr())
        else:
            st.info("Not enough numeric columns for correlation matrix.")

    st.download_button(
        "⬇️ Download Cleaned Data",
        df.to_csv(index=False),
        "cleaned_data.csv",
        mime="text/csv"
    )

    st.markdown("---")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if numeric_cols:
        st.header("📈 Visualization")

        graph_options = [
            "Bar Chart",
            "Line Chart",
            "Histogram",
            "Scatter Plot",
            "Pie Chart",
            "Box Plot",
            "Count Plot"
        ]

        graph_type = st.selectbox("Graph Type", graph_options)

        x_col = None
        y_col = None

        if graph_type in ["Histogram", "Box Plot"]:
            y_col = st.selectbox("Y Column (Numeric)", numeric_cols)

        elif graph_type == "Count Plot":
            x_col = st.selectbox("X Column", df.columns)

        else:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X Column", df.columns)
            with col2:
                y_col = st.selectbox("Y Column", numeric_cols)

            suggested_chart = smart_chart_suggestion(df, x_col, y_col)
            st.info(f"💡 Recommended Chart: {suggested_chart}")

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
            "Check relationship between selected columns"
        ]

        selected_query = st.selectbox("Choose quick analysis", ["Custom"] + predefined_queries)

        if selected_query != "Custom":
            user_query = st.text_input("Your Query", value=selected_query, disabled=True)
        else:
            user_query = st.text_input("Ask your analysis question", value="show insights")

        if st.button("🚀 Analyze Dataset"):
            charts = create_chart(df, graph_type, x_col, y_col)

            if charts:
                chart_path = charts[0]

                st.subheader("📊 Generated Chart")
                st.image(chart_path)

                st.subheader("💻 Generated Code Used")
                st.code(generate_chart_code(graph_type, x_col, y_col), language="python")

                st.subheader("🧠 AI Insights")

                insight_prompt = f"""
You are a data analyst.

Dataset columns: {df.columns.tolist()}
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Graph type: {graph_type}
X column: {x_col}
Y column: {y_col}
User query: {user_query}

Give short useful insights based on selected graph and columns.
Do not write code.
"""

                insights = ask_agent(insight_prompt)
                st.success(insights)

                pdf_path = generate_pdf_report(
                    df,
                    x_col,
                    y_col,
                    graph_type,
                    insights,
                    chart_path
                )

                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        "📄 Download PDF Report",
                        pdf_file,
                        file_name="analysis_report.pdf",
                        mime="application/pdf"
                    )

    else:
        st.error("No numeric columns found for graph.")