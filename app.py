import pandas as pd
import plotly.express as px
import streamlit as st

from anomalywatch.detector import detect_anomalies

from pathlib import Path

st.set_page_config(
    page_title="AnomalyWatch",
    page_icon="📊",
    layout="wide",
)

st.title("AnomalyWatch")
st.caption("Interactive anomaly detection using Isolation Forest")
st.caption(
    "Compare Isolation Forest, Local Outlier Factor, "
    "and One-Class SVM on numerical datasets."
)

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to detect anomalous observations. "
        "You can start with data/sample_sensors.csv."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

numeric_columns = df.select_dtypes(include="number").columns.tolist()
numeric_columns = [
    c for c in numeric_columns if c not in ("anomaly_score", "is_anomaly")
]

if not numeric_columns:
    st.error("No numerical features found in the dataset.")
    st.stop()

st.sidebar.header("Detection settings")

features = st.sidebar.multiselect(
    "Select features",
    options=numeric_columns,
    default=numeric_columns[: min(3, len(numeric_columns))],
)

algorithm = st.sidebar.selectbox(
    "Detection algorithm",
    [
        "Isolation Forest",
        "Local Outlier Factor",
        "One-Class SVM",
    ],
)

contamination = st.sidebar.slider(
    "Expected anomaly proportion",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01,
)

x_axis = st.sidebar.selectbox(
    "Plot horizontal axis",
    options=df.columns.tolist(),
    index=0,
)

y_axis = st.sidebar.selectbox(
    "Plot vertical axis",
    options=numeric_columns,
    index=0,
)

if not features:
    st.warning("Select at least one feature.")
    st.stop()

try:
    result = detect_anomalies(
        df,
        features=features,
        contamination=contamination,
        algorithm=algorithm,
    )

except ValueError as exc:
    st.error(str(exc))
    st.stop()

n_anomalies = int(result["is_anomaly"].sum())
anomaly_rate = 100 * n_anomalies / len(result)

col1, col2, col3 = st.columns(3)

col1.metric("Observations", f"{len(result):,}")
col2.metric("Detected anomalies", f"{n_anomalies:,}")
col3.metric("Anomaly rate", f"{anomaly_rate:.1f}%")

st.subheader("Anomaly visualization")

result["Detection"] = result["is_anomaly"].map({True: "Anomaly", False: "Normal"})

fig = px.scatter(
    result,
    x=x_axis,
    y=y_axis,
    color="Detection",
    color_discrete_map={
        "Normal": "#2563eb",
        "Anomaly": "#dc2626",
    },
    hover_data=["anomaly_score"],
    title=f"{y_axis} — Detected anomalies",
)

fig.update_traces(marker={"size": 7})

st.plotly_chart(fig, width="stretch")

st.subheader("Most anomalous observations")

st.dataframe(
    result.sort_values("anomaly_score", ascending=False).head(20),
    width="stretch",
)

st.subheader("Dataset with predictions")

st.dataframe(result, width="stretch")

st.download_button(
    label="Download results as CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="anomalywatch_results.csv",
    mime="text/csv",
)

benchmark_file = Path(
    "results/benchmarks/benchmark_results.csv"
)

with st.expander("View algorithm benchmark results"):
    if benchmark_file.exists():
        benchmark_results = pd.read_csv(benchmark_file)

        st.dataframe(
            benchmark_results,
            width="stretch",
        )

        figure = px.bar(
            benchmark_results,
            x="dataset",
            y="f1_score",
            color="algorithm",
            barmode="group",
            title="Algorithm performance by anomaly type",
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )
    else:
        st.info(
            "Run python -m scripts.benchmark_suite "
            "to generate benchmark results."
        )
