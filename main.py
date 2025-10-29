# app.py
# Streamlit dashboard for LLM performance vs efficiency trade-offs — Enhanced UI
# Dependencies: streamlit, pandas, numpy, plotly, openpyxl
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------- Page setup & theme helpers ----------------
st.set_page_config(
    page_title="LLM Efficiency Dashboard",
    page_icon="⚡",
    layout="wide",
)

def inject_css() -> None:
    """Lightweight CSS to give the app a cleaner, more cohesive look."""
    st.markdown(
        """
        <style>
        /* Headline block */
        .hero {
            padding: 18px 22px; border-radius: 16px;
            background: linear-gradient(90deg, #0ea5e9, #6366f1);
            color: #fff; margin-bottom: 12px;
        }
        .hero h1 { margin: 0 0 6px 0; font-weight: 800; }
        .hero .sub { opacity: .95; }

        /* Soft card look for metrics */
        .stMetric { border: 1px solid rgba(0,0,0,.05); border-radius: 12px; padding: 10px; background: #fff; }

        /* Tighter legend spacing */
        .js-plotly-plot .legend { font-size: 12px !important; }

        /* Dataframe header weight */
        .stDataFrame th { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# ---------------- Constants & text ----------------
EXPECTED_COLUMNS = [
    "Prompt",
    "Model",
    "Answer quality",
    "Electricity consumption (Wh)",
    "CO2 emission (g)",
    "Inference timing (seconds)",
]

CANONICAL_MAP = {
    "prompt": "Prompt",
    "model": "Model",
    "Answer quality": "Answer quality",
    "answer_quality": "Answer quality",
    "electricity consumption (wh)": "Electricity consumption (Wh)",
    "electricity (wh)": "Electricity consumption (Wh)",
    "energy (wh)": "Electricity consumption (Wh)",
    "co2 emission (g)": "CO2 emission (g)",
    "co₂ emission (g)": "CO2 emission (g)",
    "inference timing (seconds)": "Inference timing (seconds)",
    "latency (s)": "Inference timing (seconds)",
}

NUMERIC_COLS = [
    "Answer quality",
    "Electricity consumption (Wh)",
    "CO2 emission (g)",
    "Inference timing (seconds)",
]

DERIVED_COLS = {
    "Wh per quality": "Electricity consumption (Wh) / Answer quality",
    "Quality per Wh": "Answer quality / Electricity consumption (Wh)",
    "Quality per second": "Answer quality / Inference timing (seconds)",
    "CO2 per Wh": "CO2 emission (g) / Electricity consumption (Wh)",
}

METRIC_HELP = {
    "Answer quality": "Task-specific quality score (higher is better).",
    "Electricity consumption (Wh)": "Measured energy used for the inference (lower is better).",
    "CO2 emission (g)": "Estimated carbon emissions from the run (lower is better).",
    "Inference timing (seconds)": "End-to-end latency for the response (lower is better).",
    "Wh per quality": "Energy used per quality point — efficiency penalty (lower is better).",
    "Quality per Wh": "Quality delivered per Wh — energy efficiency (higher is better).",
    "Quality per second": "Quality delivered per second — speed-normalized quality (higher is better).",
    "CO2 per Wh": "Emissions intensity — how green your energy was (lower is better).",
}

# ---------------- Small utilities ----------------
def coerce_numeric(series: pd.Series, assume_decimal_comma: bool | None) -> pd.Series:
    """Convert a column to numeric, handling comma decimal formats and stray characters."""
    if pd.api.types.is_numeric_dtype(series):
        return series

    s = series.astype(str).str.strip()
    s = (
        s.str.replace(r"[^\d,.\-eE+]", "", regex=True)
         .str.replace("\u00A0", "", regex=False)
         .str.replace(" ", "", regex=False)
    )

    if assume_decimal_comma is None:
        comma_count = s.str.contains(",", regex=False).sum()
        dot_count = s.str.contains(r"\.", regex=True).sum()
        assume_decimal_comma = comma_count > dot_count and comma_count > 0

    if assume_decimal_comma:
        s = s.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True)
        s = s.str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(r",(?!\d{1,3}\b)", "", regex=True)
        s = s.str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = str(c).strip().lower()
        mapping[c] = CANONICAL_MAP.get(key, c)
    return df.rename(columns=mapping)


def normalize_minmax(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = out[c].to_numpy(dtype=float)
        if np.all(np.isnan(x)):
            out[c] = np.nan
            continue
        minv = np.nanmin(x)
        maxv = np.nanmax(x)
        if maxv - minv == 0:
            out[c] = 0.5
        else:
            out[c] = (x - minv) / (maxv - minv)
    return out


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Compute Pareto frontier for minimizing x and maximizing y."""
    d = df[[x_col, y_col, "Model"]].dropna().sort_values(by=[x_col, y_col], ascending=[True, False])
    frontier = []
    best_y = -np.inf
    for _, row in d.iterrows():
        y = row[y_col]
        if y > best_y:
            frontier.append(row)
            best_y = y
    if not frontier:
        return d.head(0)
    return pd.DataFrame(frontier)


def styled_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    try:
        import matplotlib  # noqa: F401
        return df.style.background_gradient(cmap="Blues", axis=None, vmin=None, vmax=None, low=0, high=0)
    except Exception:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        def css_gradient(data: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            for col in data.columns:
                if col not in numeric_cols:
                    continue
                col_vals = pd.to_numeric(data[col], errors="coerce")
                minv = col_vals.min(skipna=True)
                maxv = col_vals.max(skipna=True)
                if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
                    continue
                norm = (col_vals - minv) / (maxv - minv)
                start = np.array([255, 255, 255], dtype=float)
                end = np.array([59, 130, 246], dtype=float)
                for idx in data.index:
                    t = norm.get(idx)
                    if pd.isna(t):
                        continue
                    rgb = (1 - t) * start + t * end
                    r, g, b = [int(round(v)) for v in rgb]
                    styles.loc[idx, col] = f"background-color: rgb({r},{g},{b});"
            return styles
        return df.style.apply(css_gradient, axis=None)


def to_long(df: pd.DataFrame, cols: list[str], value_name="Value"):
    return df.melt(id_vars=["Model"], value_vars=cols, var_name="Metric", value_name=value_name)


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Model", as_index=False).agg({
        "Answer quality": "mean",
        "Electricity consumption (Wh)": "mean",
        "CO2 emission (g)": "mean",
        "Inference timing (seconds)": "mean",
    })


def jitter_values(x: pd.Series, pct: float, seed: int = 42) -> np.ndarray:
    if pct <= 0 or x.empty:
        return x
    vals = x.to_numpy(dtype=float)
    span = np.nanmax(vals) - np.nanmin(vals)
    if not np.isfinite(span) or span == 0:
        span = 1.0
    rng = np.random.default_rng(seed)
    noise = (rng.random(len(vals)) - 0.5) * 2.0 * (pct / 100.0) * span
    return vals + noise


def apply_fig_theme(fig: go.Figure, template: str = "plotly_white") -> go.Figure:
    fig.update_layout(
        template=template,
        hovermode="closest",
        margin=dict(l=24, r=24, t=40, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0, title=""),
        font=dict(size=14),
    )
    return fig

# ---------------- Sidebar: structured, with tooltips ----------------
with st.sidebar:
    st.title("Controls")

    with st.expander("📁 Data import", expanded=True):
        uploaded = st.file_uploader(
            "Upload your Excel/CSV file",
            type=["xlsx", "xls", "csv"],
            help="Provide a file containing the required columns. See the 'Metric glossary' for details.",
        )
        decimal_pref = st.selectbox(
            "Decimal format",
            ["Auto-detect", "Comma as decimal (e.g., 1,23)", "Dot as decimal (e.g., 1.23)"],
            index=0,
            help="If your numbers look wrong, force the decimal separator here.",
        )
        assume_decimal_comma = None if decimal_pref == "Auto-detect" else (decimal_pref.startswith("Comma"))
        sheet = st.text_input(
            "Excel sheet name (optional)", value="", help="Leave empty to read the first sheet.")
        show_raw = st.checkbox("Show raw data preview", value=False, help="Peek at the cleaned dataframe after parsing.")
        allow_row_filter = st.checkbox(
            "Enable row filters (Prompt/Model)", value=True,
            help="Turn off if you want to see the full dataset without filters.")

    with st.expander("🎨 Display & style", expanded=True):
        normalize_for_overview = st.checkbox(
            "Normalize metrics for Grouped Bar & Radar (0–1)", value=True,
            help="Rescales each metric independently to 0–1 for visual comparison.")
        label_points = st.checkbox(
            "Show labels on scatter points", value=True,
            help="Print model names next to points when there aren't too many.")
        chart_template = st.selectbox(
            "Chart theme (Plotly template)",
            ["plotly_white", "simple_white", "ggplot2", "seaborn", "plotly_dark"],
            index=0,
            help="Affects background, gridlines, and default fonts in charts.",
        )

    with st.expander("🛠️ Plotting tweaks", expanded=False):
        aggregate_models = st.checkbox(
            "Aggregate per model (mean of prompts)", value=True,
            help="Average metrics across prompts per model before plotting.")
        jitter_pct = st.slider(
            "Jitter amount (% of axis span)", 0.0, 5.0, 0.8, 0.1,
            help="Adds a small random offset to reduce overplotting.")
        marker_opacity = st.slider(
            "Marker opacity", 0.2, 1.0, 0.7, 0.05,
            help="Lower for dense plots; higher for sparse plots.")
        label_max_points = st.slider(
            "Label at most N points", 5, 200, 30, 5,
            help="Hide text labels above this point count to prevent clutter.")
        max_lines = st.slider(
            "Max lines (Parallel Coordinates)", 20, 500, 120, 10,
            help="Sample lines above this limit to keep the view readable.")
        logx_wh = st.checkbox("Log scale for Electricity (Wh)", value=False, help="Use logarithmic X-axis when energy spans many orders of magnitude.")
        logx_time = st.checkbox("Log scale for Time (s)", value=False, help="Use logarithmic X-axis for latency.")
        logy_co2 = st.checkbox("Log scale for CO₂ (g)", value=False, help="Use logarithmic Y-axis for emissions.")

    st.markdown("---")
    st.caption("Tip: You can download a filtered, cleaned CSV at the bottom of the app.")

# ---------------- Load data ----------------
@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes, is_csv: bool, sheet_name: str | None) -> pd.DataFrame:
    if is_csv:
        text = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text = file_bytes.decode(enc)
                break
            except Exception:
                continue
        if text is None:
            text = file_bytes.decode("utf-8", errors="ignore")
        try:
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception:
            pass
        for sep in ("\t", ";", r"\s+"):
            try:
                return pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            except Exception:
                continue
        return pd.read_csv(io.StringIO(text), sep=",", engine="python")
    else:
        kwargs = {}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        return pd.read_excel(io.BytesIO(file_bytes), **kwargs)


def clean_dataframe(df: pd.DataFrame, assume_decimal_comma: bool | None) -> pd.DataFrame:
    df = canonicalize_columns(df)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    for c in NUMERIC_COLS:
        df[c] = coerce_numeric(df[c], assume_decimal_comma)
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Wh per quality"] = df["Electricity consumption (Wh)"] / df["Answer quality"]
    df["Quality per Wh"] = df["Answer quality"] / df["Electricity consumption (Wh)"]
    df["Quality per second"] = df["Answer quality"] / df["Inference timing (seconds)"]
    df["CO2 per Wh"] = df["CO2 emission (g)"] / df["Electricity consumption (Wh)"]
    return df

# ---------------- Header ----------------
if 'header_shown' not in st.session_state:
    st.session_state.header_shown = True

st.markdown(
    """
    <div class="hero">
      <h1>LLM Efficiency Dashboard</h1>
      <div class="sub">Explore model <strong>quality</strong> vs <strong>energy</strong>, <strong>emissions</strong>, and <strong>latency</strong>, with clean parsing and rich visuals.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("📘 Metric glossary & how to read the charts", expanded=False):
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Base metrics**")
        st.write("\n".join([
            f"• **{k}** — {v}" for k, v in list(METRIC_HELP.items())[:4]
        ]))
    with cols[1]:
        st.markdown("**Derived metrics**")
        st.write("\n".join([
            f"• **{k}** — {v}" for k, v in list(METRIC_HELP.items())[4:]
        ]))
    st.caption("Notes: Radar and Overview can show normalized values. 'Bad' metrics (Wh, CO₂, Time) are inverted on Radar so larger areas are better.")

# ---------------- Guard if no upload ----------------
if 'uploaded_placeholder' not in st.session_state:
    st.session_state.uploaded_placeholder = None

if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

if uploaded is None:
    st.info(
        "Upload an **Excel (.xlsx/.xls)** or **CSV** file with columns: `Prompt`, `Model`, `Answer quality`, `Electricity consumption (Wh)`, `CO2 emission (g)`, `Inference timing (seconds)`.")
    st.stop()

# Safe to use uploaded.name now
is_csv = uploaded.name.lower().endswith(".csv")
try:
    df_raw = load_dataframe(uploaded.read(), is_csv, sheet or None)
    df = clean_dataframe(df_raw.copy(), assume_decimal_comma)
except Exception as e:
    st.error(f"Could not load/clean data: {e}")
    st.stop()

# ---------------- Filters ----------------
if allow_row_filter:
    f1, f2 = st.columns([1, 1])
    with f1:
        prompts = ["(All)"] + sorted([p for p in df["Prompt"].dropna().astype(str).unique()])
        sel_prompt = st.selectbox(
            "Filter by Prompt",
            prompts,
            index=0,
            help="Limit to a single prompt.",
        )
    with f2:
        models = ["(All)"] + sorted(df["Model"].unique().tolist())
        sel_models = st.multiselect(
            "Filter by Model(s)",
            models,
            default=["(All)"],
            help="Pick one or more models to focus on. Leave '(All)' for no filter.",
        )

    mask = pd.Series([True] * len(df))
    if sel_prompt != "(All)":
        mask &= df["Prompt"].astype(str) == sel_prompt
    if sel_models and "(All)" not in sel_models:
        mask &= df["Model"].isin(sel_models)
    df = df.loc[mask].copy()

if df.empty:
    st.warning("No data after filtering.")
    st.stop()

# ---------------- Top KPIs ----------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Models", df["Model"].nunique(), help="Number of unique model names in view.")
with k2:
    st.metric("Avg quality", f"{df['Answer quality'].mean():.2f}", help="Mean of 'Answer quality' over filtered rows.")
with k3:
    if df["Answer quality"].notna().any():
        best_q = df.loc[df["Answer quality"].idxmax()]
        st.metric("Top quality model", f"{best_q['Model']} ({best_q['Answer quality']:.2f})", help="Model with maximum observed quality.")
    else:
        st.metric("Top quality model", "n/a")
with k4:
    tmp = df.copy()
    tmp["Quality per Wh"] = tmp["Quality per Wh"].replace([np.inf, -np.inf], np.nan)
    if tmp["Quality per Wh"].notna().any():
        m = tmp["Quality per Wh"].idxmax()
        st.metric("Best energy efficiency", f"{tmp.loc[m,'Model']} ({tmp.loc[m,'Quality per Wh']:.2f} q/Wh)", help="Highest 'Quality per Wh'.")
    else:
        st.metric("Best energy efficiency", "n/a")

# ---------------- Raw data preview ----------------
if show_raw:
    st.subheader("Raw / Cleaned Data")
    st.caption("This is the parsed and cleaned dataframe after applying the canonical column mapping and numeric coercion.")
    st.dataframe(df, use_container_width=True, height=320)

# ---------------- Tabs ----------------
tabs = st.tabs([
    "📊 Overview (Grouped Bars)",
    "⚡ Quality vs Energy",
    "🧭 Efficiency Frontier",
    "🌿 CO₂ vs Speed",
    "🕸️ Radar (Profiles)",
    "🔌 Energy-to-Quality",
    "🧵 Parallel Coordinates",
    "📋 Summary Table",
])

# 1) Overview Grouped Bar
with tabs[0]:
    st.subheader("Model Comparison Overview (Grouped Bars)")
    st.caption("Each metric can be normalized independently to 0–1 for comparability across scales.")
    cols = [
        "Answer quality",
        "Electricity consumption (Wh)",
        "CO2 emission (g)",
        "Inference timing (seconds)",
    ]
    data_plot = df[["Model"] + cols].copy()

    if normalize_for_overview:
        data_plot_norm = normalize_minmax(data_plot, cols)
        long_df = to_long(data_plot_norm, cols)
        y_title = "Normalized value (0–1)"
    else:
        long_df = to_long(data_plot, cols)
        y_title = "Metric value"

    fig = px.bar(
        long_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        hover_data={"Metric": True, "Value": ":.3f"},
        height=520,
    )
    fig.update_layout(yaxis_title=y_title, xaxis_title="", legend_title="")
    st.plotly_chart(apply_fig_theme(fig, chart_template), use_container_width=True)

# 2) Quality vs Energy (bubble)
with tabs[1]:
    st.subheader("Quality vs Electricity Consumption")
    st.caption("Bubble size = latency (s). Bubble color = CO₂ (g). Use jitter to reduce overlap.")
    plot_df = df.copy()
    if aggregate_models:
        plot_df = aggregate_by_model(plot_df)
    plot_df = plot_df.dropna(subset=[
        "Electricity consumption (Wh)",
        "Answer quality",
        "Inference timing (seconds)",
    ])
    if plot_df.empty:
        st.info("Not enough data to plot.")
    else:
        plot_df["xj"] = jitter_values(plot_df["Electricity consumption (Wh)"], jitter_pct)
        plot_df["yj"] = jitter_values(plot_df["Answer quality"], jitter_pct * 0.25)
        fig = px.scatter(
            plot_df,
            x="xj",
            y="yj",
            size="Inference timing (seconds)",
            color="CO2 emission (g)",
            hover_name="Model",
            hover_data={
                "Electricity consumption (Wh)": ":.3f",
                "Answer quality": ":.2f",
                "CO2 emission (g)": ":.3f",
                "Inference timing (seconds)": ":.2f",
                "xj": False, "yj": False,
            },
            height=520,
            size_max=36,
        )
        if label_points and len(plot_df) <= label_max_points:
            fig.update_traces(text=plot_df["Model"], textposition="top center", mode="markers+text")
        fig.update_traces(marker=dict(opacity=marker_opacity, line=dict(width=0)))
        fig.update_layout(xaxis_title="Electricity (Wh)", yaxis_title="Answer quality", coloraxis_colorbar_title="CO₂ (g)")
        if logx_wh:
            fig.update_xaxes(type="log")
        st.plotly_chart(apply_fig_theme(fig, chart_template), use_container_width=True)

    with st.expander("Alternative view: strips per quality band"):
        tmp = plot_df.copy()
        if not tmp.empty:
            tmp["Q band"] = tmp["Answer quality"].round(1).astype(str)
            fig2 = px.strip(
                tmp,
                x="Electricity consumption (Wh)",
                y="Q band",
                color="Model",
                stripmode="overlay",
                orientation="h",
                hover_name="Model",
                height=420,
                category_orders={"Q band": sorted(tmp["Q band"].unique(), reverse=True)},
            )
            try:
                fig2.update_traces(jitter=0.35)
            except Exception:
                pass
            fig2.update_traces(marker=dict(opacity=marker_opacity, size=6))
            if logx_wh:
                fig2.update_xaxes(type="log")
            st.plotly_chart(apply_fig_theme(fig2, chart_template), use_container_width=True)

# 3) Efficiency Frontier (Pareto)
with tabs[2]:
    st.subheader("Efficiency Frontier (min Wh vs max Quality)")
    st.caption("The frontier connects models that are not dominated by any other (lower Wh and higher quality).")
    base = df.copy()
    if aggregate_models:
        base = aggregate_by_model(base)
    base = base.dropna(subset=["Electricity consumption (Wh)", "Answer quality"])
    if base.empty:
        st.info("Not enough data to plot.")
    else:
        frontier = pareto_frontier(base, "Electricity consumption (Wh)", "Answer quality")
        plot_df = base.copy()
        plot_df["xj"] = jitter_values(plot_df["Electricity consumption (Wh)"], jitter_pct)
        plot_df["yj"] = jitter_values(plot_df["Answer quality"], jitter_pct * 0.25)
        fig = px.scatter(
            plot_df,
            x="xj", y="yj",
            color="Model",
            hover_name="Model",
            height=520,
        )
        if not frontier.empty:
            frontier = frontier.sort_values("Electricity consumption (Wh)")
            fig.add_trace(
                go.Scatter(
                    x=frontier["Electricity consumption (Wh)"],
                    y=frontier["Answer quality"],
                    mode="lines+markers+text" if (label_points and len(frontier) <= label_max_points) else "lines+markers",
                    name="Pareto frontier",
                    text=frontier["Model"] if (label_points and len(frontier) <= label_max_points) else None,
                    textposition="top center",
                    line=dict(width=3),
                )
            )
        fig.update_traces(marker=dict(opacity=marker_opacity))
        fig.update_layout(xaxis_title="Electricity (Wh)", yaxis_title="Answer quality", legend_title="")
        if logx_wh:
            fig.update_xaxes(type="log")
        st.plotly_chart(apply_fig_theme(fig, chart_template), use_container_width=True)

# 4) CO2 vs Speed
with tabs[3]:
    st.subheader("CO₂ Emissions vs Inference Timing")
    st.caption("Bubble size = quality. Bubble color = energy (Wh). Consider log scales for skewed data.")
    plot_df = df.copy()
    if aggregate_models:
        plot_df = aggregate_by_model(plot_df)
    plot_df = plot_df.dropna(subset=["CO2 emission (g)", "Inference timing (seconds)", "Answer quality"])
    if plot_df.empty:
        st.info("Not enough data to plot.")
    else:
        plot_df["xj"] = jitter_values(plot_df["Inference timing (seconds)"], jitter_pct)
        plot_df["yj"] = jitter_values(plot_df["CO2 emission (g)"], jitter_pct)
        fig = px.scatter(
            plot_df,
            x="xj", y="yj",
            size="Answer quality",
            color="Electricity consumption (Wh)",
            hover_name="Model",
            hover_data={
                "Electricity consumption (Wh)": ":.3f",
                "CO2 emission (g)": ":.3f",
                "Inference timing (seconds)": ":.2f",
                "Answer quality": ":.2f",
                "xj": False, "yj": False,
            },
            height=520,
            size_max=36,
        )
        if label_points and len(plot_df) <= label_max_points:
            fig.update_traces(text=plot_df["Model"], textposition="top center", mode="markers+text")
        fig.update_traces(marker=dict(opacity=marker_opacity, line=dict(width=0)))
        fig.update_layout(xaxis_title="Inference time (s)", yaxis_title="CO₂ (g)", coloraxis_colorbar_title="Wh")
        if logx_time:
            fig.update_xaxes(type="log")
        if logy_co2:
            fig.update_yaxes(type="log")
        st.plotly_chart(apply_fig_theme(fig, chart_template), use_container_width=True)

# 5) Radar Chart
with tabs[4]:
    st.subheader("Radar Profiles (normalized)")
    st.caption("Bad metrics (Wh, CO₂, Time) are inverted so that larger areas are better.")
    show_radar_legend = st.checkbox("Show legend", value=False, help="Display model names below the chart. Turn off to keep the plot clean.")
    metrics = [
        "Answer quality",
        "Electricity consumption (Wh)",
        "CO2 emission (g)",
        "Inference timing (seconds)",
    ]
    rad = df[["Model"] + metrics].copy()
    norm = normalize_minmax(rad.copy(), metrics)
    for m in ["Electricity consumption (Wh)", "CO2 emission (g)", "Inference timing (seconds)"]:
        norm[m] = 1 - norm[m]

    pick_models = st.multiselect(
        "Select models (default: all)", sorted(df["Model"].unique()),
        default=sorted(df["Model"].unique()),
        help="Limit which models appear on the radar.",
    )
    norm = norm[norm["Model"].isin(pick_models)].dropna(subset=metrics)
    if norm.empty:
        st.info("Select models with complete data.")
    else:
        categories = ["Quality ↑", "Electricity ↓", "CO₂ ↓", "Time ↓"]
        fig = go.Figure()
        for _, row in norm.iterrows():
            vals = [row["Answer quality"], row["Electricity consumption (Wh)"], row["CO2 emission (g)"], row["Inference timing (seconds)"]]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Model"],
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,1], showticklabels=True, ticks="")), showlegend=show_radar_legend, height=520)
        st.plotly_chart(apply_fig_theme(fig, chart_template), use_container_width=True)

# 6) Energy-to-Quality ratio
with tabs[5]:
    st.subheader("Energy-to-Quality (lower is better)")
    st.caption("Energy spent per unit of quality.")
    plot_df = df[["Model", "Wh per quality"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Wh per quality"])
    if plot_df.empty:
        st.info("Not enough data to plot.")
    else:
        plot_df = plot_df.sort_values("Wh per quality", ascending=True)
        fig = px.bar(
            plot_df,
            x="Model",
            y="Wh per quality",
            hover_data={"Wh per quality": ":.4f"},
            height=520,
        )
        fig.update_layout(xaxis_title="", yaxis_title="Wh per quality point")
        st.plotly_chart(apply_fig_theme(fig, chart_template), use_container_width=True)

# 7) Parallel Coordinates
with tabs[6]:
    st.subheader("Parallel Coordinates (normalized)")
    st.caption("Lines are colored by quality. Use the slider in the sidebar to cap the number of lines.")
    metrics = ["Answer quality", "Electricity consumption (Wh)", "CO2 emission (g)", "Inference timing (seconds)"]
    plot_df = df[["Model"] + metrics].copy().dropna(subset=metrics)
    if aggregate_models:
        plot_df = aggregate_by_model(plot_df)
    if plot_df.empty:
        st.info("Not enough complete rows to plot.")
    else:
        if len(plot_df) > max_lines:
            plot_df = plot_df.sample(max_lines, random_state=42)
        norm = normalize_minmax(plot_df, metrics)
        fig = px.parallel_coordinates(
            norm,
            dimensions=metrics,
            color="Answer quality",
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={"Answer quality": "Quality", "Electricity consumption (Wh)": "Wh", "CO2 emission (g)": "CO₂ (g)", "Inference timing (seconds)": "Time (s)"},
            height=520,
        )
        fig.update_layout(font=dict(size=12), coloraxis_colorbar=dict(title="Quality"))
        st.plotly_chart(apply_fig_theme(fig, chart_template), use_container_width=True)

# 8) Summary Table (heatmap-like)
with tabs[7]:
    st.subheader("Benchmark Summary (heatmap)")
    st.caption("Includes base and derived metrics. Cells are shaded by column-wise magnitude.")
    table_cols = [
        "Model",
        "Answer quality",
        "Electricity consumption (Wh)",
        "CO2 emission (g)",
        "Inference timing (seconds)",
        "Wh per quality",
        "Quality per Wh",
        "Quality per second",
        "CO2 per Wh",
    ]
    table = df[table_cols].copy()
    st.dataframe(styled_dataframe(table), use_container_width=True, height=420)

# ---------------- Download ----------------
st.divider()
clean_csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered, cleaned dataset (CSV)",
    clean_csv,
    file_name="llm_efficiency_clean.csv",
    mime="text/csv",
    help="Exports exactly what is currently in view (after filters/cleaning).",
)
st.caption("All charts update live based on the filters and options in the sidebar.")
