from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import load_test_data
from src.inference.predict import get_latest_engine_predictions, predict_rul
from src.inference.risk_classification import classify_maintenance_risk
from src.utils.config import config


st.set_page_config(
    page_title="RUL Mission Control",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "bg": "#070B14",
    "panel": "#0D1424",
    "panel_2": "#111A2D",
    "border": "#1D2A43",
    "text": "#F5F7FB",
    "muted": "#8C9AB2",
    "cyan": "#54F0FF",
    "purple": "#A78BFA",
    "green": "#34D399",
    "amber": "#FBBF24",
    "red": "#FB7185",
}


@st.cache_data(show_spinner=False)
def load_prediction_data() -> pd.DataFrame:
    return predict_rul(load_test_data())


@st.cache_data(show_spinner=False)
def load_test_metrics() -> pd.DataFrame | None:
    path = Path(config["paths"]["metrics"]) / "nasa_test_evaluation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def add_status(df: pd.DataFrame) -> pd.DataFrame:
    return classify_maintenance_risk(df)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {COLORS['bg']};
                --panel: {COLORS['panel']};
                --panel-2: {COLORS['panel_2']};
                --border: {COLORS['border']};
                --text: {COLORS['text']};
                --muted: {COLORS['muted']};
                --cyan: {COLORS['cyan']};
                --purple: {COLORS['purple']};
                --green: {COLORS['green']};
                --amber: {COLORS['amber']};
                --red: {COLORS['red']};
            }}

            .stApp {{ background: var(--bg); color: var(--text); }}
            [data-testid="stHeader"] {{ background: transparent; }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #0A1020 0%, #080D18 100%);
                border-right: 1px solid var(--border);
            }}
            [data-testid="stSidebarContent"] {{ padding-top: 1.4rem; }}
            .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}

            .brand {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 1.6rem;
            }}
            .brand-mark {{
                width: 42px;
                height: 42px;
                border-radius: 12px;
                display: grid;
                place-items: center;
                color: #061018;
                font-size: 1.25rem;
                font-weight: 800;
                background: linear-gradient(135deg, var(--cyan), var(--purple));
                box-shadow: 0 0 26px rgba(84, 240, 255, 0.18);
            }}
            .brand-title {{ font-size: 1.05rem; font-weight: 700; letter-spacing: 0.02em; }}
            .brand-subtitle {{ color: var(--muted); font-size: 0.78rem; margin-top: 2px; }}

            .hero {{
                position: relative;
                overflow: hidden;
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1.45rem 1.6rem;
                margin-bottom: 1.25rem;
                background:
                    radial-gradient(circle at 85% 20%, rgba(84,240,255,0.11), transparent 25%),
                    radial-gradient(circle at 70% 100%, rgba(167,139,250,0.12), transparent 28%),
                    linear-gradient(135deg, #0B1220 0%, #0B1426 55%, #0C1020 100%);
            }}
            .hero-kicker {{
                color: var(--cyan);
                font-size: 0.74rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                margin-bottom: 0.45rem;
            }}
            .hero-title {{
                font-size: 2.15rem;
                line-height: 1.05;
                font-weight: 780;
                letter-spacing: -0.035em;
                margin: 0;
            }}
            .hero-text {{ color: var(--muted); margin-top: 0.55rem; font-size: 0.95rem; }}
            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: 7px;
                padding: 6px 10px;
                border: 1px solid rgba(52,211,153,0.25);
                border-radius: 999px;
                background: rgba(52,211,153,0.08);
                color: var(--green);
                font-size: 0.76rem;
                margin-top: 0.9rem;
            }}
            .dot {{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; box-shadow: 0 0 10px currentColor; }}

            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin: 0.9rem 0 1.25rem;
            }}
            .kpi {{
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1rem 1.05rem;
                background: linear-gradient(145deg, rgba(17,26,45,0.98), rgba(10,16,29,0.98));
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
            }}
            .kpi-label {{ color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; }}
            .kpi-value {{ margin-top: 0.35rem; font-size: 1.55rem; font-weight: 750; letter-spacing: -0.02em; }}
            .kpi-note {{ color: var(--muted); font-size: 0.75rem; margin-top: 0.3rem; }}
            .cyan {{ color: var(--cyan); }}
            .green {{ color: var(--green); }}
            .amber {{ color: var(--amber); }}
            .red {{ color: var(--red); }}
            .purple {{ color: var(--purple); }}

            .section-title {{
                font-size: 1.02rem;
                font-weight: 700;
                margin: 0.3rem 0 0.65rem;
            }}
            .section-caption {{ color: var(--muted); font-size: 0.82rem; margin-top: -0.4rem; margin-bottom: 0.8rem; }}

            .panel {{
                border: 1px solid var(--border);
                border-radius: 18px;
                background: rgba(13,20,36,0.75);
                padding: 0.9rem;
            }}

            .health-row {{
                display: flex;
                justify-content: space-between;
                gap: 8px;
                margin: 0.9rem 0;
                padding: 10px 12px;
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.04);
                border-radius: 12px;
                font-size: 0.85rem;
            }}
            .health-value {{ font-weight: 700; }}

            .engine-banner {{
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                background: linear-gradient(145deg, rgba(13,20,36,0.96), rgba(9,14,25,0.98));
                margin-bottom: 0.9rem;
            }}
            .engine-name {{ font-size: 1.25rem; font-weight: 750; }}
            .engine-meta {{ color: var(--muted); font-size: 0.78rem; margin-top: 4px; }}
            .risk-badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 999px;
                font-size: 0.74rem;
                font-weight: 700;
                margin-top: 7px;
            }}
            .risk-healthy {{ color: var(--green); background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.22); }}
            .risk-warning {{ color: var(--amber); background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.22); }}
            .risk-critical {{ color: var(--red); background: rgba(251,113,133,0.1); border: 1px solid rgba(251,113,133,0.22); }}

            div[data-testid="stMetric"] {{
                background: var(--panel);
                border: 1px solid var(--border);
                padding: 0.7rem 0.8rem;
                border-radius: 14px;
            }}
            div[data-testid="stMetricLabel"] {{ color: var(--muted); }}
            div[data-testid="stMetricValue"] {{ color: var(--text); }}

            .footnote {{ color: var(--muted); font-size: 0.72rem; margin-top: 0.8rem; }}
            @media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=24, b=10),
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
        hoverlabel=dict(bgcolor=COLORS["panel_2"], font_color=COLORS["text"]),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(140,154,178,0.08)",
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(140,154,178,0.08)",
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"]),
    )
    return fig


def metric_cards(total: int, average_rul: float, critical: int, warning: int) -> None:
    cards = [
        ("Engines monitored", f"{total:,}", "All unseen test engines", "cyan"),
        ("Average predicted RUL", f"{average_rul:.1f}", "cycles", "purple"),
        ("Critical engines", f"{critical:,}", "RUL ≤ 30 cycles", "red"),
        ("Warning engines", f"{warning:,}", "31–60 cycles", "amber"),
    ]
    html = '<div class="kpi-grid">'
    for label, value, note, color in cards:
        html += (
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {color}">{value}</div>'
            f'<div class="kpi-note">{note}</div></div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def show_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">Predictive maintenance · mission control</div>
            <h1 class="hero-title">{title}</h1>
            <div class="hero-text">{subtitle}</div>
            <div class="status-pill"><span class="dot"></span>Inference pipeline online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dashboard(engine_summary: pd.DataFrame) -> None:
    show_header(
        "Know which engine needs attention next.",
        "A visual RUL command center built around the final XGBoost inference pipeline.",
    )

    total = len(engine_summary)
    average_rul = engine_summary["Predicted_RUL"].mean()
    critical = int((engine_summary["Maintenance_Status"] == "Critical").sum())
    warning = int((engine_summary["Maintenance_Status"] == "Warning").sum())
    healthy = total - critical - warning

    metric_cards(total, average_rul, critical, warning)

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown('<div class="section-title">Fleet health</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Latest prediction for each engine</div>', unsafe_allow_html=True)

        fig = go.Figure(
            go.Pie(
                labels=["Healthy", "Warning", "Critical"],
                values=[healthy, warning, critical],
                hole=0.68,
                textinfo="none",
                marker=dict(colors=[COLORS["green"], COLORS["amber"], COLORS["red"]], line=dict(color=COLORS["bg"], width=3)),
                hovertemplate="%{label}: %{value} engines<extra></extra>",
            )
        )
        fig.add_annotation(
            text=f"<b>{total}</b><br><span style='color:{COLORS['muted']};font-size:11px'>engines</span>",
            showarrow=False,
            font=dict(size=21, color=COLORS["text"]),
        )
        plotly_layout(fig, 280)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        for label, value, color in [
            ("Healthy", healthy, COLORS["green"]),
            ("Warning", warning, COLORS["amber"]),
            ("Critical", critical, COLORS["red"]),
        ]:
            st.markdown(
                f'<div class="health-row"><span>{label}</span><span class="health-value" style="color:{color}">{value}</span></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-title">RUL distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Where the current fleet sits on the remaining-life spectrum</div>', unsafe_allow_html=True)

        rul = engine_summary["Predicted_RUL"]
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=rul,
                nbinsx=28,
                marker=dict(color=COLORS["cyan"], line=dict(color=COLORS["bg"], width=1)),
                opacity=0.9,
                hovertemplate="Predicted RUL: %{x:.1f}<br>Engines: %{y}<extra></extra>",
            )
        )
        fig.add_vrect(x0=0, x1=30, fillcolor=COLORS["red"], opacity=0.07, line_width=0)
        fig.add_vrect(x0=30, x1=60, fillcolor=COLORS["amber"], opacity=0.06, line_width=0)
        fig.add_vline(x=30, line_dash="dot", line_color=COLORS["red"], opacity=0.7)
        fig.add_vline(x=60, line_dash="dot", line_color=COLORS["amber"], opacity=0.7)
        fig.update_layout(bargap=0.08, xaxis_title="Predicted RUL (cycles)", yaxis_title="Engines")
        plotly_layout(fig, 360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title">Engines requiring attention</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Lowest predicted RUL first</div>', unsafe_allow_html=True)
    attention = engine_summary.nsmallest(10, "Predicted_RUL").copy()
    attention["Predicted_RUL"] = attention["Predicted_RUL"].round(1)
    attention = attention.rename(columns={
        "dataset_id": "Dataset",
        "unit_id": "Engine",
        "cycle": "Cycle",
        "Predicted_RUL": "Predicted RUL",
        "Maintenance_Status": "Status",
    })
    st.dataframe(
        attention[["Dataset", "Engine", "Cycle", "Predicted RUL", "Status"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Predicted RUL": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def show_engine_explorer(predictions: pd.DataFrame) -> None:
    show_header(
        "Inspect an engine in detail.",
        "Follow the predicted RUL trajectory and focus on the engines closest to the maintenance boundary.",
    )

    dataset_id = st.selectbox("Dataset", sorted(predictions["dataset_id"].unique()))
    dataset_data = predictions[predictions["dataset_id"] == dataset_id]
    unit_id = st.selectbox("Engine", sorted(dataset_data["unit_id"].unique()))

    engine_data = dataset_data[dataset_data["unit_id"] == unit_id].sort_values("cycle").copy()
    engine_data = add_status(engine_data)
    latest = engine_data.iloc[-1]

    risk = latest["Maintenance_Status"]
    risk_class = {"Healthy": "risk-healthy", "Warning": "risk-warning", "Critical": "risk-critical"}[risk]

    st.markdown(
        f"""
        <div class="engine-banner">
            <div class="engine-name">Engine {int(unit_id):03d} · {dataset_id}</div>
            <div class="engine-meta">Latest observed cycle: {int(latest['cycle'])} · model: Optimized XGBoost</div>
            <span class="risk-badge {risk_class}">{risk}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted RUL", f"{latest['Predicted_RUL']:.1f} cycles")
    c2.metric("Current cycle", f"{int(latest['cycle'])}")
    c3.metric("Engine status", risk)

    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">RUL trajectory</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Recent model behaviour across the engine lifetime</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["red"], opacity=0.06, line_width=0)
        fig.add_hrect(y0=30, y1=60, fillcolor=COLORS["amber"], opacity=0.05, line_width=0)
        fig.add_trace(
            go.Scatter(
                x=engine_data["cycle"],
                y=engine_data["Predicted_RUL"],
                mode="lines",
                line=dict(color=COLORS["cyan"], width=3),
                fill="tozeroy",
                fillcolor="rgba(84,240,255,0.07)",
                hovertemplate="Cycle %{x}<br>Predicted RUL: %{y:.1f}<extra></extra>",
            )
        )
        fig.add_hline(y=60, line_dash="dot", line_color=COLORS["amber"], opacity=0.65)
        fig.add_hline(y=30, line_dash="dot", line_color=COLORS["red"], opacity=0.65)
        fig.update_layout(xaxis_title="Cycle", yaxis_title="Predicted RUL")
        plotly_layout(fig, 430)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<div class="section-title">RUL signal</div>', unsafe_allow_html=True)
        gauge_color = COLORS["green"] if risk == "Healthy" else COLORS["amber"] if risk == "Warning" else COLORS["red"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(latest["Predicted_RUL"]),
                number=dict(suffix=" cycles", font=dict(size=28, color=COLORS["text"])),
                gauge=dict(
                    axis=dict(range=[0, 125], tickcolor=COLORS["muted"], tickfont=dict(color=COLORS["muted"])),
                    bar=dict(color=gauge_color, thickness=0.75),
                    bgcolor="rgba(255,255,255,0.03)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 30], color="rgba(251,113,133,0.10)"),
                        dict(range=[30, 60], color="rgba(251,191,36,0.08)"),
                        dict(range=[60, 125], color="rgba(52,211,153,0.07)"),
                    ],
                    threshold=dict(line=dict(color=COLORS["text"], width=3), thickness=0.8, value=float(latest["Predicted_RUL"])),
                ),
            )
        )
        plotly_layout(fig, 325)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="footnote">Dashboard thresholds: Critical ≤ {config["risk"]["critical_max_rul"]}, '
            f'Warning ≤ {config["risk"]["warning_max_rul"]}, Healthy above {config["risk"]["warning_max_rul"]} cycles.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Latest observations</div>', unsafe_allow_html=True)
    recent = engine_data.tail(12).copy()
    recent["Predicted_RUL"] = recent["Predicted_RUL"].round(1)
    recent = recent.rename(columns={"cycle": "Cycle", "Predicted_RUL": "Predicted RUL", "Maintenance_Status": "Status"})
    st.dataframe(recent[["Cycle", "Predicted RUL", "Status"]].sort_values("Cycle", ascending=False), use_container_width=True, hide_index=True)


def show_dataset_overview(predictions: pd.DataFrame) -> None:
    show_header(
        "Compare operating environments.",
        "A quick view of how predicted RUL is distributed across the four CMAPSS datasets.",
    )

    summary = (
        predictions.groupby("dataset_id")
        .agg(
            Engines=("unit_id", "nunique"),
            Records=("unit_id", "size"),
            Average_RUL=("Predicted_RUL", "mean"),
            Minimum_RUL=("Predicted_RUL", "min"),
            Maximum_RUL=("Predicted_RUL", "max"),
        )
        .reset_index()
    )

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=summary["dataset_id"],
                y=summary["Average_RUL"],
                marker=dict(color=COLORS["purple"], line=dict(color=COLORS["bg"], width=1)),
                text=summary["Average_RUL"].round(1),
                textposition="outside",
                hovertemplate="%{x}<br>Average RUL: %{y:.1f} cycles<extra></extra>",
            )
        )
        fig.update_layout(yaxis_title="Average predicted RUL (cycles)", xaxis_title="Dataset")
        plotly_layout(fig, 350)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        fig = go.Figure()
        for dataset in summary["dataset_id"]:
            values = predictions.loc[predictions["dataset_id"] == dataset, "Predicted_RUL"]
            fig.add_trace(
                go.Box(
                    y=values,
                    name=dataset,
                    boxmean=True,
                    line=dict(color=COLORS["cyan"]),
                    fillcolor="rgba(84,240,255,0.08)",
                    hovertemplate=f"{dataset}<br>RUL: %{{y:.1f}} cycles<extra></extra>",
                )
            )
        fig.update_layout(yaxis_title="Predicted RUL (cycles)", xaxis_title="Dataset", showlegend=False)
        plotly_layout(fig, 350)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    display = summary.rename(columns={
        "dataset_id": "Dataset",
        "Engines": "Engines",
        "Records": "Prediction records",
        "Average_RUL": "Avg RUL",
        "Minimum_RUL": "Min RUL",
        "Maximum_RUL": "Max RUL",
    }).round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)


def show_about_page() -> None:
    show_header(
        "From sensor history to maintenance decisions.",
        "A compact view of the modelling choices behind the dashboard.",
    )

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel">
                <div class="health-row"><span>01 · Data ingestion</span><span class="health-value">NASA CMAPSS</span></div>
                <div class="health-row"><span>02 · Feature engineering</span><span class="health-value">Lag + rolling features</span></div>
                <div class="health-row"><span>03 · Modelling</span><span class="health-value">Optimized XGBoost</span></div>
                <div class="health-row"><span>04 · Explainability</span><span class="health-value">Feature importance + SHAP</span></div>
                <div class="health-row"><span>05 · Evaluation</span><span class="health-value">707 unseen test engines</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        metrics = load_test_metrics()
        st.markdown('<div class="section-title">Final benchmark</div>', unsafe_allow_html=True)
        if metrics is None:
            st.info("Run the main pipeline first to generate the NASA test evaluation report.")
        else:
            overall = metrics[metrics["Dataset"] == "Overall"].iloc[0]
            a, b, c = st.columns(3)
            a.metric("MAE", f"{overall['MAE']:.2f}")
            b.metric("RMSE", f"{overall['RMSE']:.2f}")
            c.metric("R²", f"{overall['R2_Score']:.3f}")
            st.markdown(
                '<div class="footnote">Benchmark result on the unseen NASA CMAPSS test engines. '
                'RUL thresholds shown in the dashboard are decision-support thresholds, not failure probabilities.</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Design note</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The dashboard is intentionally focused on the questions an operator would ask first:
        how many engines need attention, which engines are closest to failure, and how the predicted RUL is changing over time.
        """
    )


def main() -> None:
    inject_styles()

    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-title">RUL Mission Control</div>'
            '<div class="brand-subtitle">Predictive maintenance</div></div></div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Navigate", ["Dashboard", "Engine Explorer", "Dataset Overview", "About Project"], label_visibility="collapsed")
        st.divider()
        st.caption("Model")
        st.markdown("**Optimized XGBoost**")
        st.caption("NASA CMAPSS · FD001–FD004")

    with st.spinner("Loading engine predictions..."):
        predictions = load_prediction_data()

    engine_summary = add_status(get_latest_engine_predictions(predictions))

    if page == "Dashboard":
        show_dashboard(engine_summary)
    elif page == "Engine Explorer":
        show_engine_explorer(predictions)
    elif page == "Dataset Overview":
        show_dataset_overview(predictions)
    else:
        show_about_page()


if __name__ == "__main__":
    main()
