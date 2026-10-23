from __future__ import annotations

import logging
from io import BytesIO
from numbers import Number
from typing import Any

import pandas as pd
import streamlit as st

from src.anomaly_agent.cli import add_attribution, attach_drivers
from src.anomaly_agent.config import load_config
from src.anomaly_agent.detection import Detector
from src.anomaly_agent.forecast_context import add_forecast_context
from src.anomaly_agent.ingestion import Ingestion
from src.anomaly_agent.metrics import MetricEngine
from src.anomaly_agent.severity import SeverityScorer
from src.anomaly_agent.variance_context import add_variance_context

logger = logging.getLogger(__name__)

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="KPI Anomaly Detection Agent",
    page_icon="📊",
    layout="wide",
)

# Preserve Streamlit's original page background.
st.markdown(
    """
    <style>
    #MainMenu,
    footer,
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }

    :root {
        --accent: #168b87;
        --accent-hover: #11736f;
        --accent-soft: rgba(22, 139, 135, 0.08);
        --border-subtle: rgba(148, 163, 184, 0.20);
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .app-title {
        color: #e5e7eb;
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        line-height: 1.3;
        margin: 0;
        padding: 0;
    }

    .app-title span {
        color: inherit;
        background: none;
        -webkit-text-fill-color: currentColor;
    }

    .app-sub {
        color: #a7b0be;
        font-size: 0.94rem;
        line-height: 1.6;
        margin: 0.55rem 0 1.5rem 0;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 16px !important;
        background: rgba(255, 255, 255, 0.025);
        padding: 0.8rem 1rem;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.10);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(22, 139, 135, 0.50);
        border-radius: 12px;
        background: var(--accent-soft);
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
        background: rgba(22, 139, 135, 0.12);
    }

    .stButton > button[kind="primary"] {
        background: var(--accent);
        border: 1px solid var(--accent);
        color: #ffffff;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.62rem 1rem;
        box-shadow: none;
        transition: background 150ms ease, border-color 150ms ease;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #ffffff;
    }

    .stButton > button:not([kind="primary"]),
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 550;
        border: 1px solid var(--border-subtle);
    }

    [data-testid="stMetric"] {
        border-radius: 12px;
        padding: 0.85rem 1rem;
        border: 1px solid var(--border-subtle);
        border-top: 3px solid var(--accent);
        background: rgba(255, 255, 255, 0.025);
    }

    [data-testid="stMetricLabel"] {
        color: #a7b0be;
        font-size: 0.84rem;
    }

    [data-testid="stMetricValue"] {
        color: #e5e7eb;
        font-size: 1.55rem;
        font-weight: 700;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        background: transparent;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CONFIG_PATH = "config/dataco_supply_chain.yaml"

SEVERITY_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


# ============================================================
# Data and display helpers
# ============================================================


@st.cache_data
def read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))


def display_value(value: Any) -> str:
    """Format values consistently for the UI."""
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, Number) and not isinstance(value, bool):
        number = float(value)

        if abs(number) >= 1000:
            return f"{number:,.0f}"

        return f"{number:,.4g}"

    return str(value)


def display_relative_deviation(value: Any) -> str:
    """Display relative deviation as a percentage."""
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass

    return f"{float(value):.1%}"


def deduplicate_alerts(
    alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the latest alert per metric, segment and direction."""
    best: dict[tuple[str, str, str], dict[str, Any]] = {}

    for anomaly in alerts:
        key = (
            anomaly["metric"],
            anomaly["segment"],
            anomaly["direction"],
        )

        current = best.get(key)

        if current is None:
            best[key] = anomaly
            continue

        current_date = pd.Timestamp(current["date"])
        new_date = pd.Timestamp(anomaly["date"])

        current_severity = SEVERITY_RANK.get(current.get("severity", "low").lower(), 0)
        new_severity = SEVERITY_RANK.get(anomaly.get("severity", "low").lower(), 0)

        if (new_date, new_severity) > (
            current_date,
            current_severity,
        ):
            best[key] = anomaly

    return sorted(
        best.values(),
        key=lambda item: pd.Timestamp(item["date"]),
        reverse=True,
    )


def generate_narrations(
    alerts: list[dict[str, Any]],
    config: Any,
) -> None:
    """Generate validated LLM narrations or fall back to templates."""
    from src.anomaly_agent.narration.client import (
        Narrator,
        build_client,
    )
    from src.anomaly_agent.narration.fallback import template_summary

    try:
        narrator = Narrator(
            build_client(config),
            max_retries=config.llm.max_retries,
        )
    except Exception as exc:
        logger.warning(
            "Could not initialize narration client; using template fallback: %s",
            exc,
        )
        narrator = None

    for anomaly in alerts:
        if narrator is not None:
            narrated_before = narrator.narrated
            fallback_before = narrator.fallbacks

            try:
                anomaly["narration"] = narrator.narrate(anomaly)

                if narrator.narrated > narrated_before:
                    anomaly["narration_source"] = "LLM response — numeric validation passed"
                elif narrator.fallbacks > fallback_before:
                    anomaly["narration_source"] = "Template fallback"
                else:
                    anomaly["narration_source"] = "Narration generated"

                continue

            except Exception as exc:
                logger.warning(
                    "Narration failed for metric %s; using template fallback: %s",
                    anomaly.get("metric", "KPI"),
                    exc,
                )

        try:
            anomaly["narration"] = template_summary(anomaly)
            anomaly["narration_source"] = "Template fallback"
        except Exception as exc:
            logger.warning(
                "Template narration failed for metric %s: %s",
                anomaly.get("metric", "KPI"),
                exc,
            )
            anomaly["narration"] = (
                f"{anomaly.get('metric', 'KPI')} was flagged for an "
                f"{anomaly.get('direction', 'unexpected')} movement on "
                f"{display_value(anomaly.get('date'))}. "
                f"Observed: {display_value(anomaly.get('observed'))}; "
                f"expected: {display_value(anomaly.get('expected'))}. "
                "Review the source data and supporting context."
            )
            anomaly["narration_source"] = "Basic fallback — narration generation failed"


# ============================================================
# Analysis pipeline
# ============================================================


def analyze_data(
    uploaded_df: pd.DataFrame,
    config_path: str,
    scan_history: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    config = load_config(config_path)
    date_col = config.dataset.date_column

    # Validate required columns.
    required = {date_col}

    for metric in config.metrics:
        if metric.column:
            required.add(metric.column)

        if metric.numerator and metric.numerator.column:
            required.add(metric.numerator.column)

        if metric.denominator and metric.denominator.column:
            required.add(metric.denominator.column)

    required.update(config.dataset.dimensions)

    missing = sorted(required - set(uploaded_df.columns))

    if missing:
        raise ValueError("The uploaded file is missing required columns: " + ", ".join(missing))

    uploaded_rows = len(uploaded_df)

    df = uploaded_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    invalid_date_rows = int(df[date_col].isna().sum())
    df = df.dropna(subset=[date_col])

    if df.empty:
        raise ValueError("No valid dates were found in the uploaded file.")

    rows_before_window = len(df)

    # Apply the same configured date window as CLI ingestion.
    if config.dataset.start_date:
        start_date = pd.Timestamp(config.dataset.start_date)
        df = df[df[date_col] >= start_date]

    if config.dataset.end_date:
        # The configured end date is inclusive.
        end_exclusive = pd.Timestamp(config.dataset.end_date) + pd.Timedelta(days=1)
        df = df[df[date_col] < end_exclusive]

    rows_outside_window = rows_before_window - len(df)

    if df.empty:
        raise ValueError(
            "No valid rows remain inside the configured date window. "
            f"Start: {config.dataset.start_date or 'earliest'}, "
            f"end: {config.dataset.end_date or 'latest'}. "
            "Check config/dataco_supply_chain.yaml."
        )

    df = df.sort_values(date_col)

    ingestion = Ingestion(config)
    df, quality_report = ingestion.validate_and_clean(df)

    if df.empty:
        raise ValueError("No usable rows remain after validation.")

    quality_report["uploaded_rows"] = uploaded_rows
    quality_report["invalid_date_rows_removed"] = invalid_date_rows
    quality_report["rows_outside_monitoring_window"] = rows_outside_window
    quality_report["rows_after_validation"] = len(df)

    # Compute additive and ratio KPIs using the existing metric engine.
    series_dict = MetricEngine(config).process(df)

    if not series_dict:
        raise ValueError("No KPI time series could be generated.")

    # None means full history; otherwise scan the latest configured periods.
    window = None if scan_history else config.alerting.alert_window

    anomalies = Detector(config).detect(
        series_dict,
        window=window,
    )

    add_forecast_context(anomalies, series_dict, config)

    # Match CLI behavior: only total-series anomalies raise headline alerts.
    if config.alerting.headline_only:
        segment_anomalies = [anomaly for anomaly in anomalies if anomaly["segment"] != "total"]

        anomalies = [anomaly for anomaly in anomalies if anomaly["segment"] == "total"]

        attach_drivers(anomalies, segment_anomalies)

    if anomalies:
        add_attribution(anomalies, df, config)
        add_variance_context(anomalies, df, config)

    scorer = SeverityScorer(config)

    minimum_severity = SEVERITY_RANK.get(config.alerting.min_severity.lower(), 1)

    qualifying = []

    for anomaly in anomalies:
        anomaly["severity"] = scorer.score(anomaly)

        rank = SEVERITY_RANK.get(anomaly["severity"].lower(), 0)

        if rank >= minimum_severity:
            qualifying.append(anomaly)

    qualifying = deduplicate_alerts(qualifying)

    return qualifying, df, quality_report


# ============================================================
# App header
# ============================================================

st.markdown(
    '<h1 class="app-title">📊 <span>KPI Anomaly Detection &amp; '
    "Alerting Agent</span></h1>"
    '<p class="app-sub">Upload an updated KPI workbook to run the existing '
    "anomaly detection pipeline on demand.</p>",
    unsafe_allow_html=True,
)


# The checkbox is on the main page, not in the hidden sidebar.
scan_history = st.checkbox(
    "Scan full history",
    key="scan_history",
    value=False,
    help=(
        "When unchecked, only the latest configured periods are checked. "
        "When checked, scan all eligible history inside the configured "
        "monitoring date window."
    ),
)


# ============================================================
# Workbook upload
# ============================================================

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload your KPI Excel workbook",
        type=["xlsx"],
        help=(
            "Use the tidy dataset schema expected by this project, including "
            "order_date, configured KPI source columns, and dimensions."
        ),
    )

    if uploaded_file:
        st.write(f"**File:** {uploaded_file.name}")

        try:
            preview_df = read_excel(uploaded_file.getvalue())

            st.write(f"**Rows:** {len(preview_df):,}  |  **Columns:** {len(preview_df.columns)}")

            with st.expander("Preview uploaded data"):
                st.dataframe(
                    preview_df.head(20),
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Could not read this Excel file: {type(exc).__name__}: {exc}")
            st.stop()

        if st.button(
            "Analyze workbook",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Validating data and running anomaly detection..."):
                    alerts, cleaned_df, quality = analyze_data(
                        preview_df,
                        CONFIG_PATH,
                        scan_history,
                    )

                    config = load_config(CONFIG_PATH)

                    # Narration failure should not invalidate a successful
                    # analysis, so keep narration generation best-effort.
                    try:
                        generate_narrations(alerts, config)
                    except Exception:
                        logger.exception("Unexpected narration pipeline error; using basic fallback where needed.")
                        for anomaly in alerts:
                            if not anomaly.get("narration"):
                                anomaly["narration"] = (
                                    f"{anomaly['metric']} was flagged as "
                                    f"{anomaly['direction']} on "
                                    f"{display_value(anomaly['date'])}. "
                                    f"Observed: "
                                    f"{display_value(anomaly.get('observed'))}; "
                                    f"expected: "
                                    f"{display_value(anomaly.get('expected'))}."
                                )
                                anomaly["narration_source"] = "Basic fallback — narration unavailable"

                date_col = config.dataset.date_column

                st.session_state["analysis_alerts"] = alerts
                st.session_state["analysis_rows"] = len(cleaned_df)
                st.session_state["analysis_start"] = cleaned_df[date_col].min()
                st.session_state["analysis_end"] = cleaned_df[date_col].max()
                st.session_state["quality_report"] = quality
                st.session_state["analysis_done"] = True
                st.session_state["analysis_file_name"] = uploaded_file.name

            except Exception as exc:
                st.session_state["analysis_done"] = False

                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")


# ============================================================
# Analysis results
# ============================================================

if st.session_state.get("analysis_done"):
    alerts = st.session_state["analysis_alerts"]

    st.write("")

    with st.container(border=True):
        st.subheader("Analysis results")

        col_rows, col_alerts = st.columns(2)

        col_rows.metric(
            "Rows analyzed",
            f"{st.session_state['analysis_rows']:,}",
        )

        col_alerts.metric(
            "Qualifying alerts",
            len(alerts),
        )

        start_date = pd.Timestamp(st.session_state["analysis_start"])
        end_date = pd.Timestamp(st.session_state["analysis_end"])

        st.caption(
            f"File: {st.session_state.get('analysis_file_name', 'Workbook')} "
            f"| Data coverage: {start_date:%Y-%m-%d} — {end_date:%Y-%m-%d}"
        )

        # --------------------------------------------------------
        # Data quality summary
        # --------------------------------------------------------

        with st.expander("Data quality summary"):
            quality = st.session_state["quality_report"]

            st.write(
                "Uploaded rows:",
                f"{quality.get('uploaded_rows', 0):,}",
            )

            st.write(
                "Invalid date rows removed:",
                f"{quality.get('invalid_date_rows_removed', 0):,}",
            )

            st.write(
                "Rows outside configured date window:",
                f"{quality.get('rows_outside_monitoring_window', 0):,}",
            )

            st.write(
                "Duplicate rows removed:",
                f"{quality.get('duplicates_removed', 0):,}",
            )

            st.write(
                "Rows remaining after validation:",
                f"{quality.get('rows_after_validation', 0):,}",
            )

            st.write("Null rates by column")

            st.json({key: round(float(value), 4) for key, value in quality.get("null_rates", {}).items()})

        # --------------------------------------------------------
        # Alerts and details
        # --------------------------------------------------------

        if not alerts:
            st.success("No qualifying anomalies detected for the selected analysis window.")

        else:
            result_rows = []

            for anomaly in alerts:
                result_rows.append(
                    {
                        "Severity": anomaly["severity"].upper(),
                        "KPI": anomaly["metric"],
                        "Segment": anomaly["segment"],
                        "Date": display_value(anomaly["date"]),
                        "Direction": anomaly["direction"],
                        "Actual": display_value(anomaly.get("observed", anomaly.get("actual"))),
                        "Expected": display_value(anomaly.get("expected")),
                        "Z-score": display_value(anomaly.get("z_score")),
                        "Relative deviation": display_relative_deviation(anomaly.get("rel_deviation")),
                    }
                )

            results_df = pd.DataFrame(result_rows)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download alerts as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="kpi_anomaly_alerts.csv",
                mime="text/csv",
            )

            for index, anomaly in enumerate(alerts, start=1):
                title = (
                    f"{index}. {anomaly['severity'].upper()} — {anomaly['metric']} on {display_value(anomaly['date'])}"
                )

                with st.expander(title):
                    st.write(f"**Segment:** {anomaly['segment']}")
                    st.write(f"**Direction:** {anomaly['direction']}")

                    st.write(
                        "**Actual:**",
                        display_value(anomaly.get("observed", anomaly.get("actual"))),
                    )

                    st.write(
                        "**Expected:**",
                        display_value(anomaly.get("expected")),
                    )

                    st.write(
                        "**Absolute deviation:**",
                        display_value(anomaly.get("abs_deviation")),
                    )

                    st.write(
                        "**Relative deviation:**",
                        display_relative_deviation(anomaly.get("rel_deviation")),
                    )

                    st.write(
                        "**Z-score:**",
                        display_value(anomaly.get("z_score")),
                    )

                    st.write("**Alert summary**")

                    st.markdown(
                        anomaly.get(
                            "narration",
                            "No narration was generated for this alert.",
                        )
                    )

                    if anomaly.get("narration_source"):
                        st.caption(f"Summary source: {anomaly['narration_source']}")

                    if anomaly.get("driver_note"):
                        st.write("**Driver note**")
                        st.write(anomaly["driver_note"])

                    if anomaly.get("drivers"):
                        st.write("**Likely driver segments**")

                        for driver in anomaly["drivers"]:
                            st.write(f"- {driver}")

                    if anomaly.get("attribution"):
                        st.write("**Attribution details**")
                        st.json(anomaly["attribution"])

                    if anomaly.get("forecast_context"):
                        st.write("**Forecast context**")
                        st.json(anomaly["forecast_context"])

                    if anomaly.get("variance_context"):
                        st.write("**Variance context**")
                        st.json(anomaly["variance_context"])

            # ----------------------------------------------------
            # Email button at the bottom of analysis results
            # ----------------------------------------------------

            st.divider()

            st.caption("Send the detected alerts to the recipients configured in your YAML settings.")

            if st.button(
                "✉️ Send Email Alert",
                type="primary",
                use_container_width=True,
                key="send_alert_email",
            ):
                try:
                    from src.anomaly_agent.alerting.email import EmailAlerter

                    config = load_config(CONFIG_PATH)

                    with st.spinner("Sending email alert digest..."):
                        EmailAlerter(config).send_digest(
                            alerts,
                            dry_run=False,
                        )

                    st.success("Email alert sent successfully.")

                except Exception as exc:
                    st.error(f"Could not send the email: {type(exc).__name__}: {exc}")
