"""
Streamlit ECB (SDW) Explorer

Quick start:
1) Save this file as app_ecb.py
2) Create a virtual env and install deps:
   pip install streamlit plotly pandas python-dateutil requests
3) Run the app:
   streamlit run app_ecb.py

Data source: European Central Bank Statistical Data Warehouse (SDW)
API docs: https://sdw-wsrest.ecb.europa.eu 
No API key required.

Notes:
- Use the search to find a dataset ("dataflow").
- Paste one or more **series keys** for that dataset to fetch/plot data.
  Example for EXR (exchange rates): EXR.D.USD.EUR.SP00.A
"""

import io
import os
from datetime import date
from typing import List, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta
import requests

APP_TITLE = "ECB SDW Data Explorer"
DEFAULT_START = date.today() - relativedelta(years=10)
DEFAULT_END = date.today()
ECB_BASE = "https://sdw-wsrest.ecb.europa.eu/service"

# -------------------- Helpers -------------------- #
@st.cache_data(show_spinner=False)
def ecb_search_dataflows(query: str, limit: int = 50) -> pd.DataFrame:
    """Search dataflows (datasets) by name/description (case-insensitive contains)."""
    if not query.strip():
        return pd.DataFrame()
    url = f"{ECB_BASE}/dataflow"
    # sdmx-json is easier to parse
    resp = requests.get(url, params={"format": "sdmx-json"}, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    flows = j.get("dataflows", {}).get("dataflow", [])
    rows = []
    q = query.lower()
    for f in flows:
        flow_id = f.get("id")
        names = f.get("name", [])
        name_en = next((n["#text"] for n in names if n.get("@xml:lang", "en").startswith("en") and "#text" in n), None)
        name_any = name_en or (names[0]["#text"] if names else "")
        if name_any and q in name_any.lower() or (flow_id and q in flow_id.lower()):
            rows.append({"flow_id": flow_id, "name": name_any})
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def ecb_fetch_series_csv(flow_id: str, series_key: str, start: date, end: date) -> pd.DataFrame:
    """Fetch a single ECB series using the CSV data endpoint and return a tidy DataFrame with Date and Value.
    The CSV includes dimension columns; we keep them for labeling when useful.
    """
    start_s = start.isoformat()
    end_s = end.isoformat()
    params = {
        "detail": "dataonly",
        "startPeriod": start_s,
        "endPeriod": end_s,
        "format": "csvdata",
    }
    url = f"{ECB_BASE}/data/{flow_id}/{series_key}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Expect TIME_PERIOD, OBS_VALUE at minimum
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise RuntimeError("Unexpected CSV format from ECB SDW.")
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    df = df.sort_values("Date")
    df = df.rename(columns={"OBS_VALUE": "Value"})
    # Build a friendly series name from available columns
    label_parts = []
    for c in ["TITLE", "TITLE_COMPL", "CURRENCY", "CURRENCY_DENOM", "EXR_TYPE", "EXR_SUFFIX", "UNIT", "UNIT_MULT"]:
        if c in df.columns and pd.notna(df[c]).any():
            val = str(df[c].dropna().iloc[0])
            if val and val != "nan":
                label_parts.append(f"{c.split('_')[0].title()}: {val}")
    label = f"{flow_id}:{series_key}"
    if label_parts:
        label += " — " + ", ".join(label_parts)
    # Reduce to Date/Value and attach column name
    out = df[["Date", "Value"]].copy()
    out = out.set_index("Date").rename(columns={"Value": label})
    return out

@st.cache_data(show_spinner=False)
def ecb_fetch_many(flow_id: str, series_keys: List[str], start: date, end: date) -> pd.DataFrame:
    frames = []
    for key in series_keys:
        try:
            s = ecb_fetch_series_csv(flow_id, key, start, end)
            frames.append(s)
        except Exception as e:
            st.warning(f"Failed to fetch {flow_id}/{key}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index.name = "Date"
    return df

# -------------------- UI -------------------- #
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.markdown("### Date Range")
    start_date = st.date_input("Start", value=DEFAULT_START)
    end_date = st.date_input("End", value=DEFAULT_END)
    st.markdown("---")
    st.markdown("### Help")
    st.caption("Search a dataset, pick its ID (flow), then paste full series keys for that dataset. Example: EXR.D.USD.EUR.SP00.A")

# Shared containers
results_df: pd.DataFrame = pd.DataFrame()
chosen_flow = st.text_input("Selected dataset (flow ID)", placeholder="e.g., EXR, ICP, BSI, MNA")

explore_tab, analyze_tab = st.tabs(["Explore", "Deep analysis"]) 

with explore_tab:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Search ECB datasets (dataflows)")
        query = st.text_input("Keyword(s)", placeholder="e.g., exchange rate, HICP, GDP, M3")
        limit = st.slider("Max results", min_value=10, max_value=200, value=50, step=10)
        do_search = st.button("Search", type="primary", use_container_width=True)
        if do_search:
            try:
                results_df = ecb_search_dataflows(query, limit)
                if results_df.empty:
                    st.warning("No matching datasets.")
            except Exception as e:
                st.error(f"Search failed: {e}")
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Pick dataset and series keys")
        if not results_df.empty:
            flows = results_df["flow_id"].tolist()
            chosen_flow = st.selectbox("Dataset (flow)", options=flows, index=0)
        series_text = st.text_input("Series keys (comma-separated)", placeholder="EXR.D.USD.EUR.SP00.A, EXR.D.GBP.EUR.SP00.A")
        series_keys = [s.strip() for s in series_text.split(",") if s.strip()]
        st.caption("Tip: Keys are dot-separated dimension codes specific to each dataset. Check ECB SDW for structure.")

    st.markdown("---")

    df = pd.DataFrame()
    if chosen_flow and series_keys:
        df = ecb_fetch_many(chosen_flow, series_keys, start_date, end_date)
        if df.empty:
            st.warning("No data for the selected keys/range.")
        else:
            fig = px.line(df, x=df.index, y=df.columns, labels={"x": "Date", "value": "Value", "variable": "Series"})
            fig.update_layout(legend_title_text="Series", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Summary stats"):
                st.dataframe(df.describe().T, use_container_width=True)

            csv = df.to_csv(index=True).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="ecb_data.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Search a dataset (flow) and enter at least one full series key.")

with analyze_tab:
    st.subheader("Deep analysis: rolling means & normalization")
    if 'df' not in locals() or df is None or df.empty:
        st.info("Load one or more series in the **Explore** tab first.")
    else:
        norm_mode = st.selectbox(
            "Normalize series (helps compare different units)",
            options=["None", "Rebase to 100 at start", "Z-score (standardize)"]
        )
        preset_windows = [3, 6, 12, 24, 60]
        windows = st.multiselect("Rolling mean windows (in periods)", options=preset_windows, default=[6, 12])
        custom_windows = st.text_input("Add custom windows (comma-separated integers)", placeholder="e.g., 9, 18")
        if custom_windows.strip():
            try:
                extra = [int(x.strip()) for x in custom_windows.split(',') if x.strip()]
                windows = sorted(list(dict.fromkeys(windows + extra)))
            except Exception:
                st.warning("Couldn't parse custom windows; please enter integers separated by commas.")
        show_raw = st.checkbox("Show raw series", value=True)

        base_df = df.copy()
        if norm_mode == "Rebase to 100 at start":
            base_df = base_df.apply(lambda s: (s / s.dropna().iloc[0]) * 100)
        elif norm_mode == "Z-score (standardize)":
            base_df = base_df.apply(lambda s: (s - s.mean()) / s.std(ddof=0))

        long_frames = []
        if show_raw:
            raw_long = base_df.reset_index().melt(id_vars=["Date"], var_name="Series", value_name="Value")
            raw_long["Line"] = "Raw"
            long_frames.append(raw_long)
        for w in windows:
            ma = base_df.rolling(window=w, min_periods=w).mean()
            ma_long = ma.reset_index().melt(id_vars=["Date"], var_name="Series", value_name="Value")
            ma_long["Line"] = f"MA{w}"
            long_frames.append(ma_long)
        if not long_frames:
            st.warning("Choose at least one window or enable raw series.")
        else:
            plot_df = pd.concat(long_frames, ignore_index=True)
            fig2 = px.line(plot_df, x="Date", y="Value", color="Series", line_dash="Line",
                           labels={"Date": "Date", "Value": "Value", "Series": "Series", "Line": "Type"})
            fig2.update_layout(legend_title_text="Series / Type", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Download analysis data"):
            st.download_button(
                label="Download analysis CSV",
                data=plot_df.to_csv(index=False).encode("utf-8"),
                file_name="ecb_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

# Footer
st.markdown(
    """
    <div style='text-align:center; opacity:0.7; font-size:0.9em;'>
    Built with <a href='https://streamlit.io' target='_blank'>Streamlit</a> · Powered by <a href='https://sdw.ecb.europa.eu' target='_blank'>ECB SDW</a>
    </div>
    """,
    unsafe_allow_html=True,
)
