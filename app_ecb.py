"""
Streamlit ECB Data Portal Explorer (uses data-api.ecb.europa.eu)

Quick start:
1) Save this file as app_ecb.py
2) Create a virtual env and install deps:
   pip install streamlit plotly pandas python-dateutil requests
3) Run the app:
   streamlit run app_ecb.py

Data source: European Central Bank Data Portal (SDMX 2.1 REST)
API docs: https://data.ecb.europa.eu/help/api/overview
Base: https://data-api.ecb.europa.eu/service
No API key required.

Notes:
- Search dataflows (datasets), then paste one or more full series keys for that flow.
  Example for EXR (exchange rates): EXR.D.USD.EUR.SP00.A
"""

import io
from datetime import date
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta
import requests
import time
from requests.exceptions import HTTPError, RequestException
import xml.etree.ElementTree as ET

APP_TITLE = "ECB Data Portal Explorer"
DEFAULT_START = date.today() - relativedelta(years=10)
DEFAULT_END = date.today()
ECB_BASE = "https://data-api.ecb.europa.eu/service"

# -------------------- Helpers -------------------- #
@st.cache_data(show_spinner=False, ttl=3600)
def ecb_fetch_dataflows() -> pd.DataFrame:
    """Fetch all dataflows once (cached for 1 hour).
    Tries multiple Accept/format combinations to satisfy the ECB Data Portal and avoid 406/unsupported content.
    Returns a DataFrame with columns [flow_id, name].
    """
    url = f"{ECB_BASE}/dataflow"

    header_options = [
        {"Accept": "application/vnd.sdmx.structure+json;version=1.0"},
        {"Accept": "application/vnd.sdmx.dataflow+json;version=1.0"},
        {"Accept": "application/vnd.sdmx+json;version=1.0"},
        {"Accept": "application/json"},
        {},
        {"Accept": "*/*"},
    ]
    param_options = [
        {},
        {"format": "sdmx-json"},
        {"format": "jsondata"},
    ]

    last_err = None
    for headers in header_options:
        for params in param_options:
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "")
                # Some gateways reply XML by default; skip if not JSON
                if "json" not in ctype.lower():
                    continue
                j = resp.json()
                flows = j.get("dataflows", {}).get("dataflow", [])
                rows = []
                for f in flows:
                    flow_id = f.get("id")
                    names = f.get("name", [])
                    name_en = next((n.get("#text") for n in names if str(n.get("@xml:lang", "en")).startswith("en") and "#text" in n), None)
                    name_any = name_en or (names[0].get("#text") if names else "")
                    rows.append({"flow_id": flow_id, "name": name_any})
                if rows:
                    return pd.DataFrame(rows)
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Unable to fetch dataflows from ECB endpoint with any Accept/format combination: {last_err}")

@st.cache_data(show_spinner=False)
def ecb_search_dataflows_resilient(query: str, limit: int = 50) -> pd.DataFrame:
    """Resilient dataflow search with exponential backoff retries."""
    if not query.strip():
        return pd.DataFrame()

    last_err = None
    for i in range(5):
        try:
            all_flows = ecb_fetch_dataflows_xml()
            q = query.lower()
            matches = all_flows[
                all_flows.apply(
                    lambda r: q in str(r["name"]).lower() or q in str(r["flow_id"]).lower(),
                    axis=1,
                )
            ]
            return matches.head(limit).reset_index(drop=True)
        except (HTTPError, RequestException, ValueError) as e:
            last_err = e
            time.sleep(0.8 * (2 ** i))
    raise RuntimeError(f"ECB dataflow search failed after retries: {last_err}")
            q = query.lower()
            matches = all_flows[
                all_flows.apply(
                    lambda r: q in str(r["name"]).lower() or q in str(r["flow_id"]).lower(),
                    axis=1,
                )
            ]
            return matches.head(limit).reset_index(drop=True)
        except (HTTPError, RequestException, ValueError) as e:
            last_err = e
            time.sleep(0.8 * (2 ** i))
    raise RuntimeError(f"ECB dataflow search failed after retries: {last_err}")

@st.cache_data(show_spinner=False)
def ecb_fetch_series_csv(flow_id: str, series_key: str, start: date, end: date) -> pd.DataFrame:
    """Fetch a single ECB series using the CSV data endpoint and return a tidy DataFrame with Date and Value."""
    params = {
        "detail": "dataonly",
        "startPeriod": start.isoformat(),
        "endPeriod": end.isoformat(),
        "format": "csvdata",
    }
    url = f"{ECB_BASE}/data/{flow_id}/{series_key}"
    headers = {"Accept": "text/csv"}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise RuntimeError("Unexpected CSV format from ECB Data Portal.")
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    df = df.sort_values("Date")
    df = df.rename(columns={"OBS_VALUE": "Value"})
    # Build a friendly label
    label_parts = []
    for c in [
        "TITLE",
        "TITLE_COMPL",
        "CURRENCY",
        "CURRENCY_DENOM",
        "EXR_TYPE",
        "EXR_SUFFIX",
        "UNIT",
        "UNIT_MULT",
    ]:
        if c in df.columns and pd.notna(df[c]).any():
            val = str(df[c].dropna().iloc[0])
            if val and val != "nan":
                label_parts.append(f"{c.split('_')[0].title()}: {val}")
    label = f"{flow_id}:{series_key}"
    if label_parts:
        label += " — " + ", ".join(label_parts)
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

@st.cache_data(show_spinner=False, ttl=3600)
def ecb_fetch_dataflows_xml() -> pd.DataFrame:
    """Fetch dataflows using SDMX-ML (XML), which the ECB Data Portal always supports.
    Returns DataFrame with [flow_id, name].
    """
    url = f"{ECB_BASE}/dataflow"
    headers = {"Accept": "application/vnd.sdmx.structure+xml;version=2.1"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    # Parse SDMX-ML
    root = ET.fromstring(resp.content)
    ns = {
        "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
        "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
        "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    }
    flows = []
    for df in root.findall('.//str:Dataflow', namespaces=ns):
        fid = df.get('{http://www.w3.org/2001/XMLSchema-instance}id') or df.get('id')
        if not fid:
            fid_el = df.find('str:ID', ns)
            fid = fid_el.text if fid_el is not None else None
        # Name in English if available
        name_el = None
        for n in df.findall('com:Name', ns):
            if (n.get('{http://www.w3.org/XML/1998/namespace}lang') or '').startswith('en'):
                name_el = n
                break
        if name_el is None:
            name_el = df.find('com:Name', ns)
        name_txt = name_el.text if name_el is not None else ''
        if fid:
            flows.append({"flow_id": fid, "name": name_txt})
    return pd.DataFrame(flows)

# -------------------- UI -------------------- #
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.markdown("### Date Range")
    start_date = st.date_input("Start", value=DEFAULT_START)
    end_date = st.date_input("End", value=DEFAULT_END)
    st.markdown("---")
    st.markdown("### Help")
    st.caption("Search a dataset (flow), pick its ID, then paste full series keys. Example: EXR.D.USD.EUR.SP00.A")

# Shared state
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
                results_df = ecb_search_dataflows_resilient(query, limit)
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
        st.caption("Tip: Keys are dot-separated dimension codes specific to each dataset. Check ECB Data Portal for structure.")

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
    Built with <a href='https://streamlit.io' target='_blank'>Streamlit</a> · Powered by <a href='https://data.ecb.europa.eu' target='_blank'>ECB Data Portal</a>
    </div>
    """,
    unsafe_allow_html=True,
)
