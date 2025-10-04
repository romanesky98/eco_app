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
- Pick a dataset (flow) from the dropdown (loaded from ECB).
- Browse/Filter **named** series and select them to plot.
- We fetch the **full history** by default; adjust analysis in the second tab.
"""

import io
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import time
from requests.exceptions import HTTPError, RequestException
import xml.etree.ElementTree as ET

APP_TITLE = "ECB Data Portal Explorer"
ECB_BASE = "https://data-api.ecb.europa.eu/service"

# -------------------- Helpers -------------------- #
@st.cache_data(show_spinner=False, ttl=3600)
def ecb_fetch_dataflows_xml() -> pd.DataFrame:
    """Fetch dataflows using SDMX-ML (XML). Returns DataFrame [flow_id, name]."""
    url = f"{ECB_BASE}/dataflow"
    headers = {"Accept": "application/vnd.sdmx.structure+xml;version=2.1"}
    resp = requests.get(url, headers=headers, timeout=45)
    resp.raise_for_status()
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
        # English name preferred
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
    df_flows = pd.DataFrame(flows)
    df_flows["label"] = df_flows.apply(lambda r: f"{r['name']}  ({r['flow_id']})".strip(), axis=1)
    return df_flows.sort_values("name", key=lambda s: s.str.lower())

@st.cache_data(show_spinner=True)
def ecb_list_series(flow_id: str, max_rows: int = 10000) -> pd.DataFrame:
    """List series for a flow using serieskeysonly CSV.
    Returns DataFrame with columns: series_key, name (human title if present), plus dimension columns.
    """
    if not flow_id:
        return pd.DataFrame()
    url = f"{ECB_BASE}/data/{flow_id}"
    params = {"detail": "serieskeysonly", "format": "csvdata"}
    headers = {"Accept": "text/csv"}
    r = requests.get(url, params=params, headers=headers, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return pd.DataFrame()

    # Dimension columns (heuristic)
    exclude = {
        "TIME_PERIOD","OBS_VALUE","DECIMALS","TIME_FORMAT","OBS_STATUS","OBS_CONF","OBS_PRE_BREAK","OBS_COM",
        # Keep TITLE columns for naming
    }
    dim_cols = [c for c in df.columns if c.isupper() and (" " not in c) and c not in exclude and not c.startswith("OBS_")]

    # Create series_key
    if "SERIES_KEY" in df.columns:
        df["series_key"] = df["SERIES_KEY"].astype(str)
    else:
        df["series_key"] = df[dim_cols].astype(str).agg(".".join, axis=1)

    # Build human-readable name: prefer TITLE_COMPL > TITLE; else join key dims
    name_col = None
    if "TITLE_COMPL" in df.columns and df["TITLE_COMPL"].notna().any():
        name_col = "TITLE_COMPL"
    elif "TITLE" in df.columns and df["TITLE"].notna().any():
        name_col = "TITLE"

    if name_col:
        df["name"] = df[name_col].astype(str)
    else:
        # Compact composite label from key parts (skip generic freq if present)
        parts_cols = [c for c in dim_cols if c != "FREQ"]
        df["name"] = df[parts_cols].astype(str).agg(" / ".join, axis=1)

    # Deduplicate and limit
    out = df.drop_duplicates(subset=["series_key"]).copy()
    if max_rows:
        out = out.head(max_rows)

    # Order columns
    ordered = ["series_key", "name"] + [c for c in dim_cols if c in out.columns]
    return out[ordered]

@st.cache_data(show_spinner=False)
def ecb_fetch_series_csv(flow_id: str, series_key: str) -> pd.DataFrame:
    """Fetch a single series full history as CSV -> DataFrame indexed by Date with a nice column label."""
    params = {"detail": "dataonly", "format": "csvdata"}
    url = f"{ECB_BASE}/data/{flow_id}/{series_key}"
    headers = {"Accept": "text/csv"}
    r = requests.get(url, params=params, headers=headers, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise RuntimeError("Unexpected CSV format from ECB Data Portal.")
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    df = df.sort_values("Date")
    df = df.rename(columns={"OBS_VALUE": "Value"})
    # Compose readable label from any available metadata
    label_parts = []
    for c in ["TITLE_COMPL", "TITLE", "CURRENCY", "CURRENCY_DENOM", "EXR_TYPE", "EXR_SUFFIX", "UNIT", "UNIT_MULT"]:
        if c in df.columns and pd.notna(df[c]).any():
            val = str(df[c].dropna().iloc[0])
            if val and val != "nan":
                label_parts.append(val)
    label = f"{flow_id}:{series_key}"
    if label_parts:
        label = " — ".join([label] + label_parts)
    out = df[["Date", "Value"]].copy().set_index("Date")
    out.columns = [label]
    return out

@st.cache_data(show_spinner=False)
def ecb_fetch_many(flow_id: str, series_keys: List[str]) -> pd.DataFrame:
    frames = []
    for key in series_keys:
        try:
            s = ecb_fetch_series_csv(flow_id, key)
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

# Sidebar: analysis controls only (no date preselection; full history is fetched)
with st.sidebar:
    st.markdown("### Analysis options")
    norm_mode = st.selectbox(
        "Normalize series (helps compare different units)",
        options=["None", "Rebase to 100 at start", "Z-score (standardize)"]
    )
    preset_windows = [3, 6, 12, 24, 60]
    windows = st.multiselect("Rolling mean windows (periods)", options=preset_windows, default=[6, 12])
    custom_windows = st.text_input("Custom windows (comma-separated)", placeholder="e.g., 9, 18")

# Load all datasets for dropdown
with st.spinner("Loading datasets from ECB..."):
    all_flows = ecb_fetch_dataflows_xml()

explore_tab, analyze_tab = st.tabs(["Explore", "Deep analysis"]) 

with explore_tab:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Choose a dataset (flow)")
        # Optional filter box to quickly narrow the dropdown
        flow_filter = st.text_input("Filter datasets (by name or id)", placeholder="e.g., exchange rate, EXR")
        flows_df = all_flows
        if flow_filter.strip():
            q = flow_filter.lower()
            flows_df = flows_df[flows_df.apply(lambda r: q in str(r["name"]).lower() or q in str(r["flow_id"]).lower(), axis=1)]
        flow_labels = flows_df["label"].tolist()
        default_index = 0 if len(flow_labels) else None
        chosen_label = st.selectbox("Dataset (flow)", options=flow_labels, index=default_index)
        # Map back to flow_id
        chosen_flow = flows_df.loc[flows_df["label"] == chosen_label, "flow_id"].iloc[0] if flow_labels else ""
        st.caption(f"Flow selected: **{chosen_flow}**")

    with col2:
        st.subheader("Browse & select series")
        max_rows = st.slider("Max keys to load", min_value=200, max_value=50000, step=200, value=5000)
        list_keys = st.button("Load series for this dataset", use_container_width=True)

        selected_from_list: List[str] = []
        selected_names: List[str] = []
        if chosen_flow and list_keys:
            try:
                series_df = ecb_list_series(chosen_flow, max_rows=max_rows)
                if series_df.empty:
                    st.warning("No series returned (dataset might be restricted or empty).")
                else:
                    # Filter series by text
                    key_filter = st.text_input("Filter series (by name or key)", placeholder="type to filter...")
                    show_df = series_df
                    if key_filter.strip():
                        q = key_filter.lower()
                        show_df = series_df[series_df.apply(lambda r: q in str(r["name"]).lower() or q in str(r["series_key"]).lower(), axis=1)]
                    st.dataframe(show_df, use_container_width=True, hide_index=True)

                    # Build labeled options "name — series_key"
                    options = (show_df["name"] + " — " + show_df["series_key"]).tolist()
                    pick = st.multiselect("Select series", options=options)
                    selected_from_list = [p.split(" — ")[-1] for p in pick]
                    selected_names = [p.split(" — ")[0] for p in pick]
            except Exception as e:
                st.error(f"Failed to list series: {e}")

        # Manual entry still supported
        manual_text = st.text_input("Add series keys (comma-separated)", placeholder="EXR.D.USD.EUR.SP00.A, EXR.D.GBP.EUR.SP00.A")
        manual_keys = [s.strip() for s in manual_text.split(",") if s.strip()]

        series_keys = list(dict.fromkeys(selected_from_list + manual_keys))
        st.caption("Tip: Use the table & filter above to find named series; you can also paste keys manually.")

    st.markdown("---")

    # Fetch & plot full history
    df = pd.DataFrame()
    if chosen_flow and series_keys:
        df = ecb_fetch_many(chosen_flow, series_keys)
        if df.empty:
            st.warning("No data for the selected keys.")
        else:
            # Apply normalization requested in sidebar (for immediate visual comparison)
            base_df = df.copy()
            if norm_mode == "Rebase to 100 at start":
                base_df = base_df.apply(lambda s: (s / s.dropna().iloc[0]) * 100)
            elif norm_mode == "Z-score (standardize)":
                base_df = base_df.apply(lambda s: (s - s.mean()) / s.std(ddof=0))

            fig = px.line(base_df, x=base_df.index, y=base_df.columns,
                          labels={"x": "Date", "value": "Value", "variable": "Series"})
            fig.update_layout(legend_title_text="Series", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Summary stats (on transformed data if applied)"):
                st.dataframe(base_df.describe().T, use_container_width=True)

            csv = df.to_csv(index=True).encode("utf-8")
            st.download_button("Download raw CSV", data=csv, file_name="ecb_data.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Choose a dataset and at least one series to plot.")

with analyze_tab:
    st.subheader("Deep analysis: rolling means")
    if 'df' not in locals() or df is None or df.empty:
        st.info("Load one or more series in the **Explore** tab first.")
    else:
        # Build list of windows from sidebar + custom
        windows_list = windows[:]
        if custom_windows.strip():
            try:
                extra = [int(x.strip()) for x in custom_windows.split(',') if x.strip()]
                windows_list = sorted(list(dict.fromkeys(windows_list + extra)))
            except Exception:
                st.warning("Couldn't parse custom windows; please enter integers separated by commas.")

        base_df = df.copy()
        if norm_mode == "Rebase to 100 at start":
            base_df = base_df.apply(lambda s: (s / s.dropna().iloc[0]) * 100)
        elif norm_mode == "Z-score (standardize)":
            base_df = base_df.apply(lambda s: (s - s.mean()) / s.std(ddof=0))

        long_frames = []
        show_raw = st.checkbox("Show raw series", value=True)
        if show_raw:
            raw_long = base_df.reset_index().melt(id_vars=["Date"], var_name="Series", value_name="Value")
            raw_long["Line"] = "Raw"
            long_frames.append(raw_long)
        for w in windows_list:
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
