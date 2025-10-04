"""
ECB Data Portal Explorer — intuitive & robust (data-api.ecb.europa.eu)

✔ End-to-end discovery via **dropdowns** (datasets & dimension values)
✔ No date preselection (pulls full history by default)
✔ Fetch series by **building keys from dimensions** (no guessing)
✔ Resilient to format/negotiation issues (uses SDMX-ML for structures, CSV for data)
✔ Clean data frames: wide (for charts) and tidy long (for export/analysis)

How to run
----------
1) Save as `app_ecb.py`
2) Install deps: `pip install streamlit plotly pandas python-dateutil requests`
3) Run: `streamlit run app_ecb.py`

Docs: https://data.ecb.europa.eu/help/api/overview  ·  Base: https://data-api.ecb.europa.eu/service
"""

from __future__ import annotations
import io
import itertools
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import xml.etree.ElementTree as ET

APP_TITLE = "ECB Data Portal Explorer"
BASE = "https://data-api.ecb.europa.eu/service"

# ------------------------------
# Helpers: SDMX structure (XML)
# ------------------------------
NS = {
    "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_dataflows() -> pd.DataFrame:
    """List all datasets (dataflows). Returns columns: flow_id, name, label."""
    resp = requests.get(f"{BASE}/dataflow", headers={"Accept": "application/vnd.sdmx.structure+xml;version=2.1"}, timeout=45)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    rows = []
    for df in root.findall('.//str:Dataflow', NS):
        fid = df.get('{http://www.w3.org/2001/XMLSchema-instance}id') or df.get('id')
        if not fid:
            el = df.find('str:ID', NS)
            fid = el.text if el is not None else None
        # English name preferred
        name_el = None
        for n in df.findall('com:Name', NS):
            if (n.get('{http://www.w3.org/XML/1998/namespace}lang') or '').startswith('en'):
                name_el = n; break
        if name_el is None:
            name_el = df.find('com:Name', NS)
        name = name_el.text if name_el is not None else ''
        if fid:
            rows.append({"flow_id": fid, "name": name, "label": f"{name} ({fid})"})
    out = pd.DataFrame(rows)
    return out.sort_values("name", key=lambda s: s.str.lower()).reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=3600)
def get_dsd_ref(flow_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (agencyID, id, version) for the DataStructure linked to a flow."""
    resp = requests.get(f"{BASE}/dataflow/{flow_id}", headers={"Accept": "application/vnd.sdmx.structure+xml;version=2.1"}, params={"references":"all"}, timeout=45)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    df = root.find('.//str:Dataflow', NS)
    if df is None: return (None, flow_id, None)
    ref = df.find('.//str:Structure/str:Ref', NS)
    if ref is not None and (ref.get('class') or '').lower()=="datastructure":
        return (ref.get('agencyID'), ref.get('id'), ref.get('version'))
    urn = df.find('.//str:Structure/str:URN', NS)
    if urn is not None and urn.text:
        try:
            body = urn.text.split('=')[1]
            agency, rest = body.split(':',1)
            if '(' in rest:
                did, ver = rest[:-1].split('(')
            else:
                did, ver = rest, None
            return (agency, did, ver)
        except Exception:
            pass
    return (None, flow_id, None)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_dsd(flow_id: str) -> Dict:
    """Fetch DataStructure and return:
    {
      'dimensions': [ {'id': 'FREQ', 'name': 'Frequency', 'codelist': ('ECB','CL_FREQ','1.0'), 'codes': DataFrame[code,name]} , ...],
      'dim_ids': ['FREQ','...']
    }
    """
    agency, did, ver = get_dsd_ref(flow_id)
    candidates = []
    if agency and did and ver: candidates.append(f"{agency}/{did}/{ver}")
    if agency and did: candidates.append(f"{agency}/{did}")
    if did: candidates.append(did)
    # fetch DSD
    root = None
    for path in candidates:
        r = requests.get(f"{BASE}/datastructure/{path}", headers={"Accept": "application/vnd.sdmx.structure+xml;version=2.1"}, params={"references":"children"}, timeout=45)
        if r.status_code==404: continue
        r.raise_for_status()
        root = ET.fromstring(r.content)
        break
    if root is None:
        return {"dimensions": [], "dim_ids": []}
    dims = []
    for dim in root.findall('.//str:DataStructure/str:DataStructureComponents/str:DimensionList/str:Dimension', NS):
        dim_id = dim.get('id') or dim.get('{http://www.w3.org/2001/XMLSchema-instance}id')
        if not dim_id or dim_id.upper()=="TIME_PERIOD":
            continue
        # human-readable name
        dname = None
        for n in dim.findall('com:Name', NS):
            if (n.get('{http://www.w3.org/XML/1998/namespace}lang') or '').startswith('en'):
                dname = n.text; break
        if dname is None:
            n = dim.find('com:Name', NS)
            dname = n.text if n is not None else dim_id
        # codelist ref
        cl_ref = dim.find('.//str:LocalRepresentation/str:Enumeration/str:Ref', NS)
        cl_urn = dim.find('.//str:LocalRepresentation/str:Enumeration/str:URN', NS)
        cl_tuple: Optional[Tuple[str,str,Optional[str]]] = None
        if cl_ref is not None and (cl_ref.get('class') or '').lower()=="codelist":
            cl_tuple = (cl_ref.get('agencyID'), cl_ref.get('id'), cl_ref.get('version'))
        elif cl_urn is not None and cl_urn.text:
            try:
                body = cl_urn.text.split('=')[1]
                agency, rest = body.split(':',1)
                if '(' in rest:
                    cid, ver = rest[:-1].split('(')
                else:
                    cid, ver = rest, None
                cl_tuple = (agency, cid, ver)
            except Exception:
                pass
        # fetch codes (if codelist exists)
        codes_df = pd.DataFrame(columns=["code","name"])        
        if cl_tuple:
            a,cid,v = cl_tuple
            paths = []
            if a and cid and v: paths.append(f"{a}/{cid}/{v}")
            if a and cid: paths.append(f"{a}/{cid}")
            if cid: paths.append(cid)
            for p in paths:
                rc = requests.get(f"{BASE}/codelist/{p}", headers={"Accept": "application/vnd.sdmx.structure+xml;version=2.1"}, timeout=45)
                if rc.status_code==404: continue
                rc.raise_for_status()
                cl_root = ET.fromstring(rc.content)
                rows=[]
                for it in cl_root.findall('.//str:Code', NS):
                    code = it.get('id') or it.get('{http://www.w3.org/2001/XMLSchema-instance}id')
                    nm_el=None
                    for nn in it.findall('com:Name', NS):
                        if (nn.get('{http://www.w3.org/XML/1998/namespace}lang') or '').startswith('en'):
                            nm_el = nn; break
                    if nm_el is None:
                        nm_el = it.find('com:Name', NS)
                    nm = nm_el.text if nm_el is not None else code
                    rows.append({"code":code, "name":nm})
                codes_df = pd.DataFrame(rows)
                break
        dims.append({"id": dim_id, "name": dname, "codes": codes_df})
    return {"dimensions": dims, "dim_ids": [d["id"] for d in dims]}

# ------------------------------
# Helpers: data fetching (CSV)
# ------------------------------
@st.cache_data(show_spinner=False)
def build_series_keys_from_selection(flow_id: str, dim_ids: List[str], selected: Dict[str, List[str]]) -> List[str]:
    """Create series keys from a dict {DIM: [codes,...]}. Empty list -> wildcard for that DIM.
    Limits cartesian size to protect UI.
    """
    # If a dim has no selection, use [''] to wildcard that position
    ordered_lists = [ (selected.get(dim) or [""]) for dim in dim_ids ]
    # Limit
    total = 1
    for lst in ordered_lists:
        total *= max(1, len(lst))
        if total > 5000:
            raise RuntimeError("Selection expands to >5000 series keys. Narrow your filters.")
    keys = []
    for combo in itertools.product(*ordered_lists):
        keys.append(".".join(combo))
    return keys

@st.cache_data(show_spinner=False)
def fetch_series_csv(flow_id: str, series_key: str) -> pd.DataFrame:
    """Fetch full history for one series as wide frame with named column."""
    url = f"{BASE}/data/{flow_id}/{series_key}"
    r = requests.get(url, params={"detail":"dataonly", "format":"csvdata"}, headers={"Accept":"text/csv"}, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise RuntimeError("Unexpected CSV from ECB.")
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    df = df.sort_values("Date")
    # Compose label
    label = f"{flow_id}:{series_key}"
    for c in ["TITLE_COMPL","TITLE","UNIT","CURRENCY","CURRENCY_DENOM","EXR_TYPE","EXR_SUFFIX"]:
        if c in df.columns and pd.notna(df[c]).any():
            label = f"{label} — {str(df[c].dropna().iloc[0])}"
            break
    out = df[["Date","OBS_VALUE"]].rename(columns={"OBS_VALUE":label}).set_index("Date")
    return out

@st.cache_data(show_spinner=False)
def fetch_many(flow_id: str, series_keys: List[str]) -> pd.DataFrame:
    frames=[]
    for k in series_keys:
        try:
            frames.append(fetch_series_csv(flow_id, k))
        except Exception as e:
            st.warning(f"Failed to fetch {flow_id}/{k}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index.name = "Date"
    return df

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.markdown("### View options")
    norm = st.selectbox("Normalize", ["None","Rebase to 100 at start","Z-score (standardize)"])
    rm_presets = [3,6,12,24,60]
    rm_sel = st.multiselect("Rolling means (periods)", rm_presets, default=[6,12])
    rm_custom = st.text_input("Custom windows (comma-separated)")

# 1) Choose dataset
with st.spinner("Loading datasets..."):
    FLOWS = fetch_dataflows()

c1, c2 = st.columns([1,1])
with c1:
    st.subheader("Dataset")
    filter_text = st.text_input("Filter datasets", placeholder="e.g., EXR, HICP, exchange rate")
    df_show = FLOWS
    if filter_text.strip():
        q = filter_text.lower()
        df_show = df_show[df_show.apply(lambda r: q in str(r["name"]).lower() or q in str(r["flow_id"]).lower(), axis=1)]
    flow_label = st.selectbox("Select dataset", options=df_show["label"].tolist())
    FLOW = df_show.loc[df_show["label"]==flow_label, "flow_id"].iloc[0]
    st.caption(f"Flow selected: **{FLOW}**")

with c2:
    st.subheader("Build a series (by dimensions)")
    with st.spinner("Loading structure & code lists..."):
        DSD = fetch_dsd(FLOW)
    dim_ids = DSD.get("dim_ids", [])
    dim_defs = DSD.get("dimensions", [])
    if not dim_defs:
        st.error("Could not load structure for this dataset. Try another, or paste keys manually below.")
    SELECTED: Dict[str,List[str]] = {}
    # Render one multiselect per dimension
    for d in dim_defs:
        codes = d["codes"]
        if codes.empty:
            st.write(f"**{d['name']}** ({d['id']}): *no code list available* — leave empty to wildcard.")
            SELECTED[d['id']] = []
            continue
        options = (codes["code"] + " — " + codes["name"]).tolist()
        pick = st.multiselect(f"{d['name']} ({d['id']})", options=options, help="Leave empty to wildcard this dimension")
        SELECTED[d['id']] = [p.split(" — ")[0] for p in pick]

    # Manual key input (comma separated) still possible
    manual_keys_txt = st.text_input("Or paste full series keys (comma-separated)")
    manual_keys = [s.strip() for s in manual_keys_txt.split(',') if s.strip()]

# 2) Build keys and fetch
st.markdown("---")
series_keys: List[str] = []
try:
    if dim_ids:
        series_keys = build_series_keys_from_selection(FLOW, dim_ids, SELECTED)
except Exception as e:
    st.error(str(e))

# Merge manual keys and dedupe
if manual_keys:
    series_keys = list(dict.fromkeys(series_keys + manual_keys))

# Show a preview of key pattern
if dim_ids:
    preview = ".".join(["{"+d+"}" if not SELECTED.get(d) else "+".join(SELECTED[d]) for d in dim_ids])
    st.caption(f"Key template: `{FLOW}/{preview}` (empty braces mean wildcard)")

# Fetch button
fetch_now = st.button("Fetch & plot selected series", type="primary")

WIDE = pd.DataFrame()
if fetch_now and series_keys:
    with st.spinner("Fetching data..."):
        WIDE = fetch_many(FLOW, series_keys)
    if WIDE.empty:
        st.warning("No data returned for the current selection.")

if not WIDE.empty:
    # Normalization
    df_plot = WIDE.copy()
    if norm == "Rebase to 100 at start":
        df_plot = df_plot.apply(lambda s: (s / s.dropna().iloc[0]) * 100)
    elif norm == "Z-score (standardize)":
        df_plot = df_plot.apply(lambda s: (s - s.mean()) / s.std(ddof=0))

    fig = px.line(df_plot, x=df_plot.index, y=df_plot.columns, labels={"x":"Date","value":"Value","variable":"Series"})
    fig.update_layout(legend_title_text="Series", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Long/tidy export
    LONG = WIDE.reset_index().melt(id_vars=["Date"], var_name="Series", value_name="Value")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Preview (wide)")
        st.dataframe(WIDE.tail(20), use_container_width=True)
        st.download_button("Download wide CSV", WIDE.to_csv().encode("utf-8"), "ecb_wide.csv", "text/csv", use_container_width=True)
    with colB:
        st.subheader("Preview (long/tidy)")
        st.dataframe(LONG.tail(40), use_container_width=True)
        st.download_button("Download long CSV", LONG.to_csv(index=False).encode("utf-8"), "ecb_long.csv", "text/csv", use_container_width=True)

    # Rolling means (secondary view)
    st.markdown("### Rolling means")
    rms = rm_sel[:]
    if rm_custom.strip():
        try:
            extra = [int(x.strip()) for x in rm_custom.split(',') if x.strip()]
            rms = sorted(list(dict.fromkeys(rms + extra)))
        except Exception:
            st.warning("Could not parse custom windows; enter comma-separated integers.")
    if rms:
        # Apply to normalized plot frame
        long_frames = []
        raw_long = df_plot.reset_index().melt(id_vars=["Date"], var_name="Series", value_name="Value")
        raw_long["Line"] = "Raw"
        long_frames.append(raw_long)
        for w in rms:
            ma = df_plot.rolling(window=w, min_periods=w).mean()
            ma_long = ma.reset_index().melt(id_vars=["Date"], var_name="Series", value_name="Value")
            ma_long["Line"] = f"MA{w}"
            long_frames.append(ma_long)
        plot_df = pd.concat(long_frames, ignore_index=True)
        fig2 = px.line(plot_df, x="Date", y="Value", color="Series", line_dash="Line")
        fig2.update_layout(legend_title_text="Series / Type", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select a dataset, pick dimension values (or paste keys), then **Fetch & plot**.")

st.markdown(
    """
    <div style='text-align:center; opacity:0.7; font-size:0.9em;'>
    Built with <a href='https://streamlit.io' target='_blank'>Streamlit</a> · Powered by <a href='https://data.ecb.europa.eu' target='_blank'>ECB Data Portal</a>
    </div>
    """,
    unsafe_allow_html=True,
)
