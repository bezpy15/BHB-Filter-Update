# ==========================================================
# BHB Study Finder — Dynamic Column Filters (repo CSV)
# ==========================================================
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder

# ---------- Page ----------
st.set_page_config(page_title="BHB Study Finder", page_icon="🔬", layout="wide")
st.title("🔬 BHB Study Finder")

st.markdown("""
This is a tool that let's you filter BHB studies according to several criteria such as study model, tissue or organ of interest, organelle of interest, method of increasing BHB level and other. You can see all the filter categories in the left panel.

**Why is this better than searching through literature via pubmed and keywords?**

This filter tool uses AI to extract information from abstracts. The text exactly extracted from the abstract is searchable via the "Raw" filters. AI is then used again to put all of the raw output into more general and easy to search categories. For example, "HK-2 cells" is the raw text extracted from the abstract and it is then categorized under the specific "Renal tubular category" and broader "Epithelial - Renal & Urothelial". Similarly, extracted raw mechanisms like "NADH supply to respiratory chain" are categorized under broader category "Mitochondrial Function & Bioenergetics".

Processing of the extracted data should make much easier for researchers to find studies most relevant to their interest. As big part of BHB research focus on its signalling effect, AI was also used to extract proposed targets of BHB from each of the abstracts. Targets are standardized to their official gene names so they can be quickly used for enrichment analysis and other bioinformatics tools.
""")

# ---------- Config ----------
DELIMS_PATTERN = r"[;,+/|]"
MAX_MULTISELECT_OPTIONS = 200
BOOL_TRUE  = {"true","1","yes","y","t"}
BOOL_FALSE = {"false","0","no","n","f"}

# Path inside your repo (commit your CSV here)
DEFAULT_LOCAL_PATH = os.environ.get("DATASET_PATH", "data/bhb_studies.csv")
# You can also set DATASET_PATH in Streamlit Secrets if you prefer:
if "DATASET_PATH" in st.secrets:
    DEFAULT_LOCAL_PATH = st.secrets["DATASET_PATH"]

# ---------- Helpers ----------
def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def is_id_like(colname: str) -> bool:
    n = normalize(colname)
    return n in {"pmid","abstractid","pmcid","id","aid"} or n.endswith("id")

def sanitize_key(s: str) -> str:
    return "flt_" + re.sub(r"[^a-zA-Z0-9_]", "_", s)

def tokenize_options(series: pd.Series) -> list:
    vals = set()
    for v in series.dropna().astype(str):
        for tok in re.split(DELIMS_PATTERN, v):
            tok = tok.strip()
            if tok:
                vals.add(tok)
    return sorted(vals)

def match_tokens(cell, selected_set):
    if not selected_set:
        return True
    if pd.isna(cell):
        return False
    toks = {t.strip() for t in re.split(DELIMS_PATTERN, str(cell)) if t.strip()}
    return bool(toks & selected_set)

def to_excel_bytes(df: pd.DataFrame):
    bio = BytesIO()
    df.to_excel(bio, index=False)
    return bio.getvalue()

def is_numeric_series(series: pd.Series, min_frac_numeric: float = 0.8) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    frac = s.notna().mean() if len(s) else 0.0
    return frac >= min_frac_numeric

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def is_datetime_series(series: pd.Series, min_frac_dt: float = 0.8) -> bool:
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, utc=False)
    frac = s.notna().mean() if len(s) else 0.0
    return frac >= min_frac_dt

def coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True, utc=False)

def is_booleanish_series(series: pd.Series, min_frac_bool: float = 0.9) -> bool:
    s = series.dropna().astype(str).str.strip().str.lower()
    ok = s.isin(BOOL_TRUE | BOOL_FALSE)
    frac = ok.mean() if len(s) else 0.0
    return frac >= min_frac_bool

def coerce_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map(lambda x: True if x in BOOL_TRUE else (False if x in BOOL_FALSE else np.nan))

def parse_id_equals_any(text: str) -> set:
    if not text:
        return set()
    parts = re.split(r"[;\s,]+", text)
    return {p.strip() for p in parts if p.strip()}

def clear_all_filters():
    for k in list(st.session_state.keys()):
        if k.startswith("flt_"):
            del st.session_state[k]
    st.rerun()

@st.cache_data(show_spinner=False)
def load_local_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. Commit your CSV there or set DATASET_PATH.")
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    else:
        return pd.read_excel(path)

# ---------- Data source (repo file by default; optional upload override) ----------
with st.sidebar:
    st.header("📦 Dataset")
    st.caption("By default, the app loads a CSV committed to your repo.")
    st.write(f"**Path:** `{DEFAULT_LOCAL_PATH}`")
    uploaded = st.file_uploader("Override (CSV/Excel, optional)", type=["csv", "xlsx", "xls"])
    st.button("🔁 Reset all filters", on_click=clear_all_filters)

df = None
src_label = ""

if uploaded is not None:
    name_lower = uploaded.name.lower()
    try:
        if name_lower.endswith(".csv"):
            df = pd.read_csv(uploaded, low_memory=False)
        else:
            df = pd.read_excel(uploaded)
        src_label = f"Loaded: {uploaded.name}"
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if df is None:
    try:
        df = load_local_file(DEFAULT_LOCAL_PATH)
        src_label = f"Loaded: {DEFAULT_LOCAL_PATH}"
    except Exception as e:
        st.error(str(e))
        st.stop()

st.caption(src_label)

# ---------- Sidebar: dynamic filters (ALL in left menu) ----------
filters_meta = []
with st.sidebar:
    st.header("🔎 Column Filters")
    st.caption("Filters apply cumulatively.")

    for col in df.columns:
        series = df[col]
        keybase = sanitize_key(col)
        st.markdown(f"**{col}**")

        # ID-like columns → equals-any input (no sliders)
        if is_id_like(col):
            help_txt = "Enter one or more IDs separated by spaces, commas, or ';' (matches any)."
            txt = st.text_input("Equals any of …", value="", key=keybase+"_idany", help=help_txt)
            filters_meta.append({"col": col, "type": "id_any", "value": txt})
            st.divider()
            continue

        # Type detection: numeric > datetime > booleanish > text/categorical
        try_numeric = is_numeric_series(series)
        try_dt = False if try_numeric else is_datetime_series(series)
        if try_dt and not coerce_datetime(series).notna().any():
            try_dt = False
        try_bool = False if (try_numeric or try_dt) else is_booleanish_series(series)

        if try_numeric:
            s_num = coerce_numeric(series)
            if s_num.notna().any():
                vmin = float(np.nanmin(s_num)); vmax = float(np.nanmax(s_num))
            else:
                vmin = 0.0; vmax = 0.0
            rng = st.slider("Range", min_value=float(vmin), max_value=float(vmax),
                            value=(float(vmin), float(vmax)), key=keybase+"_range")
            excl_na = st.checkbox("Exclude missing", value=False, key=keybase+"_exclna")
            filters_meta.append({"col": col, "type": "range", "value": rng, "excl_na": excl_na})

        elif try_dt:
            s_dt = coerce_datetime(series)
            dmin = s_dt.min().date(); dmax = s_dt.max().date()
            date_range = st.date_input("Date range", (dmin, dmax), key=keybase+"_daterange")
            excl_na = st.checkbox("Exclude missing", value=False, key=keybase+"_exclna_dt")
            filters_meta.append({"col": col, "type": "date_range", "value": date_range, "excl_na": excl_na})

        elif try_bool:
            choice = st.selectbox("Value", ["Any", "True", "False"], key=keybase+"_bool")
            filters_meta.append({"col": col, "type": "bool", "value": choice})

        else:
            tokens = tokenize_options(series.astype(str))
            if 0 < len(tokens) <= MAX_MULTISELECT_OPTIONS:
                opts = ["Any"] + tokens
                sel = st.multiselect("Select", opts, default=["Any"], key=keybase+"_multi")
                filters_meta.append({"col": col, "type": "multi", "value": sel})
            else:
                help_txt = "Type one or more values separated by ';'. Matches if any appear (case-insensitive)."
                query = st.text_input("Contains any of …", value="", help=help_txt, key=keybase+"_contains")
                filters_meta.append({"col": col, "type": "contains_any", "value": query})

        st.divider()

# ---------- Apply filters ----------
mask = pd.Series([True] * len(df))

for f in filters_meta:
    col = f["col"]
    typ = f["type"]
    val = f["value"]

    if typ == "id_any":
        ids = parse_id_equals_any(val)
        if ids:
            mask &= df[col].astype(str).isin(ids)

    elif typ == "range":
        lo, hi = val
        s_num = coerce_numeric(df[col])
        cond = s_num.between(lo, hi)
        if not f.get("excl_na", False):
            cond = cond | s_num.isna()
        mask &= cond

    elif typ == "date_range":
        s_dt = coerce_datetime(df[col])
        if isinstance(val, tuple) and len(val) == 2:
            lo, hi = pd.to_datetime(val[0]), pd.to_datetime(val[1])
            cond = s_dt.between(lo, hi)
            if not f.get("excl_na", False):
                cond = cond | s_dt.isna()
            mask &= cond

    elif typ == "bool":
        if val in ("True", "False"):
            s_b = coerce_bool(df[col])
            want = True if val == "True" else False
            mask &= (s_b == want)

    elif typ == "multi":
        sel = [s for s in val if s != "Any"]
        if sel:
            sel_set = set(sel)
            mask &= df[col].apply(lambda v: match_tokens(v, sel_set))

    elif typ == "contains_any":
        query = str(val).strip()
        if query:
            needles = [q.strip() for q in query.split(";") if q.strip()]
            patt = "|".join(re.escape(q) for q in needles)
            mask &= df[col].astype(str).str.contains(patt, case=False, na=False, regex=True)

result = df.loc[mask].copy()

# ---------- Results + downloads ----------
st.subheader(f"📑 {len(result)} row{'s' if len(result)!=1 else ''} match your filters")

gob = GridOptionsBuilder.from_dataframe(result)
gob.configure_pagination(paginationPageSize=20)
gob.configure_default_column(filter=True, sortable=True, resizable=True)
AgGrid(result, gridOptions=gob.build(), height=450, theme="alpine")

st.download_button(
    "💾 Excel",
    to_excel_bytes(result),
    "filtered_rows.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.download_button(
    "🗒️ CSV",
    result.to_csv(index=False),
    "filtered_rows.csv",
    mime="text/csv",
)
