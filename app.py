# ==========================================================
# BHB Study Finder — Dynamic Column Filters (repo CSV/root + AgGrid debug)
# ==========================================================
from io import BytesIO
import os, re, sys, traceback, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
import warnings

# ---------- Page ----------
st.set_page_config(page_title="BHB Study Finder", page_icon="🔬", layout="wide")
st.title("🔬 BHB Study Finder")

st.markdown("""
This is a tool that let's you filter BHB studies according to several criteria such as study model, tissue or organ of interest, organelle of interest, method of increasing BHB level and other. You can see all the filter categories in the left panel.

**Why is this better than searching through literature via pubmed and keywords?**

This filter tool uses AI to extract information from abstracts. The text exactly extracted from the abstract is searchable via the "Raw" filters. AI is then used again to put all of the raw output into more general and easy to search categories. For example, "HK-2 cells" is the raw text extracted from the abstract and it is then categorized under the specific "Renal tubular category" and broader "Epithelial - Renal & Urothelial". Similarly, extracted raw mechanisms like "NADH supply to respiratory chain" are categorized under broader category "Mitochondrial Function & Bioenergetics".

Processing of the extracted data should make much easier for researchers to find studies most relevant to their interest. As big part of BHB research focus on its signalling effect, AI was also used to extract proposed targets of BHB from each of the abstracts. Targets are standardized to their official gene names so they can be quickly used for enrichment analysis and other bioinformatics tools.
""")

# ---------- AgGrid import with diagnostics ----------
AGGRID_IMPORT_ERR = None
HAVE_AGGRID = False
ColumnsAutoSizeMode = None
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    try:
        from st_aggrid.shared import ColumnsAutoSizeMode
    except Exception:
        ColumnsAutoSizeMode = None  # older versions don't expose this
    HAVE_AGGRID = True
    print("[AgGrid] Import successful.")
except Exception as e:
    AGGRID_IMPORT_ERR = e
    HAVE_AGGRID = False
    AgGrid = GridOptionsBuilder = None
    print("[AgGrid] Import failed:", repr(e))
    print("[AgGrid] Traceback:\n", traceback.format_exc())

# ---------- Config ----------
DELIMS_PATTERN = r"[;,+/|]"
MAX_MULTISELECT_OPTIONS = 200
BOOL_TRUE  = {"true","1","yes","y","t"}
BOOL_FALSE = {"false","0","no","n","f"}
APP_DIR = Path(__file__).resolve().parent

# ---------- Tips (permanent) ----------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+","", s.lower())

TIPS = {
    _norm("BHB Filter"): "Filter via official gene names like **NLRP3**, **UCP1**, etc.",
    _norm("Model Raw"): "Study model as extracted from the abstract, e.g. **Wistar rat**, **healthy human adults**.",
    _norm("Model Global"): "Select **in vitro**, **animal**, or **human**.",
    _norm("Model Canonical"): "Categorized broad models such as **Ex vivo**, **Mouse**, **In silico**, etc.",
    _norm("Model Cannonical"): "Categorized broad models such as **Ex vivo**, **Mouse**, **In silico**, etc.",
    _norm("Mechanism Raw"): "Studied mechanism as extracted from the abstract, e.g. **lipolysis**, **mitochondrial complex II activation**.",
    _norm("Mechanism Canonical"): "Broader categories such as **Mitochondrial Function & Bioenergetics** or **Metabolic Regulation**.",
    _norm("Mechanism Cannonical"): "Broader categories such as **Mitochondrial Function & Bioenergetics** or **Metabolic Regulation**.",
}

# ---------- Helpers ----------
def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def is_id_like(colname: str) -> bool:
    n = normalize(colname)
    return n in {"abstractid","pmcid","id","aid"} or n.endswith("id")

def is_pmid_col(colname: str) -> bool:
    return normalize(colname) == "pmid"

def sanitize_key(s: str) -> str:
    return "flt_" + re.sub(r"[^a-zA-Z0-9_]", "_", s)

def tip_for(colname: str) -> str | None:
    return TIPS.get(_norm(colname))

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
    fmt = guess_datetime_format(series)
    s = safe_to_datetime(series, fmt)
    frac = s.notna().mean() if len(s) else 0.0
    return frac >= min_frac_dt

def coerce_datetime(series: pd.Series, fmt: str | None = None) -> pd.Series:
    if fmt is None:
        fmt = guess_datetime_format(series)
    return safe_to_datetime(series, fmt)


def coerce_datetime(series: pd.Series, fmt: str | None = None) -> pd.Series:
    if fmt is None:
        fmt = guess_datetime_format(series)
    return safe_to_datetime(series, fmt)

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

def discover_repo_csv() -> Path | None:
    if "DATASET_PATH" in st.secrets:
        p = (APP_DIR / st.secrets["DATASET_PATH"]).resolve()
        if p.exists(): return p
    if os.environ.get("DATASET_PATH"):
        p = (APP_DIR / os.environ["DATASET_PATH"]).resolve()
        if p.exists(): return p
    for name in ("bhb_studies.csv", "studies.csv", "dataset.csv"):
        p = (APP_DIR / name).resolve()
        if p.exists(): return p
    candidates = list(APP_DIR.glob("*.csv"))
    data_dir = APP_DIR / "data"
    if not candidates and data_dir.exists():
        candidates = list(data_dir.glob("*.csv"))
    return candidates[0].resolve() if candidates else None

def guess_datetime_format(series: pd.Series, sample_size: int = 500) -> str | None:
    """
    Try common date formats on a sample; return the best format if it parses ≥80% of values.
    """
    candidates = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",         # ISO without timezone
        "%Y-%m-%dT%H:%M:%S.%f",      # ISO with ms
    ]
    s = series.dropna().astype(str)
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=0)
    best_fmt, best_rate = None, 0.0
    for fmt in candidates:
        parsed = pd.to_datetime(s, errors="coerce", format=fmt, utc=False)
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_rate, best_fmt = rate, fmt
    return best_fmt if best_rate >= 0.8 else None

def safe_to_datetime(series: pd.Series, fmt: str | None) -> pd.Series:
    """
    Parse datetimes quietly (no pandas 2.3 warnings). If fmt is None, still parse but suppress
    'Could not infer format' UserWarnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*infer_datetime_format.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
        return pd.to_datetime(series, errors="coerce", format=fmt, utc=False)


# ---------- Data source (no dataset UI; auto-load from repo) ----------
APP_DIR = Path(__file__).resolve().parent
csv_path = discover_repo_csv()
if not csv_path:
    st.error("No dataset found. Put a CSV in the repo root (e.g., `bhb_studies.csv`) or set `DATASET_PATH` in Secrets.")
    st.stop()

df = pd.read_csv(csv_path, low_memory=False)
st.caption(f"Loaded dataset: `{csv_path.name}` • {len(df):,} rows, {df.shape[1]} columns")

# ---------- Sidebar: dynamic filters with permanent tips ----------
filters_meta = []
with st.sidebar:
    st.header("🔎 Column Filters")
    st.caption("Filters apply cumulatively.")
    st.button("🔁 Reset all filters", on_click=clear_all_filters)

    for col in df.columns:
        if is_pmid_col(col):  # keep column visible but don't render a filter for it
            continue

        series = df[col]
        keybase = sanitize_key(col)
        st.markdown(f"**{col}**")
        hint = tip_for(col)
        if hint:
            st.caption(hint)

        # ID-like (except PMID) → equals-any input
        if is_id_like(col):
            txt = st.text_input("Equals any of …", value="", key=keybase+"_idany")
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
    # Detect best format once and reuse (faster + no warnings)
            dt_fmt = guess_datetime_format(series)
            s_dt = coerce_datetime(series, dt_fmt)
            dmin = s_dt.min().date(); dmax = s_dt.max().date()
            date_range = st.date_input("Date range", (dmin, dmax), key=keybase+"_daterange")
            excl_na = st.checkbox("Exclude missing", value=False, key=keybase+"_exclna_dt")
            filters_meta.append({
                "col": col, "type": "date_range",
                "value": date_range, "excl_na": excl_na,
                "dt_format": dt_fmt  # <- store format so we can reuse during filtering
            })


        elif try_bool:
            choice = st.selectbox("Value", ["Any", "True", "False"], key=keybase+"_bool")
            filters_meta.append({"col": col, "type": "bool", "value": choice})

        else:
            tokens = tokenize_options(series.astype(str))
            if 0 < len(tokens) <= MAX_MULTISELECT_OPTIONS:
                sel = st.multiselect("Select", ["Any"] + tokens, default=["Any"], key=keybase+"_multi")
                filters_meta.append({"col": col, "type": "multi", "value": sel})
            else:
                query = st.text_input("Contains any of …", value="", key=keybase+"_contains")
                filters_meta.append({"col": col, "type": "contains_any", "value": query})

        st.divider()

# ---------- Apply filters ----------
mask = pd.Series([True] * len(df))
for f in filters_meta:
    col = f["col"]; typ = f["type"]; val = f["value"]
    if typ == "id_any":
        ids = parse_id_equals_any(val)
        if ids:
            mask &= df[col].astype(str).isin(ids)
    elif typ == "range":
        lo, hi = val
        s_num = coerce_numeric(df[col])
        cond = s_num.between(lo, hi)
        if not f.get("excl_na", False): cond = cond | s_num.isna()
        mask &= cond
    elif typ == "date_range":
        fmt = f.get("dt_format")
        s_dt = coerce_datetime(df[col], fmt)
        if isinstance(val, tuple) and len(val) == 2:
            lo, hi = pd.to_datetime(val[0]), pd.to_datetime(val[1])
            cond = s_dt.between(lo, hi)
        if not f.get("excl_na", False):
            cond = cond | s_dt.isna()
        mask &= cond
    elif typ == "bool":
        if val in ("True", "False"):
            s_b = coerce_bool(df[col]); want = (val == "True")
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

# Classic AgGrid look: paginated, readable column widths, horizontal scroll allowed
PAGE_SIZE = 20
GRID_HEIGHT = 600

if HAVE_AGGRID:
    gob = GridOptionsBuilder.from_dataframe(result)

    # Pagination like your original screenshot/code
    try:
        gob.configure_pagination(paginationAutoPageSize=False, paginationPageSize=PAGE_SIZE)
    except TypeError:
        gob.configure_pagination(paginationPageSize=PAGE_SIZE)

    # Column defaults: keep funnels, sorting, resizing; do NOT force-fit
    gob.configure_default_column(filter=True, sortable=True, resizable=True)

    # Make sure the grid doesn't expand to full height (which hides pagination)
    gob.configure_grid_options(domLayout="normal")

    # Build & render
    grid_opts = gob.build()
    AgGrid(
        result,
        gridOptions=grid_opts,
        height=GRID_HEIGHT,
        theme="alpine",
        fit_columns_on_grid_load=False,  # critical: no sizeColumnsToFit squish
        # If your st-aggrid exposes this import, FIT_CONTENTS keeps widths natural.
        columns_auto_size_mode=(
            ColumnsAutoSizeMode.FIT_CONTENTS if ColumnsAutoSizeMode else None
        ),
    )
else:
    st.info("Interactive grid unavailable (streamlit-aggrid not installed). Showing a simple table instead.")
    st.dataframe(result, use_container_width=True, height=GRID_HEIGHT)

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


# ---------- Debug panel (shows in UI AND prints to logs) ----------
with st.sidebar.expander("🪲 Debug", expanded=False):
    import platform
    st.write("**Python**:", sys.version.split()[0], platform.platform())
    st.write("**Streamlit**:", st.__version__)
    try:
        import importlib.metadata as ilm
        ag_ver = ilm.version("streamlit-aggrid")
        st.write("**streamlit-aggrid**:", ag_ver)
        print("[AgGrid] Detected streamlit-aggrid version:", ag_ver)
    except Exception as e:
        st.write("**streamlit-aggrid**: not installed or not detected:", repr(e))
        print("[AgGrid] streamlit-aggrid not detected:", repr(e))
    st.write("**HAVE_AGGRID**:", HAVE_AGGRID)
    if AGGRID_IMPORT_ERR:
        st.write("**Import error**:", repr(AGGRID_IMPORT_ERR))
        st.code(traceback.format_exc(), language="text")
