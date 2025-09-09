# utils/utils.py
"""
Shared helpers for the Streamlit Home Credit dashboard.

WHY THIS FILE?
- Keeps all common logic (filters, KPIs, correlation helpers) in one place.
- Each page stays small and easy to read.
- If you change a filter/KPI once here, ALL pages update.

HOW TO USE
----------
from utils.utils import (
    TARGET, TGT_LABELS,
    get_filtered_df, kpi, default_rate, by_category_rate,
    sample_df, compute_corr, bar_corr_to_target,
    missingness_topk, add_status_col, counts_repaid_default
)
"""

from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# CONSTANTS
# =============================================================================
# The label column. In Home Credit Application Train, TARGET=1 means default, 0=repaid.
TARGET: str = "TARGET"

# Human-friendly mapping for TARGET values (used for legends, labels).
TGT_LABELS: Dict[int, str] = {0: "Repaid", 1: "Default"}


# =============================================================================
# SMALL / GENERIC HELPERS
# =============================================================================
def kpi(label: str, value: Any, fmt: str = "{:,.2f}", help_text: Optional[str] = None) -> None:
    """
    Show a KPI tile with safe formatting.

    Parameters
    ----------
    label : str
        KPI title shown above the value.
    value : Any
        The numeric (or string) value to display.
    fmt : str, default "{:,.2f}"
        Format string for numeric values (thousand separators with 2 decimals).
        If you want a percent, pass fmt="{:.2f}%".
        If value is not numeric, it is shown as-is.
    help_text : str, optional
        Small help tooltip on the KPI.

    Notes
    -----
    - Gracefully handles NaN/None: shows "N/A".
    """
    is_num = isinstance(value, (int, float, np.number))
    if is_num:
        # Handle NaN/Inf nicely
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            st.metric(label, "N/A", help=help_text)
            return
        st.metric(label, fmt.format(value), help=help_text)
    else:
        st.metric(label, value if value is not None else "N/A", help=help_text)


def default_rate(target_series: pd.Series) -> float:
    """
    Compute default rate (%) from a 0/1 TARGET series.

    Returns
    -------
    float
        Default rate in percent. Example: 8.5 means 8.5% default rate.
    """
    return float(target_series.mean() * 100.0)


def by_category_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Compute default percentage by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with TARGET and the category column.
    col : str
        Column name to group by (e.g., "CODE_GENDER").

    Returns
    -------
    pd.DataFrame
        Columns: [col, "Default_%", "count"], sorted by Default_% desc.
        If inputs are missing, returns empty dataframe.
    """
    if TARGET not in df.columns or col not in df.columns:
        return pd.DataFrame(columns=[col, "Default_%", "count"])

    # .mean() on 0/1 gives the rate; multiply by 100 to get percentage
    t = (
        df.groupby(col, dropna=False)[TARGET]
          .mean().mul(100.0)
          .rename("Default_%")
          .reset_index()
    )
    # Add group sizes so charts/narratives can show n=
    t["count"] = df.groupby(col, dropna=False)[TARGET].size().values
    return t.sort_values("Default_%", ascending=False)


def sample_df(df: pd.DataFrame, n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Uniformly sample a dataframe for heavy plots (scatter/heatmaps).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already filtered).
    n : int, default 50_000
        Max rows to keep.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Same df if len(df) <= n, else a random sample of size n.
    """
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)


def _rerun() -> None:
    """Internal: Streamlit changed API name; support both."""
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# =============================================================================
# GLOBAL FILTERS (SIDEBAR)
# =============================================================================
def get_filtered_df(
    df: pd.DataFrame,
    section_title: str = "Global Filters",
    show_reset_button: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Render the global filter controls in the sidebar and return the filtered dataframe.

    This function ONLY renders controls for columns that exist in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe (clean/preprocessed).
    section_title : str, default "Global Filters"
        Sidebar header text.
    show_reset_button : bool, default True
        If True, show a "Reset Filters" button that reruns the app.

    Returns
    -------
    (filtered_df, values) : (pd.DataFrame, dict)
        filtered_df : pd.DataFrame
            A COPY of df after applying all selected filters.
        values : dict
            The user's selections (useful for showing active filter badges if you want).
            Keys include: "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE", "AGE_YEARS", "INCOME_BRACKET", "EMPLOYMENT_YEARS",
            "INCLUDE_UNEMPLOYED" (bool).
    """
    f = df.copy()
    values: Dict[str, Any] = {}

    st.sidebar.header(section_title)

    # ----- 1) Categorical filters (multi-selects) -----

    # Gender
    if "CODE_GENDER" in f.columns:
        genders = sorted(f["CODE_GENDER"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Gender", genders, default=genders)
        values["CODE_GENDER"] = sel
        if sel:
            f = f[f["CODE_GENDER"].isin(sel)]

    # Education
    if "NAME_EDUCATION_TYPE" in f.columns:
        edus = sorted(f["NAME_EDUCATION_TYPE"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Education", edus, default=edus)
        values["NAME_EDUCATION_TYPE"] = sel
        if sel:
            f = f[f["NAME_EDUCATION_TYPE"].isin(sel)]

    # Family Status
    if "NAME_FAMILY_STATUS" in f.columns:
        fams = sorted(f["NAME_FAMILY_STATUS"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Family Status", fams, default=fams)
        values["NAME_FAMILY_STATUS"] = sel
        if sel:
            f = f[f["NAME_FAMILY_STATUS"].isin(sel)]

    # Housing Type
    if "NAME_HOUSING_TYPE" in f.columns:
        hous = sorted(f["NAME_HOUSING_TYPE"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Housing Type", hous, default=hous)
        values["NAME_HOUSING_TYPE"] = sel
        if sel:
            f = f[f["NAME_HOUSING_TYPE"].isin(sel)]

    # ----- 2) Numeric / range filters (sliders) -----

    # Age (derived in preprocessing as AGE_YEARS)
    if "AGE_YEARS" in f.columns:
        a_min = float(np.floor(f["AGE_YEARS"].min()))
        a_max = float(np.ceil(f["AGE_YEARS"].max()))
        rng = st.sidebar.slider("Age (years)", a_min, a_max, (a_min, a_max), step=1.0)
        values["AGE_YEARS"] = rng
        f = f[(f["AGE_YEARS"] >= rng[0]) & (f["AGE_YEARS"] <= rng[1])]

    # Income bracket (categorical created in preprocessing: Low / Mid / High)
    if "INCOME_BRACKET" in f.columns:
        pref = ["Low", "Mid", "High"]
        present = [b for b in pref if b in f["INCOME_BRACKET"].dropna().unique().tolist()]
        sel = st.sidebar.multiselect("Income Bracket", present, default=present)
        values["INCOME_BRACKET"] = sel
        if sel:
            f = f[f["INCOME_BRACKET"].isin(sel)]

    # Employment tenure (years), with option to include those with missing tenure (unemployed)
    if "EMPLOYMENT_YEARS" in f.columns:
        emp_non_na = f["EMPLOYMENT_YEARS"].dropna()
        if not emp_non_na.empty:
            e_min = float(np.floor(emp_non_na.min()))
            e_max = float(np.ceil(emp_non_na.max()))
        else:
            e_min, e_max = 0.0, 0.0

        include_unemp = st.sidebar.checkbox("Include Unemployed (missing tenure)", value=True)
        rng = st.sidebar.slider("Employment Tenure (years)", e_min, e_max, (e_min, e_max), step=1.0)

        mask_range = f["EMPLOYMENT_YEARS"].between(rng[0], rng[1])
        if include_unemp:
            f = f[mask_range | f["EMPLOYMENT_YEARS"].isna()]
        else:
            f = f[mask_range & f["EMPLOYMENT_YEARS"].notna()]

        values["EMPLOYMENT_YEARS"] = rng
        values["INCLUDE_UNEMPLOYED"] = include_unemp

    # ----- 3) Reset button -----
    if show_reset_button and st.sidebar.button("Reset Filters"):
        _rerun()

    return f, values


# =============================================================================
# CORRELATION HELPERS
# =============================================================================
def compute_corr(df: pd.DataFrame, method: str = "spearman") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute correlations on numeric columns that have variance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already filtered).
    method : {"spearman","pearson"}, default "spearman"
        Spearman is rank-based (robust when distributions are skewed).
        Pearson is linear (sensitive to outliers but familiar).

    Returns
    -------
    (corr, tgt_corr) : (pd.DataFrame, pd.Series)
        corr : full correlation matrix across kept numeric columns.
        tgt_corr : correlation of each feature with TARGET (TARGET column removed).
                   Empty if TARGET is not in df or no numeric columns retained.

    Notes
    -----
    - We drop numeric columns with zero variance or with too few values to avoid NaNs.
    """
    # Pick numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Keep columns that have some variance and enough values
    keep: List[str] = []
    for c in num_cols:
        s = df[c].dropna()
        if s.size > 2 and s.std() > 0:
            keep.append(c)

    if not keep:
        return pd.DataFrame(), pd.Series(dtype=float)

    corr = df[keep].corr(method=method)

    # If TARGET is in the correlation matrix, pull its column and drop TARGET itself
    tgt_corr = corr[TARGET].drop(labels=[TARGET], errors="ignore") if TARGET in corr.columns else pd.Series(dtype=float)
    return corr, tgt_corr


def bar_corr_to_target(tgt_corr: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """
    Build a table of the top-N strongest absolute correlations to TARGET.

    Parameters
    ----------
    tgt_corr : pd.Series
        Series where index=feature name, value=corr(feature, TARGET).
    top_n : int, default 20
        Number of features to keep.

    Returns
    -------
    pd.DataFrame
        Columns: ["feature","corr","abs_corr"], sorted by abs_corr desc.
        Empty if tgt_corr is empty.
    """
    if tgt_corr is None or tgt_corr.empty:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])

    t = tgt_corr.dropna().to_frame("corr")
    t["abs_corr"] = t["corr"].abs()
    t = (
        t.sort_values("abs_corr", ascending=False)
         .head(top_n)
         .reset_index()
         .rename(columns={"index": "feature"})
    )
    return t


# =============================================================================
# DATA QUALITY HELPERS
# =============================================================================
def missingness_topk(artifacts: Dict, df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Return the top-k features by missing percentage.

    Parameters
    ----------
    artifacts : dict
        The 'artifacts' returned by preprocessing (session_state["artifacts"]).
        If it contains 'missingness_before', we use that (pre-imputation view).
    df : pd.DataFrame
        Fallback dataframe if artifacts are not available.
    k : int, default 20
        How many features to show.

    Returns
    -------
    pd.DataFrame
        Columns: ["column","missing_pct"], sorted desc by missing_pct.
    """
    # Preferred: use pre-imputation stats captured during preprocessing
    if artifacts and isinstance(artifacts.get("missingness_before"), list) and len(artifacts["missingness_before"]) > 0:
        miss_df = pd.DataFrame(artifacts["missingness_before"]).copy()
        miss_df = miss_df.sort_values("missing_pct", ascending=False).head(k)
        return miss_df[["column", "missing_pct"]]

    # Fallback: compute on the (possibly filtered) dataframe
    miss = df.isna().mean().mul(100.0).sort_values(ascending=False)
    return miss.head(k).reset_index().rename(columns={"index": "column", 0: "missing_pct"})


# =============================================================================
# TARGET STATUS HELPERS
# =============================================================================
def add_status_col(df: pd.DataFrame, status_col: str = "Status") -> pd.DataFrame:
    """
    Return a copy of df with an extra 'Status' column mapped from TARGET (0/1 -> Repaid/Default).

    Useful when you need a color legend in plots based on TARGET.

    Parameters
    ----------
    df : pd.DataFrame
    status_col : str, default "Status"

    Returns
    -------
    pd.DataFrame
    """
    if TARGET not in df.columns:
        return df.copy()
    out = df.copy()
    out[status_col] = out[TARGET].map(TGT_LABELS)
    return out


def counts_repaid_default(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a robust 2-column dataframe for counts of Repaid/Default.

    Why needed?
    -----------
    Pandas changed the column names returned by value_counts().reset_index() in some versions,
    which can break code that expects a specific column name. This helper guarantees:
      Columns: ["Status", "Count"]

    Parameters
    ----------
    df : pd.DataFrame
        Must contain TARGET.

    Returns
    -------
    pd.DataFrame
        Two columns: "Status" (Repaid/Default) and "Count" (int).
        Empty if TARGET not found.
    """
    if TARGET not in df.columns:
        return pd.DataFrame(columns=["Status", "Count"])

    t = (
        df[TARGET].map(TGT_LABELS)   # map 0/1 -> "Repaid"/"Default"
          .value_counts(dropna=False)
          .rename_axis("Status")     # the index name becomes a real column on reset_index
          .reset_index(name="Count") # force the counts column to be called "Count"
    )
    return t
