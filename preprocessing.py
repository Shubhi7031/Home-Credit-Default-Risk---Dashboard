# preprocessing.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict

TARGET_COL = "TARGET"
ID_COLS = {"SK_ID_CURR"}   # <-- keep IDs here (add more if needed)

# ---------- helpers ----------
def _winsorize(series: pd.Series, lower=0.01, upper=0.99):
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi), {"low": float(lo), "high": float(hi)}

def _merge_rare_levels(s: pd.Series, min_share=0.01) -> tuple[pd.Series, dict]:
    """
    Merge categories with share < min_share into 'Other'.
    Keep 'XNA' explicit if present.
    """
    freq = s.value_counts(dropna=False, normalize=True)
    rare = freq[freq < min_share].index
    mapping = {str(k): ("Other" if k in rare else str(k)) for k in freq.index}
    if "XNA" in mapping:
        mapping["XNA"] = "XNA"
    s2 = s.astype("string").map(lambda x: mapping.get(str(x), str(x)))
    return s2, mapping

def _guard_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    r = numer.replace([np.inf, -np.inf], np.nan) / denom.replace([0, np.inf, -np.inf], np.nan)
    return r.replace([np.inf, -np.inf], np.nan)

def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().mean().sort_values(ascending=False) * 100
    return m.to_frame("missing_pct").reset_index(names="column")

# ---------- main ----------
def preprocess_application_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Steps:
    - type casting
    - derived: AGE_YEARS, EMPLOYMENT_YEARS, DTI, LTI, ANNUITY_TO_CREDIT
    - missingness report; drop >60% missing; impute numeric median / categorical mode
    - rare-label merge (<1%)  [EXCLUDES ID_COLS & TARGET]
    - winsorize 1% tails for skewed numerics
    - income brackets (Low, Mid, High)
    """
    df = df.copy()

    # --- IDs as strings (never category/rare-merged) ---
    for c in ID_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # --- Cast objects (EXCEPT IDs & TARGET) to category ---
    obj_cols = [c for c in df.columns
                if df[c].dtype == "object" and c not in ID_COLS and c != TARGET_COL]
    for c in obj_cols:
        df[c] = df[c].astype("category")

    # --- Derived features ---
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    if "DAYS_EMPLOYED" in df.columns:
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan  # special code = not employed
        df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["DTI"] = _guard_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["LTI"] = _guard_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])

    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["ANNUITY_TO_CREDIT"] = _guard_ratio(df["AMT_ANNUITY"], df["AMT_CREDIT"])

    # --- Missingness & drop >60% ---
    miss_tbl_before = summarize_missingness(df)
    drop_cols = miss_tbl_before.loc[miss_tbl_before["missing_pct"] > 60, "column"].tolist()
    # Never drop TARGET or IDs even if missing (for safety)
    drop_cols = [c for c in drop_cols if c not in ID_COLS and c != TARGET_COL]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # --- Column lists (exclude IDs & TARGET where appropriate) ---
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in ID_COLS and c != TARGET_COL]
    cat_cols = [c for c in df.columns
                if (pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == "object")
                and c not in ID_COLS and c != TARGET_COL]

    # --- Impute ---
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown").astype("category")

    # --- Rare-label merge (ONLY on cat_cols, i.e., not IDs/TARGET) ---
    rare_maps: Dict[str, Dict] = {}
    for c in cat_cols:
        merged, mapping = _merge_rare_levels(df[c].astype("string"), min_share=0.01)
        df[c] = merged.astype("category")
        rare_maps[c] = mapping

    # --- Winsorize skewed numerics for charts/KPIs ---
    winsor_fields = [x for x in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                                 "AMT_GOODS_PRICE", "DTI", "LTI"] if x in df.columns]
    winsor_params: Dict[str, Dict] = {}
    for c in winsor_fields:
        df[c], params = _winsorize(df[c].astype(float), 0.01, 0.99)
        winsor_params[c] = params

    # --- Income Brackets ---
    if "AMT_INCOME_TOTAL" in df.columns:
        q1 = df["AMT_INCOME_TOTAL"].quantile(0.25)
        q3 = df["AMT_INCOME_TOTAL"].quantile(0.75)
        def _br(x):
            if x <= q1: return "Low"
            if x > q3:  return "High"
            return "Mid"
        df["INCOME_BRACKET"] = df["AMT_INCOME_TOTAL"].apply(_br).astype("category")

    miss_tbl_after = summarize_missingness(df)

    artifacts = {
        "dropped_cols": drop_cols,
        "winsor_params": winsor_params,
        "missingness_before": miss_tbl_before.to_dict(orient="records"),
        "missingness_after": miss_tbl_after.to_dict(orient="records"),
        "rare_maps": rare_maps,
        "notes": [
            "IDs excluded from categorical/rare-merge.",
            "Imputed numeric=median; categorical=mode.",
            "Winsorized 1% tails for income/credit/annuity/goods, DTI, LTI.",
            "DAYS_EMPLOYED==365243 treated as missing (unemployed).",
            "Ratios guard against invalid denominators."
        ]
    }
    return df, artifacts
