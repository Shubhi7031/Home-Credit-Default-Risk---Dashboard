# pages/5_Correlations_and_drivers.py
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except ModuleNotFoundError:
    st.error("Plotly is required. Install:  pip install plotly")
    st.stop()

from utils.utils import (
    TARGET, TGT_LABELS,
    get_filtered_df, kpi, sample_df,
    compute_corr, bar_corr_to_target, by_category_rate
)

st.title("Correlations, Drivers & Slice-and-Dice")

if "clean_df" not in st.session_state:
    st.warning("Cleaned data not found. Open the **Home** page first to load & preprocess the dataset.")
    st.stop()

df = st.session_state["clean_df"]
fdf, _ = get_filtered_df(df)
st.caption(f"Filtered sample size: **{len(fdf):,}** rows (of {len(df):,})")

if len(fdf) == 0:
    st.warning("No rows after applying filters. Adjust filters to see results.")
    st.stop()

# Options
st.subheader("Correlation Options")
method = st.radio("Method", ["spearman", "pearson"], index=0, horizontal=True)
topn = st.slider("Top-N features for bar chart", 5, 40, 20, step=1)

# Correlations
corr, tgt_corr = compute_corr(fdf, method=method)

# =================== KPIs (10) ===================
if not tgt_corr.empty:
    sorted_corr = tgt_corr.sort_values(ascending=False)
    top_pos = ", ".join([f"{i} ({v:.2f})" for i, v in sorted_corr.head(5).items()])
    top_neg = ", ".join([f"{i} ({v:.2f})" for i, v in sorted_corr.tail(5).items()])
else:
    top_pos = top_neg = "N/A"

def _most(colname):
    if corr.empty or colname not in corr.columns: return "N/A"
    s = corr[colname].drop(labels=[colname], errors="ignore").abs().sort_values(ascending=False)
    if s.empty: return "N/A"
    best = s.index[0]
    return f"{best} ({corr.loc[best, colname]:.2f})"

most_with_income = _most("AMT_INCOME_TOTAL")
most_with_credit = _most("AMT_CREDIT")
inc_credit_corr = float(corr.loc["AMT_INCOME_TOTAL", "AMT_CREDIT"]) if {"AMT_INCOME_TOTAL","AMT_CREDIT"}.issubset(corr.columns) else np.nan
age_tgt_corr = float(tgt_corr.get("AGE_YEARS", np.nan))
emp_tgt_corr = float(tgt_corr.get("EMPLOYMENT_YEARS", np.nan))
fam_tgt_corr = float(tgt_corr.get("CNT_FAM_MEMBERS", np.nan))
var_explained = float(100.0 * tgt_corr.abs().sort_values(ascending=False).head(5).sum() / tgt_corr.abs().sum()) if not tgt_corr.empty and tgt_corr.abs().sum() > 0 else np.nan
strong_n = int((tgt_corr.abs() > 0.5).sum()) if not tgt_corr.empty else 0

c = st.columns(5)
with c[0]: kpi("Top 5 +Corr with TARGET", top_pos, fmt="{}")
with c[1]: kpi("Top 5 −Corr with TARGET", top_neg, fmt="{}")
with c[2]: kpi("Most correlated with Income", most_with_income, fmt="{}")
with c[3]: kpi("Most correlated with Credit", most_with_credit, fmt="{}")
with c[4]: kpi("Corr(Income, Credit)", inc_credit_corr)

c = st.columns(5)
with c[0]: kpi("Corr(Age, TARGET)", age_tgt_corr)
with c[1]: kpi("Corr(Employment Yrs, TARGET)", emp_tgt_corr)
with c[2]: kpi("Corr(Family Size, TARGET)", fam_tgt_corr)
with c[3]: kpi("Variance Explained by Top 5 |corr| (%)", var_explained)
with c[4]: kpi("|corr(feature, TARGET)| > 0.5 (count)", strong_n, fmt="{:,}")

st.divider()

# =================== Charts (10) ===================
sel_cols = [c for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DTI","LTI",
                        "AGE_YEARS","EMPLOYMENT_YEARS","CNT_FAM_MEMBERS","CNT_CHILDREN", TARGET] if c in fdf.columns]
if len(sel_cols) >= 2:
    hm = fdf[sel_cols].corr(method=method)
    st.plotly_chart(px.imshow(hm, text_auto=True, aspect="auto",
                              title=f"{method.title()} Correlation — Selected Numerics"),
                    use_container_width=True)

ct = bar_corr_to_target(tgt_corr, top_n=topn)
if not ct.empty:
    st.plotly_chart(px.bar(ct, x="feature", y="abs_corr",
                           title=f"|Correlation| to TARGET — Top {topn} (method={method})",
                           text_auto=".2f"),
                    use_container_width=True)

fdf_s = sample_df(fdf, n=40000)

if {"AGE_YEARS","AMT_CREDIT", TARGET}.issubset(fdf_s.columns):
    tmp = fdf_s[["AGE_YEARS","AMT_CREDIT", TARGET]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.scatter(tmp, x="AGE_YEARS", y="AMT_CREDIT", color="Status", opacity=0.35,
                               title="Age vs Credit (colored by Target)"),
                    use_container_width=True)

if {"AGE_YEARS","AMT_INCOME_TOTAL", TARGET}.issubset(fdf_s.columns):
    tmp = fdf_s[["AGE_YEARS","AMT_INCOME_TOTAL", TARGET]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.scatter(tmp, x="AGE_YEARS", y="AMT_INCOME_TOTAL", color="Status", opacity=0.35,
                               title="Age vs Income (colored by Target)"),
                    use_container_width=True)

if {"EMPLOYMENT_YEARS", TARGET}.issubset(fdf_s.columns):
    tmp = fdf_s[["EMPLOYMENT_YEARS", TARGET]].dropna().copy()
    rng = np.random.default_rng(42)
    tmp["TARGET_JIT"] = tmp[TARGET] + rng.uniform(-0.05, 0.05, size=len(tmp))
    st.plotly_chart(px.scatter(tmp, x="EMPLOYMENT_YEARS", y="TARGET_JIT",
                               title="Employment Years vs TARGET (with jitter)"),
                    use_container_width=True)

if {"AMT_CREDIT","NAME_EDUCATION_TYPE"}.issubset(fdf.columns):
    st.plotly_chart(px.box(fdf, x="NAME_EDUCATION_TYPE", y="AMT_CREDIT", points=False,
                           title="Credit by Education"),
                    use_container_width=True)

if {"AMT_INCOME_TOTAL","NAME_FAMILY_STATUS"}.issubset(fdf.columns):
    st.plotly_chart(px.box(fdf, x="NAME_FAMILY_STATUS", y="AMT_INCOME_TOTAL", points=False,
                           title="Income by Family Status"),
                    use_container_width=True)

pp_cols = [c for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY"] if c in fdf_s.columns]
if len(pp_cols) >= 2 and TARGET in fdf_s.columns:
    tmp = fdf_s[pp_cols + [TARGET]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.scatter_matrix(tmp, dimensions=pp_cols, color="Status",
                                      title="Scatter Matrix — Income, Credit, Annuity (sampled)"),
                    use_container_width=True)

if {"CODE_GENDER", TARGET}.issubset(fdf.columns):
    g = by_category_rate(fdf, "CODE_GENDER")
    st.plotly_chart(px.bar(g, x="CODE_GENDER", y="Default_%", text_auto=".2f",
                           title="Default Rate by Gender (responds to sidebar filters)"),
                    use_container_width=True)

if {"NAME_EDUCATION_TYPE", TARGET}.issubset(fdf.columns):
    e = by_category_rate(fdf, "NAME_EDUCATION_TYPE")
    st.plotly_chart(px.bar(e, x="NAME_EDUCATION_TYPE", y="Default_%", text_auto=".2f",
                           title="Default Rate by Education (responds to sidebar filters)"),
                    use_container_width=True)

st.caption("Correlations use your selected method (Spearman default). Numeric fields were winsorized during preprocessing; heavy plots use a sampled subset for speed.")

st.subheader("Narrative Insights")
ins = []
if not tgt_corr.empty:
    fp = tgt_corr.sort_values(ascending=False).head(1)
    if not fp.empty:
        f, v = fp.index[0], fp.iloc[0]
        ins.append(f"- **Highest positive correlation with default**: **{f}** ({v:.2f}). Consider caps/extra docs for high values.")
    fn = tgt_corr.sort_values(ascending=True).head(1)
    if not fn.empty:
        f, v = fn.index[0], fn.iloc[0]
        ins.append(f"- **Most protective factor**: **{f}** ({v:.2f}). Consider leniency where this is strong.")
if {"LTI","DTI", TARGET}.issubset(fdf.columns):
    high_lti = fdf["LTI"] > fdf["LTI"].quantile(0.75)
    high_dti = fdf["DTI"] > fdf["DTI"].quantile(0.75)
    seg = high_lti & high_dti
    if seg.any():
        dr_seg = 100 * fdf.loc[seg, TARGET].mean()
        dr_all = 100 * fdf[TARGET].mean()
        ins.append(f"- **High LTI × High DTI** cohort default ~**{dr_seg:.2f}%** vs overall **{dr_all:.2f}%**.")
st.markdown("\n".join(ins) if ins else "- No dominant drivers under current filters; try Pearson or adjust Top-N.")
