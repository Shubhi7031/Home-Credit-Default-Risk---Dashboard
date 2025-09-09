# pages/4_Financial_health.py
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except ModuleNotFoundError:
    st.error("Plotly is required. Install:  pip install plotly")
    st.stop()

from utils.utils import TARGET, TGT_LABELS, get_filtered_df, kpi, sample_df

HIGH_CREDIT_THRESH = 1_000_000

st.title("Financial Health & Affordability")

if "clean_df" not in st.session_state:
    st.warning("Cleaned data not found. Open the **Home** page first to load & preprocess the dataset.")
    st.stop()

df = st.session_state["clean_df"]
fdf, _ = get_filtered_df(df)
st.caption(f"Filtered sample size: **{len(fdf):,}** rows (of {len(df):,})")

if len(fdf) == 0:
    st.warning("No rows after applying filters. Adjust filters to see results.")
    st.stop()

# =================== KPIs (10) ===================
_mean = lambda s: float(s.mean()) if isinstance(s, pd.Series) and s.size > 0 else np.nan
avg_income  = _mean(fdf["AMT_INCOME_TOTAL"]) if "AMT_INCOME_TOTAL" in fdf.columns else np.nan
med_income  = float(fdf["AMT_INCOME_TOTAL"].median()) if "AMT_INCOME_TOTAL" in fdf.columns else np.nan
avg_credit  = _mean(fdf["AMT_CREDIT"]) if "AMT_CREDIT" in fdf.columns else np.nan
avg_annuity = _mean(fdf["AMT_ANNUITY"]) if "AMT_ANNUITY" in fdf.columns else np.nan
avg_goods   = _mean(fdf["AMT_GOODS_PRICE"]) if "AMT_GOODS_PRICE" in fdf.columns else np.nan
avg_dti     = _mean(fdf["DTI"]) if "DTI" in fdf.columns else np.nan
avg_lti     = _mean(fdf["LTI"]) if "LTI" in fdf.columns else np.nan

income_gap = ( _mean(fdf.loc[fdf[TARGET]==0, "AMT_INCOME_TOTAL"]) - _mean(fdf.loc[fdf[TARGET]==1, "AMT_INCOME_TOTAL"]) ) if TARGET in fdf.columns and "AMT_INCOME_TOTAL" in fdf.columns else np.nan
credit_gap = ( _mean(fdf.loc[fdf[TARGET]==0, "AMT_CREDIT"])        - _mean(fdf.loc[fdf[TARGET]==1, "AMT_CREDIT"]) )        if TARGET in fdf.columns and "AMT_CREDIT" in fdf.columns else np.nan
pct_high_credit = 100.0 * fdf["AMT_CREDIT"].gt(HIGH_CREDIT_THRESH).mean() if "AMT_CREDIT" in fdf.columns else np.nan

c = st.columns(5)
with c[0]: kpi("Avg Annual Income", avg_income)
with c[1]: kpi("Median Annual Income", med_income)
with c[2]: kpi("Avg Credit Amount", avg_credit)
with c[3]: kpi("Avg Annuity", avg_annuity)
with c[4]: kpi("Avg Goods Price", avg_goods)
c = st.columns(5)
with c[0]: kpi("Avg DTI", avg_dti)
with c[1]: kpi("Avg LTI", avg_lti)
with c[2]: kpi("Income Gap (Non-def − Def)", income_gap)
with c[3]: kpi("Credit Gap (Non-def − Def)", credit_gap)
with c[4]: kpi(f"% High Credit (> {HIGH_CREDIT_THRESH:,.0f})", pct_high_credit, fmt="{:.2f}%")

st.divider()

# =================== Charts (10) ===================
if "AMT_INCOME_TOTAL" in fdf.columns:
    st.plotly_chart(px.histogram(fdf, x="AMT_INCOME_TOTAL", nbins=60, title="Income Distribution"),
                    use_container_width=True)
if "AMT_CREDIT" in fdf.columns:
    st.plotly_chart(px.histogram(fdf, x="AMT_CREDIT", nbins=60, title="Credit Amount Distribution"),
                    use_container_width=True)
if "AMT_ANNUITY" in fdf.columns:
    st.plotly_chart(px.histogram(fdf, x="AMT_ANNUITY", nbins=60, title="Annuity Distribution"),
                    use_container_width=True)

# sample for heavy plots
fdf_s = sample_df(fdf, n=50000)

if {"AMT_INCOME_TOTAL","AMT_CREDIT"}.issubset(fdf_s.columns):
    st.plotly_chart(px.scatter(fdf_s, x="AMT_INCOME_TOTAL", y="AMT_CREDIT", opacity=0.35,
                               title="Income vs Credit (sampled)"),
                    use_container_width=True)
if {"AMT_INCOME_TOTAL","AMT_ANNUITY"}.issubset(fdf_s.columns):
    st.plotly_chart(px.scatter(fdf_s, x="AMT_INCOME_TOTAL", y="AMT_ANNUITY", opacity=0.35,
                               title="Income vs Annuity (sampled)"),
                    use_container_width=True)

if {"AMT_CREDIT", TARGET}.issubset(fdf.columns):
    tmp = fdf[[TARGET, "AMT_CREDIT"]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.box(tmp, x="Status", y="AMT_CREDIT", points=False, title="Credit by Target"),
                    use_container_width=True)

if {"AMT_INCOME_TOTAL", TARGET}.issubset(fdf.columns):
    tmp = fdf[[TARGET, "AMT_INCOME_TOTAL"]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.box(tmp, x="Status", y="AMT_INCOME_TOTAL", points=False, title="Income by Target"),
                    use_container_width=True)

if {"AMT_INCOME_TOTAL","AMT_CREDIT"}.issubset(fdf_s.columns):
    st.plotly_chart(px.density_heatmap(fdf_s, x="AMT_INCOME_TOTAL", y="AMT_CREDIT",
                                       nbinsx=40, nbinsy=40, title="Joint Density: Income × Credit (sampled)"),
                    use_container_width=True)

if "INCOME_BRACKET" in fdf.columns and TARGET in fdf.columns:
    t = (fdf.groupby("INCOME_BRACKET")[TARGET].mean().mul(100.0)
           .rename("Default_%").reset_index())
    order = [b for b in ["Low","Mid","High"] if b in t["INCOME_BRACKET"].tolist()]
    t["INCOME_BRACKET"] = pd.Categorical(t["INCOME_BRACKET"], categories=order, ordered=True)
    t = t.sort_values("INCOME_BRACKET")
    st.plotly_chart(px.bar(t, x="INCOME_BRACKET", y="Default_%", text_auto=".2f",
                           title="Default Rate by Income Bracket"),
                    use_container_width=True)

corr_cols = [c for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DTI","LTI", TARGET] if c in fdf.columns]
if len(corr_cols) >= 2:
    corr = fdf[corr_cols].corr(method="spearman")
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto",
                              title="Spearman Correlation — Financial Variables & Target"),
                    use_container_width=True)

st.caption("Numeric fields were winsorized during preprocessing; heavy plots use a sampled subset for speed.")

st.subheader("Narrative Insights")
ins = []
if "LTI" in fdf.columns and TARGET in fdf.columns:
    thr = 6.0
    high = fdf["LTI"] > thr
    if high.any():
        dr_high = 100 * fdf.loc[high, TARGET].mean()
        dr_rest = 100 * fdf.loc[~high, TARGET].mean()
        ins.append(f"- **LTI > {thr}** segment default rate **{dr_high:.2f}%** vs **{dr_rest:.2f}%**.")

if "DTI" in fdf.columns and TARGET in fdf.columns:
    thr = 0.35
    high = fdf["DTI"] > thr
    if high.any():
        dr_high = 100 * fdf.loc[high, TARGET].mean()
        dr_rest = 100 * fdf.loc[~high, TARGET].mean()
        ins.append(f"- **DTI > {thr}** shows elevated risk: **{dr_high:.2f}%** vs **{dr_rest:.2f}%**.")

if {"AMT_CREDIT","AMT_INCOME_TOTAL", TARGET}.issubset(fdf.columns):
    low_inc = fdf["INCOME_BRACKET"].eq("Low") if "INCOME_BRACKET" in fdf.columns else fdf["AMT_INCOME_TOTAL"] <= fdf["AMT_INCOME_TOTAL"].quantile(0.25)
    high_credit = fdf["AMT_CREDIT"] > fdf["AMT_CREDIT"].quantile(0.75)
    mask = low_inc & high_credit
    if mask.any():
        dr_seg = 100 * fdf.loc[mask, TARGET].mean()
        ins.append(f"- **Low income × High credit** cohort has default ~**{dr_seg:.2f}%**. Consider policy caps or stricter docs.")
st.markdown("\n".join(ins) if ins else "- Under current filters, affordability thresholds don’t show strong divergence.")
