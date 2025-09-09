# pages/1_Overview_and_Data_Quality.py
import streamlit as st
import pandas as pd
import numpy as np

# plotly guard
try:
    import plotly.express as px
except ModuleNotFoundError:
    st.error("Plotly is required. Install:  pip install plotly")
    st.stop()

from utils.utils import (
    TARGET, get_filtered_df, kpi, missingness_topk, counts_repaid_default
)

st.title("Overview & Data Quality")

# Ensure Home.py loaded data
if "clean_df" not in st.session_state:
    st.warning("Cleaned data not found. Open the **Home** page first to load & preprocess the dataset.")
    st.stop()

df = st.session_state["clean_df"]
artifacts = st.session_state.get("artifacts", {})

# Global filters (shared)
fdf, _filters = get_filtered_df(df)
st.caption(f"Filtered sample size: **{len(fdf):,}** rows out of **{len(df):,}** total")

# ===================== KPIs (10) =====================
c = st.columns(5)
with c[0]: kpi("Total Applicants", len(fdf), fmt="{:,}")
with c[1]: kpi("Default Rate (%)", 100 * fdf[TARGET].mean() if TARGET in fdf.columns else np.nan, fmt="{:.2f}%")
with c[2]: kpi("Repaid Rate (%)", 100 * (1 - fdf[TARGET].mean()) if TARGET in fdf.columns else np.nan, fmt="{:.2f}%")
with c[3]: kpi("Total Features", fdf.shape[1], fmt="{:,}")
with c[4]: kpi("Avg Missing per Feature (%)", float(fdf.isna().mean().mean() * 100), fmt="{:.2f}%")

c = st.columns(5)
with c[0]: kpi("# Numerical Features", int(fdf.select_dtypes(include=[np.number]).shape[1]), fmt="{:,}")
with c[1]: kpi("# Categorical Features", int(fdf.select_dtypes(include=["category", "object"]).shape[1]), fmt="{:,}")
with c[2]: kpi("Median Age (Years)", float(fdf["AGE_YEARS"].median()) if "AGE_YEARS" in fdf.columns else np.nan)
with c[3]: kpi("Median Annual Income", float(fdf["AMT_INCOME_TOTAL"].median()) if "AMT_INCOME_TOTAL" in fdf.columns else np.nan)
with c[4]: kpi("Average Credit Amount", float(fdf["AMT_CREDIT"].mean()) if "AMT_CREDIT" in fdf.columns else np.nan)

st.divider()

# ===================== Charts (10) =====================
# 1) Target donut
if TARGET in fdf.columns:
    tgt_counts = counts_repaid_default(fdf)  # ['Status','Count']
    st.plotly_chart(px.pie(tgt_counts, names="Status", values="Count", hole=0.5,
                           title="Target Distribution (Repaid vs Default)"),
                    use_container_width=True)

# 2) Top-20 features by missing % (pre-imputation if available)
miss_df = missingness_topk(artifacts, df, k=20)
fig2 = px.bar(miss_df, x="missing_pct", y="column", orientation="h",
              title="Top 20 Features by Missing % (pre-imputation when available)")
fig2.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig2, use_container_width=True)

# 3–5) Histograms
g = st.columns(3)
with g[0]:
    if "AGE_YEARS" in fdf.columns:
        st.plotly_chart(px.histogram(fdf, x="AGE_YEARS", nbins=50, title="Age Distribution (Years)"), use_container_width=True)
with g[1]:
    if "AMT_INCOME_TOTAL" in fdf.columns:
        st.plotly_chart(px.histogram(fdf, x="AMT_INCOME_TOTAL", nbins=50, title="Income Distribution"), use_container_width=True)
with g[2]:
    if "AMT_CREDIT" in fdf.columns:
        st.plotly_chart(px.histogram(fdf, x="AMT_CREDIT", nbins=50, title="Credit Amount Distribution"), use_container_width=True)

# 6–7) Boxplots
g = st.columns(2)
with g[0]:
    if "AMT_INCOME_TOTAL" in fdf.columns:
        st.plotly_chart(px.box(fdf, y="AMT_INCOME_TOTAL", points=False, title="Income — Boxplot (winsorized)"),
                        use_container_width=True)
with g[1]:
    if "AMT_CREDIT" in fdf.columns:
        st.plotly_chart(px.box(fdf, y="AMT_CREDIT", points=False, title="Credit — Boxplot (winsorized)"),
                        use_container_width=True)

# 8–10) Categorical counts
g = st.columns(3)
with g[0]:
    if "CODE_GENDER" in fdf.columns:
        chart_df = fdf["CODE_GENDER"].value_counts().rename_axis("CODE_GENDER").reset_index(name="count")
        st.plotly_chart(px.bar(chart_df, x="CODE_GENDER", y="count", title="Gender Distribution", text_auto=True),
                        use_container_width=True)
with g[1]:
    if "NAME_FAMILY_STATUS" in fdf.columns:
        chart_df = fdf["NAME_FAMILY_STATUS"].value_counts().rename_axis("NAME_FAMILY_STATUS").reset_index(name="count")
        st.plotly_chart(px.bar(chart_df, x="NAME_FAMILY_STATUS", y="count", title="Family Status Distribution", text_auto=True),
                        use_container_width=True)
with g[2]:
    if "NAME_EDUCATION_TYPE" in fdf.columns:
        chart_df = fdf["NAME_EDUCATION_TYPE"].value_counts().rename_axis("NAME_EDUCATION_TYPE").reset_index(name="count")
        st.plotly_chart(px.bar(chart_df, x="NAME_EDUCATION_TYPE", y="count", title="Education Distribution", text_auto=True),
                        use_container_width=True)

st.caption("Numeric fields used here are winsorized at 1% tails (preprocessed).")

# ===================== Narrative =====================
st.subheader("Narrative Insights")
bullets = []
if TARGET in fdf.columns:
    dr = 100 * fdf[TARGET].mean()
    bullets.append(f"- Portfolio default rate under current filters is **{dr:.2f}%**.")
if "AMT_INCOME_TOTAL" in fdf.columns:
    sk = float(fdf["AMT_INCOME_TOTAL"].skew())
    shape = "right-skewed" if sk > 1 else ("left-skewed" if sk < -1 else "roughly symmetric")
    bullets.append(f"- Income distribution appears **{shape}** (skew={sk:.2f}); boxplot uses winsorization for stability.")
if artifacts.get("missingness_before"):
    top_miss = pd.DataFrame(artifacts["missingness_before"]).sort_values("missing_pct", ascending=False).head(3)
    tops = ", ".join(f"{r['column']} ({r['missing_pct']:.1f}%)" for _, r in top_miss.iterrows())
    bullets.append(f"- Highest pre-imputation missingness: {tops}. These were imputed/dropped per rules.")
if not bullets:
    bullets = ["- Dataset loaded and filtered; no additional notes."]
st.markdown("\n".join(bullets))
