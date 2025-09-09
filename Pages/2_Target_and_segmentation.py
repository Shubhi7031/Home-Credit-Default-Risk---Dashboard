# pages/2_Target_and_segmentation.py
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
    get_filtered_df, kpi, default_rate, by_category_rate, counts_repaid_default
)

st.title("Target & Risk Segmentation")

if "clean_df" not in st.session_state:
    st.warning("Cleaned data not found. Open the **Home** page first to load & preprocess the dataset.")
    st.stop()

df = st.session_state["clean_df"]
fdf, _ = get_filtered_df(df)
st.caption(f"Filtered sample size: **{len(fdf):,}** rows (of {len(df):,})")

if TARGET not in fdf.columns:
    st.error("Column 'TARGET' not found in data.")
    st.stop()

# =================== KPIs (10) ===================
c = st.columns(5)
with c[0]: kpi("Total Defaults", int(fdf[TARGET].sum()), fmt="{:,}")
with c[1]: kpi("Default Rate (%)", default_rate(fdf[TARGET]), fmt="{:.2f}%")
with c[2]:
    if "CODE_GENDER" in fdf.columns:
        g = by_category_rate(fdf, "CODE_GENDER")
        kpi("Default Rate by Gender (max)", g["Default_%"].iloc[0] if not g.empty else "N/A",
            fmt="{:.2f}%", help_text=(f"{g.iloc[0]['CODE_GENDER']} | n={int(g.iloc[0]['count']):,}" if not g.empty else None))
    else:
        kpi("Default Rate by Gender (max)", "N/A")
with c[3]:
    if "NAME_EDUCATION_TYPE" in fdf.columns:
        e = by_category_rate(fdf, "NAME_EDUCATION_TYPE")
        kpi("Default Rate by Education (max)", e["Default_%"].iloc[0] if not e.empty else "N/A",
            fmt="{:.2f}%", help_text=(f"{e.iloc[0]['NAME_EDUCATION_TYPE']} | n={int(e.iloc[0]['count']):,}" if not e.empty else None))
    else:
        kpi("Default Rate by Education (max)", "N/A")
with c[4]:
    if "NAME_FAMILY_STATUS" in fdf.columns:
        fam = by_category_rate(fdf, "NAME_FAMILY_STATUS")
        kpi("Default Rate by Family (max)", fam["Default_%"].iloc[0] if not fam.empty else "N/A",
            fmt="{:.2f}%", help_text=(f"{fam.iloc[0]['NAME_FAMILY_STATUS']} | n={int(fam.iloc[0]['count']):,}" if not fam.empty else None))
    else:
        kpi("Default Rate by Family (max)", "N/A")

c = st.columns(5)
with c[0]: kpi("Avg Income — Defaulters", fdf.loc[fdf[TARGET]==1, "AMT_INCOME_TOTAL"].mean() if "AMT_INCOME_TOTAL" in fdf.columns else np.nan)
with c[1]: kpi("Avg Credit — Defaulters", fdf.loc[fdf[TARGET]==1, "AMT_CREDIT"].mean() if "AMT_CREDIT" in fdf.columns else np.nan)
with c[2]: kpi("Avg Annuity — Defaulters", fdf.loc[fdf[TARGET]==1, "AMT_ANNUITY"].mean() if "AMT_ANNUITY" in fdf.columns else np.nan)
with c[3]: kpi("Avg Employment Years — Defaulters", fdf.loc[fdf[TARGET]==1, "EMPLOYMENT_YEARS"].mean() if "EMPLOYMENT_YEARS" in fdf.columns else np.nan)
with c[4]:
    if "NAME_HOUSING_TYPE" in fdf.columns:
        h = by_category_rate(fdf, "NAME_HOUSING_TYPE")
        kpi("Default Rate by Housing (max)", h["Default_%"].iloc[0] if not h.empty else "N/A",
            fmt="{:.2f}%", help_text=(f"{h.iloc[0]['NAME_HOUSING_TYPE']} | n={int(h.iloc[0]['count']):,}" if not h.empty else None))
    else:
        kpi("Default Rate by Housing (max)", "N/A")

st.divider()

# =================== Charts (10) ===================

# 1) Counts: Default vs Repaid
tgt_counts = counts_repaid_default(fdf)    # ['Status','Count']
st.plotly_chart(px.bar(tgt_counts, x="Status", y="Count", title="Counts — Default vs Repaid", text_auto=True),
                use_container_width=True)

# 2–5) Default % by categories
if "CODE_GENDER" in fdf.columns:
    g = by_category_rate(fdf, "CODE_GENDER")
    st.plotly_chart(px.bar(g, x="CODE_GENDER", y="Default_%", title="Default % by Gender", text_auto=".2f"),
                    use_container_width=True)
if "NAME_EDUCATION_TYPE" in fdf.columns:
    e = by_category_rate(fdf, "NAME_EDUCATION_TYPE")
    st.plotly_chart(px.bar(e, x="NAME_EDUCATION_TYPE", y="Default_%", title="Default % by Education", text_auto=".2f"),
                    use_container_width=True)
if "NAME_FAMILY_STATUS" in fdf.columns:
    fam = by_category_rate(fdf, "NAME_FAMILY_STATUS")
    st.plotly_chart(px.bar(fam, x="NAME_FAMILY_STATUS", y="Default_%", title="Default % by Family Status", text_auto=".2f"),
                    use_container_width=True)
if "NAME_HOUSING_TYPE" in fdf.columns:
    h = by_category_rate(fdf, "NAME_HOUSING_TYPE")
    st.plotly_chart(px.bar(h, x="NAME_HOUSING_TYPE", y="Default_%", title="Default % by Housing Type", text_auto=".2f"),
                    use_container_width=True)

# 6) Box — Income by Target
if "AMT_INCOME_TOTAL" in fdf.columns:
    tmp = fdf[[TARGET, "AMT_INCOME_TOTAL"]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.box(tmp, x="Status", y="AMT_INCOME_TOTAL", points=False, title="Income by Target"),
                    use_container_width=True)

# 7) Box — Credit by Target
if "AMT_CREDIT" in fdf.columns:
    tmp = fdf[[TARGET, "AMT_CREDIT"]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.box(tmp, x="Status", y="AMT_CREDIT", points=False, title="Credit by Target"),
                    use_container_width=True)

# 8) Violin — Age vs Target
if "AGE_YEARS" in fdf.columns:
    tmp = fdf[[TARGET, "AGE_YEARS"]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.violin(tmp, x="Status", y="AGE_YEARS", box=True, points=False, title="Age vs Target (Violin)"),
                    use_container_width=True)

# 9) Histogram — Employment Years by Target (stacked)
if "EMPLOYMENT_YEARS" in fdf.columns:
    tmp = fdf[[TARGET, "EMPLOYMENT_YEARS"]].dropna().copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.histogram(tmp, x="EMPLOYMENT_YEARS", color="Status", nbins=50,
                                 barmode="stack", title="Employment Years by Target (stacked)"),
                    use_container_width=True)

# 10) Stacked Bar — Contract Type vs Target
if "NAME_CONTRACT_TYPE" in fdf.columns:
    tmp = fdf[["NAME_CONTRACT_TYPE", TARGET]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.histogram(tmp, x="NAME_CONTRACT_TYPE", color="Status", barnorm=None,
                                 title="NAME_CONTRACT_TYPE vs Target", barmode="stack"),
                    use_container_width=True)

st.caption("All rates are computed on the filtered dataset. Numeric fields were winsorized during preprocessing for stability.")

# Narrative
st.subheader("Narrative Insights")
ins = []
for col in ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]:
    if col in fdf.columns:
        t = by_category_rate(fdf, col)
        if not t.empty:
            top = t.iloc[0]
            ins.append(f"- **Highest default** in **{col} = {top[col]}** at **{top['Default_%']:.2f}%** (n={int(top['count']):,}).")
if {"AMT_INCOME_TOTAL","LTI"}.issubset(fdf.columns):
    ins.append("- Consider interaction of **low income** and **high LTI**; verify on the Financial Health page.")
st.markdown("\n".join(ins) if ins else "- No categorical risk hotspots under current filters.")
