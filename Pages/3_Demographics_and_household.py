# pages/3_Demographics_and_household.py
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except ModuleNotFoundError:
    st.error("Plotly is required. Install:  pip install plotly")
    st.stop()

from utils.utils import TARGET, TGT_LABELS, get_filtered_df, kpi

st.title("Demographics & Household")

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
m_share = f_share = np.nan
if "CODE_GENDER" in fdf.columns:
    vc = fdf["CODE_GENDER"].value_counts(normalize=True)
    m_share = 100 * vc.get("M", 0.0)
    f_share = 100 * vc.get("F", 0.0)
gender_label = f"M {m_share:.1f}% | F {f_share:.1f}%" if not np.isnan(m_share) else "N/A"

avg_age_def = float(fdf.loc[fdf.get(TARGET, pd.Series(index=fdf.index)).eq(1), "AGE_YEARS"].mean()) if "AGE_YEARS" in fdf.columns and TARGET in fdf.columns else np.nan
avg_age_non = float(fdf.loc[fdf.get(TARGET, pd.Series(index=fdf.index)).eq(0), "AGE_YEARS"].mean()) if "AGE_YEARS" in fdf.columns and TARGET in fdf.columns else np.nan
with_children = 100.0 * (fdf["CNT_CHILDREN"] > 0).mean() if "CNT_CHILDREN" in fdf.columns else np.nan
avg_family_size = float(fdf["CNT_FAM_MEMBERS"].mean()) if "CNT_FAM_MEMBERS" in fdf.columns else np.nan

married_pct = single_pct = np.nan
if "NAME_FAMILY_STATUS" in fdf.columns:
    married_pct = 100 * (fdf["NAME_FAMILY_STATUS"] == "Married").mean()
    single_pct  = 100 * (fdf["NAME_FAMILY_STATUS"] == "Single / not married").mean()

higher_ed_pct = np.nan
if "NAME_EDUCATION_TYPE" in fdf.columns:
    mask = fdf["NAME_EDUCATION_TYPE"].isin(["Higher education", "Academic degree"])
    if mask.notna().any():
        higher_ed_pct = 100 * mask.mean()

living_with_parents_pct = 100 * (fdf["NAME_HOUSING_TYPE"] == "With parents").mean() if "NAME_HOUSING_TYPE" in fdf.columns else np.nan

currently_working_pct = np.nan
if "EMPLOYMENT_YEARS" in fdf.columns:
    currently_working_pct = 100 * fdf["EMPLOYMENT_YEARS"].gt(0).mean()
    if "OCCUPATION_TYPE" in fdf.columns:
        occ_working = fdf["OCCUPATION_TYPE"].notna() & fdf["EMPLOYMENT_YEARS"].isna()
        currently_working_pct = 100 * ((fdf["EMPLOYMENT_YEARS"].gt(0)) | occ_working).mean()

avg_emp_years = float(fdf["EMPLOYMENT_YEARS"].mean()) if "EMPLOYMENT_YEARS" in fdf.columns else np.nan

c = st.columns(5)
with c[0]: kpi("Gender Split", gender_label, fmt="{}")
with c[1]: kpi("Avg Age — Defaulters", avg_age_def)
with c[2]: kpi("Avg Age — Non-Defaulters", avg_age_non)
with c[3]: kpi("% With Children", with_children, fmt="{:.1f}%")
with c[4]: kpi("Avg Family Size", avg_family_size)
c = st.columns(5)
with c[0]: kpi("Married vs Single", f"{married_pct:.1f}% | {single_pct:.1f}%" if not np.isnan(married_pct) else "N/A", fmt="{}")
with c[1]: kpi("% Higher Education (Bachelor+)", higher_ed_pct, fmt="{:.1f}%")
with c[2]: kpi("% Living With Parents", living_with_parents_pct, fmt="{:.1f}%")
with c[3]: kpi("% Currently Working", currently_working_pct, fmt="{:.1f}%")
with c[4]: kpi("Avg Employment Years", avg_emp_years)

st.divider()

# =================== Charts (10) ===================
if "AGE_YEARS" in fdf.columns:
    st.plotly_chart(px.histogram(fdf, x="AGE_YEARS", nbins=50, title="Age Distribution (All)"), use_container_width=True)

if "AGE_YEARS" in fdf.columns and TARGET in fdf.columns:
    tmp = fdf[["AGE_YEARS", TARGET]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.histogram(tmp, x="AGE_YEARS", color="Status", barmode="overlay",
                                 opacity=0.6, nbins=50, title="Age by Target (Overlay)"),
                    use_container_width=True)

if "CODE_GENDER" in fdf.columns:
    g = fdf["CODE_GENDER"].value_counts().rename_axis("CODE_GENDER").reset_index(name="count")
    st.plotly_chart(px.bar(g, x="CODE_GENDER", y="count", text_auto=True, title="Gender Distribution"),
                    use_container_width=True)

if "NAME_FAMILY_STATUS" in fdf.columns:
    fam = fdf["NAME_FAMILY_STATUS"].value_counts().rename_axis("NAME_FAMILY_STATUS").reset_index(name="count")
    st.plotly_chart(px.bar(fam, x="NAME_FAMILY_STATUS", y="count", text_auto=True, title="Family Status Distribution"),
                    use_container_width=True)

if "NAME_EDUCATION_TYPE" in fdf.columns:
    edu = fdf["NAME_EDUCATION_TYPE"].value_counts().rename_axis("NAME_EDUCATION_TYPE").reset_index(name="count")
    st.plotly_chart(px.bar(edu, x="NAME_EDUCATION_TYPE", y="count", text_auto=True, title="Education Distribution"),
                    use_container_width=True)

if "OCCUPATION_TYPE" in fdf.columns:
    occ = fdf["OCCUPATION_TYPE"].value_counts().head(10).rename_axis("OCCUPATION_TYPE").reset_index(name="count")
    st.plotly_chart(px.bar(occ, x="OCCUPATION_TYPE", y="count", text_auto=True, title="Occupation (Top 10)"),
                    use_container_width=True)

if "NAME_HOUSING_TYPE" in fdf.columns:
    hous = fdf["NAME_HOUSING_TYPE"].value_counts().reset_index()
    hous.columns = ["NAME_HOUSING_TYPE", "count"]
    st.plotly_chart(px.pie(hous, names="NAME_HOUSING_TYPE", values="count", hole=0.4,
                           title="Housing Type Distribution"),
                    use_container_width=True)

if "CNT_CHILDREN" in fdf.columns:
    kids = fdf["CNT_CHILDREN"].value_counts().sort_index().rename_axis("CNT_CHILDREN").reset_index(name="count")
    st.plotly_chart(px.bar(kids, x="CNT_CHILDREN", y="count", text_auto=True, title="Children Count"),
                    use_container_width=True)

if "AGE_YEARS" in fdf.columns and TARGET in fdf.columns:
    tmp = fdf[["AGE_YEARS", TARGET]].copy()
    tmp["Status"] = tmp[TARGET].map(TGT_LABELS)
    st.plotly_chart(px.box(tmp, x="Status", y="AGE_YEARS", points=False, title="Age vs Target — Boxplot"),
                    use_container_width=True)

corr_cols = [c for c in ["AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", TARGET] if c in fdf.columns]
if len(corr_cols) >= 2:
    corr = fdf[corr_cols].corr(method="spearman")
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto",
                              title="Spearman Correlation — Age, Children, Family Size, Target"),
                    use_container_width=True)

st.caption("Numeric fields were winsorized during preprocessing. Correlations use Spearman for robustness.")

st.subheader("Narrative Insights")
ins = []
if "AGE_YEARS" in fdf.columns and TARGET in fdf.columns:
    def_m = fdf.loc[fdf[TARGET]==1, "AGE_YEARS"].median()
    non_m = fdf.loc[fdf[TARGET]==0, "AGE_YEARS"].median()
    if not np.isnan(def_m) and not np.isnan(non_m):
        delta = non_m - def_m
        dirn = "younger" if delta > 0 else "older"
        ins.append(f"- Defaulters skew **{dirn}** (median age: default {def_m:.1f} vs repaid {non_m:.1f}).")
if {"CNT_CHILDREN","CNT_FAM_MEMBERS", TARGET}.issubset(fdf.columns):
    ch_corr = fdf[["CNT_CHILDREN", TARGET]].corr(method="spearman").iloc[0,1]
    fs_corr = fdf[["CNT_FAM_MEMBERS", TARGET]].corr(method="spearman").iloc[0,1]
    ins.append(f"- Weak/Moderate correlation with default: children (**{ch_corr:.2f}**) and family size (**{fs_corr:.2f}**).")
if "EMPLOYMENT_YEARS" in fdf.columns and TARGET in fdf.columns:
    work_rate = 100 * fdf["EMPLOYMENT_YEARS"].gt(0).mean()
    ins.append(f"- ~**{work_rate:.1f}%** show active employment tenure; cross-check risk by tenure on the Target page.")
st.markdown("\n".join(ins) if ins else "- No strong demographic/household signals under current filters.")
