# 📘 Assignment (Markdown): Streamlit Dashboard for **Home Credit Default Risk — `application_train.csv`**

> **Goal:** Design a 5-page Streamlit dashboard (no code required in this document) that explores the Home Credit Application Train dataset and surfaces risk insights.  
> **Scope:** Each page must include **10 KPIs** and **10 graphs**, plus narrative insights. Provide formulas, data prep plan, and evaluation rubric.

---

## 0) Project Overview

**Dataset**
- File: `application_train.csv`
- Rows/Cols: ~307K × 122 (approx.)
- Target: `TARGET` → `0` (repaid), `1` (default)

**Primary Columns (commonly used)**
- IDs: `SK_ID_CURR`
- Target: `TARGET`
- Demographics: `CODE_GENDER`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `NAME_HOUSING_TYPE`
- Financials: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`
- Others: `NAME_CONTRACT_TYPE`, `REGION_RATING_CLIENT`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`

**Preprocessing (to be implemented before visualization)**
1. Convert ages: `AGE_YEARS = -DAYS_BIRTH / 365.25`
2. Employment tenure: `EMPLOYMENT_YEARS = -DAYS_EMPLOYED / 365.25` (clip huge positives used as “not employed” codes)
3. Create ratios:
   - `DTI = AMT_ANNUITY / AMT_INCOME_TOTAL`
   - `LOAN_TO_INCOME = AMT_CREDIT / AMT_INCOME_TOTAL`
   - `ANNUITY_TO_CREDIT = AMT_ANNUITY / AMT_CREDIT`
4. Handle missing values (report % and apply strategy: drop columns > 60% missing; impute median/most-frequent for others)
5. Standardize categories (merge rare categories under “Other” if share < 1%)
6. Outlier handling: Winsorize top/bottom 1% for skewed numeric features used in charts
7. Define **income brackets** (quantiles): Low (Q1), Mid (Q2–Q3), High (Q4)
8. Ensure consistent filters (gender, education, family status, housing, income bracket, age range)

**Navigation**
- Sidebar radio with 5 pages
- Global filters: Gender, Education, Family Status, Housing Type, Age range, Income bracket

---

## Page 1 — **Overview & Data Quality**

**Purpose:** Introduce dataset, quality, and high-level portfolio risk.

### KPIs (10)
1. **Total Applicants** = `count(SK_ID_CURR)`
2. **Default Rate (%)** = `mean(TARGET) × 100`
3. **Repaid Rate (%)** = `(1 - mean(TARGET)) × 100`
4. **Total Features** = number of columns
5. **Avg Missing per Feature (%)** = `mean(isnull(col)) × 100` averaged over columns
6. **# Numerical Features**
7. **# Categorical Features**
8. **Median Age (Years)** = `median(AGE_YEARS)`
9. **Median Annual Income**
10. **Average Credit Amount**

### Graphs (10)
1. Pie / Donut — Target distribution (0 vs 1)
2. Bar — Top 20 features by missing %
3. Histogram — `AGE_YEARS`
4. Histogram — `AMT_INCOME_TOTAL`
5. Histogram — `AMT_CREDIT`
6. Boxplot — `AMT_INCOME_TOTAL`
7. Boxplot — `AMT_CREDIT`
8. Countplot — `CODE_GENDER`
9. Countplot — `NAME_FAMILY_STATUS`
10. Countplot — `NAME_EDUCATION_TYPE`

**Narrative (add below charts):** 2–3 bullet insights summarizing distribution shapes, skew, and immediate red flags (e.g., high missingness columns).

---

## Page 2 — **Target & Risk Segmentation**

**Purpose:** Understand how default varies across key segments.

### KPIs (10)
1. **Total Defaults** = `sum(TARGET)`
2. **Default Rate (%)**
3. **Default Rate by Gender (%)**
4. **Default Rate by Education (%)**
5. **Default Rate by Family Status (%)**
6. **Avg Income — Defaulters**
7. **Avg Credit — Defaulters**
8. **Avg Annuity — Defaulters**
9. **Avg Employment (Years) — Defaulters**
10. **Default Rate by Housing Type (%)**

### Graphs (10)
1. Bar — Counts: Default vs Repaid
2. Bar — Default % by Gender
3. Bar — Default % by Education
4. Bar — Default % by Family Status
5. Bar — Default % by Housing Type
6. Boxplot — Income by Target
7. Boxplot — Credit by Target
8. Violin — Age vs Target
9. Histogram (stacked) — `EMPLOYMENT_YEARS` by Target
10. Stacked Bar — `NAME_CONTRACT_TYPE` vs Target

**Narrative:** Highlight the 2–3 segments with **highest** and **lowest** default rates; propose hypotheses (e.g., low income + high LTI).

---

## Page 3 — **Demographics & Household Profile**

**Purpose:** Who are the applicants? Household structure and human factors.

### KPIs (10)
1. **% Male vs Female**
2. **Avg Age — Defaulters**
3. **Avg Age — Non-Defaulters**
4. **% With Children** = `% (CNT_CHILDREN > 0)`
5. **Avg Family Size** = `mean(CNT_FAM_MEMBERS)`
6. **% Married vs Single** (from `NAME_FAMILY_STATUS`)
7. **% Higher Education** (Bachelor+)
8. **% Living With Parents** (`NAME_HOUSING_TYPE == 'With parents'`)
9. **% Currently Working** (derive from occupation/employment)
10. **Avg Employment Years**

### Graphs (10)
1. Histogram — Age distribution (all)
2. Histogram — Age by Target (overlay)
3. Bar — Gender distribution
4. Bar — Family Status distribution
5. Bar — Education distribution
6. Bar — Occupation distribution (top 10)
7. Pie — Housing Type distribution
8. Countplot — `CNT_CHILDREN`
9. Boxplot — Age vs Target
10. Heatmap — Corr(Age, Children, Family Size, TARGET)

**Narrative:** Note “life-stage” patterns: younger vs older risk, effect of children/family size.

---

## Page 4 — **Financial Health & Affordability**

**Purpose:** Ability to repay, affordability indicators, and stress.

### KPIs (10)
1. **Avg Annual Income**
2. **Median Annual Income**
3. **Avg Credit Amount**
4. **Avg Annuity**
5. **Avg Goods Price**
6. **Avg DTI** = `mean(AMT_ANNUITY / AMT_INCOME_TOTAL)`
7. **Avg Loan-to-Income (LTI)** = `mean(AMT_CREDIT / AMT_INCOME_TOTAL)`
8. **Income Gap (Non-def − Def)** = `mean(income|0) − mean(income|1)`
9. **Credit Gap (Non-def − Def)** = `mean(credit|0) − mean(credit|1)`
10. **% High Credit (> 1M)**

### Graphs (10)
1. Histogram — Income distribution
2. Histogram — Credit distribution
3. Histogram — Annuity distribution
4. Scatter — Income vs Credit (alpha blending)
5. Scatter — Income vs Annuity
6. Boxplot — Credit by Target
7. Boxplot — Income by Target
8. KDE / Density — Joint Income–Credit
9. Bar — Income Brackets vs Default Rate
10. Heatmap — Financial variable correlations (Income, Credit, Annuity, DTI, LTI, TARGET)

**Narrative:** Identify affordability thresholds where default rises (e.g., LTI > 6, DTI > 0.35 — to be computed and validated).

---

## Page 5 — **Correlations, Drivers & Interactive Slice-and-Dice**

**Purpose:** What drives default? Combine correlation views with interactive filters.

### KPIs (10)
1. **Top 5 +Corr with TARGET** (list)
2. **Top 5 −Corr with TARGET** (list)
3. **Most correlated with Income**
4. **Most correlated with Credit**
5. **Corr(Income, Credit)**
6. **Corr(Age, TARGET)**
7. **Corr(Employment Years, TARGET)**
8. **Corr(Family Size, TARGET)**
9. **Variance Explained by Top 5 Features** (proxy via |corr|)
10. **# Features with |corr| > 0.5**

### Graphs (10)
1. Heatmap — Correlation (selected numerics)
2. Bar — |Correlation| of features vs TARGET (top N)
3. Scatter — Age vs Credit (hue=TARGET)
4. Scatter — Age vs Income (hue=TARGET)
5. Scatter — Employment Years vs TARGET (jitter/bins)
6. Boxplot — Credit by Education
7. Boxplot — Income by Family Status
8. Pair Plot — Income, Credit, Annuity, TARGET
9. Filtered Bar — Default Rate by Gender (responsive to sidebar)
10. Filtered Bar — Default Rate by Education (responsive)

**Narrative:** Translate correlations into **candidate policy rules** (e.g., set LTI caps; minimum income floors for certain risk segments).

---

## KPI & Metric Definitions (Formulas)

- **Default Rate (%)** = `100 × mean(TARGET)`
- **Repaid Rate (%)** = `100 × (1 − mean(TARGET))`
- **AGE_YEARS** = `-DAYS_BIRTH / 365.25`
- **EMPLOYMENT_YEARS** = `-DAYS_EMPLOYED / 365.25` (clip unrealistic)
- **DTI** = `AMT_ANNUITY / AMT_INCOME_TOTAL`
- **LTI (Loan-to-Income)** = `AMT_CREDIT / AMT_INCOME_TOTAL`
- **ANNUITY_TO_CREDIT** = `AMT_ANNUITY / AMT_CREDIT`
- **Income Brackets** = quantile bins of `AMT_INCOME_TOTAL` (Q1, Q2–Q3, Q4)
- **High Credit** = `AMT_CREDIT > 1_000_000` (tunable threshold)

---

## Data Quality Checklist (Dashboard Page 1)

- Report **missing%** per feature; list top 20
- Show **imputed fields** and method (median/mode)
- Flag **categorical levels** with <1% share (group into “Other”)
- Note **outlier treatment** for skewed numerics
- Confirm **data leakage** checks (avoid any post-outcome features—if present, exclude)

---

## Interactivity Requirements

- **Global Filters (Sidebar):** Gender, Education, Family Status, Housing Type, Age Range (slider), Income Bracket (select), Employment Tenure (slider)
- **Local Filters (per page):** Allow selecting **Top-N categories** to highlight in bars
- **Drill-through:** Clicking a bar or segment should update KPIs and subordinate charts (conceptually; implement with session state)
- **Reset Button:** Clear all filters
- **Download:** Export filtered dataset as CSV (concept)

---

## Expected Business Insights (Examples to write under charts)

- **Risk Hotspots:** Higher default rates in **low-income** and **high LTI** segments; confirm with Page 4 thresholds.
- **Demographic Patterns:** Younger applicants may show higher default; cross-check with employment tenure.
- **Affordability:** Default rate increases nonlinearly when **DTI > threshold**; propose credit policy caps.
- **Policy Ideas:** Minimum income floor, cap on LTI by age band, additional docs for high-risk occupations.

---

## Deliverables

1. **Markdown assignment (this file)**
2. **Dashboard spec**: 5 pages × (10 KPIs + 10 graphs)
3. **Data prep notes**: missingness, imputation, outliers, derived features
4. **Narratives**: 2–3 insights per page
5. **Screenshots** (after you build it) showing each page populated
6. **Appendix**: Calculation checks, filter logic, field dictionary

---

## Grading Rubric (Total 100)

- **Data Prep & Validity (20):** Correct formulas, clear missing/outlier strategy
- **KPI Design (20):** Relevance, correctness, and coverage
- **Visualization Quality (20):** Readability, appropriate chart types, legends/labels
- **Interactivity (15):** Useful filters; correct propagation to KPIs/graphs
- **Insights & Storytelling (15):** Actionable findings; policy suggestions
- **Professionalism (10):** Organization, naming, consistency, documentation

---

## Submission Instructions

- Submit this **Markdown** and your final **Streamlit app** (link or repo).
- Include a **README** describing:
  - Environment & dependencies
  - How to run (`streamlit run app.py`)
  - Data location (path to `application_train.csv`)
  - Any preprocessing scripts

---

## Appendix A — Field Map & Notes (examples)

- `TARGET`: 0=Repaid, 1=Default  
- `DAYS_BIRTH`: negative days; convert to years (abs/365.25)  
- `DAYS_EMPLOYED`: negative days; treat extreme positive (365243) as missing/unemployed  
- `AMT_INCOME_TOTAL`: annual income  
- `AMT_CREDIT`: total credit amount  
- `AMT_ANNUITY`: repayment annuity  
- `AMT_GOODS_PRICE`: price of goods bought with the loan  
- `CODE_GENDER`: M/F/XNA (treat XNA separately or merge)  
- `NAME_FAMILY_STATUS`: marital status categories  
- `NAME_EDUCATION_TYPE`: Basic/Secondary/Higher/Academic degree etc.

---

## Appendix B — Checklist (tick as you complete)

- [ ] Derived columns: AGE_YEARS, EMPLOYMENT_YEARS, DTI, LTI, ANNUITY_TO_CREDIT  
- [ ] Outlier handling for income/credit/annuity  
- [ ] Rare-label handling for categorical features  
- [ ] Global filters wired to all pages  
- [ ] 10 KPIs per page  
- [ ] 10 graphs per page  
- [ ] 2–3 insights written under each page  
- [ ] Final screenshot pack

---
