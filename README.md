# 🏃 Karmanta 2026 – Cracking the Attrition Code in IT Services

> **Karmanta 2026 – Case Competition**
> Team **Green Analysts** · IISWBM, Kolkata (University of Calcutta)
> Case hosted by IMI New Delhi

A predictive and prescriptive analysis of employee attrition at an IT services firm. The project combines exploratory analysis, **machine-learning risk models** (Logistic Regression and Decision Trees), **survival analysis** (Cox Proportional Hazards), a Power BI dashboard, and an interactive **Streamlit** app, then turns the findings into an "Attrition Defense System" of practical recommendations.

---

## 📌 Business Problem

Annual churn of roughly **12–20%** in IT services, especially among mid-level engineers and project managers, is disrupting client delivery, raising hiring costs and draining institutional knowledge. The firm has already tried conventional HR interventions (pay corrections, engagement initiatives, exit interviews), but leadership still lacks clarity on the **real, quantifiable drivers**.

**Objective:** build a predictive framework that identifies attrition risk, extract actionable insights, and recommend both immediate and long-term retention strategies.

---

## 🧠 Approach

1. **Data cleaning:** prepare the raw HR dataset for analysis
2. **Exploratory Data Analysis:** burnout, demographics, satisfaction, pay, career stage, structure, experience and commute
3. **Predictive modelling:** Logistic Regression and Decision Tree classifiers, with ROC comparison and risk banding
4. **Survival analysis:** Cox Proportional Hazards to understand *who leaves faster*
5. **High-risk persona analysis:** profile the employees the models flag
6. **Recommendations:** Job Characteristics Model, Total Rewards, 9-Box Grid and an HR Balanced Scorecard

---

## 🗂️ Dataset

`Karmanta - Dataset.csv` is the raw dataset provided for the case. `karmanta_clean.csv` is the cleaned version used for modelling and the app. It covers demographics, performance, project allocation, engagement metrics and exit feedback.

- **9,988 employees**, of whom **2,245 left**, giving an overall attrition rate of **22.48%**

---

## 📊 Key Findings

### Attrition is behavioural, not demographic

Demographic splits barely move the needle. Attrition stays in a narrow band of about 21–24% across gender, education, marital status, age group, native place and tenure. Behavioural and experience factors separate leavers from stayers far more sharply.

| Factor | Stayed | Left |
|---|---|---|
| Overtime (attrition rate) | 20.87% (no overtime) | **26.14%** (overtime) |
| Business travel (attrition rate) | 18.74% (no travel) | **34.24%** (travels frequently) |
| Distance from office | 8.60% (0–10 km) | **29.89%** (40–50 km) |
| Avg. job satisfaction | 3.05 | **2.77** |
| Avg. environment satisfaction | 3.14 | **2.48** |
| Avg. work-life balance | 3.12 | **2.61** |
| Avg. job involvement | 3.11 | **2.56** |
| Avg. years since last promotion | 3.81 | **4.58** |

### Compensation is not the problem

Average salary is nearly identical for stayers and leavers (about ₹114.4K vs ₹114.8K), as is average salary hike (50.03% vs 50.60%). Pay corrections alone are unlikely to fix attrition.

### Other observations

- Cyber Security has the highest attrition among departments (24.16%), but department-level differences are small
- Commute and travel show the clearest structural gradients

---

## 🤖 Models

### Predictive models

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 81% | 0.74 |
| Decision Tree | 85% | 0.79 |

- **Logistic Regression** picks up policy and tenure signals. Top signals: frequent travel, years since promotion, overtime, distance.
- **Decision Tree** picks up behaviour and experience signals. Top drivers: work-life balance, distance, environment satisfaction, job involvement, job satisfaction, promotion delay.
- Employees are banded by predicted risk as **Low (<30%)**, **Medium (30–60%)** and **High (>60%)**.

> Because about 77.5% of employees stayed, accuracy should be read alongside AUC and recall for the "left" class rather than on its own.

### Survival model

A **Cox Proportional Hazards** model (concordance ≈ 0.70) estimates how quickly different employees leave:

| Factor | Effect on attrition risk |
|---|---|
| Frequent business travel | ~**1.72×** faster exit |
| Overtime | ~**1.22×** |
| Promotion delay | ~**1.09×** per year |
| Longer commute | Increases risk |

Employees who stay longer report a better environment, better job satisfaction, better work-life balance and higher involvement.

### High-risk persona

A typical high-risk employee has a moderate-to-high commute, travels often, reports lower satisfaction, has waited longer for promotion and is more likely to work overtime. Strong indicators include low job involvement (disengaged, not necessarily low-performing), poor perception of the work environment, and stagnant career progression.

---

## 💡 Recommendations: The Attrition Defense System

| Framework | Purpose | Key actions |
|---|---|---|
| **Job Characteristics Model** | Fix the daily experience | Mini project ownership, client exposure, regular manager 1:1s, recognition moments. These raise autonomy, task significance, feedback and skill variety. |
| **Total Rewards** | Move beyond salary | Hybrid/remote options for long commuters, priority access to internal roles, fast-track promotion reviews, learning and certification fund, spot awards, manager-led stretch projects |
| **9-Box Grid** | Prioritize who to save first | Overlay attrition risk with performance. Offer improvement planning for low performers, career pathing and mentoring for mid-tier talent, and immediate retention discussions with leadership sponsors for high performers at risk |
| **HR Balanced Scorecard** | Make it sustainable | Monthly risk report, attrition probability trend, promotion delay metric, engagement index and manager accountability, tracked as leadership KPIs rather than an annual HR exercise |

Medium- and high-risk employees are the primary target group for the experience and rewards actions.

---

## 🗃️ Repository Structure

| File | Description |
|---|---|
| `cleaning.ipynb` | Data cleaning |
| `analysis.ipynb` | Exploratory data analysis |
| `ml.ipynb` | Logistic Regression and Decision Tree models |
| `survival.ipynb` | Cox Proportional Hazards survival analysis |
| `app.py` | Streamlit **Employee Attrition Prediction Dashboard** |
| `Karmanta - Dataset.csv` | Raw dataset |
| `karmanta_clean.csv` | Cleaned dataset used by the app |
| `lr_high_summary.csv` | Profile summary of high-risk employees (Logistic Regression) |
| `dt_high_summary.csv` | Profile summary of high-risk employees (Decision Tree) |
| `karmanta_dash.pbix` | Power BI dashboard: *Workforce Health – Executive Overview* and *Diagnostic Deep Dive* |
| `Karmanta - Case .pdf` | Official case statement |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/sm4ck717/karmanta-hr-attrition.git
cd karmanta-hr-attrition

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install streamlit pandas scikit-learn
```

The notebooks may need extra packages such as `jupyter`, `matplotlib`, `seaborn` and a survival-analysis library (for example `lifelines`). Check the imports at the top of each notebook.

### Run the Streamlit app

```bash
streamlit run app.py
```

Run it from the repository root so the app can find `karmanta_clean.csv`. The model trains on startup and is cached afterwards.

### Explore the notebooks

Run them in this order: `cleaning.ipynb`, `analysis.ipynb`, `ml.ipynb`, `survival.ipynb`.

### Power BI dashboard

Open `karmanta_dash.pbix` in [Power BI Desktop](https://powerbi.microsoft.com/desktop/).

---

## 🖥️ App Features (`app.py`)

- **Interactive inputs** in the sidebar: age, salary, distance from home, sub-department, overtime, business travel, job satisfaction and work-life balance
- **Attrition probability** from a Logistic Regression model (standardized features, 80/20 stratified split), with a High/Low risk label
- Features not exposed in the sidebar are set to the dataset median (numeric) or mode (categorical)
- Optional preview of the cleaned dataset

---

## 🛠️ Tech Stack

Python · Pandas · scikit-learn · Streamlit · Survival analysis (Cox PH) · Jupyter Notebook · Power BI

---

## 👤 Team – Green Analysts

- **Shreyan Sarkar**

Indian Institute of Social Welfare & Business Management (IISWBM), Kolkata, affiliated to the University of Calcutta.

---

## 📄 License

No license has been specified yet. Consider adding one (for example MIT) if you want others to reuse this work.
