import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

sns.set(style="whitegrid", palette="muted")

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="Stack Overflow 2025 Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------
# Load Data
# -----------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset/stack_overflow_2025_cleaned_eda.csv")

df = load_data()

# -----------------------------------------
# Sidebar Filters
# -----------------------------------------
st.sidebar.title("🎛️ Filters")

# Country filter
countries = sorted(df['Country'].dropna().unique())
selected_country = st.sidebar.multiselect(
    "Select Country",
    countries,
    default=[]
)

# Work Experience filter
exp_range = st.sidebar.slider(
    "Work Experience (Years)",
    int(df['WorkExp'].min()),
    int(df['WorkExp'].max()),
    (0, 30)
)

# Salary Prediction Inputs
st.sidebar.header("💸 Salary Prediction Inputs")
work_exp = st.sidebar.slider("Years of Work Experience", 0, 50, 5)
years_code = st.sidebar.slider("Years of Coding Experience", 0, 50, 5)
job_sat = st.sidebar.slider("Job Satisfaction Score", 1, 10, 7)

input_data = pd.DataFrame({
    "WorkExp": [work_exp],
    "YearsCode": [years_code],
    "JobSat": [job_sat]
})

# Filter dataframe
filtered_df = df.copy()
if selected_country:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_country)]
filtered_df = filtered_df[
    filtered_df['WorkExp'].between(exp_range[0], exp_range[1])
]

# -----------------------------------------
# Title and KPIs
# -----------------------------------------
st.title("📊 Stack Overflow 2025 – Deep EDA Dashboard")
st.caption("Interactive analysis of developers, skills, AI adoption & compensation")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Respondents", f"{filtered_df.shape[0]:,}")
c2.metric("Countries", filtered_df['Country'].nunique())
c3.metric("Avg Experience", f"{filtered_df['WorkExp'].mean():.1f} yrs")
c4.metric("Median Salary", f"${filtered_df['ConvertedCompYearly'].median():,.0f}")

st.markdown("---")

# =========================================
# Tabs Layout (Fixed: Proper list syntax)
# =========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Demographics",
    "💰 Salary Analysis",
    "👨‍💻 Roles & Experience",
    "🤖 AI Insights",
    "📚 Learning Impact",
    "🧠 Future ML",
    "💡 Key Insights"
])

# =========================================
# TAB 1: Demographics
# =========================================
with tab1:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Age', order=filtered_df['Age'].value_counts().index, ax=ax)
    ax.set_title('Age Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Top 20 Countries by Respondents")
    country_counts = filtered_df[filtered_df['Country'] != 'Unknown']['Country'].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x=country_counts.values, y=country_counts.index, ax=ax)
    ax.set_title('Top 20 Countries by Respondents')
    ax.set_xlabel('Number of Respondents')
    ax.set_ylabel('Country')
    st.pyplot(fig)

    st.subheader("Education Level Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, y='EdLevel', order=filtered_df['EdLevel'].value_counts().index, ax=ax)
    ax.set_title('Education Level Distribution')
    st.pyplot(fig)

    st.subheader("Employment Status")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=filtered_df, x='Employment', order=filtered_df['Employment'].value_counts().index, ax=ax)
    ax.set_title('Employment Status')
    plt.xticks(rotation=60, ha='right')
    st.pyplot(fig)

    st.subheader("Developer Types")
    dev_types = filtered_df['DevType'].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig = px.bar(x=dev_types.index, y=dev_types.values, title='Developer Types',
                 labels={'x': 'Developer Type', 'y': 'Count'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Organization Size Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=filtered_df, x='OrgSize', order=filtered_df['OrgSize'].value_counts().index, ax=ax)
    ax.set_title('Organization Size Distribution')
    plt.xticks(rotation=70)
    st.pyplot(fig)

    st.subheader("Remote Work Preferences")
    remote_map = {
        'Remote': 'Remote',
        'In-person': 'In-person',
        'Hybrid (some remote, leans heavy to in-person)': 'Hybrid (Mostly In-person)',
        'Hybrid (some in-person, leans heavy to flexibility)': 'Hybrid (Mostly Remote)',
        'Your choice (very flexible, you can come in when you want or just as needed)': 'Fully Flexible',
        'Unknown': 'Unknown'
    }
    filtered_df['RemoteWork_Short'] = filtered_df['RemoteWork'].map(remote_map)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='RemoteWork_Short', hue='RemoteWork_Short',
                  order=filtered_df['RemoteWork_Short'].value_counts().index,
                  palette='magma', legend=False, ax=ax)
    ax.set_title('Remote Work Preferences')
    plt.xticks(rotation=20, ha='right')
    ax.set_xlabel('Work Type')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# =========================================
# TAB 2: Salary Analysis
# =========================================
with tab2:
    st.subheader("Salary vs Work Experience Groups")
    salary_exp = filtered_df[filtered_df['ConvertedCompYearly'].between(1000, 300000)].copy()
    salary_exp['WorkExpGroup'] = pd.cut(salary_exp['WorkExp'],
                                       bins=[0, 2, 5, 10, 15, 20, 30, 50],
                                       labels=['0–2', '3–5', '6–10', '11–15', '16–20', '21–30', '30+'])
    fig = px.scatter(salary_exp, x=salary_exp['WorkExpGroup'].astype(str), y='ConvertedCompYearly',
                     opacity=0.45, title="Salary vs Work Experience",
                     labels={'x': 'Work Experience (Years)', 'ConvertedCompYearly': 'Yearly Salary (USD)'})
    fig.update_layout(yaxis_tickformat=',', height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Annual Compensation Distribution (0–300k USD)")
    comp_data_filtered = filtered_df['ConvertedCompYearly'][filtered_df['ConvertedCompYearly'].between(0, 300000)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(comp_data_filtered, bins=50, kde=True, color='royalblue', alpha=0.6, edgecolor='white', ax=ax)
    from matplotlib.ticker import StrMethodFormatter
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_title('Annual Compensation Distribution (0–300k USD)')
    ax.set_xlabel('Yearly Compensation (USD)')
    ax.set_ylabel('Number of Respondents')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)

    st.subheader("Job Satisfaction Scores")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df['JobSat'].dropna(), bins=10, kde=True, ax=ax)
    ax.set_title('Job Satisfaction Scores')
    st.pyplot(fig)

    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_cols = ['WorkExp', 'YearsCode', 'ConvertedCompYearly', 'JobSat']
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

# =========================================
# TAB 3: Roles & Experience
# =========================================
with tab3:
    st.subheader("Top Paying Developer Roles")
    df_dev_salary = filtered_df[['DevType', 'ConvertedCompYearly']].dropna()
    df_dev_salary = df_dev_salary[df_dev_salary['ConvertedCompYearly'] <= 300000]
    df_dev_salary = df_dev_salary.assign(DevType=df_dev_salary['DevType'].str.split(',')).explode('DevType')
    df_dev_salary['DevType'] = df_dev_salary['DevType'].str.strip()
    top_roles = df_dev_salary.groupby('DevType')['ConvertedCompYearly'].median().sort_values(ascending=False).head(15)
    fig = px.bar(top_roles, x=top_roles.values, y=top_roles.index, orientation='h',
                 title="Median Salary by Developer Role")
    fig.update_layout(xaxis_tickformat=',')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Paying Countries")
    country_salary = filtered_df[filtered_df['ConvertedCompYearly'].between(1000, 300000)] \
        .groupby('Country')['ConvertedCompYearly'].median().sort_values(ascending=False).head(20)
    fig = px.bar(country_salary, x=country_salary.values, y=country_salary.index, orientation='h',
                 title="Median Salary by Country")
    fig.update_layout(xaxis_tickformat=',')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 Programming Languages Worked With")
    df_languages = filtered_df['LanguageHaveWorkedWith'].str.split(';').explode()
    lang_counts = df_languages.value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=lang_counts.values, y=lang_counts.index, ax=ax)
    ax.set_title('Top 20 Languages Respondents Have Worked With')
    st.pyplot(fig)

    st.subheader("Top 15 Databases Worked With (excluding Unknown)")
    df_db = filtered_df['DatabaseHaveWorkedWith'].str.split(';').explode()
    db_counts = df_db[df_db != 'Unknown'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=db_counts.values, y=db_counts.index, ax=ax)
    ax.set_title('Top 15 Databases Worked With (excluding Unknown)')
    ax.set_xlabel('Number of Respondents')
    ax.set_ylabel('Database')
    st.pyplot(fig)

# =========================================
# TAB 4: AI Insights
# =========================================
with tab4:
    st.subheader("AI Tool Learning Adoption")
    label_map = {
        'Yes, I learned how to use AI-enabled tools required for my job or to benefit my career': 'Yes: Career',
        'Yes, I learned how to use AI-enabled tools for my personal curiosity and/or hobbies': 'Yes: Personal',
        'No, I didn\'t spend time learning in the past year': 'No Learning',
        'No, I learned something that was not related to AI or AI enablement for my personal curiosity and/or hobbies': 'Non-AI: Personal',
        'No, I learned something that was not related to AI or AI enablement as required for my job or to benefit my career': 'Non-AI: Career',
    }
    filtered_df['Short_LearnCodeAI'] = filtered_df['LearnCodeAI'].map(label_map).fillna('Unknown')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=filtered_df, x='Short_LearnCodeAI', hue='Short_LearnCodeAI',
                  order=filtered_df['Short_LearnCodeAI'].value_counts().index,
                  palette='viridis', legend=False, ax=ax)
    ax.set_title('AI Tool Learning Adoption (Survey Results)')
    plt.xticks(rotation=15, ha='right')
    ax.set_xlabel('Learning Category')
    ax.set_ylabel('Number of Respondents')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AI Sentiment among Respondents")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='AISent', order=filtered_df['AISent'].value_counts().index, ax=ax)
        plt.xticks(rotation=45)
        ax.set_title('AI Sentiment')
        st.pyplot(fig)

    with col2:
        st.subheader("Perceived Threat of AI")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='AIThreat', order=filtered_df['AIThreat'].value_counts().index, ax=ax)
        ax.set_title('Perceived Threat of AI')
        st.pyplot(fig)

    st.subheader("AI Usage vs Job Satisfaction")
    ai_usage_mean = filtered_df[['AISelect', 'JobSat']].dropna() \
        .groupby('AISelect')['JobSat'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=ai_usage_mean.values, y=ai_usage_mean.index, ax=ax)
    ax.set_xlabel("Average Job Satisfaction")
    st.pyplot(fig)

    st.subheader("AI Threat Perception vs Job Satisfaction")
    ai_threat_median = filtered_df[['AIThreat', 'JobSat']].dropna() \
        .groupby('AIThreat')['JobSat'].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=ai_threat_median.values, y=ai_threat_median.index, ax=ax)
    ax.set_xlabel("Median Job Satisfaction")
    st.pyplot(fig)

# =========================================
# TAB 5: Learning Impact
# =========================================
with tab5:
    st.subheader("Learning Methods vs Salary")
    learn_salary = filtered_df[['LearnCode', 'ConvertedCompYearly']].dropna()
    learn_salary = learn_salary[learn_salary['ConvertedCompYearly'] <= 300000]
    learn_exploded = learn_salary.assign(LearnCode=learn_salary['LearnCode'].str.split(';')).explode('LearnCode')
    learn_exploded['LearnCode'] = learn_exploded['LearnCode'].str.strip()
    learn_stats = learn_exploded.groupby('LearnCode')['ConvertedCompYearly'].median().sort_values()
    fig = px.bar(learn_stats, x=learn_stats.values, y=learn_stats.index, orientation='h',
                 title="Median Salary by Learning Method")
    fig.update_layout(xaxis_tickformat=',', height=600)
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 6: Future ML – Salary Prediction
# =========================================
with tab6:
    st.header("🔮 Salary Prediction ML Models")
    st.markdown("Input the parameters on the **sidebar** to predict salary using multiple models.")

    # Assuming models exist in the folder
    gb_model = joblib.load("models/GradientBoosting_model.pkl")
    rf_model = joblib.load("models/RandomForest_model.pkl")
    ridge_model = joblib.load("models/Ridge_model.pkl")

    with st.form("salary_form"):
        submitted = st.form_submit_button("Predict Salary")
        if submitted:
            gb_pred = gb_model.predict(input_data)[0]
            rf_pred = rf_model.predict(input_data)[0]
            ridge_pred = ridge_model.predict(input_data)[0]
            st.success("Predictions Complete ✅")
            col1, col2, col3 = st.columns(3)
            col1.metric("Gradient Boosting", f"${gb_pred:,.0f}")
            col2.metric("Random Forest", f"${rf_pred:,.0f}")
            col3.metric("Ridge Regression", f"${ridge_pred:,.0f}")
            pred_df = pd.DataFrame({
                "Model": ["GradientBoosting", "RandomForest", "Ridge"],
                "Predicted Salary": [gb_pred, rf_pred, ridge_pred]
            }).set_index("Model")
            st.bar_chart(pred_df)

    st.info("""
    🔮 **Planned ML Features**
    - Role recommendation
    - Job satisfaction modeling
    - Explainable AI insights (SHAP/LIME)
    """)

# =========================================
# TAB 7: 💡 Key Insights (Greatly Enhanced)
# =========================================
with tab7:
    st.header("💡 Strategic Intelligence: Key Trends from Stack Overflow 2025")
    st.markdown("### Insights from 49,000+ global developer responses")
    st.markdown("---")

    # Row 1: AI Paradox & Adoption
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 The AI Paradox: High Usage, Low Trust")
        st.warning("""
        - **84%** of developers now use or plan to use AI tools (up from 76% in 2024).
        - Only **3%** "highly trust" AI code/output accuracy.
        - **46%** distrust AI-generated code due to frequent inaccuracies.
        - **66%** say AI suggestions are "almost right but require fixes" — creating hidden productivity tax.
        - Developers using AI report **higher job satisfaction**, suggesting perceived value despite risks.
        """)

    with col2:
        st.subheader("💰 Compensation Leaders & Geography")
        st.success("""
        - **Top Roles by Median Salary**: Engineering Managers, Executives, Cloud Architects.
        - **Highest-Paying Countries**: United States, Israel, Switzerland, Australia, Ireland.
        - **Nomadic/Remote-First** developers earn significantly more than in-person counterparts.
        - **Hybrid (Mostly Remote)** is the dominant work model globally.
        """)

    st.markdown("---")

    # Row 2: Tech Stack Dominance
    st.subheader("🛠️ The 2025 Tech Stack Landscape")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Programming Languages**")
        st.write("""
        - **JavaScript** remains #1 (66% usage).
        - **Python** solid at #2 (58%), growing fastest among professionals.
        - **Rust** and **Go** continue strong growth in system-level roles.
        """)

    with c2:
        st.markdown("**Databases**")
        st.write("""
        - **PostgreSQL** is now the undisputed leader (~60% usage).
        - **MySQL** declining but still widely used.
        - **MongoDB** dominates NoSQL space.
        """)

    with c3:
        st.markdown("**Cloud & Tools**")
        st.write("""
        - **AWS** leads cloud platforms.
        - **Docker** and **Git** near-universal adoption.
        - **GitHub Copilot** most widely used AI coding assistant.
        """)

    st.markdown("---")

    # Row 3: Workforce & Satisfaction
    st.subheader("👥 Workforce Dynamics & Developer Happiness")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Remote Work Revolution**")
        st.write("""
        - Only **~20%** of developers work fully in-person.
        - **Fully Remote** and **Hybrid (Mostly Remote)** are preferred by majority.
        - Remote workers report **higher satisfaction** and **higher pay**.
        """)

    with col_b:
        st.markdown("**Job Satisfaction Peak**")
        st.write("""
        - Massive spike at **8/10** satisfaction score.
        - Strongest predictors: **fair compensation**, **work-life balance**, **autonomy**, and **learning opportunities**.
        - Lowest satisfaction: micromanagement, bureaucracy, legacy codebases.
        """)

    st.markdown("---")

    # Final Strategic Takeaways
    st.subheader("🎯 Strategic Recommendations for 2026")
    st.markdown("""
    1. **Invest in AI literacy** — not just tool usage, but critical evaluation of outputs.
    2. **Prioritize flexibility** — remote/hybrid models attract and retain top talent.
    3. **Focus on PostgreSQL + Python + Cloud** stack for hiring competitive advantage.
    4. **Upskill in management & architecture** — highest salary growth paths.
    5. **Self-learners win** — developers using diverse/novel learning methods earn the most.
    6. **Trust is the next AI battleground** — tools that prove accuracy will dominate.
    """)

    st.divider()
    
st.markdown("---")
st.caption(" – Stack Overflow 2025 EDA & ML Dashboard")