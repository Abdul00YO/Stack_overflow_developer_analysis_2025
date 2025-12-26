# 📊 Stack Overflow 2025 – EDA & ML Dashboard

![Dashboard Preview](images/dashboard_preview.png)

A comprehensive **interactive dashboard** built using **Python, Streamlit, Plotly, and Seaborn** for **analyzing Stack Overflow 2025 developer survey data**. It includes **EDA (Exploratory Data Analysis)**, **visualizations**, and **ML-based salary predictions**.

---

## 🚀 Features

### 1️⃣ Deep EDA
- **Age Distribution**: Visualize respondents’ age groups.
- **Country Insights**: Top countries by respondents and median salary.
- **Education Level**: Count of developers by education.
- **Employment Status**: Developer employment trends.
- **Developer Types**: Distribution of various developer roles.
- **Organization Size**: Visual analysis of company sizes.
- **Learning Methods**: Impact of learning resources on salaries.

### 2️⃣ Interactive Dashboards
- Built with **Streamlit** for real-time filtering and interactive exploration.
- Sidebar filters for **country**, **work experience**, and other parameters.
- Visualizations using **Plotly** and **Seaborn** for better interactivity.

### 3️⃣ AI & ML Insights
- **AI Adoption vs Job Satisfaction**: Analyze how AI usage impacts satisfaction.
- **AI Threat Perception**: Median satisfaction based on perceived AI threat.
- **Learning vs Salary**: Median salaries by different learning resources.

### 4️⃣ Salary Prediction (ML Models)
- Predict salary based on:
  - Work Experience (`WorkExp`)
  - Coding Experience (`YearsCode`)
  - Job Satisfaction (`JobSat`)
- Models used:
  - Gradient Boosting
  - Random Forest
  - Ridge Regression
- Input via **sidebar sliders** for real-time predictions.
- Compare predictions across models visually.

---

## 🗂 Repository Structure

```
Stack_Overflow_2025/
│
├─ dataset/              # CSV datasets (cleaned for dashboard)
├─ images/               # Dashboard screenshots and images
├─ models/               # Pretrained ML models (.pkl/.joblib)
├─ app.py                # Streamlit dashboard main app
├─ data_clean_1.ipynb    # Initial data cleaning notebook
├─ eda_2.ipynb           # EDA notebook 1
├─ eda_3.ipynb           # EDA notebook 2
├─ eda_4.ipynb           # EDA notebook 3
├─ model_5.ipynb         # ML model training notebook
├─ README.md             # This file
├─ .gitignore            # Git ignore file
└─ LICENSE               # License
```

---

## ⚡ Tech Stack
- **Python 3.10+**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn, Plotly** – Visualizations
- **Scikit-learn** – Machine Learning
- **Joblib** – Model serialization
- **Streamlit** – Interactive dashboard

---

## 🛠 Installation & Setup

1. **Clone the repo:**
```bash
git clone https://github.com/Abdul00YO/Stack_overflow_developer_analysis_2025.git
cd Stack_Overflow_2025
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scriptsctivate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the dashboard:
```bash
streamlit run app.py
```

---

## 📊 Screenshots

![Dashboard Home](images/home.png)  
*Dashboard Home*

![Salary Prediction](images/salary_prediction.png)  
*Salary Prediction*

![Learning vs Salary](images/learning_salary.png)  
*Learning vs Salary*

*(Replace with your actual screenshots in the `images/` folder)*

---

## 💡 Future Improvements
- Role recommendation based on experience & skills.
- Job satisfaction prediction & insights.
- Explainable AI for model transparency.
- Additional interactive visualizations.

---

## 📝 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author
**Abdullah Arshad** – Data Analyst & Developer  
Portfolio: [developer.hatissports.com](https://developer.hatissports.com)  
GitHub: [@Abdul00YO](https://github.com/Abdul00YO)

---

Thank you — the file now renders as Markdown on GitHub and in editors.
