<p align="center">

# ✈️ Predictive Maintenance using Machine Learning

### Predicting Remaining Useful Life (RUL) of Aircraft Engines using NASA CMAPSS Dataset

</p>

<p align="center">

<!-- Replace after generating the final banner -->

<img src="assets/banner.png" width="100%">

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# 🚀 Project Highlights

- End-to-End Machine Learning Pipeline
- Professional Modular Architecture
- Advanced Feature Engineering
- Engine-wise Train/Test Split (No Data Leakage)
- Automated Reporting & Visualization
- Baseline Models: Linear Regression & Random Forest
- Easily Extendable to Gradient Boosting Models

---

# 📌 Project Overview

This project implements an end-to-end Machine Learning pipeline to predict the **Remaining Useful Life (RUL)** of aircraft engines using the **NASA CMAPSS turbofan engine dataset**.

The repository focuses not only on building accurate predictive models but also on following production-inspired software engineering practices including modular architecture, reusable workflows, automated evaluation, and reproducible experimentation.

---

# 🏗️ Project Architecture

<p align="center">

<img src="assets/project_pipeline.png" width="95%">

</p>

---

# ⚙️ Machine Learning Pipeline

| Stage | Status |
|---------------------------|:------:|
| Data Ingestion | ✅ |
| RUL Calculation | ✅ |
| Feature Engineering | ✅ |
| Exploratory Data Analysis | ✅ |
| Engine-wise Train/Test Split | ✅ |
| Data Preprocessing | ✅ |
| Linear Regression | ✅ |
| Random Forest | ✅ |
| XGBoost | 🚧 |
| LightGBM | 🚧 |
| CatBoost | 🚧 |
| Hyperparameter Tuning | 🚧 |
| SHAP Explainability | 🚧 |
| Streamlit Dashboard | 🚧 |

---

# 📂 Project Structure

```text
predictive_maintenance_rul/

├── configs/
├── dashboard/
├── data/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tests/
├── main.py
├── README.md
└── requirements.txt
```

---

# 📊 Dataset

The project uses the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.

Supported datasets

- FD001
- FD002
- FD003
- FD004

Each dataset contains

- Operational Settings
- Sensor Measurements
- Engine Cycles
- Remaining Useful Life (RUL)

---

# 🤖 Models

| Model | Status |
|-----------------------|:------:|
| Linear Regression | ✅ |
| Random Forest Regressor | ✅ |
| XGBoost | 🚧 |
| LightGBM | 🚧 |
| CatBoost | 🚧 |

---

# 📈 Current Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|----------------------|---------:|---------:|---------:|
| Random Forest | **29.950** | **42.339** | **0.701** |
| Linear Regression | 36.711 | 47.331 | 0.626 |

🏆 **Current Best Model : Random Forest Regressor**

---

# 📸 Sample Results

<p align="center">

<img src="assets/results_preview.png" width="90%">

</p>

The project automatically generates

- Feature Consistency Heatmap
- Sensor Trend Analysis
- Correlation Analysis
- Actual vs Predicted Plots
- Model Comparison Reports

---

# 💡 Engineering Decisions

Some important engineering choices made during development

- Engine-wise Train/Test Split to eliminate data leakage.
- Removed identifier columns before training.
- Scaling applied only where required.
- Configuration-driven architecture using YAML.
- Modular pipeline for maintainability.
- Automatic report generation.
- Reusable workflow architecture for every ML model.

---

# 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib

### Utilities

- Joblib
- YAML

---

# 🚀 Getting Started

Clone repository

```bash
git clone https://github.com/<your-username>/predictive-maintenance-rul.git
```

Move into project

```bash
cd predictive-maintenance-rul
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

# 🔮 Future Improvements

- XGBoost
- LightGBM
- CatBoost
- Hyperparameter Optimization (Optuna)
- SHAP Explainability
- MLflow Experiment Tracking
- Interactive Streamlit Dashboard
- Docker Deployment

---

# 👨‍💻 Author

**Gaurav Pardeshi**

GitHub

https://github.com/<your-username>

LinkedIn

https://linkedin.com/in/<your-profile>

---

# 📄 License

This project is licensed under the MIT License.

---

<p align="center">

⭐ If you found this project interesting, consider giving it a Star!

</p>