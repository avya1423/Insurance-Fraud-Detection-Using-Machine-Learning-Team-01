# 🛡️ Insurance Fraud Detection System
### Machine Learning — College Mini-Project

A complete fraud detection system for **Automobile**, **Health**, and **Property** insurance claims, powered by Logistic Regression, Decision Tree, and Random Forest classifiers.

---

## 📁 Project Structure

```
insurance_fraud_detection/
│
├── app.py                      # 🌐 Streamlit web interface (main UI)
├── train.py                    # 🏋️  Train all models (run this first!)
├── requirements.txt            # 📦 Python dependencies
│
├── data/
│   ├── __init__.py
│   └── generate_data.py        # 🔢 Synthetic dataset generator
│
├── models/
│   ├── __init__.py
│   └── train_models.py         # 🤖 Model training & evaluation logic
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py        # 🔧 Data preprocessing utilities
│
├── scenarios/
│   ├── __init__.py
│   ├── auto_fraud.py           # 🚗 Automobile insurance pipeline
│   ├── health_fraud.py         # 🏥 Health insurance pipeline
│   └── property_fraud.py       # 🏠 Property insurance pipeline
│
├── saved_models/               # 💾 Persisted models (auto-created)
│   ├── auto_model.pkl
│   ├── auto_scaler.pkl
│   ├── health_model.pkl
│   ├── health_scaler.pkl
│   ├── property_model.pkl
│   └── property_scaler.pkl
│
└── plots/                      # 🖼️ Evaluation charts (auto-created)
    ├── auto_confusion.png
    ├── auto_roc.png
    ├── auto_model_comparison.png
    ├── auto_feature_importance.png
    ├── health_*.png
    └── property_*.png
```

---

## 🚀 How to Run Locally

### 1. Clone / Download the Project
```bash
cd insurance_fraud_detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train All Models
```bash
python train.py
```
This will:
- Generate synthetic datasets for all 3 scenarios
- Train 3 ML models per scenario (9 models total)
- Print accuracy, AUC-ROC, and classification reports
- Save best models to `saved_models/`
- Save evaluation plots to `plots/`

### 5. Launch the Web Interface
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

---

## 🧠 Machine Learning Models

| Model               | Type              | Key Hyperparameters              |
|---------------------|-------------------|----------------------------------|
| Logistic Regression | Linear Classifier | C=0.5, max_iter=1000             |
| Decision Tree       | Tree-based        | max_depth=8                      |
| Random Forest       | Ensemble          | n_estimators=150, max_depth=12   |

The **best model** (by AUC-ROC) is automatically saved and used for predictions.

---

## 📊 Evaluation Metrics

For each scenario, the system generates:
- ✅ **Accuracy** — percentage of correct predictions
- ✅ **AUC-ROC** — ability to distinguish fraud from legitimate
- ✅ **Confusion Matrix** — visualised as a heatmap
- ✅ **Classification Report** — precision, recall, F1-score per class
- ✅ **Feature Importance** — top features driving fraud detection
- ✅ **ROC Curves** — comparison across all models

---

## 🔍 Scenarios Explained

### Scenario 1: 🚗 Automobile Insurance
Detects fraud based on:
- Inflated claim amounts vs. vehicle price
- High-frequency past claimants
- New policy + immediate claim
- Odd-hour incidents, no witnesses/police report

### Scenario 2: 🏥 Health Insurance
Detects fraud based on:
- Excessive diagnoses and procedures
- Overbilling ratio > 1.5×
- Duplicate claim flags
- Very short interval between claims
- Too many physicians for one patient

### Scenario 3: 🏠 Property Insurance
Detects fraud based on:
- Claim amount ≈ total property value
- New policy + immediate claim
- Old property, maximum damage severity
- No photos, no police/fire report
- No third-party assessment

---

## 📦 Dataset

All datasets are **synthetically generated** (no real personal data):
- 2,000 records per scenario
- 75% Legitimate, 25% Fraud (realistic class imbalance)
- Features engineered to reflect real-world fraud patterns

---

## 🏆 Expected Results

| Scenario   | Best Model    | Accuracy | AUC-ROC |
|------------|---------------|----------|---------|
| Auto       | Random Forest | ~96%     | ~0.99   |
| Health     | Random Forest | ~97%     | ~0.99   |
| Property   | Random Forest | ~97%     | ~0.99   |

---

## 📋 Tech Stack

| Component       | Technology          |
|-----------------|---------------------|
| Language        | Python 3.9+         |
| Data Processing | Pandas, NumPy        |
| ML Models       | Scikit-learn         |
| Visualisation   | Matplotlib, Seaborn  |
| Web Interface   | Streamlit            |
| Model Saving    | Joblib               |

---

## 🎓 College Demonstration Notes

This project demonstrates:
1. **End-to-end ML pipeline**: from raw data → trained model → web UI
2. **Multi-class problem handling**: 3 independent fraud domains
3. **Model comparison**: objective selection using AUC-ROC
4. **Real-world feature engineering**: domain-specific fraud indicators
5. **Production-ready patterns**: model persistence, modular code, clean UI

---

*Built as a college mini-project for demonstration purposes.*
