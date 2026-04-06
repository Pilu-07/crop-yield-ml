# 🌾 Smart Crop Recommendation System

A production-ready agricultural dashboard for Odisha-based crop recommendations and yield prediction. Powered by **FastAPI** and **CatBoost ML** models.

---

## 🚀 Quick Start (Local Setup)

Follow these steps to run the dashboard on your machine:

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Crops.git
cd Crops
```

### 2. Create and active a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

Open your browser and go to:
→ **http://127.0.0.1:8000**

---

## 🏗️ Project Architecture

- **`index.html`**: Premium, high-end agricultural dashboard UI.
- **`style.css`**: Professional dark-themed design system with responsive grid layout.
- **`script.js`**: Frontend logic for API communication, sliders, and result animation.
- **`app/main.py`**: FastAPI backend serving predictions and agricultural insights.
- **`models/`**: Pre-trained machine learning artifacts (.pkl).
- **`phase2_pipeline.py`**: Advanced ML training pipeline (Regression + Classification).
- **`requirements.txt`**: List of all required Python libraries.

---

## 🧪 Machine Learning Models
- **Yield Prediction**: CatBoost Regressor (RMSE Optimized for Odisha regional data).
- **Crop Recommendation**: Multi-class Classification targeting 13 unique crops.
- **Feature Engineering**: Automated soil fertility index (SFI) and water stress index (WSI) calculations.

---

## 📜 Dataset
The model is trained on the `Odisha_farming_final_dataset.csv`, covering agricultural performance data from 2015 to 2025 across all 30 districts of Odisha.

---

## 🛠️ Tech Stack
- **Frontend**: Vanilla HTML5, CSS3 (Modern Grid), JavaScript (ES6+).
- **Backend**: FastAPI, Pydantic.
- **ML**: CatBoost, Scikit-Learn, Pandas, NumPy.
- **Server**: Uvicorn.
