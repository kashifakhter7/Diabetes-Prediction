# 🩺 Diabetes Prediction System Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](#)

A clinical diagnostic support system leveraging Machine Learning to predict the onset of diabetes based on standard health diagnostic measurements.

Featuring an exploratory analysis pipeline, comparative benchmarking of 11 machine learning algorithms, model serialization, and an interactive **Streamlit** web application for real-time risk assessment.

---

## 📌 Project Overview

Diabetes is a chronic metabolic disease characterized by elevated levels of blood glucose. Early risk detection enables preventative lifestyle modifications and timely clinical treatment.

This project uses clinical diagnostic metrics from female patients of Pima Indian heritage to classify individuals into:
- **Diabetic (Outcome = 1)**: High risk of diabetes.
- **Non-Diabetic (Outcome = 0)**: Low risk / healthy diagnostic profile.

The system scales patient measurements using `StandardScaler` and predicts diabetes probability with an optimized **AdaBoost Classifier** model.

---

## ✨ Features

- **Multi-Algorithm Benchmarking**: Evaluated 11 classical and ensemble ML algorithms to identify the optimal classifier.
- **Top-Tier Ensemble Model**: AdaBoost classifier achieving the highest diagnostic accuracy (~78%).
- **Interactive Streamlit Web Interface**: Clean, accessible clinical form for inputting patient measurements and getting immediate risk feedback.
- **Standardized Preprocessing**: Robust data normalization with Scikit-Learn's `StandardScaler`.
- **Zero Retraining Required**: Pre-trained model serialized into `model.pkl` for immediate plug-and-play inference.

---

## 🏗️ Project Architecture

```plaintext
Diabetes-Prediction/
│
├── data_set/
│   └── diabetes.csv                    # Pima Indians Diabetes Dataset
│
├── ipynb/
│   └── Diabetes_Prediction.ipynb       # Jupyter Notebook with exploratory data analysis & training
│
├── pkl_folder/
│   └── model.pkl                       # Serialized trained AdaBoost model
│
├── python_folder/
│   ├── diabetes_prediction.py          # Data preprocessing, model training & evaluation script
│   └── frontend.py                     # Streamlit interactive web dashboard
│
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Comprehensive documentation
```

---

## 📊 Dataset Description

The project is trained on the **Pima Indians Diabetes Database** (`data_set/diabetes.csv`):

- **Total Records**: 768 female patients
- **Target Distribution**:
  - `Non-Diabetic (0)`: 500 instances (65.1%)
  - `Diabetic (1)`: 268 instances (34.9%)
- **Diagnostic Features (8 Clinical Attributes)**:
  1. **Pregnancies**: Number of times pregnant
  2. **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
  3. **Blood Pressure**: Diastolic blood pressure ($mm\ Hg$)
  4. **Skin Thickness**: Triceps skin fold thickness ($mm$)
  5. **Insulin**: 2-Hour serum insulin ($\mu U/ml$)
  6. **BMI**: Body Mass Index ($weight\ in\ kg / (height\ in\ m)^2$)
  7. **Diabetes Pedigree Function**: Genetic scoring metric representing diabetic family history
  8. **Age**: Patient age in years

---

## 📈 Model Performance & Comparison

All models were evaluated using an 80/20 train-test split (`random_state=42`):

| Algorithm | Test Accuracy | Status / Key Observation |
|:---|:---:|:---|
| 🏆 **AdaBoost Classifier** | **77.92%** | **Top Performer** (Best ensemble generalization) |
| **Voting Classifier** | 77.27% | Strong ensemble blending SVM, RF, and Logistic Regression |
| **Naive Bayes (Gaussian)** | 76.62% | Fast probabilistic baseline |
| **Extra Trees Classifier** | 75.97% | Reduced variance through randomized splits |
| **Logistic Regression** | 75.32% | Standard linear clinical benchmark |
| **Decision Tree** | 75.32% | Transparent decision rule thresholds |
| **Random Forest** | 74.68% | Multi-tree bagging ensemble |
| **Gradient Boosting** | 74.68% | Boosting baseline |
| **Support Vector Machine (SVM)** | 73.38% | Effective margin boundary separation |
| **XGBoost** | 70.78% | Gradient boosted trees |
| **K-Nearest Neighbors (KNN)** | 69.48% | Sensitive to distance variations |

> **Model Selection**: **AdaBoost Classifier** was selected as the final production model for deployment as it achieved the highest accuracy of **77.92%** on unseen test data.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Diabetes-Prediction.git
cd Diabetes-Prediction
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
Ensure `model.pkl` from `pkl_folder/` is accessible in your execution directory, then launch:
```bash
streamlit run python_folder/frontend.py
```

The web dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 💻 How to Use the Web App

1. Launch the Streamlit application using the command above.
2. Enter the patient's diagnostic values in the input fields:
   - Pregnancies, Glucose level, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
3. Click the **Predict** button.
4. View the instant diagnosis:
   - 🚨 **High risk of diabetes detected!** (Follow-up clinical consultation suggested)
   - ✅ **Low risk of diabetes.** (Healthy profile indicated)

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Analysis**: Pandas, NumPy
- **Interactive Frontend**: Streamlit
- **Model Storage**: Pickle

---

## 📜 Medical Disclaimer

This project is created strictly for **educational, experimental, and research purposes**. It should never replace professional medical diagnosis, advice, or treatment by a qualified healthcare professional.

---

## 🤝 Contributing

Contributions, feedback, and suggestions are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
