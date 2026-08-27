# Heart Disease Prediction Using Machine Learning

# Project Overview

This project is a Heart Disease Prediction System developed using Python and Machine Learning. It predicts the possibility of heart disease based on patient medical information.

The project also includes a Flask web application that allows users to enter medical details and receive a prediction.

# Objective

To develop a machine learning model that can predict the possibility of heart disease using patient medical data.

# Dataset

The project uses a Heart Disease dataset containing medical information such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* Slope
* Number of Major Vessels
* Thalassemia

*Target:*

* `0` – No heart disease
* `1` – Possibility of heart disease

## Machine Learning

The project uses classification algorithms including:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost

The models are evaluated using **Accuracy, Precision, Recall, and F1-Score**.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Flask
* HTML & CSS
* Joblib

## Project Structure

text
Heart-Disease-Prediction/
│
├── dataset/
│   └── heart.csv
│
├── model/
│   └── heart_model.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── train_model.py
├── disease_prediction.py
├── requirements.txt
├── .gitignore
└── README.md


## How to Run

## 1. Clone the repository
bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Heart-Disease-Prediction


## 2. Create a virtual environment
bash
python -m venv venv


## 3. Activate the virtual environment

Windows PowerShell:
bash
venv\Scripts\Activate.ps1

## 4. Install dependencies

bash
pip install -r requirements.txt

## 5. Run the Flask application
bash
python disease_prediction.py


Open the URL shown in the terminal, usually:

text
http://127.0.0.1:5000/



