from flask import Flask, render_template, request
import joblib
import numpy as np


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("model/heart_model.pkl")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from HTML form

        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])


        # Put all values into correct order

        input_data = np.array([
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]).reshape(1, -1)


        # Make prediction

        prediction = model.predict(input_data)[0]


        # Get probability if available

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)

            probability = round(
                float(np.max(probabilities)) * 100,
                2
            )


        # Check prediction

        if prediction == 1:

            result = " Possibility of Heart Disease"

            result_class = "danger"

        else:

            result = "No Heart Disease Detected"

            result_class = "success"


        return render_template(
            "index.html",
            prediction=result,
            probability=probability,
            result_class=result_class
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error: Please check your input values.",
            result_class="error"
        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
