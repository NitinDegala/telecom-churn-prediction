from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standar_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            features = [
                float(request.form["SeniorCitizen"]),
                float(request.form["tenure"]),
                float(request.form["MonthlyCharges"]),
                float(request.form["TotalCharges"]),
                float(request.form["SeniorCitizen_yeo"]),
                float(request.form["gender_Male"]),
                float(request.form["Dependents_Yes"]),
                float(request.form["PhoneService_Yes"]),
                float(request.form["MultipleLines_No phone service"]),
                float(request.form["MultipleLines_Yes"]),
                float(request.form["InternetService_Fiber optic"]),
                float(request.form["InternetService_No"]),
                float(request.form["OnlineSecurity_No internet service"]),
                float(request.form["OnlineSecurity_Yes"]),
                float(request.form["OnlineBackup_No internet service"]),
                float(request.form["OnlineBackup_Yes"]),
                float(request.form["DeviceProtection_No internet service"]),
                float(request.form["DeviceProtection_Yes"]),
                float(request.form["TechSupport_No internet service"]),
                float(request.form["TechSupport_Yes"]),
                float(request.form["StreamingTV_No internet service"]),
                float(request.form["StreamingTV_Yes"]),
                float(request.form["StreamingMovies_No internet service"]),
                float(request.form["StreamingMovies_Yes"]),
                float(request.form["PaperlessBilling_Yes"]),
                float(request.form["PaymentMethod_Credit card (automatic)"]),
                float(request.form["PaymentMethod_Electronic check"]),
                float(request.form["PaymentMethod_Mailed check"]),
                float(request.form["Provider_BSNL"]),
                float(request.form["Provider_Jio"]),
                float(request.form["Provider_VI"]),
                float(request.form["Contract_od"])
            ]

            # Convert to numpy array
            features_array = np.array([features])

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Make prediction
            prediction = model.predict(features_scaled)[0]
            if prediction == 0:
                return render_template("index.html", prediction='Bad Customer')
            else:
                return render_template("index.html", prediction="Good Customer")

        except Exception as e:
            # If an error occurs during POST, render the template with the error message
            prediction = f"Error: {str(e)}"
            return render_template("index.html", prediction=prediction) # <--- Added return here for error handling

    # This handles the initial GET request (when the page is loaded)
    # and any scenario where the POST block didn't execute or encountered an error.
    return render_template("index.html", prediction=prediction) # <--- ADDED REQUIRED RETURN STATEMENT

if __name__ == "__main__":
    app.run(debug=True)