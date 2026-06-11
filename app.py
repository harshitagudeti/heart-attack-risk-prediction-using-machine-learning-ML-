from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# ===== LOAD MODEL & SCALER =====
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===== HOME PAGE =====
@app.route("/")
def home():
    return render_template("index.html")


# ===== PREDICTION ROUTE =====
@app.route("/predict", methods=["POST"])
def predict():

    # ===== GET VALUES FROM FORM =====
    age = float(request.form["age"])
    chol = float(request.form["cholesterol"])
    hr = float(request.form["heartrate"])
    bmi = float(request.form["bmi"])
    stress = float(request.form["stress"])

    # ===== PREPARE INPUT =====
    input_data = np.array([[age, chol, hr, bmi, stress]])
    input_scaled = scaler.transform(input_data)

    # ===== MODEL PREDICTION =====
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    risk_percent = int(probability * 100)

    if prediction[0] == 1:
        result = "⚠️ High Heart Attack Risk"
        color = "red"
    else:
        result = "✅ Low Heart Attack Risk"
        color = "green"

    # ===== CREATE GRAPH =====
    labels = ["Age", "Cholesterol", "Heart Rate", "BMI", "Stress"]
    values = [age, chol, hr, bmi, stress]

    plt.figure(figsize=(7,4))
    bars = plt.bar(labels, values)

    # color bars
    for bar in bars:
        bar.set_color("#1976d2")

    plt.title("Patient Health Parameters")
    plt.ylabel("Values")
    plt.tight_layout()

    # ensure static folder exists
    if not os.path.exists("static"):
        os.makedirs("static")

    graph_path = os.path.join("static", "graph.png")
    plt.savefig(graph_path)
    plt.close()

    # ===== SEND DATA TO RESULT PAGE =====
    return render_template(
        "result.html",
        result=result,
        risk_percent=risk_percent,
        color=color,
        graph="graph.png"
    )


# ===== RUN SERVER =====
if __name__ == "__main__":
    app.run(debug=True)