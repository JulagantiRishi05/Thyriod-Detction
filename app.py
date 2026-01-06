from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

scaler = pickle.load(open("scaler.pkl", "rb"))
models = {
    "Logistic Regression": pickle.load(open("logistic.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest.pkl", "rb")),
    "SVM": pickle.load(open("svm.pkl", "rb")),
    "KNN": pickle.load(open("knn.pkl", "rb"))
}

def predict(age, sex, T3, T4, TSH):
    data = np.array([[age, sex, T3, T4, TSH]])
    data_scaled = scaler.transform(data)

    votes = []
    for name, model in models.items():
        pred = model.predict(data_scaled)[0]
        votes.append(pred)

    # Majority voting
    final = "Thyroid Detected" if sum(votes) > len(votes)/2 else "Normal"
    return final

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    T3 = float(request.form["t3"])
    T4 = float(request.form["t4"])
    TSH = float(request.form["tsh"])

    result = predict(age, sex, T3, T4, TSH)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
