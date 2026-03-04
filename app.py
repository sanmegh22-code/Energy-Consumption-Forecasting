from flask import Flask, render_template, request, send_file
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = None
accuracy = None
train_size = None
test_size = None
actual_values = None
predicted_values = None


# TRAIN MODEL FUNCTION
def train_model(data):
    global model, accuracy, train_size, test_size, actual_values, predicted_values

    features = ["T1", "RH_1", "T2", "RH_2"]
    target = "Appliances"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_size = len(X_train)
    test_size = len(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = round(r2_score(y_test, y_pred), 3)

    actual_values = y_test[:20].tolist()
    predicted_values = y_pred[:20].tolist()


# HOME
@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=accuracy,
        train_size=train_size,
        test_size=test_size,
        actual_values=actual_values,
        predicted_values=predicted_values
    )


# TRAIN PREDEFINED DATASET
@app.route("/train_predefined", methods=["POST"])
def train_predefined():
    try:
        data = pd.read_csv("energydata_complete.csv", encoding="latin1")
        train_model(data)

        return render_template(
            "index.html",
            message="Predefined dataset trained successfully!",
            accuracy=accuracy,
            train_size=train_size,
            test_size=test_size,
            actual_values=actual_values,
            predicted_values=predicted_values
        )

    except Exception as e:
        return render_template("index.html", message=str(e))


# UPLOAD CUSTOM DATASET
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["dataset"]

        if not file:
            return render_template("index.html", message="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        if file.filename.endswith(".csv"):
            data = pd.read_csv(filepath, encoding="latin1")

        elif file.filename.endswith(".xlsx"):
            data = pd.read_excel(filepath)

        else:
            return render_template("index.html", message="Unsupported file type")

        train_model(data)

        return render_template(
            "index.html",
            message="Custom dataset trained successfully!",
            accuracy=accuracy,
            train_size=train_size,
            test_size=test_size,
            actual_values=actual_values,
            predicted_values=predicted_values
        )

    except Exception as e:
        return render_template("index.html", message=str(e))


# PREDICT ENERGY
@app.route("/predict", methods=["POST"])
def predict():

    global model

    if model is None:
        return render_template("index.html", message="Train the model first!")

    try:
        t1 = float(request.form["T1"])
        rh1 = float(request.form["RH_1"])
        t2 = float(request.form["T2"])
        rh2 = float(request.form["RH_2"])

        prediction = model.predict([[t1, rh1, t2, rh2]])

        energy_wh = round(prediction[0], 2)
        energy_kwh = round(energy_wh / 1000, 3)

        if energy_wh < 100:
            level = "Low Usage"
        elif energy_wh < 300:
            level = "Moderate Usage"
        else:
            level = "High Usage"

        result = f"""
        Appliance Electricity Consumption:
        {energy_wh} Wh ({energy_kwh} kWh)
        Usage Level: {level}
        """

        return render_template(
            "index.html",
            result=result,
            accuracy=accuracy,
            train_size=train_size,
            test_size=test_size,
            actual_values=actual_values,
            predicted_values=predicted_values
        )

    except:
        return render_template("index.html", message="Invalid input values")


# DOWNLOAD PDF REPORT
@app.route("/download_report")
def download_report():

    file_path = "report.pdf"

    doc = SimpleDocTemplate(file_path)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Energy Consumption Estimation Report", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    data = [
        ["Model", "Random Forest Regressor"],
        ["R² Score", str(accuracy)],
        ["Training Samples", str(train_size)],
        ["Testing Samples", str(test_size)]
    ]

    table = Table(data)

    elements.append(table)

    doc.build(elements)

    return send_file(file_path, as_attachment=True)


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)