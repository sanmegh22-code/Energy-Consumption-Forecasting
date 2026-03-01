from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables
model = None
model_name = None
r2 = None
mae = None
rmse = None
train_size = None
test_size = None
feature_importance = None

actual_sample = None
predicted_sample = None
sample_error = None


# ------------------------------------------------
# Train Model Function
# ------------------------------------------------
def train_model(data):
    global model, model_name, r2, mae, rmse
    global train_size, test_size, feature_importance
    global actual_sample, predicted_sample, sample_error

    features = ["T1", "RH_1", "T2", "RH_2"]
    target = "Appliances"

    if not all(col in data.columns for col in features + [target]):
        raise Exception("Dataset must contain columns: T1, RH_1, T2, RH_2, Appliances")

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_size = len(X_train)
    test_size = len(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    # Select Best Model
    if rf_r2 > lr_r2:
        model = rf
        model_name = "Random Forest Regressor"
        predictions = rf_pred
        r2 = round(rf_r2, 3)
        feature_importance = dict(zip(features, rf.feature_importances_))
    else:
        model = lr
        model_name = "Linear Regression"
        predictions = lr_pred
        r2 = round(lr_r2, 3)
        feature_importance = dict(zip(features, np.abs(lr.coef_)))

    mae = round(mean_absolute_error(y_test, predictions), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, predictions)), 3)

    # Take first test sample for display
    actual_sample = round(y_test.iloc[0], 2)
    predicted_sample = round(predictions[0], 2)
    sample_error = round(abs(actual_sample - predicted_sample), 2)


# ------------------------------------------------
# Home Route
# ------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html",
                           model_name=model_name,
                           r2=r2,
                           mae=mae,
                           rmse=rmse,
                           train_size=train_size,
                           test_size=test_size,
                           feature_importance=feature_importance,
                           actual_sample=actual_sample,
                           predicted_sample=predicted_sample,
                           sample_error=sample_error)


# ------------------------------------------------
# Train Predefined Dataset
# ------------------------------------------------
@app.route("/train_predefined", methods=["POST"])
def train_predefined():
    data = pd.read_csv("energydata_complete.csv", encoding="latin1")
    train_model(data)
    return home()


# ------------------------------------------------
# Upload Dataset
# ------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
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
    return home()


# ------------------------------------------------
# Prediction Route
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", message="Train the model first!")

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

    return render_template("index.html",
                           result=True,
                           t1=t1,
                           rh1=rh1,
                           t2=t2,
                           rh2=rh2,
                           energy_wh=energy_wh,
                           energy_kwh=energy_kwh,
                           level=level,
                           model_name=model_name,
                           r2=r2,
                           mae=mae,
                           rmse=rmse,
                           train_size=train_size,
                           test_size=test_size,
                           feature_importance=feature_importance,
                           actual_sample=actual_sample,
                           predicted_sample=predicted_sample,
                           sample_error=sample_error)


if __name__ == "__main__":
    app.run()