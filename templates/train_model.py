import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


model = None
accuracy = None
train_size = None
test_size = None


def train_model(data):
    global model, accuracy, train_size, test_size

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