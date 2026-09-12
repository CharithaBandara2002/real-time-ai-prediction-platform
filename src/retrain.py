import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("Starting automated retraining...")

df = pd.read_csv("data/raw/current_data.csv")

df = df.drop("Address", axis=1)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment("House Price Prediction")

with mlflow.start_run(run_name="Automated Retraining"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    mlflow.log_param("retraining", True)
    mlflow.log_metric("r2_score", score)

    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="house_price_model"
    )

    print(f"Retraining completed. R² Score: {score:.4f}")