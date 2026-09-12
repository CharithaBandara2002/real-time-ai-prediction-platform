import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/raw/housing.csv")

# Remove Address column
df = df.drop("Address", axis=1)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")
print("Saved in models/model.pkl")