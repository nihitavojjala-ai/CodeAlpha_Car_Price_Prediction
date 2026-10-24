	
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("car data.csv")
df.columns = df.columns.str.strip()
df = df.dropna().copy()

print("CAR PRICE PREDICTION")
print("-" * 30)
print("Dataset shape:", df.shape)
print(df.head())

# Select features and target
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numerical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200, random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nModel Evaluation")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# Save results and graph
os.makedirs("charts", exist_ok=True)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})
results.to_csv("prediction_results.csv", index=False)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.savefig("charts/price_prediction.png")
plt.close()

print("\nGraph saved in charts/price_prediction.png")
print("Results saved in prediction_results.csv")
print("Project completed successfully!")

