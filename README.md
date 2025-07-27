# computational-RF-using-AI-ML-in-CO2-EOR

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


# Load Dataset

data = pd.read_csv('eclipse_data.csv')


# Define Features and Target

X = data[['InjectionRate', 'Porosity', 'Permeability', 'DistanceToInjector']]
y = data['Recovery']


# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)


# Predict on Test Data

y_pred = model.predict(X_test)


# 7. Evaluate the Model

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")


# Save the Trained Model

joblib.dump(model, "co2_eor_linear_model.joblib")
print("\nModel saved as 'co2_eor_linear_model.joblib'")

