# housing_task.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("housing_labelled.csv")

# Identify target and features
target = "median_house_value"  # Assuming the dataset has 'price' as the target
features = df.columns.drop([target]).tolist()

X = df[features]
y = df[target]

print(y.describe())


# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define 5 feature sets (adapt based on your actual column names)
feature_sets = {
    # All raw features (baseline)
    "1. All features": X_scaled,
    
    # Only location-based features: longitude, latitude, median_income
    "2. Location + Income": np.hstack([
        X_scaled[:, [0]],  # longitude
        X_scaled[:, [1]],  # latitude
        X_scaled[:, [7]]   # median_income
    ]),
    
    # House size features + income (drop population/demographics)
    "3. Size + Income": np.hstack([
        X_scaled[:, [2]],  # housing_median_age
        X_scaled[:, [3]],  # total_rooms
        X_scaled[:, [4]],  # total_bedrooms
        X_scaled[:, [7]]   # median_income
    ]),
    
    # Population density features
    "4. Density Features": np.hstack([
        X_scaled[:, [5]],  # population
        X_scaled[:, [6]],  # households
        (X_scaled[:, [5]] / (X_scaled[:, [6]] + 1e-5))  # population per household
    ]),
    
    # PCA on all features except median_house_value
    "5. PCA (3 components)": PCA(n_components=3).fit_transform(X_scaled)
}

# Store results for table
results = []

for name, X_feat in feature_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds_train = lr_model.predict(X_train)
    lr_mse = mean_squared_error(y_train, lr_preds_train)

    # Neural Network Regressor
    nn_model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=3000,early_stopping=True, random_state=42)
    nn_model.fit(X_train, y_train)
    nn_preds_train = nn_model.predict(X_train)
    nn_mse = mean_squared_error(y_train, nn_preds_train)

    results.append((name, lr_mse, nn_mse))

    # Detailed per feature set
    print(f"\n=== Feature Set: {name} ===")
    print(f"[Linear Regression] Training MSE: {lr_mse:.4f}")
    print(f"[Neural Network]   Training MSE: {nn_mse:.4f}")


y_mean = np.mean(y_train)
baseline_preds = np.full_like(y_train, y_mean)
baseline_mse = mean_squared_error(y_train, baseline_preds)
print("\nBaseline MSE (mean predictor):", baseline_mse)

# Summary table
print("\nSummary of Training MSE:")
print(f"{'Features Used':<25} {'LinReg MSE':<15} {'NeuralNet MSE'}")
print("-" * 55)
for r in results:
    print(f"{r[0]:<25} {r[1]:<15.4f} {r[2]:.4f}")

plt.scatter(y_train, lr_preds_train, alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Linear Regression Predictions")
plt.title("Linear Regression Predictions vs True")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.show()
