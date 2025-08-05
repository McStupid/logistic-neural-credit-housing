# credit_card_fraud_task1.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("card_labelled.csv")

# Features and target
target = "fraudulent"
features = df.columns.drop(["Time", target]).tolist()

# print(df.columns)

X = df[features]
y = df[target]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define 5 feature sets
feature_sets = {
    "1. All features": X_scaled,
    "2. V1–V10": X_scaled[:, 1:11],  # V1 to V10 (excluding Time and Amount)
    "3. V1–V5 + Amount": pd.concat([
        pd.DataFrame(X_scaled[:, 1:6]),           # V1 to V5
        pd.DataFrame(X_scaled[:, -2])             # Amount 
    ], axis=1).values,
    "4. PCA (5 components)": PCA(n_components=5).fit_transform(X_scaled),
    "5. Time + Amount": X_scaled[:, [0, -2]]      # Time, Amount
}

# Store results
results = []

for name, X_feat in feature_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_precision = precision_score(y_test, lr_preds)
    lr_recall = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_cm = confusion_matrix(y_test, lr_preds)

    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
    nn_model.fit(X_train, y_train)
    nn_preds = nn_model.predict(X_test)

    nn_acc = accuracy_score(y_test, nn_preds)
    nn_precision = precision_score(y_test, nn_preds)
    nn_recall = recall_score(y_test, nn_preds)
    nn_f1 = f1_score(y_test, nn_preds)
    nn_cm = confusion_matrix(y_test, nn_preds)

    # Store summary for table
    results.append((name, lr_acc, nn_acc))

    # Print detailed metrics cleanly
    print(f"\n=== Feature Set: {name} ===\n")

    print("[Logistic Regression]")
    print(f"Accuracy:  {lr_acc:.4f}")
    print(f"Precision: {lr_precision:.4f}")
    print(f"Recall:    {lr_recall:.4f}")
    print(f"F1 Score:  {lr_f1:.4f}")
    print("Confusion Matrix:")
    for row in lr_cm:
        print("  ", row)

    print("\n[Neural Network]")
    print(f"Accuracy:  {nn_acc:.4f}")
    print(f"Precision: {nn_precision:.4f}")
    print(f"Recall:    {nn_recall:.4f}")
    print(f"F1 Score:  {nn_f1:.4f}")
    print("Confusion Matrix:")
    for row in nn_cm:
        print("  ", row)

# After all feature sets, print summary table cleanly
print("\nSummary of Training Accuracies:")
print(f"{'Features Used':<25} {'LogReg Accuracy':<18} {'NeuralNet Accuracy'}")
print("-" * 60)
for r in results:
    print(f"{r[0]:<25} {r[1]:<18.4f} {r[2]:.4f}")
