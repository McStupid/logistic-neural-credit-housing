import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("card_labelled.csv")

# Target and features
target = "fraudulent"
features = ['Time'] + [f"V{i}" for i in range(1,29)] + ['Amount']

X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define feature sets
feature_sets = {
    "1. All features": X_scaled,
    "2. V1–V10 + Time + Amount": np.hstack([
        X_scaled[:, 1:11],
        X_scaled[:, [0, -1]]
    ]),
    "3. V1–V5 + Norm Amount": np.hstack([
        X_scaled[:, 1:6],
        (X_scaled[:, -1] / (X_scaled[:, 0] + 1)).reshape(-1, 1)
    ]),
    "4. PCA(5) + Amount": np.hstack([
        PCA(n_components=5).fit_transform(X_scaled),
        X_scaled[:, [-1]]
    ]),
    "5. Time + Amount + Interaction": np.hstack([
        X_scaled[:, [0]],
        X_scaled[:, [-1]],
        (X_scaled[:, 0] * X_scaled[:, -1]).reshape(-1, 1)
    ])
}

results = []

for name, X_feat in feature_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds_train = lr_model.predict(X_train)
    lr_train_acc = accuracy_score(y_train, lr_preds_train)
    
    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
    nn_model.fit(X_train, y_train)
    nn_preds_train = nn_model.predict(X_train)
    nn_train_acc = accuracy_score(y_train, nn_preds_train)
    
    results.append({
        "Feature Set": name,
        "LogReg Train Acc": lr_train_acc,
        "NeuralNet Train Acc": nn_train_acc
    })
    
    # Print detailed metrics per feature set inside the loop
    print(f"\n=== Feature Set: {name} ===")
    
    print("[Logistic Regression]")
    print(f"Training Accuracy: {lr_train_acc:.4f}")
    print("Confusion Matrix (Training):")
    for row in confusion_matrix(y_train, lr_preds_train):
        print("  ", row)

    print("\n[Neural Network]")
    print(f"Training Accuracy: {nn_train_acc:.4f}")
    print("Confusion Matrix (Training):")
    for row in confusion_matrix(y_train, nn_preds_train):
        print("  ", row)

# Summary table
print("\nSummary of Training Accuracies:")
print(f"{'Feature Set':<30} {'LogReg Accuracy':<20} {'NeuralNet Accuracy'}")
print("-" * 65)
for r in results:
    print(f"{r['Feature Set']:<30} {r['LogReg Train Acc']:<20.4f} {r['NeuralNet Train Acc']:.4f}")
