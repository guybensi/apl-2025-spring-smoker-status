import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Split features and labels
X = train.drop(columns=["id", "smoking"])
y = train["smoking"].astype(int)
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# Initialize model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_estimators=100,
    max_depth=4
)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n🌀 Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_pred)
    val_scores.append(score)
    
    print(f"✅ AUC Score: {score:.5f}")
    
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

# Summary of validation results
print("\n" + "="*50)
print("📊 Validation AUCs:", [f"{s:.5f}" for s in val_scores])
print(f"📈 Mean Validation AUC (simulated Kaggle score): {np.mean(val_scores):.5f}")
print("="*50)
