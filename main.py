import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def add_features(df):
    df = df.copy()
    df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["LDL_to_HDL"] = df["LDL"] / (df["HDL"] + 1e-6)
    df["waist_to_height"] = df["waist(cm)"] / (df["height(cm)"] + 1e-6)
    df["liver_ratio"] = (df["AST"] + df["ALT"]) / (df["Gtp"] + 1e-6)
    return df

# Feature engineering
X = add_features(train.drop(columns=["id", "smoking"]))
y = train["smoking"].astype(int)
X_test = add_features(test.drop(columns=["id"]))
test_ids = test["id"]

# Final model with best params from Optuna
model = XGBClassifier(
    learning_rate=0.0534,
    n_estimators=936,
    max_depth=3,
    subsample=0.7003,
    colsample_bytree=0.7548,
    gamma=1.2670,
    min_child_weight=2,
    reg_alpha=4.8035,
    reg_lambda=2.9730,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)

# Cross-validation
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

# Summary
print("\n" + "=" * 50)
print("📊 Validation AUCs:", [f"{s:.5f}" for s in val_scores])
print(f"📈 Mean Validation AUC (simulated Kaggle score): {np.mean(val_scores):.5f}")
print("=" * 50)

# Create submission file
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")
