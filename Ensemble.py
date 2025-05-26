import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

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

X = add_features(train.drop(columns=["id", "smoking"]))
y = train["smoking"].astype(int)
X_test = add_features(test.drop(columns=["id"]))
test_ids = test["id"]

# Best params for XGB
xgb_model = XGBClassifier(
    learning_rate=0.056090339198925074,
    n_estimators=987,
    max_depth=3,
    subsample=0.7046993932017644,
    colsample_bytree=0.8657596472956113,
    gamma=1.8178856894204485,
    min_child_weight=10,
    reg_alpha=4.834402348462552,
    reg_lambda=4.53906682715448,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)

# Best params for HGB
hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.12361308801324597,
    max_iter=674,
    max_depth=4,
    l2_regularization=3.9070637612932817,
    min_samples_leaf=17,
    random_state=42
)

# Weights by AUC (normalized)
xgb_auc = 0.89389
hgb_auc = 0.88987
xgb_w, hgb_w = xgb_auc / (xgb_auc + hgb_auc), hgb_auc / (xgb_auc + hgb_auc)

# Cross-validation + prediction
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n🌀 Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb_model.fit(X_train, y_train)
    hgb_model.fit(X_train, y_train)

    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    hgb_val = hgb_model.predict_proba(X_val)[:, 1]

    blended_val = xgb_w * xgb_val + hgb_w * hgb_val
    score = roc_auc_score(y_val, blended_val)
    val_scores.append(score)
    print(f"✅ AUC Score: {score:.5f}")

    xgb_test = xgb_model.predict_proba(X_test)[:, 1]
    hgb_test = hgb_model.predict_proba(X_test)[:, 1]
    test_preds += (xgb_w * xgb_test + hgb_w * hgb_test) / kf.n_splits

# Summary
print("\n" + "=" * 50)
print("📊 Blended Validation AUCs:", [f"{s:.5f}" for s in val_scores])
print(f"📈 Mean Validation AUC: {np.mean(val_scores):.5f}")
print("=" * 50)

# Create submission
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Blended submission.csv saved!")
