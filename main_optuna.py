import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Columns to drop
DROP_COLS = ["id", "eyesight(left)", "eyesight(right)", "hearing(left)", "hearing(right)", "dental caries", "Urine protein"]

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
X_test = add_features(test.drop(columns=DROP_COLS))
test_ids = test["id"]

# Optuna objective
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
        "use_label_encoder": False
    }

    model = XGBClassifier(**params)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        score = roc_auc_score(y.iloc[val_idx], pred)
        scores.append(score)

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Best model
print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)

# Final model train
final_model = XGBClassifier(**study.best_params, use_label_encoder=False)
final_model.fit(X, y)
test_preds = final_model.predict_proba(X_test)[:, 1]

# Save submission
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Final submission.csv saved!")
