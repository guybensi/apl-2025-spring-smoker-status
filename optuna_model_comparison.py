import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import optuna

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Feature Engineering
def add_features(df):
    df = df.copy()
    df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["LDL_to_HDL"] = df["LDL"] / (df["HDL"] + 1e-6)
    df["waist_to_height"] = df["waist(cm)"] / (df["height(cm)"] + 1e-6)
    df["liver_ratio"] = (df["AST"] + df["ALT"]) / (df["Gtp"] + 1e-6)
    return df

# Use all columns except id and target
X = add_features(train.drop(columns=["id", "smoking"]))
y = train["smoking"].astype(int)
X_test = add_features(test.drop(columns=["id"]))
test_ids = test["id"]

# Optimization function per model
def objective(trial, model_type):
    if model_type == "xgb":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "use_label_encoder": False,
            "verbosity": 0
        }
        model = XGBClassifier(**params)

    elif model_type == "hist":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "random_state": 42
        }
        model = HistGradientBoostingClassifier(**params)

    elif model_type == "gbr":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": 42
        }
        model = GradientBoostingClassifier(**params)

    # Cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], pred))

    return np.mean(scores)

# Run Optuna for each model
results = {}

for model_type in ["xgb", "hist", "gbr"]:
    print(f"\n🔍 Optimizing {model_type.upper()}")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_type), n_trials=50)

    results[model_type] = {
        "auc": study.best_value,
        "params": study.best_params
    }

# Print summary
print("\n📊 Summary:")
for name, res in results.items():
    print(f"🔹 {name.upper()} AUC: {res['auc']:.5f}")

best_model = max(results, key=lambda m: results[m]["auc"])
print(f"\n✅ Best model: {best_model.upper()} with AUC = {results[best_model]['auc']:.5f}")
