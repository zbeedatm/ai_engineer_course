"""
Gradient Boosting on Titanic (robust to older library versions)
- Features: is_female, is_child, is_first_class
- Models: XGBoost, LightGBM, CatBoost
- Prints accuracy, feature importances, and one example probability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1) Load Titanic and build 3 binary features
file_path = "titanic3.csv"   # ensure this file is in your working directory
df = pd.read_csv(file_path)[["sex", "age", "pclass", "survived"]].copy()

# Basic cleanup
df["sex"] = df["sex"].astype(str).str.lower().str.strip()
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["pclass"] = pd.to_numeric(df["pclass"], errors="coerce")
df["survived"] = pd.to_numeric(df["survived"], errors="coerce").fillna(0).astype(int)

# Minimal imputations to keep the demo simple
df["age"] = df["age"].fillna(df["age"].median())
df["pclass"] = df["pclass"].fillna(df["pclass"].mode()[0])

# Tiny, readable features (0/1)
X_df = pd.DataFrame({
    "is_female":      (df["sex"] == "female").astype(int),
    "is_child":       (df["age"] < 16).astype(int),
    "is_first_class": (df["pclass"] == 1).astype(int),
})
y = df["survived"].values
feat_names = list(X_df.columns)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_df.values, y, test_size=0.25, random_state=42, stratify=y
)

print("Feature sample (first 8 rows):")
print(X_df.assign(survived=y).head(8).to_string(index=False))

# ==================== XGBoost ====================
print("\n=== XGBoost ===")
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

# Try to use callbacks-based early stopping if available; otherwise, fit normally.
xgb_callbacks = []
try:
    from xgboost.callback import EarlyStopping as XGBEarlyStopping
    xgb_callbacks = [XGBEarlyStopping(rounds=50, save_best=True, maximize=False)]
except Exception:
    xgb_callbacks = []

try:
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=xgb_callbacks)  # some versions may not support callbacks
except TypeError:
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # fallback: no callbacks

pred = xgb.predict(X_val)
print("Accuracy:", accuracy_score(y_val, pred))
print("Feature importances:", dict(zip(feat_names, xgb.feature_importances_)))
p = xgb.predict_proba([[1, 0, 1]])[0, 1]
print("Example prob (is_female=1,is_child=0,is_first_class=1):", round(p, 3))

# ==================== LightGBM (version-safe) ====================
print("\n=== LightGBM ===")
from lightgbm import LGBMClassifier
try:
    # Newer LightGBM
    from lightgbm import early_stopping, log_evaluation
except ImportError:
    # Older LightGBM
    from lightgbm.callback import early_stopping, log_evaluation

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
# Fit with callbacks if supported; otherwise, fallback to plain fit.
try:
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)]
    )
except TypeError:
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # fallback

pred = lgbm.predict(X_val)
print("Accuracy:", accuracy_score(y_val, pred))
print("Feature importances:", dict(zip(feat_names, lgbm.feature_importances_)))
p = lgbm.predict_proba([[1, 0, 1]])[0, 1]
print("Example prob (is_female=1,is_child=0,is_first_class=1):", round(p, 3))

# ==================== CatBoost (version-safe) ====================
print("\n=== CatBoost ===")
from catboost import CatBoostClassifier

# Prefer constructor-only params; fit may not accept early_stopping_rounds in some versions.
cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=4,
    loss_function="Logloss",
    random_state=42,
    verbose=False
)

# Try early stopping via fit kw if available; otherwise, plain fit.
try:
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
except TypeError:
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

pred = cat.predict(X_val)
try:
    pred = pred.astype(int)  # some versions return strings
except Exception:
    pass

print("Accuracy:", accuracy_score(y_val, pred))
importances = cat.get_feature_importance()
print("Feature importances:", dict(zip(feat_names, importances)))
p = cat.predict_proba([[1, 0, 1]])[0, 1]
print("Example prob (is_female=1,is_child=0,is_first_class=1):", round(p, 3))
