"""Model A: standard population-level models (pre-session features only)."""
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data.preprocessing import FEATURES


def make_lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=0.3, max_iter=3000))])


def make_rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(300, min_samples_leaf=15, n_jobs=-1, random_state=0))])


def train_baselines(X, y):
    return {"lr": make_lr().fit(X[FEATURES], y), "rf": make_rf().fit(X[FEATURES], y)}


def lr_contributions(lr, row: pd.Series):
    """Exact linear contribution (coef * standardised value) of each feature to the logit. NOT causal."""
    x = lr.named_steps["imp"].transform(row[FEATURES].to_frame().T.astype(float))
    z = lr.named_steps["sc"].transform(x)[0]
    return pd.Series(z * lr.named_steps["clf"].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
