"""Feature definitions and evaluation shared by training and scoring."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score,
                             brier_score_loss, log_loss)
from xgboost import XGBClassifier
from portfolio_common import SEED

BASIC_NUMERIC = ["age", "annual_income", "years_employed", "num_dependents",
                 "years_at_residence", "loan_amount", "tenure_months",
                 "collateral_flag", "loan_to_income"]
CATEGORICAL = ["employment_type", "education_level", "occupation_sector",
               "housing_type", "region", "loan_type", "loan_purpose", "disbursal_channel"]
ENRICHED_NUMERIC = [
    "bur_n_lines", "bur_n_active", "bur_n_written_off", "bur_total_sanctioned",
    "bur_total_outstanding", "bur_max_dpd", "bur_mean_dpd", "bur_n_dpd30",
    "bur_nonbank_share", "bur_n_dpd_known", "bur_n_dpd_missing",
    "bur_dpd_coverage", "bur_utilisation", "bur_thin_file",
    "rp_n_missed", "rp_n_late", "rp_max_dpd", "rp_mean_dpd", "rp_max_consec_missed",
    "rp_pct_on_time", "rp_paid_to_due", "rp_dpd_first3", "rp_dpd_last3", "rp_dpd_trend",
    "emi_to_income",
]
FEATURE_SETS = {
    "basic": {"numeric": BASIC_NUMERIC, "categorical": CATEGORICAL},
    "enriched": {"numeric": BASIC_NUMERIC + ENRICHED_NUMERIC, "categorical": CATEGORICAL},
}

def feature_frame(a, spec):
    names = spec["numeric"] + spec["categorical"]
    missing = set(names) - set(a.columns)
    if missing:
        raise ValueError(f"Missing model input columns: {sorted(missing)}")
    x = a[names].copy()
    for col in spec["numeric"]:
        x[col] = pd.to_numeric(x[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    for col in spec["categorical"]:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    return x

def build_pipeline(algorithm, spec, params):
    if algorithm == "logistic":
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler())])
        model = LogisticRegression(max_iter=2500, solver="lbfgs", C=params["C"], random_state=SEED)
    else:
        numeric = "passthrough"  # XGBoost handles NaN; dense output preserves literal zeros.
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", learning_rate=.04, subsample=.85, colsample_bytree=.85,
            reg_lambda=5, min_child_weight=15, n_jobs=4, random_state=SEED, **params)
    categories = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([
        ("numeric", numeric, spec["numeric"]),
        ("categorical", categories, spec["categorical"])], remainder="drop", sparse_threshold=0)
    return Pipeline([("preprocess", preprocess), ("model", model)])

def evaluate(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    fpr, tpr, _ = roc_curve(y, p) if both else (np.array([0]), np.array([0]), None)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p)) if both else np.nan
    return {
        "loans": len(y), "defaults": int(y.sum()), "default_rate": float(y.mean()),
        "auc": auc, "ks": float(np.max(tpr-fpr)) if both else np.nan,
        "gini": 2*auc-1, "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(p.mean()), "threshold": float(threshold),
        "flagged_share": float(pred.mean()), "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp)}

def score_bands(p, boundaries):
    # Bands describe score ranks against training distribution, not lending decisions.
    return np.where(np.asarray(p) >= boundaries[1], "High relative score",
           np.where(np.asarray(p) >= boundaries[0], "Medium relative score", "Low relative score"))
