"""Apply the saved selected model without retraining or computing new performance claims."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from portfolio_common import PROCESSED, EXCEL, MODELS, MODEL_REPORTS, write_workbook, fingerprint
from model_utils import feature_frame, score_bands

def run(input_path=None, output_path=None):
    input_path = Path(input_path) if input_path else PROCESSED / "analytics_base.csv"
    a = pd.read_csv(input_path)
    if not a.account_id.notna().all() or not a.account_id.is_unique:
        raise ValueError("Scoring requires unique, non-null account_id.")
    bundle = joblib.load(MODELS / "selected_model.joblib")
    x = feature_frame(a, bundle["feature_spec"])
    p = bundle["pipeline"].predict_proba(x)[:, 1]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid model probabilities")
    out = a[[c for c in ["account_id", "customer_id", "target_default"] if c in a]].copy()
    out["default_probability"] = p
    out["risk_band"] = score_bands(p, bundle["risk_band_boundaries"])
    out["flagged_for_review"] = (p >= bundle["threshold"]).astype(int)
    out["model"] = bundle["selected_model"]
    tr, va, te = (set(bundle[key]) for key in
                  ["training_account_ids", "validation_account_ids", "test_account_ids"])
    out["evaluation_status"] = np.select(
        [a.account_id.isin(tr), a.account_id.isin(va), a.account_id.isin(te)],
        ["training_in_sample", "validation_used_for_selection", "held_out_test"],
        default="new_records_not_evaluated")
    if fingerprint(input_path) != bundle["training_data_sha256"]:
        out["evaluation_status"] = "new_or_modified_records_not_evaluated"
    out["limitation"] = bundle["limitation"]
    path = Path(output_path) if output_path else PROCESSED / "loan_risk_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        notes = pd.DataFrame([
            ("default_probability", "Model estimate for supplied default label; horizon unspecified; not separately recalibrated."),
            ("risk_band", "Relative to training score distribution: below median / median-80th / top 20%."),
            ("flagged_for_review", "Above threshold fixed using validation capacity; not a credit decision."),
            ("evaluation_status", "Only held_out_test rows form the independent model-performance comparison."),
        ], columns=["field", "meaning"])
        drivers_path = MODEL_REPORTS / "individual_drivers.csv"
        if drivers_path.exists():
            drivers = pd.read_csv(drivers_path)
            out = out.merge(drivers.drop(columns=["customer_id", "partition"]), on="account_id",
                            how="left", validate="one_to_one")
        write_workbook(EXCEL / "loan_risk_scores.xlsx", {"loan_scores": out, "definitions": notes})
    out.to_csv(path, index=False)
    print(f"Scored {len(out):,} loans: {path}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.input, args.output)
