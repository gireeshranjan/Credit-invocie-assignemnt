"""Compare logistic regression and XGBoost; preserve a customer-disjoint holdout."""
import argparse
import importlib.metadata
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from xgboost import DMatrix
from portfolio_common import (PROCESSED, EXCEL, MODELS, MODEL_REPORTS, SEED, LIMITATION,
                              load_base, write_json, write_workbook, fingerprint)
from model_utils import FEATURE_SETS, feature_frame, build_pipeline, evaluate

def split_customers(a):
    y, groups = a.target_default, a.customer_id
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev, test = next(outer.split(a, y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED+1)
    tr, va = next(inner.split(a.iloc[dev], y.iloc[dev], groups.iloc[dev]))
    parts = {"train": dev[tr], "validation": dev[va], "test": test}
    sets = {k: set(a.iloc[idx].customer_id) for k, idx in parts.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or
                sets["validation"] & sets["test"])
    assert len(set(np.concatenate(list(parts.values())))) == len(a)
    for key, idx in parts.items():
        assert a.iloc[idx].target_default.nunique() == 2, f"One-class partition: {key}"
    return parts

def threshold_for_capacity(probabilities, review_share):
    # Ties can make the actual flagged share differ; report that share explicitly.
    return float(np.quantile(probabilities, 1-review_share))

def drift_comparison(train, test, cols, train_p, test_p):
    rows, reference = [], {}
    features = [(c, train[c], test[c]) for c in cols]
    features += [("model_score", pd.Series(train_p), pd.Series(test_p))]
    for name, ref, cur in features:
        if pd.api.types.is_numeric_dtype(ref):
            vals = ref.dropna().to_numpy()
            cuts = np.unique(np.quantile(vals, np.arange(.1, 1, .1))) if len(vals) else np.array([])
            edges = np.r_[-np.inf, cuts, np.inf]
            rb = pd.cut(ref, edges, duplicates="drop").astype(str).where(ref.notna(), "Missing")
            cb = pd.cut(cur, edges, duplicates="drop").astype(str).where(cur.notna(), "Missing")
        else:
            rb, cb = ref.fillna("Missing").astype(str), cur.fillna("Missing").astype(str)
        rp, cp = rb.value_counts(normalize=True), cb.value_counts(normalize=True)
        categories = rp.index.union(cp.index)
        r = rp.reindex(categories, fill_value=0).to_numpy().clip(1e-6)
        c = cp.reindex(categories, fill_value=0).to_numpy().clip(1e-6)
        r, c = r/r.sum(), c/c.sum()
        reference[name] = {
            "kind": "numeric" if pd.api.types.is_numeric_dtype(ref) else "categorical",
            "interior_cutpoints": [float(v) for v in cuts] if pd.api.types.is_numeric_dtype(ref) else [],
            "reference_shares": {str(k): float(v) for k, v in rp.items()},
            "missing_share": float(ref.isna().mean()), "training_rows": len(ref),
            "numeric_bin_rule": "pd.cut with negative/positive infinity endpoints; missing separate"}
        rows.append({"feature": name, "psi_or_csi": float(np.sum((c-r)*np.log(c/r))),
                     "comparison": "train vs random grouped test; diagnostic, not observed production drift"})
    return pd.DataFrame(rows), reference

def explain(pipeline, x):
    transformed = pipeline.named_steps["preprocess"].transform(x)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = transformed * model.coef_[0]
        bias = np.repeat(float(model.intercept_[0]), len(x))
        method = "standardized feature contributions to log-odds"
        raw = model.decision_function(transformed)
    else:
        matrix = DMatrix(transformed)
        contributions = model.get_booster().predict(matrix, pred_contribs=True)
        values, bias = contributions[:, :-1], contributions[:, -1]
        method = "TreeSHAP contributions to raw log-odds"
        raw = model.get_booster().predict(matrix, output_margin=True)
    if not np.allclose(values.sum(axis=1)+bias, raw, rtol=1e-4, atol=1e-4):
        raise ValueError("Explanation contributions do not reconstruct the model margin.")
    return values, bias, names, method

def run(review_share=.20):
    if not 0 < review_share < 1:
        raise ValueError("review_share must be between 0 and 1")
    MODELS.mkdir(parents=True, exist_ok=True)
    MODEL_REPORTS.mkdir(parents=True, exist_ok=True)
    a = load_base()
    parts = split_customers(a)
    membership = a[["account_id", "customer_id", "target_default"]].copy()
    membership["partition"] = ""
    for part, idx in parts.items():
        membership.loc[idx, "partition"] = part
    membership.to_csv(MODEL_REPORTS / "split_membership.csv", index=False)
    split_table = membership.groupby("partition").agg(
        loans=("account_id", "size"), customers=("customer_id", "nunique"),
        defaults=("target_default", "sum"), default_rate=("target_default", "mean")).reset_index()
    validation_rows, trials, fitted = [], [], {}
    grids = {
        "logistic": [{"C": .05}, {"C": .5}, {"C": 5.0}],
        "xgboost": [{"max_depth": 2, "n_estimators": 220},
                    {"max_depth": 3, "n_estimators": 300},
                    {"max_depth": 4, "n_estimators": 220}]}
    for scenario, spec in FEATURE_SETS.items():
        x = feature_frame(a, spec)
        tr, va = parts["train"], parts["validation"]
        for algorithm, configs in grids.items():
            best = None
            for params in configs:
                pipe = build_pipeline(algorithm, spec, params)
                pipe.fit(x.iloc[tr], a.target_default.iloc[tr])
                p = pipe.predict_proba(x.iloc[va])[:, 1]
                threshold = threshold_for_capacity(p, review_share)
                metrics = evaluate(a.target_default.iloc[va], p, threshold)
                trials.append({"scenario": scenario, "algorithm": algorithm,
                               "parameters": json.dumps(params), **metrics})
                # Validation AUC then lower Brier; final test never guides tuning.
                key = (metrics["auc"], -metrics["brier"])
                if best is None or key > best["key"]:
                    best = dict(key=key, pipe=pipe, params=params, threshold=threshold, metrics=metrics)
            name = scenario + "_" + algorithm
            fitted[name] = dict(**best, scenario=scenario, algorithm=algorithm, spec=spec)
            validation_rows.append({"model": name, "scenario": scenario, "algorithm": algorithm,
                                    "parameters": json.dumps(best["params"]), **best["metrics"]})
            joblib.dump(best["pipe"], MODELS / f"{name}.joblib")
            print(f"Validation {name}: AUC={best['metrics']['auc']:.4f}", flush=True)
    validation = pd.DataFrame(validation_rows)
    # Predeclared selection: among models within .005 of best validation AUC,
    # prefer lower Brier, then simpler feature set / algorithm on exact ties.
    eligible = validation[validation.auc.ge(validation.auc.max()-.005)].copy()
    eligible["complexity"] = eligible.scenario.eq("enriched").astype(int)*2 + eligible.algorithm.eq("xgboost").astype(int)
    winner = eligible.sort_values(["brier", "complexity", "model"]).iloc[0]["model"]
    validation["selected"] = validation.model.eq(winner)
    validation.to_csv(MODEL_REPORTS / "validation_comparison.csv", index=False)
    # Selection is fixed here, before any test predictions or test metrics are computed.
    chosen = fitted[winner]
    write_json(MODEL_REPORTS / "selection_before_test.json", {
        "selected_model": winner, "rule": "Within .005 of best validation AUC, lowest Brier; complexity tie-break.",
        "review_capacity_assumption": review_share, "hyperparameters": chosen["params"],
        "test_used_for_selection": False})
    test_rows, test_predictions, curve_rows = [], [], []
    te = parts["test"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for name, bundle in fitted.items():
        x = feature_frame(a, bundle["spec"])
        p = bundle["pipe"].predict_proba(x.iloc[te])[:, 1]
        y = a.target_default.iloc[te].to_numpy()
        test_rows.append({"model": name, "selected": name == winner,
                          **evaluate(y, p, bundle["threshold"])})
        pred = a.iloc[te][["account_id", "customer_id", "target_default"]].copy()
        pred["model"], pred["default_probability"] = name, p
        pred["flagged_for_review"] = (p >= bundle["threshold"]).astype(int)
        test_predictions.append(pred)
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[2].plot(predicted, observed, marker=".", label=name)
        for b, (pr, ob) in enumerate(zip(predicted, observed), 1):
            curve_rows.append({"model": name, "bin": b, "mean_probability": pr, "observed_default_rate": ob})
    for ax, title, xlabel, ylabel in [
        (axes[0], "ROC", "False-positive rate", "Recall"),
        (axes[1], "Precision-recall", "Recall", "Precision"),
        (axes[2], "Calibration (quantile bins)", "Mean predicted probability", "Observed default rate")]:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.legend(fontsize=7)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].axhline(a.target_default.iloc[te].mean(), linestyle="--", color="grey")
    axes[2].plot([0, 1], [0, 1], "--", color="grey")
    fig.tight_layout(); fig.savefig(MODEL_REPORTS / "model_comparison.png", dpi=160); plt.close(fig)
    test = pd.DataFrame(test_rows)
    predictions = pd.concat(test_predictions, ignore_index=True)
    ytest = a.target_default.iloc[te].to_numpy()
    baseline = evaluate(ytest, np.repeat(a.target_default.iloc[parts["train"]].mean(), len(te)), 1.0)
    baseline_table = pd.DataFrame([{"model": "always_performing_training_prevalence_score", **baseline}])
    spec, pipe = chosen["spec"], chosen["pipe"]
    x = feature_frame(a, spec)
    train_p = pipe.predict_proba(x.iloc[parts["train"]])[:, 1]
    boundaries = np.quantile(train_p, [.5, .8]).tolist()
    artifact = {
        "pipeline": pipe, "feature_spec": spec, "selected_model": winner,
        "threshold": chosen["threshold"], "risk_band_boundaries": boundaries,
        "limitation": LIMITATION, "seed": SEED,
        "training_data_sha256": fingerprint(PROCESSED / "analytics_base.csv"),
        "training_account_ids": a.iloc[parts["train"]].account_id.tolist(),
        "validation_account_ids": a.iloc[parts["validation"]].account_id.tolist(),
        "test_account_ids": a.iloc[te].account_id.tolist(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["pandas", "numpy", "scikit-learn", "xgboost", "joblib"]}}
    joblib.dump(artifact, MODELS / "selected_model.joblib")
    metadata = {k: v for k, v in artifact.items() if k not in
                ["pipeline", "training_account_ids", "validation_account_ids", "test_account_ids"]}
    write_json(MODELS / "model_metadata.json", metadata)
    # Permutation importance uses validation only, and groups original columns.
    perm = permutation_importance(pipe, x.iloc[parts["validation"]],
        a.target_default.iloc[parts["validation"]], n_repeats=4, random_state=SEED,
        scoring="roc_auc", n_jobs=1)
    importance = pd.DataFrame({"feature": x.columns, "auc_drop_mean": perm.importances_mean,
        "auc_drop_std": perm.importances_std}).sort_values("auc_drop_mean", ascending=False)
    # Exact additive contributions for the selected model: all loans, labelled by partition later.
    contributions, bias, names, method = explain(pipe, x)
    top = np.argsort(np.abs(contributions), axis=1)[:, -3:][:, ::-1]
    local = a[["account_id", "customer_id"]].copy()
    local["partition"] = membership.partition.values
    local["method"], local["base_log_odds"] = method, bias
    for rank in range(3):
        local[f"driver_{rank+1}"] = names[top[:, rank]]
        local[f"contribution_{rank+1}"] = contributions[np.arange(len(a)), top[:, rank]]
    local.to_csv(MODEL_REPORTS / "individual_drivers.csv", index=False)
    contribution_importance = pd.DataFrame({
        "encoded_feature": names, "mean_absolute_log_odds_contribution": np.abs(contributions[te]).mean(axis=0)
    }).sort_values("mean_absolute_log_odds_contribution", ascending=False)
    # Evaluate gender etc. after selection, using the same global threshold.
    selected_p = pipe.predict_proba(x.iloc[te])[:, 1]
    subgroup_rows = []
    for col in ["gender", "loan_type", "employment_type", "bur_thin_file"]:
        for value in a.iloc[te][col].drop_duplicates():
            mask = a.iloc[te][col].eq(value).to_numpy()
            subgroup_rows.append({"dimension": col, "group": str(value),
                "small_group": int(mask.sum()) < 100,
                **evaluate(ytest[mask], selected_p[mask], chosen["threshold"])})
    scored_test = a.iloc[te][["account_id", "target_default"]].copy()
    scored_test["default_probability"] = selected_p
    scored_test["risk_decile"] = pd.qcut(
        scored_test.default_probability.rank(method="first", ascending=False), 10, labels=False)+1
    deciles = scored_test.groupby("risk_decile").agg(
        loans=("account_id", "size"), defaults=("target_default", "sum"),
        observed_default_rate=("target_default", "mean"), mean_probability=("default_probability", "mean")
    ).reset_index()
    deciles["cumulative_default_capture"] = deciles.defaults.cumsum()/deciles.defaults.sum()
    drift, reference = drift_comparison(x.iloc[parts["train"]], x.iloc[te], list(x.columns), train_p, selected_p)
    write_json(MODEL_REPORTS / "monitoring_reference.json", reference)
    ledger = []
    for col in a.columns:
        use = "enriched" if col in spec["numeric"]+spec["categorical"] else "excluded"
        if col in FEATURE_SETS["basic"]["numeric"]+FEATURE_SETS["basic"]["categorical"]:
            use = "basic and enriched"
        elif col in FEATURE_SETS["enriched"]["numeric"]:
            use = "enriched only"
        if col in ["account_id", "customer_id"]:
            reason = "Join/group identifiers only; never predictors."
        elif col == "target_default":
            reason = "Outcome only; never a predictor."
        elif col.startswith("coll_"):
            reason = "Post-delinquency selection and unreliable event dates; analytics only."
        elif col.startswith(("bur_", "rp_")) or col == "emi_to_income":
            reason = "Retrospective snapshot; timing relative to outcome unverified."
        elif col == "gender":
            reason = "Subgroup audit only; excluded from initial predictors."
        else:
            reason = "Observed snapshot; historical availability not guaranteed." if use != "excluded" else "Not in explicit feature allow-list."
        ledger.append({"column": col, "use": use, "timing_or_exclusion_note": reason})
    tables = {
        "split_summary": split_table, "validation_comparison": validation, "test_comparison": test,
        "baseline": baseline_table, "test_calibration": pd.DataFrame(curve_rows),
        "risk_deciles": deciles, "subgroup_test": pd.DataFrame(subgroup_rows),
        "permutation_importance": importance, "contribution_importance": contribution_importance,
        "split_drift_diagnostic": drift, "feature_ledger": pd.DataFrame(ledger),
        "hyperparameter_trials": pd.DataFrame(trials)}
    for name, frame in tables.items():
        frame.to_csv(MODEL_REPORTS / f"{name}.csv", index=False)
    predictions.to_csv(MODEL_REPORTS / "test_predictions.csv", index=False)
    write_workbook(EXCEL / "risk_model_results.xlsx", tables)
    selected_test = test.set_index("model").loc[winner]
    text = [
        "# Risk modelling results", "", LIMITATION, "",
        f"Selected before test evaluation: {winner}.",
        "Selection: best validation AUC shortlist within 0.005, then lowest validation Brier; complexity breaks exact ties.",
        f"Selected test AUC {selected_test.auc:.4f}; KS {selected_test.ks:.4f}; Gini {selected_test.gini:.4f}; "
        f"precision {selected_test.precision:.2%}; recall {selected_test.recall:.2%}; Brier {selected_test.brier:.4f}.",
        f"Review threshold {chosen['threshold']:.6f} targets approximately {review_share:.0%} of validation loans. "
        "This is an explicit demo review-capacity assumption, not an approval/rejection rule. Ties can change coverage.",
        "Models are trained only on the training partition. Validation guides tuning and selection. "
        "The final test set is used once for comparison. No refitting on test data is performed.",
        "No class resampling or class weights: retain the observed prevalence and assess precision/recall. "
        "Probabilities are assessed for calibration but are not separately recalibrated or guaranteed calibrated.",
        "Gender is excluded from predictors; subgroup metrics are descriptive and do not constitute a complete fairness audit.",
        "Explanations are additive contributions in log-odds, not percentage-point changes, causal effects or rejection reasons.",
        "", "## Monitoring plan",
        "Freeze training feature distributions and score bins as the reference. On each new batch, monitor "
        "row/key validity, missingness, unseen categories, feature CSI and score PSI. The included train/test "
        "comparison is only a split diagnostic, not evidence of production drift. Investigate changes against "
        "approved tolerances and sample sizes; do not automatically retrain from a PSI threshold alone. "
        "When matured labels and a fixed outcome horizon become available, track AUC, KS, precision/recall "
        "and calibration overall and by segment. Retraining requires a reviewed data window, grouped/time-aware "
        "validation, comparison to the current model and human approval before replacing it.",
        "", "## Limitations",
        "Random customer-disjoint testing demonstrates generalisation to unseen customers within this synthetic snapshot. "
        "It does not establish performance on future vintages. The basic feature comparison removes bureau/repayment "
        "history but cannot establish historical availability of customer attributes. "
        "Collections, IDs, customer_since, raw dates and existing portfolio loan counts are excluded.",
    ]
    (MODEL_REPORTS / "model_report.md").write_text("\n\n".join(text), encoding="utf-8")
    print(f"Selected {winner}; test AUC={selected_test.auc:.4f}; results: {EXCEL / 'risk_model_results.xlsx'}")
    return tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-share", type=float, default=.20)
    run(parser.parse_args().review_share)
