# Credit Risk Portfolio Analytics

Run from this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 run_pipeline.py
```

The installed environment is isolated in .venv. If rebuilding it:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Individual stages:
- scripts/prepare_data.py: validate the five source tables, apply the agreed cleaning rules, aggregate child records and retain one row per loan. Real unknowns stay blank. Caps at age minus 18/current age are explicit inherited assumptions, not corrections to source files.
- scripts/analytics.py: the five agreed analysis areas, customer-level education/income, supporting gender/age/region/employment/vintage tables, prior bureau write-offs, outside-credit activity, small-cell flags, charts, five numerical insights with actions and further observations.
  Time series: default rate by origination quarter with +/-2 standard-error control limits and a least-squares trend test, 30+ days past due by months since origination (seasoning) and by calendar month. Calendar months assume one instalment per month starting the month after origination.
- scripts/train_models.py: logistic regression and XGBoost, each using basic and enriched features. Customers never cross partitions. Fit preprocessing on training data only. Tune on validation; lock selection before evaluating the held-out test set.
- scripts/score_loans.py: apply the saved selected model. Scores include whether records were training, validation, test or new.
- scripts/verify_outputs.py: check row retention, missingness, Excel/CSV agreement, customer isolation and saved predictions.
- run_pipeline.py: run all stages; --skip-prepare uses the current cleaned CSV.
- scripts/export_missing_review.py: optional original-information workbook with source missing cells displayed as Missing; this is distinct from cleaned analytics.

## Open these outputs

All active Excel workbooks are in exports/excel:
- analytics_base.xlsx: all cleaned loan rows, dictionary, quality log and missing-value policy.
- portfolio_analytics.xlsx: summary and cross-dimensional tables.
- risk_model_results.xlsx: model comparisons, metrics, explanations, subgroup checks and feature ledger.
- loan_risk_scores.xlsx: individual scores, relative risk bands and three additive model drivers.
- original_data_review.xlsx: previously generated source-review workbook; genuine missing values displayed as Missing.
- missing_values_review.xlsx: existing detailed missing-data review.

Readable reports:
- reports/analytics/analytics_report.html and analysis_summary.md.
- reports/models/model_report.md and model_comparison.png.
- reports/models/individual_drivers.csv explains individual loans.
- reports/models/test_predictions.csv contains held-out predictions for all four comparisons.
- reports/models/split_membership.csv records the exact customer-disjoint split.
- reports/models/monitoring_reference.json freezes training distributions for later monitoring.
- models/selected_model.joblib includes preprocessing, selected estimator, threshold and metadata.
- data/processed/loan_risk_scores.csv contains scores, evaluation status and explanations.

Superseded generated backups and notebook copies were removed during folder cleanup.
Raw CSVs and the assignment PDF are preserved. Notebook rationale is in docs/DATA_PREPARATION.md; use scripts/prepare_data.py to execute preparation.

## Interpretation and assumptions

Annual income is USD; the dictionary does not specify the currency of all loan amounts.
Default rates are loan-weighted. Education-income distributions count customers once.
Customer counts across product cells can overlap. Unknowns remain in their groups.
Cells below 100 loans are flagged; 100 is a display convention, not a significance threshold.
Original loan amounts on defaulted loans are not losses, EAD or expected loss.
Month-12 balances are not balances from a shared calendar snapshot.

The default event date/horizon and bureau observation dates are missing. These are synthetic,
retrospective default-label models, not validated future-default forecasts. Full repayment features
can overlap the outcome. Compare basic with enriched models to expose sensitivity. Neither
feature set establishes historical availability. Collections and IDs are excluded from predictors;
gender is retained for audits only. Interest rate, customer_since, date-quality flags and
customer_n_loans are also excluded from the predictor allow-list.

The split is approximately 60/20/20 by loans, with grouping at customer level. No model is refitted
on the held-out test set. Hyperparameter choices and the final model are selected using validation.
Among models within 0.005 of best validation AUC, choose lowest Brier, breaking exact ties by
complexity. All four test results are shown without revising that selection.

Thresholds target the highest-scored 20% of validation loans as an explicit review-capacity
assumption; ties can change coverage. Low/medium/high are relative training-score bands,
not approved credit policies. No automated approve/reject decision is made.

Logistic numeric imputation, scaling and category encoding are fitted on training only.
XGBoost receives numeric NaN directly and dense one-hot categories so zeros are not mistaken
for missing sparse entries. No SMOTE or class weights are used at the observed ~16% prevalence.
Calibration is measured, not assumed or guaranteed; no separate recalibration is applied.

ROC-AUC, KS (maximum TPR-FPR), Gini (2*AUC-1), average precision, accuracy, precision,
recall, Brier, log loss and confusion counts are reported for both algorithms.
Feature explanations are log-odds contributions, not causal effects or percentage-point changes.
Logistic uses scaled feature times coefficient; XGBoost uses native TreeSHAP contributions.
Permutation importance uses validation data; contribution importance uses held-out rows.

The analytics/modelling pipeline contains no external AI calls. The separate chatbot is now available;
run .\.venv\Scripts\python.exe launch_chatbot.py and see CHATBOT.md.
Natural-language chat uses Claude Sonnet 5.5 and requires an Anthropic API key; verified local examples do not.
The wider portfolio dashboard remains a subsequent component.

Technical references:
- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- https://xgboost.readthedocs.io/en/stable/python/python_api.html
