# Risk modelling results



Synthetic, retrospective default-label classification. Default event dates, outcome horizon and bureau observation dates are unavailable. Repayment features may overlap the outcome. Scores are not validated future default probabilities.



Selected before test evaluation: enriched_logistic.

Selection: best validation AUC shortlist within 0.005, then lowest validation Brier; complexity breaks exact ties.

Selected test AUC 0.6770; KS 0.2790; Gini 0.3539; precision 30.12%; recall 37.63%; Brier 0.1272.

Review threshold 0.216796 targets approximately 20% of validation loans. This is an explicit demo review-capacity assumption, not an approval/rejection rule. Ties can change coverage.

Models are trained only on the training partition. Validation guides tuning and selection. The final test set is used once for comparison. No refitting on test data is performed.

No class resampling or class weights: retain the observed prevalence and assess precision/recall. Probabilities are assessed for calibration but are not separately recalibrated or guaranteed calibrated.

Gender is excluded from predictors; subgroup metrics are descriptive and do not constitute a complete fairness audit.

Explanations are additive contributions in log-odds, not percentage-point changes, causal effects or rejection reasons.



## Monitoring plan

Freeze training feature distributions and score bins as the reference. On each new batch, monitor row/key validity, missingness, unseen categories, feature CSI and score PSI. The included train/test comparison is only a split diagnostic, not evidence of production drift. Investigate changes against approved tolerances and sample sizes; do not automatically retrain from a PSI threshold alone. When matured labels and a fixed outcome horizon become available, track AUC, KS, precision/recall and calibration overall and by segment. Retraining requires a reviewed data window, grouped/time-aware validation, comparison to the current model and human approval before replacing it.



## Limitations

Random customer-disjoint testing demonstrates generalisation to unseen customers within this synthetic snapshot. It does not establish performance on future vintages. The basic feature comparison removes bureau/repayment history but cannot establish historical availability of customer attributes. Collections, IDs, customer_since, raw dates and existing portfolio loan counts are excluded.