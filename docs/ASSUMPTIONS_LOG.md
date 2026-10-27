# Assumptions log

Three timing gaps matter most: the data has no default dates, no bureau pull dates and no point-in-time payment history. Until a bank supplies them, scores classify a recorded outcome rather than forecast a future default. Rows are ordered by impact.

| Area | Assumption or approximation | Impact | Needed from the bank's IT/data team |
| --- | --- | --- | --- |
| Model | The default label has no event date or outcome window, so scores classify the recorded outcome, not a 12-month forward probability of default. | High | Default definition (for example 90+ days past due), default dates and a fixed performance window. |
| Model | Bureau timing is unknown: 5,760 lines were opened after the 2023-12-31 reference date and some closures are dated up to 2029. | High | The bureau pull date for each application, and bureau data as at that date. |
| Model | Repayment features use the same 12-month window as the outcome, so they may overlap it. | High | Payment history as at each scoring date (point-in-time snapshots). |
| Model | Validation uses a random split that keeps each customer in one partition, not a later time period. | High | At least two years of matured outcomes for out-of-time validation. |
| Data | Collections data is excluded from the model: it is recorded after delinquency, and 3,875 events are dated before the loan existed. | Medium | Collections event timestamps from the source system. |
| Data | customer_since is unreliable (7,293 loans start before the relationship) and is not used. | Medium | Corrected relationship start dates. |
| Data | Years employed is capped at age minus 18 for 1,004 customers (assumes work starts at 18); years at address is capped at age for 5. | Medium | Employment and address start dates. |
| Data | The 1,008 customers with no bureau record form their own group, not zero debt. Utilisation is left blank for 3,347 loans with no active outside credit. | Medium | Whether "no record" means no bureau hit or no bureau pull. |
| Analytics | Utilisation shows no link to default here, unlike real portfolios; the data is synthetic and findings are associations, not causes. | Medium | Production data, to re-test every finding. |
| Model | The review flag covers 20% of loans; risk bands sit at the training median and 80th percentile. Both are demo choices, not lending policy. | Medium | Review capacity and risk appetite from Credit Policy. |
| Model | Gender is excluded and subgroup results are descriptive; this is not a full fairness audit. | Medium | Fairness policy, protected-attribute list and legal review. |
| AI | Ask the Portfolio sends only the typed question, field names, filters and at most one loan ID to the Claude API; it runs on the local machine with no login, screens English questions only, and a live call is not yet verified. | Medium | Single sign-on, role-based access, vendor-risk, data-residency and retention approval, managed API keys. |
| Monitoring | PSI/CSI tolerances, control limits and retraining cadence are proposed, not approved. | Medium | Model risk management standards, approved thresholds and an independent validation. |
| Data | Missing tenure for 23 unemployed customers is set to 0, matching all 471 other unemployed customers. Every other blank stays blank, with a flag. | Low | Source codes that separate "not provided" from "not applicable". |
| Model | Blanks are filled with the training median plus a "was missing" flag. Mean, KNN, MICE, a random draw and zero-fill all scored within 0.001 AUC. | Low | Nothing further; the choice is documented and tested. |
| Data | 600 gender labels coded M/F are mapped to Male/Female. 76 incomes at exactly 12,000 look like an upstream floor and are kept. | Low | Income verification source and any floors applied upstream. |
| Data | Income is in USD per the data dictionary; the currency of loan amounts is not stated. | Low | Currency and units for every monetary field. |
| Analytics | A repayment's calendar month is taken as the origination month plus its instalment number. | Low | Actual payment due dates. |
