# Product brief: Credit Risk Workspace

Credit Risk Workspace gives a bank's credit-risk team one place to see where portfolio risk sits, score every loan with its reasons, and ask questions in plain English. It is built and verified on the supplied synthetic portfolio of 10,000 customers and 14,685 loans.

**Problem.** Risk information sits in five separate tables: customers, loans, credit bureau, repayments and collections. Answering "where is our risk?" takes manual joins and one-off spreadsheets. Concentrations are seen late: mortgages are 9.8% of loans but 52% of lending, and they default at 24.7%.

**Target users.** Credit-risk analysts and portfolio managers in the Chief Risk Officer's team are the primary users. Credit officers use it to review individual loans; the Chief Risk Officer uses it for the monthly portfolio review.

**What it does**

- Cleans and joins all five tables into one row per loan, keeping every loan and flagging every fix.
- Profiles risk by income, employment, region, age, product and over time, with control limits on default trends.
- Scores every loan with logistic regression (test AUC 0.677, Gini 0.354, KS 0.279) and shows its top three drivers.
- Interactive Streamlit dashboard: seven pages, sidebar filters, and a loan-level review.
- Ask the Portfolio: plain-English questions answered by fixed, read-only queries on the data, never by numbers the model writes.

**Scope**

| In scope | Out of scope |
| --- | --- |
| The five supplied tables, cleaned and joined | Real customer data and production deployment |
| Portfolio profiling and trends over time | Expected loss, pricing and profitability |
| Default classification model with per-loan drivers | A validated 12-month probability of default (no default dates) |
| Dashboard with filters and loan review | Automatic approve or decline decisions |
| AI Q&A grounded in the data, with an audit log | Login, user roles and multi-user hosting |
| Monitoring plan (PSI/CSI, control limits) | Collections data as a model input (recorded after delinquency) |

**Key results**

- Half of all lending is in the riskiest product: mortgages, 1.5x the 16.1% portfolio default rate.
- Each prior write-off at another lender raises risk: 14.6% with none, 31.4% with three or more.
- Bureau and repayment history lift the model from AUC 0.651 (application data only) to 0.675 in cross-validation.
- No deterioration over time: quarterly default rates show no significant trend (p = 0.69).

**Risk and compliance.** Gender is excluded from the model, and results are reported by subgroup. Scores support human review and never make lending decisions. The AI assistant receives only the question and field names; customer rows, incomes and scores never leave the bank's machine. Every cleaning step, check and AI query is logged without storing raw questions or keys. The model is explainable per loan and needs an independent model-risk validation before any real use.

See also: [architecture and workflow](../README.md#architecture-and-workflow) and the [assumptions log](ASSUMPTIONS_LOG.md).
