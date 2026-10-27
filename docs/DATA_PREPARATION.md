# Data preparation rationale

Preserved from the retired exploratory notebook. The executable implementation is scripts/prepare_data.py.

# 01 — Data Ingestion & Cleaning

**Goal:** load the 5 raw tables, surface and fix data-quality issues, and join everything into **one clean,
analysis-ready dataset** (one row per loan account) that every later stage — analytics, model, dashboard — reads from.

**Inputs** (`data/raw/`): `customers`, `loan_accounts`, `bureau_credit`, `repayment_history`, `collections_events`

**Outputs**
| File | What |
|---|---|
| data/processed/analytics_base.csv (Parquet optional) | joined, cleaned dataset — 1 row per account |
| `data/processed/analytics_base.xlsx` | same, plus a data-quality log sheet and a column dictionary |
| `reports/data_quality_log.csv` | every check run, rows affected, action taken |

**Join logic** — `loan_accounts` is the driving table because it holds `target_default`:
```
loan_accounts (account_id)
  ⟕ customers            on customer_id   many → one
  ⟕ bureau  (aggregated) on customer_id   many → one
  ⟕ repayment (aggreg.)  on account_id    one  → one
  ⟕ collections (aggr.)  on account_id    one  → one   (analytics only — see §7)
```
**Rule throughout: no rows are dropped.** Unknown numeric values remain null. Only missing current-employment tenure for unemployed customers is filled with zero.
Missingness and correction flags make the policy traceable. Model-specific imputers must be fitted on training data later.
The Excel export includes missing-value policy and coverage sheets. CSV/Excel blanks are genuine unknowns.

## 1. Ingest

## 2. Profile — missing values and key integrity

`bureau.date_closed` being null is **expected** (the line is still active) — not a data-quality issue. The rest are handled table by table below.

**Key checks:** primary keys must be unique, and every foreign key must point to a real parent row.

Keys are clean — no duplicates, no orphans. Relationships really are one-to-many, so the child tables must be
**aggregated before joining** (a plain join would multiply rows):

## 3. Clean `customers`

### 3.1 Inconsistent gender labels

### 3.2 Missing employment tenure
The dictionary defines this as tenure in current employment. Missing tenure for a customer explicitly
classified as Unemployed is set to zero, with a flag. Missing tenure for others stays unknown.
All customers remain in portfolio totals; tenure comparisons include an Unknown group.
Any statistical imputation for modelling belongs in a training-only pipeline.


### 3.3 Implausible `years_employed`
Some customers have worked longer than they could have been an adult (e.g. age 25 with 20 years employed).
The existing cap at **age - 18** is retained and flagged. This is a declared assumption about
starting employment at 18, rather than proof every earlier start is impossible. Missing values
are not capped or filled.

### 3.4 Missing dependents and residence tenure
Keep missing values unknown and add flags. Zero dependents and zero residence tenure have specific
meanings. Keep these accounts in the portfolio; use an Unknown group for comparisons and disclose
the number of known observations for averages.


### 3.5 Income floor and segment bands
The minimum income (12,000) appears 76 times — looks like a floor/clip applied upstream. Kept as-is, recorded as an assumption. Bands are added for profiling in Stage 2.

## 4. Clean `loan_accounts`

### 4.1 Missing interest rate

Missing rates remain unknown. Blank does not imply an interest-free loan. These loans remain in
portfolio totals; interest-rate summaries use known rates and report coverage. Similar observed
default rates do not establish that missingness is random.


### 4.2 Consistency checks

Half the loans pre-date `customer_since` — impossible, so that field is unreliable. We flag it and won't use it as a feature.

## 5. Clean `bureau_credit`

### 5.1 Missing days past due

All missing bureau DPD values remain unknown, including closed accounts. Observed medians do not
establish the missing account's actual delay. Keep every bureau record; maximum/mean use observed
DPD, accompanied by known/missing counts and coverage. If all DPD values are missing, maximum/mean
remain null. Active accounts' closure dates remain null because closure has not occurred.


### 5.2 Future-dated records

Bureau dates extend beyond the declared reference cutoff. Preserve these records for descriptive
analytics and document the inconsistency. The dataset does not establish when bureau information
was available relative to each loan. Model development must define an observation window and discuss
this limitation; retaining a record does not establish its availability at application time.


## 6. Validate `repayment_history`

Fully consistent: On-Time = 0 DPD, Late = 1–28 DPD, Missed = 30+ DPD with nothing paid. Every account has exactly **12 months** — a fixed observation window. No changes needed.

## 7. Validate `collections_events` — and why it's kept out of the model

Two reasons to keep collections **out of the model**:
1. **Leakage** — collections happen *after* an account goes delinquent. At application time a bank wouldn't have this data, so a model using it would look better than it really is.
2. **Unreliable dates** — 45% of events are dated before the loan existed.

They're still useful for analytics (recovery rates, collections effort by segment), so we keep them with a `coll_` prefix.

## 8. Aggregate child tables to the join grain

### 8.1 Bureau → one row per customer

### 8.2 Repayment → one row per account
Behaviour over the 12-month window: misses, lateness, worst DPD, longest run of consecutive misses, and whether DPD is **getting worse** (last 3 months vs first 3).

### 8.3 Collections → one row per account (analytics only)

## 9. Join into one account-level dataset
`validate=` makes pandas raise an error if a join would duplicate rows.

## 10. Final checks

### Data-quality log

## 11. Export
CSV and Excel preserve unknown numeric values as blanks. Parquet is optional; CSV is the canonical
input for the next pipeline stage. The workbook includes data, quality log, column dictionary, missing-value
coverage and policy.


## Data-quality notes
- All loan accounts are retained; keys and exact duplicate rows are checked after joining.
- Missing employment tenure for Unemployed customers is filled with zero and flagged.
- Other missing tenure, dependents, residence tenure, rates and bureau DPD stay unknown.
- Bureau maximum/mean DPD use observed values, with known/missing counts and coverage.
- No bureau records means zero recorded counts and unknown DPD, amounts and ratios.
  No collections records means zero recorded events/recovery and an absence flag.
- Existing label standardisation and tenure caps remain documented and flagged.
- Unknown groups remain in grouped analytics; averages disclose their known-value denominator.
- The model stage must define observation/target timing, split the data and fit any imputer on
  training data only. Collections remain for descriptive analytics, not candidate predictors.
- Next: portfolio analytics reads data/processed/analytics_base.csv. A trained model is not needed first.
