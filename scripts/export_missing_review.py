from pathlib import Path
import pandas as pd

from portfolio_common import ROOT, RAW

# Recognise empty CSV cells as missing.
# No values are imputed or corrected.
def read_table(name):
    return pd.read_csv(
        RAW / f"{name}.csv",
        keep_default_na=False,
        na_values=[""],
    )

customers = read_table("customers")
loans = read_table("loan_accounts")
bureau = read_table("bureau_credit")
repayment = read_table("repayment_history")
collections = read_table("collections_events")

# Preserve missing totals when every contributing value is missing.
def total(series):
    return series.sum(min_count=1)

# Bureau: one summary per customer.
bureau_summary = bureau.groupby("customer_id").agg(
    bur_record_count=("bureau_id", "size"),
    bur_total_sanctioned=("sanctioned_amount", total),
    bur_total_outstanding=("outstanding_amount", total),
    bur_max_dpd=("days_past_due", "max"),
    bur_mean_dpd=("days_past_due", "mean"),
    bur_missing_dpd_count=("days_past_due", lambda s: s.isna().sum()),
    bur_missing_closure_date_count=("date_closed", lambda s: s.isna().sum()),
)

# Repayment: one summary per loan account.
repayment_summary = repayment.groupby("account_id").agg(
    rp_record_count=("payment_month", "size"),
    rp_total_due=("emi_due", total),
    rp_total_paid=("emi_paid", total),
    rp_max_dpd=("days_past_due", "max"),
    rp_mean_dpd=("days_past_due", "mean"),
    rp_missing_dpd_count=("days_past_due", lambda s: s.isna().sum()),
)

# Collections: one summary per loan account.
collections_summary = collections.groupby("account_id").agg(
    coll_event_count=("collection_event_id", "size"),
    coll_total_recovered=("amount_recovered", total),
    coll_missing_recovery_count=(
        "amount_recovered", lambda s: s.isna().sum()
    ),
)

# Left joins preserve every original loan account.
base = (
    loans
    .merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    .merge(
        bureau_summary,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    .merge(
        repayment_summary,
        on="account_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        collections_summary,
        on="account_id",
        how="left",
        validate="one_to_one",
    )
)

# Check duplicates; do not silently delete anything.
assert len(base) == len(loans), "Loan count changed during joining."
assert base["account_id"].notna().all(), "Missing account IDs found."
assert base["account_id"].is_unique, "Duplicate account IDs found."
assert not base.duplicated().any(), "Complete duplicate rows found."

xlsx_path = ROOT / "exports" / "excel" / "original_data_review.xlsx"
xlsx_path.parent.mkdir(parents=True, exist_ok=True)

# 'Missing' is written only when exporting.

sheets = {
    "loan_review": base,
    "original_customers": customers,
    "original_loans": loans,
    "original_bureau": bureau,
    "original_repayment": repayment,
    "original_collections": collections,
}

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    for sheet_name, frame in sheets.items():
        frame.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            na_rep="Missing",
        )
        sheet = writer.sheets[sheet_name]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

print(f"Loan rows: {len(base):,}")
print(f"Complete duplicate rows: {base.duplicated().sum()}")
print(f"Excel: {xlsx_path}")