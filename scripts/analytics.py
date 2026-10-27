"""Produce the agreed portfolio analytics as CSV, Excel, charts and a readable report."""
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from portfolio_common import (RAW, EXCEL, ANALYTICS, load_base, add_segments,
                              portfolio_summary, write_json, write_workbook, add_credit_history_groups,
                              vintage_trend, trend_test, load_repayment, repayment_timeline,
                              seasoning_curve, calendar_delinquency)

def heatmap(frame, row, col, path, title, order=None):
    rates = frame.pivot(index=row, columns=col, values="default_rate")
    counts = frame.pivot(index=row, columns=col, values="loan_count")
    if order:
        rates = rates.reindex(columns=order)
        counts = counts.reindex(columns=order)
    fig, ax = plt.subplots(figsize=(11, max(4, len(rates) * .65)))
    image = ax.imshow(np.ma.masked_invalid(rates.to_numpy(dtype=float)), cmap="YlOrRd", aspect="auto",
                      vmin=0, vmax=max(.4, float(np.nanmax(rates.to_numpy(dtype=float)))))
    ax.set_xticks(range(len(rates.columns)), rates.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(rates.index)), rates.index)
    for i in range(len(rates.index)):
        for j in range(len(rates.columns)):
            value, n = rates.iloc[i, j], counts.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1%}\nn={int(n):,}" + (" *" if n < 100 else ""),
                        ha="center", va="center", fontsize=9, color="white" if value > .5 else "#112533")
    ax.set_title(title + "\n* fewer than 100 loans; interpret cautiously")
    fig.colorbar(image, ax=ax, label="Observed default rate", format=matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("Annual income (USD)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def run():
    ANALYTICS.mkdir(parents=True, exist_ok=True)
    charts = ANALYTICS / "charts"
    charts.mkdir(exist_ok=True)
    a = add_segments(load_base())
    # Customer income distributions count every person once, independently of loan count.
    customers = pd.read_csv(RAW / "customers.csv")
    education = customers.groupby("education_level", dropna=False).annual_income.agg(
        customer_count="size", median_annual_income="median", mean_annual_income="mean",
        p25_annual_income=lambda s: s.quantile(.25), p75_annual_income=lambda s: s.quantile(.75)
    ).reset_index()
    tables = {}
    for name, keys in {
        "product": ["loan_type"], "income": ["income_group"],
        "product_income": ["loan_type", "income_group"],
        "education_income": ["education_level", "income_group"],
        "education_income_gender": ["education_level", "income_group", "gender"],
        "employment_income": ["employment_type", "income_group"],
        "gender": ["gender"], "income_gender": ["income_group", "gender"],
        "age": ["age_group"], "region": ["region"], "region_product": ["region", "loan_type"],
    }.items():
        tables[name] = portfolio_summary(a, keys)
        if len(keys) == 1:
            assert tables[name].loan_count.sum() == len(a)
    a["missed_payment_group"] = pd.cut(a.rp_n_missed, [-1, 0, 1, 2, np.inf],
                                               labels=["0", "1", "2", "3+"])
    a["missed_payment_group"] = a.missed_payment_group.cat.add_categories(["Unknown"]).fillna("Unknown")
    a["bureau_dpd_group"] = pd.cut(a.bur_max_dpd, [-np.inf, 1, 30, 60, np.inf], right=False,
                                   labels=["0", "1-29", "30-59", "60+"])
    a["bureau_dpd_group"] = a.bureau_dpd_group.cat.add_categories(
        ["No bureau records", "DPD unknown in existing records"])
    a.loc[a.bur_thin_file.eq(1), "bureau_dpd_group"] = "No bureau records"
    a["bureau_dpd_group"] = a.bureau_dpd_group.fillna("DPD unknown in existing records")
    tables["missed_payments"] = portfolio_summary(a, ["missed_payment_group"])
    tables["bureau_dpd"] = portfolio_summary(a, ["bureau_dpd_group"])
    a = add_credit_history_groups(a)
    tables["write_offs"] = portfolio_summary(a, ["write_off_group"])
    tables["credit_activity"] = portfolio_summary(a, ["credit_activity"])
    tables["employment"] = portfolio_summary(a, ["employment_type"])
    tables["vintage"] = portfolio_summary(a, ["vintage_year"])
    # Time series: origination cohorts, months since origination and calendar months.
    timeline = repayment_timeline(a, load_repayment())
    tables["vintage_quarterly"] = vintage_trend(a, "Q")
    tables["seasoning"] = seasoning_curve(timeline)
    tables["calendar_delinquency"] = calendar_delinquency(timeline)
    tables["education_customer_income"] = education
    tables["missing_coverage"] = pd.DataFrame({
        "column": a.columns, "missing_rows": a.isna().sum().values,
        "known_rows": a.notna().sum().values, "missing_share": a.isna().mean().values})
    overview = {
        "loan_count": len(a), "customer_count": int(a.customer_id.nunique()),
        "defaults": int(a.target_default.sum()), "default_rate": float(a.target_default.mean()),
        "original_loan_amount": float(a.loan_amount.sum()),
        "original_amount_on_defaulted": float(a.loc[a.target_default.eq(1), "loan_amount"].sum()),
        "last_recorded_balance": float(a.rp_last_outstanding.sum()),
    }
    tables["overview"] = pd.DataFrame([overview])
    definitions = [
        ("default_rate", "Defaulted loan count / loan count within each group; loan-weighted."),
        ("customer_count", "Distinct customers within a group; not additive across product/income intersections."),
        ("education_customer_income", "One customer once; annual_income is USD per source dictionary."),
        ("original_loan_amount", "Dataset monetary units; dictionary does not confirm currency for loan amounts."),
        ("original_amount_on_defaulted", "Original amounts on labelled defaulted loans; not expected loss or realised loss."),
        ("last_recorded_balance", "Sum after each loan's month 12; not a common calendar-date balance."),
        ("small_group", "Fewer than 100 loans: descriptive flag, not a statistical confidence guarantee."),
        ("income_group", "Left-closed intervals: 30,000 belongs to 30k-under 60k, etc."),
        ("missing", "Unknowns retained. No bureau records differs from observed zero overdue days."),
        ("interpretation", "Synthetic descriptive associations; no causal or lending-policy conclusions."),
        ("vintage_quarterly", "Loans grouped by origination quarter; the default rate is each cohort's eventual "
         "labelled outcome, not the date of default."),
        ("control_limits", "Portfolio default rate +/- 2 binomial standard errors for each cohort's size; "
         "a cohort outside the band differs by more than random variation would usually explain."),
        ("calendar_month", "Origination month + instalment number; assumes one instalment per month, "
         "starting the month after origination."),
        ("dpd30_rate", "Share of loan-months 30+ days past due (the same months as Missed in this data)."),
    ]
    tables["definitions"] = pd.DataFrame(definitions, columns=["measure", "meaning"])
    for name, frame in tables.items():
        frame.to_csv(ANALYTICS / f"{name}.csv", index=False)
    write_workbook(EXCEL / "portfolio_analytics.xlsx", tables)
    write_json(ANALYTICS / "overview.json", overview)
    order = ["Under 30k", "30k-under 60k", "60k-under 90k", "90k-under 120k", "120k+"]
    for name, row, title in [
        ("product_income", "loan_type", "Product and income"),
        ("education_income", "education_level", "Education and income"),
        ("employment_income", "employment_type", "Employment and income")]:
        heatmap(tables[name], row, "income_group", charts / f"{name}.png", title, order)
    fig, ax = plt.subplots(figsize=(9, 5))
    g = tables["income_gender"].pivot(index="income_group", columns="gender", values="default_rate").reindex(order)
    g.plot.bar(ax=ax, color=["#7657A4", "#247BA0"])
    ax.set_ylabel("Observed default rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Gender within income groups: descriptive comparison")
    ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(charts / "income_gender.png", dpi=160); plt.close(fig)
    for name, key in [("missed_payments", "missed_payment_group"), ("bureau_dpd", "bureau_dpd_group"),
                      ("write_offs", "write_off_group"), ("credit_activity", "credit_activity"),
                      ("age", "age_group"), ("region", "region")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(tables[name][key].astype(str), tables[name].default_rate, color="#247BA0")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.set_title("Observed default rate: " + name.replace("_", " "))
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout(); fig.savefig(charts / f"{name}.png", dpi=160); plt.close(fig)
    vq = tables["vintage_quarterly"]
    pos = np.arange(len(vq))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(pos, vq.lower_limit, vq.upper_limit, color="#247BA0", alpha=.12, label="Control band (+/-2 SE)")
    ax.plot(pos, vq.expected_rate, color="#7a8c9c", linestyle="--", linewidth=1, label="Portfolio rate")
    ax.plot(pos, vq.default_rate, color="#247BA0", marker="o", label="Cohort default rate")
    ax.scatter(pos[vq.outside_limits], vq.default_rate[vq.outside_limits], color="#c65b4b", s=70, zorder=3,
               label="Outside band")
    ax.set_xticks(pos, vq.vintage, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Default rate by origination quarter, with control limits")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(charts / "vintage_control.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for outcome, color in [("Defaulted", "#c65b4b"), ("Performing", "#247BA0")]:
        curve = tables["seasoning"][tables["seasoning"].outcome.eq(outcome)]
        ax.plot(curve.payment_month, curve.dpd30_rate, marker="o", color=color, label=outcome + " loans")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Months since origination")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(bottom=0)
    ax.set_title("Share of loans 30+ days past due, by loan age")
    ax.legend()
    fig.tight_layout(); fig.savefig(charts / "seasoning.png", dpi=160); plt.close(fig)
    cal = tables["calendar_delinquency"][~tables["calendar_delinquency"].small_group].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cal.index, cal.dpd30_rate, color="#247BA0")
    ticks = cal.index[::6]
    ax.set_xticks(ticks, cal.calendar_month[ticks], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(bottom=0)
    ax.set_title("Share of active loans 30+ days past due, by calendar month (months with 500+ active loans)")
    fig.tight_layout(); fig.savefig(charts / "calendar_delinquency.png", dpi=160); plt.close(fig)
    overall = overview["default_rate"]
    def rate(name, key):
        return tables[name].set_index(key).default_rate
    mortgage = tables["product"].set_index("loan_type").loc["Mortgage"]
    income, employment = rate("income", "income_group"), rate("employment", "employment_type")
    low = a[a.annual_income.lt(60000)]
    unemp = low[low.employment_type.eq("Unemployed")]
    write_offs, age = rate("write_offs", "write_off_group"), rate("age", "age_group")
    under_30_share = tables["age"].set_index("age_group").loan_share["Under 30"]
    missed = rate("missed_payments", "missed_payment_group")
    activity = rate("credit_activity", "credit_activity")
    customer_activity = a.drop_duplicates("customer_id").credit_activity.value_counts(normalize=True)
    closed_only = a[a.credit_activity.eq("Bureau history, none active")].drop_duplicates("customer_id")
    region, vintage = rate("region", "region"), rate("vintage", "vintage_year")
    cells = tables["region_product"][~tables["region_product"].small_group]
    hotspot = cells.loc[cells.default_rate.idxmax()]
    gender = tables["gender"].set_index("gender")
    findings = [
        f"Mortgage concentration: mortgages are {mortgage.loan_share:.1%} of loans but {mortgage.amount_share:.1%} "
        f"of original lending, and default at {mortgage.default_rate:.1%} ({mortgage.default_rate / overall:.1f}x "
        f"the {overall:.1%} portfolio rate). Action: review mortgage affordability and concentration limits first.",
        f"Income: default rates fall as income rises, from {income['Under 30k']:.1%} under USD 30k to "
        f"{income['120k+']:.1%} at USD 120k+. Below USD 60k ({len(low):,} loans) the rate is "
        f"{low.target_default.mean():.1%}. Action: tighten affordability and loan-size checks below USD 60k.",
        f"Employment x income: unemployed customers default at {employment['Unemployed']:.1%}; unemployed and "
        f"below USD 60k reach {unemp.target_default.mean():.1%} ({len(unemp):,} loans, "
        f"{int(unemp.target_default.sum()):,} defaults), more than double the portfolio rate. "
        "Action: route this segment to manual underwriting review.",
        f"Prior write-offs: risk rises with each debt another lender has written off: {write_offs['0']:.1%} with none, "
        f"{write_offs['1']:.1%} with one, {write_offs['2']:.1%} with two and {write_offs['3+']:.1%} with three or more. "
        f"{a.bur_n_written_off.gt(0).mean():.0%} of loans belong to customers with at least one. "
        "Action: make a prior write-off a mandatory review trigger at application.",
        f"Age: default rates decline steadily with age, from {age['Under 30']:.1%} under 30 "
        f"({under_30_share:.0%} of loans) to {age['60+']:.1%} at 60+. "
        "Action: apply affordability buffers and lower starting limits for younger borrowers.",
    ]
    observations = [
        f"Missed payments as early warning: {missed['0']:.1%} default with no missed instalments, "
        f"{missed['1']:.1%} with one, {missed['2']:.1%} with two and {missed['3+']:.1%} with three or more. "
        "This is the same 12-month window as the outcome, so use it as a monitoring trigger, not an application rule.",
        f"Credit activity: {customer_activity['Active outside credit']:.0%} of customers hold active credit elsewhere "
        f"({activity['Active outside credit']:.1%} defaults). Customers with no bureau record default at "
        f"{activity['No bureau records']:.1%}; of customers whose outside credit is all closed, "
        f"{closed_only.bur_has_written_off.mean():.0%} have a recorded write-off.",
        f"What does not drive risk: region ({region.min():.1%}-{region.max():.1%}) and origination year "
        f"({vintage.min():.1%}-{vintage.max():.1%}) barely move default rates. The one hotspot is "
        f"{hotspot.region} {hotspot.loan_type.lower()}s at {hotspot.default_rate:.1%} ({int(hotspot.loan_count):,} loans).",
        f"Gender: female loans {gender.loc['Female', 'default_rate']:.1%}, male loans "
        f"{gender.loc['Male', 'default_rate']:.1%}; differences vary by income group, and gender is excluded from the model.",
    ]
    slope, p_value = trend_test(vq)
    outside = vq[vq.outside_limits]
    early = tables["seasoning"].pivot(index="payment_month", columns="outcome", values="dpd30_rate").iloc[0]
    trends = [
        f"Lending volume was steady: {vq.loan_count.min():,}-{vq.loan_count.max():,} loans per origination quarter "
        f"from {vq.vintage.iloc[0]} to {vq.vintage.iloc[-1]}.",
        f"Cohort default rates ranged {vq.default_rate.min():.1%}-{vq.default_rate.max():.1%} with no significant "
        f"trend ({slope * 100:+.2f} percentage points per year, p = {p_value:.2f}). "
        + (f"Outside the +/-2 standard-error control band: "
           + ", ".join(f"{v} ({r:.1%})" for v, r in zip(outside.vintage, outside.default_rate))
           + f"; about {0.0455 * len(vq):.0f} of {len(vq)} quarters would fall outside by chance alone, "
           "so these merit review rather than alarm."
           if len(outside) else "No quarter falls outside the +/-2 standard-error control band."),
        f"Loans that later defaulted were 30+ days past due more often from the first month "
        f"({early['Defaulted']:.1%} vs {early['Performing']:.1%}), and the gap holds across all 12 months: "
        "delinquency is an early signal rather than something that builds with loan age.",
        f"Calendar-month delinquency stayed between {cal.dpd30_rate.min():.1%} and {cal.dpd30_rate.max():.1%} of "
        f"active loans ({cal.calendar_month.iloc[0]} to {cal.calendar_month.iloc[-1]}) with no sustained "
        "deterioration. Action: track cohort default and 30+ DPD rates against these control limits each quarter.",
    ]
    report = ["# Portfolio analytics", "", f"{len(a):,} loans; {a.customer_id.nunique():,} customers; "
              f"{a.target_default.sum():,} labelled defaults ({a.target_default.mean():.2%}).", "",
              "## Key insights", ""]
    report += [f"{i}. {text}" for i, text in enumerate(findings, 1)]
    report += ["", "## Trends over time", ""] + [f"- {text}" for text in trends]
    report += ["", "## Further observations", ""] + [f"- {text}" for text in observations]
    report += ["", "Observed associations in a synthetic, retrospective dataset: not causal, and not lending rules "
               "on their own. Unknowns are retained, groups under 100 loans are flagged, and no expected-loss or "
               "profit claim is made.", "",
               "Full tables: exports/excel/portfolio_analytics.xlsx; CSVs and charts: reports/analytics."]
    (ANALYTICS / "analysis_summary.md").write_text("\n".join(report), encoding="utf-8")
    def html_line(text):
        return f"<h2>{html.escape(text[3:])}</h2>" if text.startswith("## ") else "<p>" + html.escape(text) + "</p>"
    sections = "<h1>Portfolio analytics</h1>" + "".join(html_line(t) for t in report[1:] if t)
    images = "".join(f'<img src="charts/{name}.png" style="max-width:100%;margin:20px 0" alt="{name}">'
                     for name in ["product_income", "vintage_control", "seasoning", "calendar_delinquency",
                                  "employment_income", "write_offs", "age", "credit_activity",
                                  "region", "missed_payments", "bureau_dpd", "education_income", "income_gender"])
    (ANALYTICS / "analytics_report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Portfolio analytics</title>'
        '<body style="font:16px system-ui;max-width:1100px;margin:40px auto;color:#19334a">'
        + sections + images + "</body>", encoding="utf-8")
    print(f"Analytics complete: {EXCEL / 'portfolio_analytics.xlsx'}")
    return tables

if __name__ == "__main__":
    run()
