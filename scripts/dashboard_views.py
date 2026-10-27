"""Charts and employee-facing views, calculated from the same data as the assistant."""
import html
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.metrics import roc_curve
from portfolio_common import (MODEL_REPORTS, portfolio_summary, add_credit_history_groups, vintage_trend,
                              trend_test, load_repayment, repayment_timeline, seasoning_curve,
                              calendar_delinquency)

NAVY = "#12304a"
TEAL = "#087f8c"
CORAL = "#c65b4b"
INCOME_ORDER = ["Under 30k", "30k-under 60k", "60k-under 90k", "90k-under 120k", "120k+", "Unknown"]
LABELS = {
 "loan_type":"Loan product", "income_group":"Annual income (USD)", "employment_type":"Employment",
 "education_level":"Education", "gender":"Gender", "region":"Region", "age_group":"Age",
 "housing_type":"Housing", "bureau_history":"Bureau history", "missed_payment_group":"Missed payments",
 "loan_count":"Loans", "customer_count":"Customers", "defaults":"Defaulted loans", "default_rate":"Default rate",
 "original_loan_amount":"Original lending", "median_annual_income":"Median income (loan-weighted)",
 "small_group":"Below 100 loans", "mean_income_usd":"Mean annual income (USD)",
 "median_income_usd":"Median annual income (USD)", "column":"Field",
 "missing_loan_rows":"Missing loan rows", "total_loan_rows":"Total loans", "missing_share":"Missing share",
 "model":"Model", "selected":"Selected", "loans":"Loans", "auc":"ROC-AUC", "ks":"KS", "gini":"Gini",
 "precision":"Precision", "recall":"Recall", "accuracy":"Accuracy", "brier":"Brier score",
 "default_probability":"Saved model score", "risk_band":"Relative risk band",
 "evaluation_status":"Evaluation partition", "account_id":"Loan account", "feature":"Model input",
 "log_odds_contribution":"Contribution (log-odds)", "direction":"Direction",
 "write_off_group":"Prior write-offs", "credit_activity":"Outside credit", "vintage":"Origination cohort",
 "mean_absolute_log_odds_contribution":"Average contribution (log-odds)"}
MODEL_LABELS = {"basic_logistic":"Logistic | customer & loan", "basic_xgboost":"XGBoost | customer & loan",
 "enriched_logistic":"Logistic | enriched", "enriched_xgboost":"XGBoost | enriched"}
DECIMAL_METRICS = {"auc", "log_odds_contribution", "mean_absolute_log_odds_contribution"}
FEATURE_LABELS = {
 "age":"Age", "annual_income":"Annual income", "years_employed":"Years employed", "num_dependents":"Dependents",
 "years_at_residence":"Years at address", "loan_amount":"Loan amount", "tenure_months":"Loan term",
 "collateral_flag":"Secured loan", "loan_to_income":"Loan / income", "bur_n_lines":"Outside credit lines",
 "bur_n_active":"Active outside credit lines", "bur_n_written_off":"Prior write-offs",
 "bur_total_sanctioned":"Outside credit limits", "bur_total_outstanding":"Outside debt",
 "bur_max_dpd":"Worst bureau days overdue", "bur_mean_dpd":"Average bureau days overdue",
 "bur_n_dpd30":"Bureau lines 30+ days overdue", "bur_nonbank_share":"Share from non-bank lenders",
 "bur_n_dpd_known":"Bureau lines with known overdue days", "bur_n_dpd_missing":"Bureau lines with unknown overdue days",
 "bur_dpd_coverage":"Bureau overdue-days coverage", "bur_utilisation":"Outside credit utilisation",
 "bur_thin_file":"No bureau record", "rp_n_missed":"Missed payments", "rp_n_late":"Late payments",
 "rp_max_dpd":"Worst days overdue", "rp_mean_dpd":"Average days overdue",
 "rp_max_consec_missed":"Longest run of missed payments", "rp_pct_on_time":"Share paid on time",
 "rp_paid_to_due":"Paid / due", "rp_dpd_first3":"Days overdue, months 1-3", "rp_dpd_last3":"Days overdue, months 10-12",
 "rp_dpd_trend":"Change in days overdue", "emi_to_income":"Instalments / income",
 "employment_type":"Employment", "education_level":"Education", "occupation_sector":"Sector",
 "housing_type":"Housing", "region":"Region", "loan_type":"Loan product", "loan_purpose":"Loan purpose",
 "disbursal_channel":"Channel"}

def feature_label(encoded):
    """Readable name for an encoded model input such as categorical__loan_type_Mortgage."""
    kind, _, name = encoded.partition("__")
    if name.startswith("missingindicator_"):
        return feature_label("numeric__" + name.removeprefix("missingindicator_")) + " unknown"
    if kind == "categorical":
        column = next((c for c in FEATURE_LABELS if name.startswith(c + "_")), None)
        if column:
            return f"{FEATURE_LABELS[column]}: {name[len(column)+1:]}"
    return FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())

@st.cache_data(show_spinner=False)
def repayment_rows():
    return load_repayment()

def selected_data(data, filters):
    x = data.copy()
    for item in filters:
        x = x[x[item["column"]].isin(item["values"])]
    return x

def short_number(value):
    return f"{value/1e6:,.1f}M" if abs(value)>=1e6 else f"{value:,.0f}"

def friendly_frame(frame):
    x=frame.copy()
    for col in x:
        if col.endswith("_rate") or col.endswith("_share") or col in ["default_probability","precision","recall","accuracy"]:
            x[col]=x[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "Unknown")
        elif col in ["auc","ks","gini","brier"]:
            x[col]=x[col].map(lambda v:f"{v:.3f}" if pd.notna(v) else "Unknown")
        elif col=="model":
            x[col]=x[col].map(MODEL_LABELS).fillna(x[col])
    return x.rename(columns=LABELS)

def table(frame, height=None):
    kwargs={"hide_index":True,"width":"stretch"}
    if height:kwargs["height"]=height
    st.dataframe(friendly_frame(frame),**kwargs)

def chart_style(chart):
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelColor="#526477",titleColor="#526477",gridColor="#edf1f5",
        labelFontSize=11,titleFontSize=11,titlePadding=12).configure_legend(
        labelColor="#526477",title=None).properties(padding={"left":12,"right":12,"top":12,"bottom":12})

def bar(frame, dimension, metric="default_rate", height=240, horizontal=True, color=None):
    percent=metric in ["default_rate","missing_share"]
    number_format=".0%" if percent else ".2f" if metric in DECIMAL_METRICS else "~s"
    label=LABELS.get(metric,metric.replace("_"," ").title())
    tooltip=[alt.Tooltip(dimension+":N",title=LABELS.get(dimension,dimension))]
    for c,fmt in [("loan_count",",d"),("defaults",",d"),("default_rate",".2%"),("original_loan_amount",",.0f")]:
        if c in frame:
            tooltip.append(alt.Tooltip(c+":Q",title=LABELS.get(c,c),format=fmt))
    if metric not in [t.shorthand.split(":")[0] for t in tooltip]:
        tooltip.append(alt.Tooltip(metric+":Q",title=label,format=".2%" if percent else ",.2f"))
    sort=INCOME_ORDER if dimension=="income_group" else "-x" if horizontal else None
    quantitative=alt.X(metric+":Q",title=label,axis=alt.Axis(format=number_format))
    category=alt.Y(dimension+":N",title=None,sort=sort,axis=alt.Axis(labelLimit=220))
    plot=alt.Chart(frame).mark_bar(cornerRadiusEnd=4,color=color or TEAL).encode(
        x=quantitative,y=category,tooltip=tooltip)
    if not horizontal:
        plot=alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4,color=color or TEAL).encode(
            x=alt.X(dimension+":N",title=None,sort=sort,axis=alt.Axis(labelAngle=-20,labelLimit=140)),
            y=alt.Y(metric+":Q",title=label,axis=alt.Axis(format=number_format)),tooltip=tooltip)
    return chart_style(plot.properties(height=height))

def heatmap(frame, row, col="income_group"):
    base=alt.Chart(frame).encode(
        x=alt.X(col+":N",sort=INCOME_ORDER if col=="income_group" else None,title=LABELS.get(col,col),
                axis=alt.Axis(labelAngle=-20,labelLimit=140)),
        y=alt.Y(row+":N",title=None,axis=alt.Axis(labelLimit=150)),
        tooltip=[alt.Tooltip(row+":N",title=LABELS.get(row,row)),
                 alt.Tooltip(col+":N",title=LABELS.get(col,col)),
                 alt.Tooltip("loan_count:Q",title="Loans",format=",d"),
                 alt.Tooltip("defaults:Q",title="Defaulted loans",format=",d"),
                 alt.Tooltip("default_rate:Q",title="Default rate",format=".2%"),
                 alt.Tooltip("small_group:N",title="Below 100 loans")])
    cells=base.mark_rect(cornerRadius=3).encode(
        color=alt.Color("default_rate:Q",title="Default rate",scale=alt.Scale(range=["#e8f3f4",TEAL,"#0f364a"]),
                        legend=alt.Legend(format=".0%")),
        opacity=alt.condition("datum.small_group",alt.value(.5),alt.value(1)))
    text=base.mark_text(fontSize=12,fontWeight=600).encode(
        text=alt.Text("default_rate:Q",format=".1%"),
        color=alt.condition("datum.default_rate > 0.25 && !datum.small_group",alt.value("white"),alt.value(NAVY)))
    return chart_style((cells+text).properties(height=max(220,frame[row].nunique()*44)))

def kpis(x):
    metrics=[("LOAN ACCOUNTS",f"{len(x):,}","Current selection"),
             ("CUSTOMERS",f"{x.customer_id.nunique():,}","Distinct borrowers"),
             ("ORIGINAL LENDING",short_number(x.loan_amount.sum()),"Dataset monetary units"),
             ("OBSERVED DEFAULT RATE",f"{x.target_default.mean():.2%}" if len(x) else "Unknown",
              f"{int(x.target_default.sum()):,} defaulted loans")]
    for col,(label,value,note) in zip(st.columns(4),metrics):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

def overview(x):
    if x.empty:
        st.info("No loans match the selected filters.");return
    product=portfolio_summary(x,["loan_type"])
    with st.container(border=True):
        st.subheader("Where the portfolio is concentrated")
        st.caption("Compare original lending with observed default frequency.")
        a,b=st.columns(2)
        with a:
            st.markdown("**Lending by product**")
            st.altair_chart(bar(product,"loan_type","original_loan_amount"),width="stretch")
        with b:
            st.markdown("**Default rate by product**")
            st.altair_chart(bar(product,"loan_type",color=CORAL),width="stretch")
        largest=product.loc[product.original_loan_amount.idxmax()]
        st.caption(f"{largest.loan_type} accounts for {largest.amount_share:.1%} of selected lending and has a {largest.default_rate:.1%} observed default rate.")
    with st.container(border=True):
        st.subheader("Income and repayment")
        a,b=st.columns(2)
        with a:
            st.markdown("**Default rate across income bands**")
            st.altair_chart(bar(portfolio_summary(x,["income_group"]),"income_group",horizontal=False),width="stretch")
        with b:
            st.markdown("**Default rate by missed payments**")
            st.altair_chart(bar(portfolio_summary(x,["missed_payment_group"]),"missed_payment_group",horizontal=False,color=CORAL),width="stretch")
        st.caption("Hover to see loan and default counts. Repayment patterns describe the recorded outcome; they do not establish a future forecast.")
    with st.expander("View and export the product summary"):
        table(product[["loan_type","loan_count","defaults","default_rate","original_loan_amount"]])
        st.download_button("Download summary",product.to_csv(index=False),"product_summary.csv","text/csv",key="overview_export")

def trends(x):
    if x.empty:
        st.info("No loans match the selected filters.");return
    grain=st.radio("Group loans by origination",["Quarter","Year"],horizontal=True)
    freq,per_year={"Quarter":("Q",4),"Year":("Y",1)}[grain]
    cohorts=vintage_trend(x,freq)
    with st.container(border=True):
        st.subheader("Lending and default rate by origination cohort")
        a,b=st.columns(2)
        with a:
            st.markdown("**Original lending**")
            st.altair_chart(bar(cohorts,"vintage","original_loan_amount",horizontal=False),width="stretch")
        with b:
            st.markdown("**Default rate with control limits**")
            axis=alt.X("vintage:N",title=None,sort=None,axis=alt.Axis(labelAngle=-45))
            band=alt.Chart(cohorts).mark_area(opacity=.15,color=TEAL).encode(
                x=axis,y=alt.Y("lower_limit:Q",title="Default rate",axis=alt.Axis(format=".0%"),scale=alt.Scale(zero=False)),
                y2="upper_limit:Q")
            centre=alt.Chart(cohorts).mark_line(strokeDash=[4,4],color="#7a8c9c").encode(x=axis,y="expected_rate:Q")
            line=alt.Chart(cohorts).mark_line(point=True,color=NAVY).encode(x=axis,y="default_rate:Q",tooltip=[
                alt.Tooltip("vintage:N",title="Cohort"),alt.Tooltip("loan_count:Q",title="Loans",format=",d"),
                alt.Tooltip("default_rate:Q",title="Default rate",format=".2%"),
                alt.Tooltip("lower_limit:Q",title="Lower limit",format=".2%"),
                alt.Tooltip("upper_limit:Q",title="Upper limit",format=".2%")])
            flags=alt.Chart(cohorts[cohorts.outside_limits]).mark_point(filled=True,size=90,color=CORAL).encode(
                x=axis,y="default_rate:Q")
            st.altair_chart(chart_style((band+centre+line+flags).properties(height=240)),width="stretch")
        note=f"The band is the selection's default rate +/- 2 standard errors; red points fall outside it."
        if len(cohorts)>=3:
            slope,p_value=trend_test(cohorts,per_year)
            note+=f" Trend: {slope*100:+.2f} percentage points per year (p = {p_value:.2f}"
            note+="; not significant)." if p_value>=.05 else "; significant at 5%)."
        st.caption(note+" Default rates are each cohort's eventual outcome, not the date of default.")
    timeline=repayment_timeline(x,repayment_rows())
    curve=seasoning_curve(timeline)
    months=calendar_delinquency(timeline)
    months["small_group"]=months.active_loans.lt(max(30,.2*months.active_loans.max()))
    months=months[~months.small_group].assign(month=lambda m:pd.to_datetime(m.calendar_month))
    with st.container(border=True):
        st.subheader("Delinquency over time")
        a,b=st.columns(2)
        with a:
            st.markdown("**30+ days past due by loan age**")
            chart=alt.Chart(curve).mark_line(point=True).encode(
                x=alt.X("payment_month:O",title="Months since origination",axis=alt.Axis(labelAngle=0)),
                y=alt.Y("dpd30_rate:Q",title="Share of loans",axis=alt.Axis(format=".0%")),
                color=alt.Color("outcome:N",title=None,scale=alt.Scale(domain=["Defaulted","Performing"],range=[CORAL,TEAL]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["outcome",alt.Tooltip("payment_month:O",title="Month"),
                         alt.Tooltip("dpd30_rate:Q",title="30+ days past due",format=".2%"),
                         alt.Tooltip("loan_count:Q",title="Loans",format=",d")])
            st.altair_chart(chart_style(chart.properties(height=240)),width="stretch")
        with b:
            st.markdown("**30+ days past due by calendar month**")
            chart=alt.Chart(months).mark_line(color=TEAL).encode(
                x=alt.X("month:T",title=None),y=alt.Y("dpd30_rate:Q",title="Share of active loans",axis=alt.Axis(format=".0%")),
                tooltip=[alt.Tooltip("calendar_month:N",title="Month"),
                         alt.Tooltip("dpd30_rate:Q",title="30+ days past due",format=".2%"),
                         alt.Tooltip("active_loans:Q",title="Active loans",format=",d")])
            st.altair_chart(chart_style(chart.properties(height=240)),width="stretch")
        st.caption("Calendar months assume one instalment per month after origination. Months with few active loans "
                   "at the start and end of the series are hidden.")
    with st.expander("View and export the cohort table"):
        table(cohorts[["vintage","loan_count","defaults","default_rate","lower_limit","upper_limit","original_loan_amount"]])
        st.download_button("Download cohorts",cohorts.to_csv(index=False),"cohort_trends.csv","text/csv",key="trend_export")

def segments(x):
    if x.empty:
        st.info("No loans match the selected filters.");return
    controls=st.columns([2,1])
    choice=controls[0].selectbox("Compare segments",["Education | income","Employment | income","Product | income","Gender | income"])
    gender=controls[1].selectbox("Gender comparison",["All","Female","Male"])
    z=x if gender=="All" else x[x.gender.eq(gender)]
    if z.empty:
        st.info("No loans match this gender selection.");return
    row={"Education | income":"education_level","Employment | income":"employment_type",
         "Product | income":"loan_type","Gender | income":"gender"}[choice]
    summary=portfolio_summary(z,[row,"income_group"])
    with st.container(border=True):
        st.subheader("Default rates within comparable income groups")
        st.altair_chart(heatmap(summary,row),width="stretch")
        st.caption(f"{len(z):,} loans in this comparison. Faded cells contain fewer than 100 loans; hover for counts. Associations are descriptive.")
    if row=="education_level":
        income=z.drop_duplicates("customer_id").groupby("education_level",observed=True).annual_income.agg(
            customer_count="size",median_income_usd="median").reset_index()
        with st.container(border=True):
            st.subheader("Education and annual income")
            st.altair_chart(bar(income,"education_level","median_income_usd",height=200),width="stretch")
            st.caption("Annual income in USD. Each selected customer is counted once.")
    with st.expander("Supporting figures"):
        table(summary)
    with st.container(border=True):
        st.subheader("External credit history")
        b = z.copy()
        b["bureau_dpd_group"] = pd.cut(b.bur_max_dpd, [-np.inf, 1, 30, 60, np.inf], right=False,
            labels=["0 overdue days", "1-29 days", "30-59 days", "60+ days"]).astype(object)
        b.loc[b.bur_thin_file.eq(1), "bureau_dpd_group"] = "No bureau records"
        b["bureau_dpd_group"] = b.bureau_dpd_group.fillna("DPD unknown in existing records")
        bureau=portfolio_summary(b,["bureau_dpd_group"])
        st.altair_chart(bar(bureau,"bureau_dpd_group",height=240),width="stretch")
        st.caption("No bureau records and records with unknown overdue days are shown separately.")
    with st.container(border=True):
        st.subheader("Credit history and age")
        h=add_credit_history_groups(z)
        for col,(key,title) in zip(st.columns(3),[("write_off_group","Debts written off by other lenders"),
                                                  ("credit_activity","Outside credit activity"),("age_group","Age")]):
            with col:
                st.markdown(f"**{title}**")
                st.altair_chart(bar(portfolio_summary(h,[key]),key,horizontal=False,height=230,
                                    color=CORAL if key=="write_off_group" else None),width="stretch")
        st.caption("Default rate by group. Customers without bureau records are shown separately, not counted as zero write-offs.")

def loan_review(x,data):
    if x.empty:
        st.info("No loans match the selected filters.");return None
    scored=data.scores[data.scores.account_id.isin(x.account_id)].sort_values("default_probability",ascending=False)
    if not data.model_fresh or scored.empty:
        st.warning("Saved scores are unavailable for this portfolio snapshot.");return None
    account=st.selectbox("Select a loan account",scored.account_id.tolist(),key="review_account")
    s=scored.set_index("account_id").loc[account]
    row=x.set_index("account_id").loc[account]
    with st.container(border=True):
        st.subheader(account)
        a,b,c=st.columns(3)
        a.metric("Saved risk score",f"{s.default_probability:.1%}")
        b.metric("Relative risk band",str(s.risk_band).replace(" relative score",""))
        c.metric("Recorded outcome","Defaulted" if row.target_default else "Performing")
        st.caption(f"{row.loan_type} | {row.region} | {row.employment_type} | Annual income USD {row.annual_income:,.0f}")
        profile=pd.DataFrame([{"Original loan amount":row.loan_amount,"Tenure (months)":row.tenure_months,
            "Loan / annual income":round(row.loan_to_income,2),
            "Missed payments":row.rp_n_missed,"Maximum bureau overdue days":row.bur_max_dpd}])
        st.dataframe(profile,hide_index=True,width="stretch")
    with st.container(border=True):
        st.subheader("What contributes to this score?")
        drivers=pd.DataFrame([{"feature":feature_label(str(s[f"driver_{i}"])),
            "contribution":s[f"contribution_{i}"],"direction":"Raises score" if s[f"contribution_{i}"]>0 else "Lowers score"} for i in [1,2,3]])
        chart=alt.Chart(drivers).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("feature:N",title=None,sort="-x",axis=alt.Axis(labelLimit=240)),
            x=alt.X("contribution:Q",title="Contribution to log-odds"),
            color=alt.Color("direction:N",scale=alt.Scale(domain=["Raises score","Lowers score"],range=[CORAL,TEAL])),
            tooltip=["feature",alt.Tooltip("contribution:Q",format=".3f"),"direction"]).properties(height=170)
        st.altair_chart(chart_style(chart),width="stretch")
        st.caption("Three largest additive model contributions. These are not causal effects or percentage-point changes.")
    status={"training_in_sample":"Training data | score is in-sample",
            "validation_used_for_selection":"Validation data | used for model selection",
            "held_out_test":"Held-out test data"}.get(s.evaluation_status,s.evaluation_status)
    st.info(status+". The dataset does not define a validated future default horizon. Review indicators support human assessment.")
    return account

def model_performance(data):
    if not data.model_fresh:
        st.warning("Model results do not match the current data snapshot.");return
    st.caption("Evaluation scope: fixed held-out test set. Portfolio filters do not change model-validation results.")
    selected=data.model_test[data.model_test.selected.eq(True)].iloc[0]
    a,b,c=st.columns(3)
    a.metric("Selected model AUC",f"{selected.auc:.3f}")
    b.metric("KS",f"{selected.ks:.3f}")
    c.metric("Gini",f"{selected.gini:.3f}")
    st.caption(f"Selected using validation data: {MODEL_LABELS.get(selected.model,selected.model)}.")
    prediction=pd.read_csv(MODEL_REPORTS/"test_predictions.csv")
    curves=[]
    for model,g in prediction.groupby("model"):
        fpr,tpr,_=roc_curve(g.target_default,g.default_probability)
        curves.extend({"False-positive rate":float(a),"Recall":float(b),"Model":MODEL_LABELS.get(model,model)} for a,b in zip(fpr,tpr))
    with st.container(border=True):
        a,b=st.columns(2)
        with a:
            st.markdown("**ROC discrimination**")
            curve=alt.Chart(pd.DataFrame(curves)).mark_line().encode(
                x=alt.X("False-positive rate:Q",scale=alt.Scale(domain=[0,1])),
                y=alt.Y("Recall:Q",scale=alt.Scale(domain=[0,1])),
                color=alt.Color("Model:N",legend=alt.Legend(orient="bottom")),
                tooltip=["Model","False-positive rate","Recall"])
            guide=alt.Chart(pd.DataFrame({"False-positive rate":[0,1],"Recall":[0,1]})).mark_line(strokeDash=[4,4],color="#aebcca").encode(x="False-positive rate:Q",y="Recall:Q")
            st.altair_chart(chart_style((curve+guide).properties(height=240)),width="stretch")
        with b:
            st.markdown("**Probability calibration**")
            cal=pd.read_csv(MODEL_REPORTS/"test_calibration.csv")
            cal["Model"]=cal.model.map(MODEL_LABELS)
            chart=alt.Chart(cal).mark_line(point=True).encode(
                x=alt.X("mean_probability:Q",title="Predicted probability",axis=alt.Axis(format=".0%"),scale=alt.Scale(domain=[0,1])),
                y=alt.Y("observed_default_rate:Q",title="Observed default rate",axis=alt.Axis(format=".0%"),scale=alt.Scale(domain=[0,1])),
                color=alt.Color("Model:N",legend=alt.Legend(orient="bottom")),
                tooltip=["Model",alt.Tooltip("mean_probability:Q",format=".2%"),alt.Tooltip("observed_default_rate:Q",format=".2%")])
            cal_guide=alt.Chart(pd.DataFrame({"mean_probability":[0,1],"observed_default_rate":[0,1]})).mark_line(strokeDash=[4,4],color="#aebcca").encode(x="mean_probability:Q",y="observed_default_rate:Q")
            st.altair_chart(chart_style((chart+cal_guide).properties(height=240)),width="stretch")
    table(data.model_test[["model","selected","auc","ks","gini","precision","recall","accuracy","brier"]])
    st.caption("Both algorithms use the same test loans. Accuracy alone is insufficient at this portfolio's default prevalence. Selection was fixed before test evaluation.")
    with st.container(border=True):
        st.subheader("What drives the selected model")
        importance=pd.read_csv(MODEL_REPORTS/"contribution_importance.csv").head(12)
        importance["feature"]=importance.encoded_feature.map(feature_label)
        st.altair_chart(bar(importance,"feature","mean_absolute_log_odds_contribution",height=330),width="stretch")
        st.caption("Average size of each input's contribution to the score across held-out test loans. "
                   "Larger means more influence, in either direction; it is not a causal effect.")

def data_quality(x,data):
    if x.empty:
        st.info("No loans match the selected filters.");return
    counts=x[data.source_columns].isna().sum()
    missing=pd.DataFrame({"column":counts.index,"missing_loan_rows":counts.values,
                          "missing_share":counts.values/len(x)})
    missing=missing[missing.missing_loan_rows.gt(0)].sort_values("missing_loan_rows",ascending=False)
    with st.container(border=True):
        st.subheader("Where information is unavailable")
        st.altair_chart(bar(missing,"column","missing_share",height=max(250,len(missing)*24)),width="stretch")
        st.caption("Counts are loan rows, so a customer with several loans may appear more than once. Unknown values remain visible.")
    a,b=st.columns(2)
    a.metric("Loans without bureau records",f"{int(x.bur_thin_file.sum()):,}")
    b.metric("Existing bureau / unknown DPD",f"{int((x.bur_thin_file.eq(0)&x.bur_n_dpd_known.eq(0)).sum()):,}")
    with st.expander("Missing-value counts and interpretation",expanded=False):
        table(missing)
        st.write("No recorded credit history is different from observed zero overdue days. Missing current-employment tenure is zero only for unemployed customers. Other genuine unknowns stay blank in analytics.")
        st.caption("Bureau observation dates and the default outcome horizon are not established; model scores are retrospective.")

def result_dashboard(result):
    st.subheader("Analysis from your question")
    st.write(result["text"])
    cohort=result.get("cohort")
    if cohort:
        st.caption(f"Answer selection: {cohort['loans']:,} loans | {cohort['customers']:,} customers")
        pieces=[LABELS.get(f["column"],f["column"])+": "+", ".join(f["values"]) for f in cohort["filters"]]
        if cohort.get("income_min_inclusive") is not None:pieces.append(f"Income >= {cohort['income_min_inclusive']:,.0f}")
        if cohort.get("income_max_exclusive") is not None:pieces.append(f"Income < {cohort['income_max_exclusive']:,.0f}")
        st.caption("Applied filters: "+"; ".join(pieces) if pieces else "Applied filters: All portfolio")
    for i,item in enumerate(result.get("tables",[])):
        f=pd.DataFrame(item["rows"])
        with st.container(border=True):
            st.markdown("**"+item["title"]+"**")
            if f.empty:
                st.info("No groups meet this selection.");continue
            dims=[c for c in f if c in LABELS and c in ["loan_type","income_group","employment_type","education_level","gender","region","age_group","housing_type","bureau_history","missed_payment_group"]]
            if "default_rate" in f and dims:
                if len(dims)==2:
                    xcol="income_group" if "income_group" in dims else dims[1]
                    row=next(c for c in dims if c!=xcol)
                    st.altair_chart(heatmap(f,row,xcol),width="stretch",key=f"answer_{i}")
                else:
                    f["Segment"]=f[dims].astype(str).agg(" / ".join,axis=1)
                    st.altair_chart(bar(f,"Segment",height=min(650,max(200,len(f)*25))),width="stretch",key=f"answer_{i}")
            elif "missing_share" in f:
                st.altair_chart(bar(f,"column","missing_share",height=320),width="stretch",key=f"answer_{i}")
            elif "median_income_usd" in f:
                f["Segment"]=f[[c for c in ["education_level","gender"] if c in f]].astype(str).agg(" / ".join,axis=1)
                st.altair_chart(bar(f,"Segment","median_income_usd",height=250),width="stretch",key=f"answer_{i}")
            elif "auc" in f:
                f["Model label"]=f.model.map(MODEL_LABELS)
                st.altair_chart(bar(f,"Model label","auc",height=220),width="stretch",key=f"answer_{i}")
            elif "log_odds_contribution" in f:
                f["feature"]=f.feature.map(feature_label)
                st.altair_chart(bar(f,"feature","log_odds_contribution",height=180),width="stretch",key=f"answer_{i}")
            table(f.drop(columns=["Segment","Model label"],errors="ignore"))
    if result.get("limitation"):st.caption(result["limitation"])
