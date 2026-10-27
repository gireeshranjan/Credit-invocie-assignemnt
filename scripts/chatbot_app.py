"""Employee banking workspace: portfolio dashboard and integrated assistant."""
import html
import re
import pandas as pd
import streamlit as st
from portfolio_chat import PortfolioChat, make_plan
from portfolio_common import ROOT
from server_config import configured_client
import dashboard_views as views

st.set_page_config(page_title="Portfolio | Credit Risk Workspace",page_icon=":material/account_balance:",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f5f7fb;color:#16324b;}
.block-container {padding-top:2rem;padding-bottom:2rem;max-width:1800px;}
[data-testid="stSidebar"] {background:#102a43;min-width:245px;max-width:265px;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#c7d7e4;}
[data-testid="stSidebar"] hr {border-color:#28445c;}
[data-testid="stToolbar"] {visibility:hidden;}
[data-testid="stHeader"] {background:transparent;}
h1,h2,h3 {color:#16324b;letter-spacing:-.025em;}
h1 {font-size:2.1rem !important;} h3 {font-size:1.15rem !important;}
[data-testid="stVerticalBlockBorderWrapper"]>div {background:white;border-radius:14px;}
[data-testid="stMetricLabel"] p {font-size:.8rem;color:#64788a;}
[data-testid="stMetricValue"] {color:#12304a;}
.stButton>button {border-radius:8px;font-size:.83rem;border-color:#d8e3eb;}
.stButton>button:hover {border-color:#087f8c;color:#087f8c;}
.kpi {background:#fff;border:1px solid #e1e8f0;border-radius:13px;padding:18px 20px;margin:6px 0 20px;min-height:125px;}
.kpi-label {font-size:10px;font-weight:700;color:#6c8092;letter-spacing:.09em;}
.kpi-value {font-size:30px;font-weight:700;line-height:1.5;color:#12304a;}
.kpi-note {font-size:11px;color:#7a8c9c;}
.brand {color:white;font-size:21px;font-weight:700;letter-spacing:.08em;margin-bottom:2px;}
.brand-sub {color:#7fa8bf;font-size:10px;letter-spacing:.15em;margin-bottom:28px;}
.eyebrow {font-size:10px;color:#087f8c;font-weight:700;letter-spacing:.16em;}
.tag {display:inline-block;border:1px solid #d8e4ed;background:#edf5f7;color:#416276;border-radius:20px;padding:5px 12px;font-size:11px;}
.assistant-dot {color:#087f8c;font-size:14px;}
[data-testid="stChatMessage"] {background:#f6f9fc;border-radius:10px;font-size:.85rem;}
footer {visibility:hidden;}
@media(max-width:950px){.kpi-value{font-size:24px}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""",unsafe_allow_html=True)

# Discard the retired credential widget and old session object on migration.
for legacy in ["portfolio_api_key","portfolio_chat","chat_history"]:
    st.session_state.pop(legacy,None)
if "workspace_engine" not in st.session_state:
    try:
        st.session_state.workspace_engine=PortfolioChat()
        st.session_state.workspace_history=[]
    except (OSError,ValueError,KeyError):
        st.error("Portfolio data is temporarily unavailable. Please contact your administrator.")
        st.stop()
engine=st.session_state.workspace_engine
engine.client=configured_client()
data=engine.data
if "pending_view" in st.session_state:
    st.session_state.workspace_view=st.session_state.pop("pending_view")

def reset_filters():
    for field in ["loan_type","region","income_group","employment_type"]:
        st.session_state["filter_"+field]=[]
    st.session_state.pop("analysis_result",None)

with st.sidebar:
    st.markdown('<div class="brand">PORTFOLIO</div><div class="brand-sub">CREDIT RISK WORKSPACE</div>',unsafe_allow_html=True)
    page=st.radio("Workspace",["Portfolio overview","Trends over time","Customer segments","Loan review","Model performance","Data quality","Assistant analysis"],
                  key="workspace_view",label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="brand-sub" style="margin-bottom:10px">PORTFOLIO FILTERS</div>',unsafe_allow_html=True)
    filters=[]
    for field,label in [("loan_type","Loan product"),("region","Region"),("income_group","Income band"),("employment_type","Employment")]:
        options=data.catalog[field]
        if field=="income_group":options=[v for v in views.INCOME_ORDER if v in options]
        selected=st.multiselect(label,options,key="filter_"+field,placeholder="All")
        if selected:filters.append({"column":field,"values":selected})
    st.button("Reset filters",on_click=reset_filters,width="stretch")
    st.divider()
    st.caption("Synthetic portfolio | Assignment workspace")
    st.caption("Use scores to support review. Lending decisions require human judgement.")

x=views.selected_data(data.a,filters)
heading=st.columns([4,1])
with heading[0]:
    st.markdown('<div class="eyebrow">RISK INTELLIGENCE / PORTFOLIO</div>',unsafe_allow_html=True)
    st.title(page)
    st.caption("Explore lending exposure, customer segments and the evidence behind risk.")
with heading[1]:
    st.markdown('<br><span class="tag">Portfolio snapshot</span>',unsafe_allow_html=True)
views.kpis(x)
left,right=st.columns([2.25,1],gap="large")
selected_account=None
with left:
    if page=="Portfolio overview":views.overview(x)
    elif page=="Trends over time":views.trends(x)
    elif page=="Customer segments":views.segments(x)
    elif page=="Loan review":selected_account=views.loan_review(x,data)
    elif page=="Model performance":views.model_performance(data)
    elif page=="Data quality":views.data_quality(x,data)
    elif st.session_state.get("analysis_result"):
        st.caption("Summary cards reflect sidebar filters. The answer below shows its own explicit selection.")
        views.result_dashboard(st.session_state.analysis_result)
    else:
        st.info("Ask a portfolio question or choose a shortcut. Its supporting chart will appear here.")
        views.overview(x)

def push_answer(question,result,mode):
    entry={"question":question,"result":result,"mode":mode}
    st.session_state.workspace_history.append(entry)
    st.session_state.workspace_history=st.session_state.workspace_history[-12:]
    if result["status"]=="answered":
        st.session_state.analysis_result=result
        st.session_state.pending_view="Assistant analysis"
    st.rerun()

with right:
    with st.container(border=True):
        st.markdown('### Portfolio assistant')
        st.caption("Ask about the current selection. Explore the evidence in the dashboard.")
        with st.expander("How the assistant works"):
            st.markdown("**Trigger:** you type a question or press a shortcut. The assistant only chooses one of a "
                        "fixed set of read-only queries; every number is calculated from the portfolio data.\n\n"
                        "**False positive:** a valid question is refused or you are asked to rephrase, for example "
                        "unusual wording or an unsupported calculation. Rephrase or use a shortcut.\n\n"
                        "**False negative:** a question is matched to the wrong but valid query, such as the wrong "
                        "filter. Check the applied filters shown with every answer.\n\n"
                        "**Human review:** answers and scores support analysis only. A credit officer checks the "
                        "filters and tables and makes any lending decision.")
        if engine.client is None:
            st.caption("Natural-language assistance is temporarily unavailable. Portfolio shortcuts are available.")
        else:
            st.markdown('<span class="tag">Ready for portfolio questions</span>',unsafe_allow_html=True)
        if selected_account:st.caption("Selected loan: "+selected_account)
        st.markdown("**Explore this selection**")
        shortcuts=[
            ("Compare loan products",{"action":"segment","dimensions":["loan_type"]}),
            ("Explore education and income",{"action":"education_income"}),
            ("Review missing information",{"action":"missing_values"})]
        if selected_account:
            shortcuts.insert(0,("Explain the selected loan",{"action":"loan_score","account_id":selected_account}))
        for label,overrides in shortcuts:
            if st.button(label,key="shortcut_"+label,width="stretch"):
                plan=make_plan(**overrides)
                if plan["action"]!="loan_score":plan["filters"]=filters
                result=data.execute(plan)
                result["query_plan"]=plan
                engine.last_plan=plan
                push_answer(label,result,"Portfolio shortcut")
        st.divider()
        with st.container(height=390 if st.session_state.workspace_history else 180,border=False):
            if not st.session_state.workspace_history:
                st.markdown("**Start with a question**")
                st.write("Which products have the highest observed default rate?")
                st.caption("Or compare income, employment and education. All figures come from this portfolio.")
            for i,entry in enumerate(st.session_state.workspace_history[-5:]):
                with st.chat_message("user"):
                    st.write(entry["question"])
                with st.chat_message("assistant"):
                    result=entry["result"]
                    if entry["mode"]=="Portfolio shortcut":st.caption("Portfolio shortcut")
                    if result["status"]=="answered":
                        st.write(result["text"])
                        st.caption("Supporting figures are available in the analysis view.")
                        if st.button("View this analysis",key=f"history_{i}"):
                            st.session_state.analysis_result=result
                            st.session_state.pending_view="Assistant analysis"
                            st.rerun()
                    else:st.info(result["text"])
        question=st.chat_input("Ask about this portfolio...",key="workspace_question",max_chars=1500,
                               disabled=engine.client is None)
        if question:
            safe_question=re.sub(r"sk-ant-[A-Za-z0-9_-]+","[redacted]",question)
            with st.spinner("Checking the portfolio..."):
                result=engine.ask(question,dashboard_filters=filters,selected_account_id=selected_account)
            push_answer(safe_question,result,"Assistant")
        if st.session_state.workspace_history and st.button("Clear conversation",width="stretch"):
            st.session_state.workspace_history=[]
            engine.last_plan=None
            st.rerun()
st.caption("Source: supplied synthetic loan portfolio. Unknown information is retained. Scores do not establish a future default horizon.")
