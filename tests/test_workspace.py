"""Dashboard, server-secret and shared-context regression checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from portfolio_common import ROOT
from portfolio_chat import PortfolioData,PortfolioChat,make_plan
from dashboard_views import selected_data
from server_config import read_api_key
from package_submission import include_path
from test_chatbot import FakeClient,tool_response

class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=PortfolioData()

    def test_filter_totals(self):
        x=selected_data(self.data.a,[{"column":"loan_type","values":["Mortgage"]}])
        self.assertEqual(len(x),1432)
        self.assertEqual(int(x.target_default.sum()),353)

    def test_server_secret_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/".env").write_text("ANTHROPIC_API_KEY=private-test-env")
            self.assertEqual(read_api_key(root,{}),"private-test-env")
            (root/".streamlit").mkdir()
            (root/".streamlit/secrets.toml").write_text('ANTHROPIC_API_KEY="private-test-toml"')
            self.assertEqual(read_api_key(root,{}),"private-test-toml")
            self.assertEqual(read_api_key(root,{"ANTHROPIC_API_KEY":"private-test-process"}),"private-test-process")

    def test_blank_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/".env").write_text("ANTHROPIC_API_KEY=")
            self.assertIsNone(read_api_key(root,{}))

    def test_package_excludes_credentials_and_environment(self):
        for path in [".env",".env.production",".streamlit/secrets.toml","docs/.env",
                     "scripts/__pycache__/test.pyc",".venv/Scripts/python.exe",
                     "reports/chatbot/audit.jsonl","../outside.txt"]:
            self.assertFalse(include_path(path),path)
        for path in [".env.example",".streamlit/config.toml","scripts/chatbot_app.py","data/raw/customers.csv"]:
            self.assertTrue(include_path(path),path)

    def test_chat_inherits_dashboard_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            bot=PortfolioChat(self.data,FakeClient(tool_response(make_plan())),Path(directory)/"audit")
            result=bot.ask("Portfolio overview",dashboard_filters=[{"column":"loan_type","values":["Mortgage"]}])
            self.assertEqual(result["cohort"]["loans"],1432)
            self.assertEqual(result["query_plan"]["filters"][0]["values"],["Mortgage"])

    def test_chat_can_explicitly_reset_dashboard_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            bot=PortfolioChat(self.data,FakeClient(tool_response(make_plan(inherit_dashboard_filters=False))),Path(directory)/"audit")
            result=bot.ask("Show the entire portfolio",dashboard_filters=[{"column":"loan_type","values":["Mortgage"]}])
            self.assertEqual(result["cohort"]["loans"],14685)

    def test_selected_loan_context(self):
        with tempfile.TemporaryDirectory() as directory:
            bot=PortfolioChat(self.data,FakeClient(tool_response(make_plan(action="loan_score",account_id="ACC200001"))),Path(directory)/"audit")
            result=bot.ask("Explain this loan",selected_account_id="ACC200001")
            self.assertEqual(result["status"],"answered")

    def test_employee_ui_and_views(self):
        # A sentinel server key verifies it never appears in rendered employee elements.
        with patch.dict(os.environ,{"ANTHROPIC_API_KEY":"private-test-sentinel-never-render"}):
            app=AppTest.from_file(str(ROOT/"scripts/chatbot_app.py"),default_timeout=40).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.text_input),0)
            for kind in ["markdown","caption","text","info","warning","code"]:
                for element in getattr(app,kind):
                    self.assertNotIn("private-test-sentinel",str(element.value))
                    self.assertNotIn("Anthropic API key",str(element.value))
                    self.assertNotIn("Claude connection",str(element.value))
            app.multiselect(key="filter_loan_type").set_value(["Mortgage"]).run()
            self.assertFalse(app.exception)
            for view in ["Customer segments","Loan review","Model performance","Data quality"]:
                app.radio(key="workspace_view").set_value(view).run()
                self.assertFalse(app.exception,[e.message for e in app.exception])
            next(b for b in app.button if b.label=="Compare loan products").click().run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual(app.radio(key="workspace_view").value,"Assistant analysis")
            result=app.session_state["analysis_result"]
            self.assertEqual(result["cohort"]["loans"],1432)
            self.assertTrue(len(app.dataframe)>0)

if __name__=="__main__":unittest.main()
