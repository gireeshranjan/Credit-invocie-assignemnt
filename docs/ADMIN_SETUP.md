# Administrator setup

The employee interface contains no credential fields. Credentials are loaded only by the Python server.

## Start locally

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 launch_chatbot.py
```

Open http://127.0.0.1:8501. The dashboard and portfolio shortcuts work without an API key.

## Enable natural-language questions

Privately edit the project .env file and set ANTHROPIC_API_KEY to your Anthropic API credential.
For a fresh submission checkout, copy .env.example to .env first. Never paste a key in chat, a
notebook, source code or the employee interface. Restart the app after configuration.

Accepted server sources, in precedence order:
1. ANTHROPIC_API_KEY process environment variable.
2. Project .streamlit/secrets.toml, with an ANTHROPIC_API_KEY entry.
3. Project .env, with an ANTHROPIC_API_KEY entry.

The private .env and secrets.toml are excluded by .gitignore and the submission builder.
The included .env.example is blank. There is no key in the submitted application.
An Anthropic API account with access to claude-sonnet-5-5 is required; a chat subscription
alone does not configure API access. The app does not silently switch models.
Without a configured key, free-form chat is disabled and local shortcuts remain available.
A configured key does not prove account/model access; perform a live check before presenting.

## Install on another machine

Use Python 3.11 or 3.12 (the locked requirements were verified on 3.12):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 launch_chatbot.py
```

Keep the supplied data, reports and saved models together: the assistant checks their data fingerprint.
The source-data and model pipelines can be rerun using run_pipeline.py.

## Verify and submit

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/package_submission.py
```

Submit dist/Credit_Risk_Workspace.zip. The builder excludes local credentials, the Python
environment, caches and operational chat logs. It includes source code, data, reports,
workbooks, models, tests and setup documentation.

After privately enabling API access, try a portfolio overview, a filtered product comparison,
a follow-up and a selected-loan explanation. Check that unrelated celebrity questions,
credential requests and automatic approval decisions are refused.

This is a local synthetic-data demonstration. The launcher binds to 127.0.0.1.
External multi-user deployment would additionally require employee authentication,
authorization, server-wide quotas and deployment-specific secret management.
