"""Private configuration. Credentials are never exposed as Streamlit controls."""
from functools import lru_cache
from pathlib import Path
import os
import tomllib
from dotenv import dotenv_values
from portfolio_common import ROOT

def read_api_key(root=ROOT, environ=None):
    env = os.environ if environ is None else environ
    value = env.get("ANTHROPIC_API_KEY", "").strip()
    if not value:
        secret_file = Path(root) / ".streamlit" / "secrets.toml"
        if secret_file.exists():
            with secret_file.open("rb") as f:
                value = str(tomllib.load(f).get("ANTHROPIC_API_KEY", "")).strip()
    if not value:
        private_file = Path(root) / ".env"
        if private_file.exists():
            value = (dotenv_values(private_file, interpolate=False).get("ANTHROPIC_API_KEY") or "").strip()
    if value.lower() in {"", "your-key-here", "replace-me", "your_anthropic_api_key"}:
        return None
    return value

@lru_cache(maxsize=2)
def _client(key):
    from portfolio_chat import create_client
    return create_client(key)

def configured_client():
    try:
        key = read_api_key()
        return _client(key) if key else None
    except (OSError, ValueError, TypeError):
        return None
