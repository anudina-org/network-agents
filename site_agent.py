import os
from langsmith import traceable
import requests
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from sqlalchemy import null

SITE_API_URL = "http://network-site-api.ocp.anudina.com/getSiteDetails"


@tool
def get_site_details() -> dict:
    """Fetch network site details from the site API."""
    response = requests.get(SITE_API_URL, timeout=10)
    response.raise_for_status()
    return response.json()

@traceable(run_type="llm", metadata={"ls_provider": "ollama", "ls_model_name": "qwen2.5"})
def _make_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    print(f"Using LLM provider: {provider}")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            stream_usage=True,
            temperature=0,
            streaming=True,
        ).with_config({
    "metadata": {
        "ls_provider": "ollama",
        "ls_model_name": "qwen2.5" # Must match exactly what you put in Settings
    }
})
    from langchain_google_genai import ChatGoogleGenerativeAI
    return null;

    # return ChatGoogleGenerativeAI(
    #     model=os.getenv("GOOGLE_MODEL", "gemini-2.5"),
    #     google_api_key=os.getenv("GOOGLE_API_KEY"),
    #     max_retries=3,
    # )


def create_site_agent():
    llm = _make_llm()
    return create_react_agent(
        llm,
        tools=[get_site_details],
        prompt=(
            "You are a network site specialist agent. "
            "Use the get_site_details tool to fetch live site data and answer questions about network sites. "
            "Be concise and accurate."
        ),
    )
