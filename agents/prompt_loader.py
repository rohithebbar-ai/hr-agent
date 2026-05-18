"""
Prompt Loader — loads versioned prompts from YAML files.

Usage:
    from agents.prompt_loader import load_prompt
    prompt = load_prompt("generate", version="v1")
    chain = prompt | llm
"""

from pathlib import Path
from zipfile import DEFAULT_VERSION
from langchain_core.runnables import history
import yaml
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DEFAULT_VERSION = "v1"

def load_prompt(name:str, version:str = DEFAULT_VERSION) -> ChatPromptTemplate:
    path = PROMPTS_DIR / version / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    messages = [("system", config["system"])]
    if f"{history}" in config.get("system", ""):
        messages.append(("placeholder", "{history}"))
    messages.append(("human", config["human"]))

    return ChatPromptTemplate.from_messages(messages)

def get_prompt_version_info(name:str, version: str = DEFAULT_VERSION) -> dict:
    path = PROMPTS_DIR / version / f"{name}.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return {
        "name": config.get("name"),
        "version": config.get("version"),
        "description": config.get("description"),
        "created": config.get("created")
    }
