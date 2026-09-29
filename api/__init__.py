"""
Unified LLM API Wrapper
Supports convenient calling of Gemini, DeepSeek, and Claude models
"""
import json
import os
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseLLMClient
# Resolve only the requested provider, preserving public class imports.
_PROVIDER_CLASSES = {
    "GeminiClient": (".gemini", "GeminiClient"),
    "GenaiGeminiClient": (".genai_gemini", "GenaiGeminiClient"),
    "DeepSeekClient": (".deepseek", "DeepSeekClient"),
    "ClaudeClient": (".claude", "ClaudeClient"),
    "OpenAIClient": (".openai_client", "OpenAIClient"),
}

def __getattr__(name):
    if name in _PROVIDER_CLASSES:
        module, class_name = _PROVIDER_CLASSES[name]
        return getattr(importlib.import_module(module, __name__), class_name)
    raise AttributeError(name)

class _ProviderMap(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return __getattr__(value) if isinstance(value, str) else value

MODEL_CLASSES = _ProviderMap({
    "gemini": "GenaiGeminiClient", "deepseek": "DeepSeekClient",
    "claude": "ClaudeClient", "openai": "OpenAIClient", "v4flash": "OpenAIClient",
})


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file

    Args:
        config_path: Configuration file path, defaults to api/config/llm_config.json

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: Configuration file does not exist
        json.JSONDecodeError: Configuration file format error
    """
    if config_path is None:
        config_path = os.environ.get("LLM_CONFIG_PATH") or Path(__file__).parent / "config" / "llm_config.json"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            name = value[2:-1]
            if not os.environ.get(name):
                raise ValueError(f"Missing environment variable: {name}")
            return os.environ[name]
        return value
    return resolve(config)


def get_client(model: str, **config) -> BaseLLMClient:
    """
    Factory function: Create LLM client instance

    Args:
        model: Model name ("gemini" or "deepseek")
        **config: Model-specific configuration parameters

    Returns:
        BaseLLMClient: Client instance

    Raises:
        ValueError: Unsupported model type

    Example:
        >>> client = get_client("gemini",
        ...                    project="your-project",
        ...                    location="us-central1")
        >>> result = client.generate("Tell me a joke")
    """
    model = model.lower()
    if model not in MODEL_CLASSES:
        raise ValueError(
            f"Unsupported model: {model}. "
            f"Supported models: {', '.join(MODEL_CLASSES.keys())}"
        )

    client_class = MODEL_CLASSES[model]
    return client_class(**config)


def get_client_from_config(
    model: str,
    config_path: Optional[str] = None
) -> BaseLLMClient:
    """
    Create LLM client from configuration file

    Args:
        model: Model name ("gemini" or "deepseek")
        config_path: Configuration file path, defaults to api/config/llm_config.json

    Returns:
        BaseLLMClient: Client instance

    Raises:
        ValueError: Model configuration not found in configuration file

    Example:
        >>> client = get_client_from_config("gemini")
        >>> result = client.generate("Tell me a joke")
    """
    config = load_config(config_path)
    model = model.lower()

    if model not in config:
        raise ValueError(
            f"Model '{model}' configuration not found in configuration file. "
            f"Available models: {', '.join(config.keys())}"
        )

    model_config = config[model]
    return get_client(model, **model_config)


def batch_generate(
    prompts: List[str],
    model: str,
    max_workers: int = 5,
    show_progress: bool = True,
    config_path: Optional[str] = None,
    **config
) -> List[Dict[str, Any]]:
    """
    Batch generate text (with concurrent support)

    Args:
        prompts: List of prompts
        model: Model name ("gemini" or "deepseek")
        max_workers: Maximum number of concurrent threads, default 5
        show_progress: Whether to show progress bar, default True
        config_path: Configuration file path (if provided, use configuration file first)
        **config: Model configuration parameters (if not using configuration file)

    Returns:
        List[Dict]: List of results, each element contains:
            - prompt: Original prompt
            - result: Generated text (on success)
            - error: Error message (on failure)
            - success: Whether successful

    Example:
        >>> # Using configuration file
        >>> results = batch_generate(
        ...     prompts=["Question 1", "Question 2", "Question 3"],
        ...     model="gemini",
        ...     max_workers=3
        ... )

        >>> # Direct configuration
        >>> results = batch_generate(
        ...     prompts=["Question 1", "Question 2"],
        ...     model="deepseek",
        ...     api_key="your-key",
        ...     appid="your-appid"
        ... )
    """
    if config_path:
        client = get_client_from_config(model, config_path)
    else:
        client = get_client(model, **config)

    return client.batch_generate(
        prompts=prompts,
        max_workers=max_workers,
        show_progress=show_progress
    )


# Export all public interfaces
__all__ = [
    # Classes
    "BaseLLMClient",
    "GeminiClient",
    "GenaiGeminiClient",
    "DeepSeekClient",
    "ClaudeClient",
    "OpenAIClient",
    # Functions
    "get_client",
    "get_client_from_config",
    "batch_generate",
    "load_config",
]
