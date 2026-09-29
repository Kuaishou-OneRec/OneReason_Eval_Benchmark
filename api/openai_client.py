"""
OpenAI-Compatible API Client Implementation
Supports any OpenAI-compatible endpoint (e.g., vLLM, Ollama, local models).
"""
from typing import Optional
from openai import OpenAI
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    OpenAI-Compatible API Client

    Works with any server exposing an OpenAI-compatible chat completions API.

    Example:
        >>> client = OpenAIClient(
        ...     api_key="your-api-key",
        ...     base_url="http://localhost:8000/v1",
        ...     model_name="Qwen/Qwen3-1.7B",
        ... )
        >>> response = client.generate("Tell me a joke")
    """

    def _setup(self):
        """Initialize OpenAI-compatible client"""
        self.api_key = self.config.get("api_key", "EMPTY")
        self.base_url = self.config.get("base_url", "http://localhost:8000/v1")
        self.model_name = self.config.get("model_name", "default")
        self.default_max_tokens = self.config.get("max_new_tokens", 300)
        self.default_temperature = self.config.get("temperature", 0.7)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call OpenAI-compatible API to generate text

        Args:
            prompt: Input prompt
            temperature: Temperature parameter (0.0-2.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters

        Returns:
            str: Generated text content

        Raises:
            Exception: Raised when API call fails
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        request_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        request_params.update(kwargs)

        response = self.client.chat.completions.create(**request_params)

        if response and response.choices:
            content = response.choices[0].message.content
            if content:
                return content
            else:
                raise Exception("API returned empty response")
        else:
            raise Exception("API returned invalid response")
