"""
Gemini API Client using google-genai SDK
Uses Vertex AI backend with REST transport, avoiding gRPC concurrent init issues.

Install: pip install google-genai -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import os
from typing import Optional
from google import genai
from google.genai import types
from .base import BaseLLMClient


class GenaiGeminiClient(BaseLLMClient):
    """
    Gemini client using google-genai SDK (Vertex AI backend, REST transport).

    A single Client instance is created in the main thread during _setup(),
    then shared across all worker threads. Each generate_content() call is an
    independent HTTP request, so there is no gRPC channel contention under
    concurrent load.

    Config keys are identical to GeminiClient for drop-in compatibility:
        project, location, model_name, credentials_path,
        max_new_tokens, temperature, max_retries, retry_delay
    """

    def _setup(self):
        self.project = self.config.get("project")
        self.location = self.config.get("location")
        self.model_name = self.config.get("model_name", "gemini-2.5-pro")
        credentials_path = self.config.get("credentials_path")
        self.default_max_tokens = self.config.get("max_new_tokens")
        self.default_temperature = self.config.get("temperature")

        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        if not self.project or not self.location:
            raise ValueError("project and location are required parameters")

        # Created once in the main thread; all worker threads share this instance.
        self._client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            http_options=types.HttpOptions(
                base_url=self.config.get("base_url"),
                headers=self.config.get("headers")
            )
        )

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        gen_config_kwargs = {}
        if temperature is not None:
            gen_config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = max_tokens

        gen_config = (
            types.GenerateContentConfig(**gen_config_kwargs)
            if gen_config_kwargs
            else None
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=gen_config,
        )

        if response and response.text:
            return response.text
        else:
            raise Exception("API returned empty response")
