"""
API-based Generator for closed-source models (Claude, Gemini, DeepSeek, OpenAI).
Uses lightweight client wrappers to avoid importing heavy dependencies.
"""
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urllib_request

from benchmark.base_generator import Generator
from benchmark.console import *

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_api_client(model_type: str, config_path=None):
    """
    Create an API client based on model_type.
    - gemini*: uses google-genai SDK
    - others (claude, deepseek, openai): uses OpenAI-compatible SDK
    """
    import json

    from api import load_config
    config = load_config(config_path)

    if model_type not in config:
        raise ValueError(f"Model '{model_type}' not found in config. Available: {list(config.keys())}")

    model_config = config[model_type]

    if model_type.startswith("gemini"):
        return _GenaiGeminiClient(**model_config)
    else:
        return _OpenAICompatibleClient(**model_config)


class _OpenAICompatibleClient:
    """Lightweight wrapper around OpenAI SDK."""

    def __init__(self, **config):
        from openai import OpenAI

        self.model_name = config.get("model_name", "default")
        self.default_max_tokens = config.get("max_new_tokens", None)
        self.default_temperature = config.get("temperature", 0.01)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.api_key = config.get("api_key", "dummy")
        self.base_url = config.get("base_url")
        self.default_headers = config.get("default_headers", {})
        self.use_responses_api = config.get("use_responses_api", False)
        self.reasoning_effort = config.get("reasoning_effort", None)
        self.reasoning_variant = config.get("reasoning_variant", None)

        self.client = None
        if not self.use_responses_api:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.default_headers,
            )

    @staticmethod
    def _extract_responses_text(resp_json: Dict[str, Any]) -> Optional[str]:
        output_text = resp_json.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        for item in resp_json.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    text = content.get("text")
                else:
                    text = getattr(content, "text", None)
                if text:
                    return text
        return None

    def _call_responses_api(
        self,
        prompt: str,
        system: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": prompt,
        }
        if system:
            payload["instructions"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.default_headers,
        }
        req = urllib_request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")

        resp_json = json.loads(body)
        content = self._extract_responses_text(resp_json)
        if content:
            return content
        raise Exception("API returned empty response")

    @staticmethod
    def _is_bad_request_error(exc: Exception) -> bool:
        error_msg = str(exc).lower()
        return "400" in error_msg or "bad request" in error_msg

    def _create_chat_completion(self, request_params: Dict[str, Any]):
        if self.reasoning_effort is None:
            return self.client.chat.completions.create(**request_params)

        # Build all 4 variants
        all_variants = {}

        # v1: reasoning_effort as direct param, with temperature
        v1 = dict(request_params)
        v1["reasoning_effort"] = self.reasoning_effort
        all_variants[1] = v1

        # v2: reasoning_effort as direct param, without temperature
        v2 = dict(v1)
        v2.pop("temperature", None)
        all_variants[2] = v2

        # v3: reasoning via extra_body, with temperature
        v3 = dict(request_params)
        v3["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
        all_variants[3] = v3

        # v4: reasoning via extra_body, without temperature
        v4 = dict(v3)
        v4.pop("temperature", None)
        all_variants[4] = v4

        # If a specific variant is requested, only try that one
        if self.reasoning_variant is not None:
            return self.client.chat.completions.create(**all_variants[self.reasoning_variant])

        # Otherwise try all variants, skip 400 errors
        last_error = None
        for variant in all_variants.values():
            try:
                return self.client.chat.completions.create(**variant)
            except Exception as exc:
                last_error = exc
                if not self._is_bad_request_error(exc):
                    raise

        raise last_error

    def generate(self, prompt: str, **kwargs) -> str:
        import time, random

        system = kwargs.pop("system", None)
        temperature = kwargs.pop("temperature", self.default_temperature)
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay + random.uniform(0, delay * 0.3))

                request_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                }
                if temperature is not None:
                    request_params["temperature"] = temperature
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens

                if self.use_responses_api:
                    return self._call_responses_api(
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                response = self._create_chat_completion(request_params)
                if response and response.choices:
                    content = response.choices[0].message.content
                    if content:
                        return content
                raise Exception("API returned empty response")
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                retryable = any(k in error_msg for k in ['503', '429', '500', 'timeout', 'rate limit', 'overload'])
                if attempt == self.max_retries - 1 or not retryable:
                    raise Exception(f"API call failed: {last_error}")

        raise Exception(f"Max retries reached: {last_error}")


class _GenaiGeminiClient:
    """Lightweight wrapper around google-genai SDK."""

    def __init__(self, **config):
        from google import genai
        from google.genai import types

        self.model_name = config.get("model_name", "gemini-3-flash-preview")
        self.default_max_tokens = config.get("max_new_tokens", 10000)
        self.default_temperature = config.get("temperature", 0.01)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self._types = types

        self.client = genai.Client(
            api_key=config.get("api_key", "dummy"),
            http_options={
                "base_url": config.get("base_url"),
                "headers": config.get("default_headers", {}),
            }
        )

    def generate(self, prompt: str, **kwargs) -> str:
        import time, random

        temperature = kwargs.pop("temperature", self.default_temperature)
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)

        gen_config_kwargs = {}
        if temperature is not None:
            gen_config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = max_tokens

        gen_config = self._types.GenerateContentConfig(**gen_config_kwargs) if gen_config_kwargs else None

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay + random.uniform(0, delay * 0.3))

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=gen_config,
                )
                if response and response.text:
                    return response.text
                raise Exception("API returned empty response")
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                retryable = any(k in error_msg for k in ['503', '429', '500', 'timeout', 'rate limit', 'overload'])
                if attempt == self.max_retries - 1 or not retryable:
                    raise Exception(f"Gemini API call failed: {last_error}")

        raise Exception(f"Max retries reached: {last_error}")


class APIGenerator(Generator):
    """Generator that calls closed-source LLM APIs."""

    def __init__(self, model_type: str, config_path: Optional[str] = None,
                 max_workers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.max_workers = max_workers
        self.client = _get_api_client(model_type, config_path)
        self.model_name = self.client.model_name
        console.print(f"[green]APIGenerator initialized: {self.model_type} ({self.model_name})[/green]")

    def __str__(self) -> str:
        return self.model_name.replace("/", "_")

    def _call_api_for_messages(self, prompt) -> str:
        """Call API with proper message handling."""
        if isinstance(prompt, list):
            system_msg = None
            user_parts = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_msg = content
                else:
                    user_parts.append(content)

            user_text = "\n\n".join(user_parts)

            if system_msg:
                if self.model_type.startswith("gemini"):
                    user_text = f"{system_msg}\n\n{user_text}"
                else:
                    return self.client.generate(user_text, system=system_msg)

            return self.client.generate(user_text)
        else:
            return self.client.generate(prompt)

    def generate(self, prompts: Dict[str, Any], **kwargs
                 ) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]:
        """Batch generate text via API calls with retry."""
        generations: Dict[str, List[str]] = {}
        logprobs: Dict[str, List[float]] = {}

        sample_ids = list(prompts.keys())
        total = len(sample_ids)

        console.print(f"[cyan]Generating {total} samples via {self.model_type} API (max_workers={self.max_workers})...[/cyan]")

        def process_one(sample_id: str) -> Tuple[str, str, Optional[str]]:
            prompt = prompts[sample_id]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    result = self._call_api_for_messages(prompt)
                    return sample_id, result, None
                except Exception as e:
                    error_msg = str(e).lower()
                    is_retryable = any(k in error_msg for k in ['429', 'rate limit', 'tpm', 'timeout', '503'])
                    if is_retryable and attempt < max_retries - 1:
                        import time as _time
                        wait = (2 ** attempt) * 5
                        _time.sleep(wait)
                    else:
                        return sample_id, "", str(e)
            return sample_id, "", "max retries exceeded"

        completed_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_one, sid): sid for sid in sample_ids}

            for future in as_completed(futures):
                sample_id, result, error = future.result()
                if error:
                    console.print(f"[red]Error for sample {sample_id}: {error}[/red]")
                    generations[sample_id] = [f"ERROR: {error}"]
                    error_count += 1
                else:
                    generations[sample_id] = [result]

                logprobs[sample_id] = []
                completed_count += 1

                if completed_count % 10 == 0 or completed_count == total:
                    console.print(f"  Progress: {completed_count}/{total} (errors: {error_count})")

        console.print(f"[green]Generation complete: {total} samples, {error_count} errors[/green]")
        return generations, logprobs
