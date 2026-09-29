"""SID2Caption evaluator for the v3.1 Challenge task."""

import os
import re

from api.openai_client import OpenAIClient
from benchmark.tasks.v2_0.sid2caption.evaluator import Sid2CaptionEvaluator


_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def _strip_json_code_fence(response: str) -> str:
    """Normalize fenced JSON returned by the challenge judge."""
    if not response:
        return response

    cleaned_response = response.strip()
    fenced_json = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_json:
        return fenced_json.group(1).strip()
    return response


class _ChallengeOpenAIClient(OpenAIClient):
    """OpenAI-compatible client with challenge-only response normalization."""

    def generate(self, *args, **kwargs) -> str:
        response = super().generate(*args, **kwargs)
        return _strip_json_code_fence(response)


class ChallengeSid2CaptionEvaluator(Sid2CaptionEvaluator):
    """Use an explicitly configured judge without changing scoring logic."""

    def _get_llm_client(self):
        api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("WQ_API_KEY")
        base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("WQ_API_BASE_URL")
        model_override = os.environ.get("JUDGE_MODEL")
        if not api_key or not base_url:
            raise ValueError("Configure JUDGE_API_KEY and JUDGE_BASE_URL before caption evaluation")

        eval_config = self.task_config.get("evaluation_config", {})
        llm_judge_model = "openai/" + model_override if model_override else eval_config.get("llm_judge_model", "")
        if "/" not in llm_judge_model:
            raise ValueError(
                "llm_judge_model must use the 'provider/model' format"
            )
        _, model_name = llm_judge_model.split("/", 1)
        if not model_name or model_name == "configure-judge-model":
            raise ValueError("Configure JUDGE_MODEL before caption evaluation")

        base_url = base_url.rstrip("/")
        # Accept either an OpenAI SDK base URL or the full curl endpoint.
        if base_url.endswith(_CHAT_COMPLETIONS_SUFFIX):
            base_url = base_url[:-len(_CHAT_COMPLETIONS_SUFFIX)]

        return _ChallengeOpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            max_new_tokens=10000,
            temperature=0.0,
            max_retries=3,
            retry_delay=2,
        )
