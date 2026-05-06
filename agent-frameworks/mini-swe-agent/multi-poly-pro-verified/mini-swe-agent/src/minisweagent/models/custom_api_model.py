import logging
import os
from typing import Any, Literal

import requests
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("custom_api_model")


class CustomAPIModelConfig(BaseModel):
    model_name: str = "gpt-4"
    api_key: str = os.getenv("CUSTOM_API_KEY", "")
    api_base: str = os.getenv("CUSTOM_API_BASE", "")
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: int = 300
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")


class CustomAPIModel:
    """Custom API model for OpenAI-compatible endpoints."""

    def __init__(self, **kwargs):
        self.config = CustomAPIModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        if not self.config.api_key:
            raise ValueError(
                "API key not provided. Set CUSTOM_API_KEY environment variable or pass api_key in config."
            )
        if not self.config.api_base:
            raise ValueError(
                "API base URL not provided. Set CUSTOM_API_BASE environment variable or pass api_base in config."
            )

        # Ensure API base ends with /chat/completions
        if not self.config.api_base.endswith("/chat/completions"):
            if self.config.api_base.endswith("/v1"):
                self.config.api_base = f"{self.config.api_base}/chat/completions"
            elif self.config.api_base.endswith("/"):
                self.config.api_base = f"{self.config.api_base}v1/chat/completions"
            else:
                self.config.api_base = f"{self.config.api_base}/v1/chat/completions"

        logger.info(f"Initialized CustomAPIModel with endpoint: {self.config.api_base}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, ValueError)),
    )
    def _query(self, messages: list[dict[str, str]]) -> dict:
        """Make API request to custom endpoint."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens

        try:
            response = requests.post(
                self.config.api_base,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the custom API model."""
        # Extract only role and content
        clean_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

        response_data = self._query(clean_messages)

        # Extract content from response
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract content from response: {response_data}")
            raise ValueError(f"Invalid response format: {e}") from e

        # Calculate cost
        cost = 0.0
        try:
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost = (
                input_tokens * self.config.cost_per_input_token
                + output_tokens * self.config.cost_per_output_token
            )
        except Exception as e:
            if self.config.cost_tracking != "ignore_errors":
                logger.warning(f"Error calculating cost: {e}")

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": content,
            "extra": {
                "response": response_data,
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for this model."""
        return {
            "model_name": self.config.model_name,
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }






