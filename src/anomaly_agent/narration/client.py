"""LLM client interface, implementations, and the validated narration flow."""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .facts import to_llm_facts
from .fallback import template_summary
from .prompts import REQUIRED_LABELS, SYSTEM_PROMPT, build_user_prompt
from .validator import validate_numbers

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Provider interface: one method, no tools."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's text reply."""


class GroqClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        from groq import Groq  # imported lazily so tests need no SDK

        self._client = Groq(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=self._model,
        )
        return response.choices[0].message.content or ""


class FakeLLMClient(LLMClient):
    """Scripted client for offline tests. Items may be strings or exceptions to raise.

    The last item repeats once the script is exhausted.
    """

    def __init__(self, responses: Sequence[str | Exception]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        item = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def build_client(config: Config) -> LLMClient | None:
    """Return the configured client, or None when no API key is available."""
    from dotenv import load_dotenv

    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return GroqClient(model=config.llm.model, api_key=key)


class Narrator:
    """Ask the LLM, validate numbers and structure, retry, then fall back to a template."""

    def __init__(
        self,
        client: LLMClient | None,
        max_retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.attempts = 1 + max(0, max_retries)
        self._sleep = sleep
        self.fallbacks = 0
        self.narrated = 0

    @staticmethod
    def _clean(text: str) -> str:
        # Some reasoning models prepend <think>...</think>
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def narrate(self, facts: dict[str, Any]) -> str:
        label = f"{facts.get('metric')}/{facts.get('segment')}"
        if self.client is None:
            return self._fallback(facts, label, "no LLM client configured")

        bundle = to_llm_facts(facts)
        user = build_user_prompt(bundle)
        reason = "no attempts made"
        for attempt in range(self.attempts):
            try:
                text = self._clean(self.client.complete(SYSTEM_PROMPT, user))
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.warning("LLM error for %s (attempt %d/%d): %s", label, attempt + 1, self.attempts, reason)
                if attempt < self.attempts - 1:
                    self._sleep(2 ** (attempt + 1))
                continue

            if not text:
                reason = "empty response"
            elif missing := [lab for lab in REQUIRED_LABELS if lab not in text]:
                reason = f"missing sections: {missing}"
            else:
                result = validate_numbers(text, bundle)
                if result.ok:
                    self.narrated += 1
                    return text
                reason = f"numbers not in facts: {result.unmatched}"
            logger.warning("Narration rejected for %s (attempt %d/%d): %s", label, attempt + 1, self.attempts, reason)

        return self._fallback(facts, label, reason)

    def _fallback(self, facts: dict[str, Any], label: str, reason: str) -> str:
        self.fallbacks += 1
        logger.warning("FALLBACK template used for %s: %s", label, reason)
        return template_summary(facts)
