import os
import re
import time
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq

from ..config import Config

load_dotenv()

SYSTEM_PROMPT = (
    "You are a business analyst writing one alert. Use ONLY the facts provided; never name "
    "holidays, events or causes that are not in the facts, and never mention field names like "
    "is_good, priority or severity. Use plain text with no markdown, no asterisks and no "
    "headers. Write exactly four short lines starting with: What changed:, Possible driver:, "
    "Why it matters:, Next check:. In 'Possible driver' say 'not determined from the data' "
    "unless drivers are listed, and hedge any driver with 'may'. Under 90 words in total."
)


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=key) if key else None

    def narrate(self, fact_bundle: Dict[str, Any]) -> str:
        """Ask the LLM for a summary; retry with backoff; fall back to a rule-based summary."""
        if self.client is None:
            print("GROQ_API_KEY not set; using rule-based summary")
            return self._fallback_narration(fact_bundle)
        attempts = 1 + max(0, self.config.llm.max_retries)
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": str(fact_bundle)},
                    ],
                    model=self.config.llm.model,
                )
                text = response.choices[0].message.content or ""
                # Some reasoning models prepend <think>...</think>
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                if text:
                    return text
                print("LLM returned empty text")
            except Exception as e:
                print(f"LLM narration error (attempt {attempt + 1}/{attempts}): {type(e).__name__}: {e}")
            if attempt < attempts - 1:
                time.sleep(2 ** (attempt + 1))  # backoff: 2s, 4s, ...
        return self._fallback_narration(fact_bundle)

    def _fallback_narration(self, facts: Dict[str, Any]) -> str:
        metric, segment = facts.get("metric"), facts.get("segment")
        obs, exp = facts.get("observed"), facts.get("expected")
        date = facts.get("date")
        date_txt = date.date().isoformat() if hasattr(date, "date") else str(date)
        text = (
            f"[{str(metric).upper()}] {metric} in {segment} moved {facts.get('direction')} "
            f"on {date_txt}. Observed: {obs:.2f}, expected: {exp:.2f} "
            f"({facts.get('rel_deviation', 0):.1%} deviation)."
        )
        drivers = facts.get("drivers")
        if drivers:
            text += " Possible drivers: " + "; ".join(drivers) + "."
        return text + " This move warrants investigation."
