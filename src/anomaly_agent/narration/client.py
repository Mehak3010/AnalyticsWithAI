import os
from groq import Groq
from typing import Dict, Any, Optional
from ..config import Config
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def narrate(self, fact_bundle: Dict[str, Any]) -> str:
        prompt = self._build_prompt(fact_bundle)
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business analyst. Explain the metric anomaly in plain language. "
                                   "Use the facts provided. Do not invent data. Keep it under 120 words. "
                                   "Structure: Headline, What changed, Likely driver, Why it matters, Next check."
                    },
                    {
                        "role": "user",
                        "content": str(fact_bundle)
                    }
                ],
                model=self.config.llm.model,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Narration error: {e}")
            return self._fallback_narration(fact_bundle)

    def _build_prompt(self, facts: Dict[str, Any]) -> str:
        return f"Explain this anomaly: {facts}"

    def _fallback_narration(self, facts: Dict[str, Any]) -> str:
        metric = facts.get("metric")
        segment = facts.get("segment")
        obs = facts.get("observed")
        exp = facts.get("expected")
        dir = facts.get("direction")
        date = facts.get("date")
        
        return (f"[{metric.upper()}] {metric} in {segment} moved {dir} on {date}. "
                f"Observed: {obs:.2f}, Expected: {exp:.2f}. "
                f"This move of {abs(obs-exp):.2f} warrants investigation.")
