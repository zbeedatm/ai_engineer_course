"""
title: Weather Filter
version: 0.1
description: Intercept weather questions and fetch real data from local proxy.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import re
import urllib.request
import urllib.parse
import json


def fetch_weather(city: str) -> Optional[dict]:
    """
    Call the local Flask proxy from INSIDE the container.
    Your Flask must run with: app.run(host="0.0.0.0", port=5000)
    """
    try:
        base_url = "http://host.docker.internal:5000/weather"
        url = base_url + "?city=" + urllib.parse.quote(city)
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except Exception as e:
        # if anything fails, just return None and let the LLM answer normally
        print(f"[weather-filter] error calling weather API: {e}")
        return None


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        max_turns: int = Field(
            default=20, description="Maximum allowable conversation turns for a user."
        )

    class UserValves(BaseModel):
        max_turns: int = Field(
            default=20, description="Maximum allowable conversation turns for a user."
        )

    def __init__(self):
        self.valves = self.Valves()

    def _extract_city(self, text: str) -> Optional[str]:
        """
        Very simple extractor:
        - 'weather in Tel Aviv'
        - 'what is the weather in London'
        - 'get weather for Paris'
        """
        text_low = text.lower()

        # try some patterns
        m = re.search(r"weather\s+(in|for)\s+(.+)", text_low)
        if m:
            # return original-case guess (we'll just title-case it)
            city = m.group(2).strip()
            # remove trailing '?'
            city = city.rstrip(" ?.")
            # restore simple casing
            return city.title()

        return None

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Runs BEFORE LLM.
        If user asked for weather, we fetch it and inject as system message so LLM can just answer.
        """
        messages: List[Dict[str, Any]] = body.get("messages", [])
        if not messages:
            return body

        last_msg = messages[-1]
        content = last_msg.get("content") or last_msg.get("text") or ""

        city = self._extract_city(content)
        if not city:
            # no weather request → just return
            return body

        # call our local API
        weather = fetch_weather(city)
        if not weather:
            # if failed, just return original
            return body

        # build a helper message for the model
        # we inject BEFORE the user's message so the model "knows"
        helper_msg = {
            "role": "system",
            "content": (
                f"The user asked for the weather in '{city}'.\n"
                f"Real weather data (from local API): {json.dumps(weather)}\n"
                f"Please answer in a natural way. If fields are missing, say so."
            ),
        }

        # insert helper message right before the last user message
        body["messages"] = messages[:-1] + [helper_msg, last_msg]

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # you can post-process the model's answer here if you want
        return body
