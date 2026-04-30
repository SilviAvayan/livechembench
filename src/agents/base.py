"""Base agent: async OpenAI-compatible client for NVIDIA inference API."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from openai import AsyncOpenAI


class BaseAgent:
    """Thin async wrapper around any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Core completion helpers
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Return the full response content as a string (non-streaming)."""
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Yield response tokens as they arrive (streaming — preferred)."""
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Any:
        """Return parsed JSON from the model response.

        Handles responses wrapped in markdown code fences, e.g.:
            ```json
            { ... }
            ```
        """
        raw = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return _parse_json(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Any:
    """Extract and parse the first JSON object/array from *text*.

    Strips markdown code fences if present.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        text = fenced.group(1)

    # Try the whole text first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back: find the first { or [ and pair it
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")
