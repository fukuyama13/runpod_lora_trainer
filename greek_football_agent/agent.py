"""
Greek football news agent: scrape via tools, then generate an English digest with Claude.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic

import scraper

MODEL = "claude-sonnet-4-20250514"

_cached_articles: list[dict[str, Any]] | None = None

TOOLS: list[dict[str, Any]] = [
    {
        "name": "scrape_news",
        "description": (
            "Fetch the latest Greek football articles from gazzetta.gr/football and "
            "sport24.gr/football (up to 5 articles per site). Returns title, URL, and body text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "generate_digest",
        "description": (
            "Produce a polished English daily digest from articles. "
            "Call scrape_news first so articles are available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "articles": {
                    "type": "array",
                    "description": "Optional. Scraped articles; if omitted, uses the last scrape_news result.",
                    "items": {"type": "object"},
                }
            },
        },
    },
]

SYSTEM_PROMPT = """You are a Greek football news assistant. Your job is to run the digest workflow:
1) Call the scrape_news tool once to load fresh articles.
2) Call the generate_digest tool once to turn them into an English newsletter-style digest.

Do not skip tools. Do not fabricate article content."""


def _tool_scrape_news() -> str:
    global _cached_articles
    _cached_articles = scraper.scrape_all(gazzetta_limit=5, sport24_limit=5)
    return json.dumps(_cached_articles, ensure_ascii=False)


def _tool_generate_digest(articles: list[dict[str, Any]] | None) -> str:
    data = articles if articles else _cached_articles
    if not data:
        return "Error: no articles. Run scrape_news first."

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    user_content = f"""Using the following scraped articles (JSON), write a **Daily Greek Football Digest** in English.

Format requirements:
- Start with a clear **title** and **today's date** (use the date you infer from context or today's date in the user's timezone as plain text).
- Organize items into topical sections such as: Transfers, Match Results & Highlights, National Team (Greece), Greek Super League & Clubs, International Football, Injuries & Squad News, Other — use only sections that fit the content.
- Tone: professional sports newsletter, clean and scannable; short paragraphs or bullets where appropriate.
- Attribute themes to sources only implicitly (no need to cite URLs inline unless useful).
- Do not invent facts beyond the article text.

Articles JSON:
{json.dumps(data, ensure_ascii=False)}
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=(
            "You write sharp, accurate sports newsletters in English. "
            "You only use facts present in the provided articles."
        ),
        messages=[{"role": "user", "content": user_content}],
    )

    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip() or "(Empty digest.)"


def _execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    if name == "scrape_news":
        return _tool_scrape_news()
    if name == "generate_digest":
        articles = tool_input.get("articles")
        if isinstance(articles, list) and articles:
            return _tool_generate_digest(articles)
        return _tool_generate_digest(None)
    return json.dumps({"error": f"unknown tool: {name}"})


def run_agent() -> str:
    """Run the tool-use agent loop: scrape → digest → final text."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    client = Anthropic(api_key=api_key)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Run the full workflow: scrape the latest Greek football news, "
                "then generate today's English digest. Use your tools in order."
            ),
        }
    ]

    final_digest: str | None = None
    max_rounds = 8
    last_response = None

    for _ in range(max_rounds):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        last_response = response

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            if not final_digest:
                parts: list[str] = []
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)
                if parts:
                    final_digest = "\n".join(parts)
            break

        tool_result_blocks: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tid = block.id
            name = block.name
            tool_input = block.input if isinstance(block.input, dict) else {}
            result = _execute_tool(name, tool_input)
            if name == "generate_digest" and result and not result.startswith("Error:"):
                final_digest = result
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": result,
                }
            )

        if not tool_result_blocks:
            break
        messages.append({"role": "user", "content": tool_result_blocks})

    if final_digest:
        return final_digest.strip()

    if last_response:
        tail: list[str] = []
        for block in last_response.content:
            if getattr(block, "type", None) == "text":
                tail.append(block.text)
        if tail:
            return "\n".join(tail).strip()

    return "(No digest produced.)"
