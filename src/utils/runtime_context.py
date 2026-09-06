"""Runtime context shared by every prompt-producing module.

Keeping the date in one place prevents relative expressions (for example
``this year``/``今年``) from being resolved differently by Planner, workers,
and the synthesizer.
"""
from __future__ import annotations

from datetime import datetime


def current_date() -> str:
    """Return the local date in ISO format, including the host timezone."""
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def runtime_context_text(date: str | None = None) -> str:
    date = date or current_date()
    return f"""Current date: {date}.

For relative date expressions such as today, yesterday, recently, this year,
this month, 今年, 本月, and 最近, always resolve them relative to the current
date above.

For time-sensitive research, prefer current-year and recent sources. Older
sources may only be used as historical context and must not be treated as
current facts."""


def resolve_relative_dates(text: str, date: str | None = None) -> str:
    """Supplement common relative expressions with an explicit year."""
    date = date or current_date()
    year = date[:4]
    resolved = text
    for source, target in {
        "今年": f"今年（{year}年）",
        "本年": f"本年（{year}年）",
        "this year": f"this year ({year})",
    }.items():
        resolved = resolved.replace(source, target)
    return resolved


def inject_runtime_context(messages: list[dict], date: str | None = None) -> list[dict]:
    """Add the shared runtime context to the first system prompt.

    The input is copied so callers' message histories are never mutated.
    """
    result = [dict(m) if isinstance(m, dict) else m for m in messages]
    context = runtime_context_text(date)
    for message in result:
        if isinstance(message, dict) and message.get("role") == "system":
            if context not in str(message.get("content", "")):
                message["content"] = f"{context}\n\n{message.get('content', '')}"
            return result
    result.insert(0, {"role": "system", "content": context})
    return result


__all__ = ["current_date", "runtime_context_text", "resolve_relative_dates", "inject_runtime_context"]
