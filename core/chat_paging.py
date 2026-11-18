"""Bounded transcript display; complete messages remain in the SQLite archive."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptPage:
    turns: list
    start: int
    end: int
    total: int
    previewed: bool


def transcript_page(history, *, end=None, start=None, max_turns=100, max_chars=200_000, message_chars=24_000):
    if min(max_turns, max_chars, message_chars) < 1:
        raise ValueError("Transcript display limits must be positive")
    total = len(history)
    forward = start is not None
    if forward:
        start = max(0, min(total, int(start)))
        end = min(total, start + max_turns)
    else:
        end = total if end is None else max(0, min(total, int(end)))
        start = max(0, end - max_turns)
    rows = history[start:end]
    selected = []
    used = 0
    previewed = False
    for turn in rows if forward else reversed(rows):
        text = str(turn.get("content", ""))
        clipped = len(text) > message_chars
        if clipped:
            text = text[:message_chars] + "\n\n[Long message preview. Export the conversation to read the complete text.]"
        if selected and used + len(text) > max_chars:
            break
        # A very small configured page remains bounded even for one message.
        if len(text) > max_chars:
            text = text[:max_chars]
            clipped = True
        selected.append({**turn, "content": text})
        used += len(text)
        previewed |= clipped
    if forward:
        return TranscriptPage(selected, start, start + len(selected), total, previewed)
    return TranscriptPage(list(reversed(selected)), end - len(selected), end, total, previewed)


def export_transcript(history, destination):
    """Stream complete messages without materializing the archive."""
    with open(destination, "w", encoding="utf-8") as output:
        output.write("# Conversation\n\n")
        for turn in history:
            output.write("## " + str(turn.get("role", "message")).title() + "\n\n")
            output.write(str(turn.get("content", "")))
            output.write("\n\n")
