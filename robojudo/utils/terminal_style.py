"""ANSI terminal styling for user-facing console panels."""

from __future__ import annotations

import re

RESET = "\033[0m"
WHITE = "\033[97m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"

_ANSI_RE = re.compile(r"\033[^m]*m")


def white(text: str) -> str:
    return f"{WHITE}{text}{RESET}"


def key(text: str) -> str:
    """Highlight a key name or button."""
    return f"{CYAN}{BOLD}{text}{RESET}"


def mark(text: str) -> str:
    """Highlight selection marker."""
    return f"{YELLOW}{BOLD}{text}{RESET}"


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
