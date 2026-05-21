"""Shared rendering helpers for user facing workspace text files."""

from collections.abc import Iterable
from textwrap import wrap

USER_TEXT_LINE_WIDTH = 100
ELLIPSIS = "..."
_NO_WRAP_SPACE = "\x00"
_PROTECTED_PHRASES = (
    "Level 1 weights",
    "Level 1 and/or L2 in L1 weights",
    "L2 in L1",
    "L2 vs global",
    "one step",
    "two step",
)


def wrap_user_text_lines(
    lines: Iterable[str],
    *,
    width: int = USER_TEXT_LINE_WIDTH,
) -> list[str]:
    """Wrap user facing text lines while preserving blank lines and bullets."""
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_user_text_line(str(line), width=width))
    return wrapped


def join_user_text_lines(
    lines: Iterable[str],
    *,
    width: int = USER_TEXT_LINE_WIDTH,
    trailing_newline: bool = False,
) -> str:
    """Return wrapped user facing text content."""
    text = "\n".join(wrap_user_text_lines(lines, width=width))
    return f"{text}\n" if trailing_newline else text


def extend_user_text_lines(
    target: list[str],
    line: str,
    *,
    width: int = USER_TEXT_LINE_WIDTH,
) -> None:
    """Append one wrapped user facing line to an existing line list."""
    target.extend(wrap_user_text_lines([line], width=width))


def print_user_text_line(line: str, *, width: int = USER_TEXT_LINE_WIDTH) -> None:
    """Print one wrapped user facing message."""
    for wrapped_line in wrap_user_text_lines([line], width=width):
        print(wrapped_line)


def compact_user_text(text: object, *, max_chars: int = USER_TEXT_LINE_WIDTH) -> str:
    """Return one single-line label bounded for transient progress displays."""
    value = " ".join(str(text).split())
    if len(value) <= max_chars:
        return value
    if max_chars <= len(ELLIPSIS):
        return value[:max_chars]
    return f"{value[: max_chars - len(ELLIPSIS)]}{ELLIPSIS}"


def _wrap_user_text_line(line: str, *, width: int) -> list[str]:
    if line == "":
        return [""]
    indent_length = len(line) - len(line.lstrip(" "))
    indent = line[:indent_length]
    content = line[indent_length:]
    bullet = content.startswith("- ")
    body = _protect_short_phrases(content[2:] if bullet else content)
    initial_indent = f"{indent}- " if bullet else indent
    subsequent_indent = f"{indent}  " if bullet else indent
    return [
        _restore_short_phrases(wrapped_line)
        for wrapped_line in wrap(
            body,
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
    ]


def _protect_short_phrases(text: str) -> str:
    """Keep short scientific route phrases together during wrapping."""
    protected = text
    for phrase in _PROTECTED_PHRASES:
        protected = protected.replace(phrase, phrase.replace(" ", _NO_WRAP_SPACE))
    return protected


def _restore_short_phrases(text: str) -> str:
    """Restore protected phrase separators after wrapping."""
    return text.replace(_NO_WRAP_SPACE, " ")
