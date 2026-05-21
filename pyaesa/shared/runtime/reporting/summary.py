"""Structured summary sections for public runtime reports."""

from collections.abc import Iterable
from dataclasses import dataclass

from pyaesa.shared.runtime.text import extend_user_text_lines


@dataclass(frozen=True)
class SummaryWarning:
    """One user relevant warning owned by a report section."""

    message: str

    def lines(self) -> list[str]:
        """Return rendered warning lines."""
        return [f"WARNING: {self.message}"]


@dataclass(frozen=True)
class SummaryInfo:
    """One user relevant information record owned by a report section."""

    message: str

    def lines(self) -> list[str]:
        """Return rendered information lines."""
        return [f"INFO: {self.message}"]


@dataclass(frozen=True)
class SummarySection:
    """One structured public summary section."""

    title: str
    lines: tuple[str, ...] = ()
    infos: tuple[SummaryInfo, ...] = ()
    warnings: tuple[SummaryWarning, ...] = ()
    children: tuple["SummarySection", ...] = ()

    def is_empty(self) -> bool:
        """Return whether the section has no renderable payload."""
        return not self.lines and not self.infos and not self.warnings and not self.children


@dataclass(frozen=True)
class SummaryDocument:
    """Top level structured summary for one public report."""

    title: str
    lines: tuple[str, ...] = ()
    sections: tuple[SummarySection, ...] = ()


def warning(message: object) -> SummaryWarning:
    """Return one normalized warning record."""
    return SummaryWarning(message=str(message).strip())


def info(message: object) -> SummaryInfo:
    """Return one normalized information record."""
    return SummaryInfo(message=str(message).strip())


def section(
    title: str,
    *,
    lines: Iterable[str] = (),
    infos: Iterable[SummaryInfo] = (),
    warnings: Iterable[SummaryWarning] = (),
    children: Iterable[SummarySection] = (),
) -> SummarySection:
    """Return one normalized structured summary section."""
    clean_title = str(title).strip()
    return SummarySection(
        title=clean_title,
        lines=tuple(str(line).rstrip() for line in lines if str(line).strip()),
        infos=tuple(infos),
        warnings=tuple(warnings),
        children=tuple(child for child in children if not child.is_empty()),
    )


def document(
    title: str,
    *,
    lines: Iterable[str] = (),
    sections: Iterable[SummarySection] = (),
) -> SummaryDocument:
    """Return one normalized structured summary document."""
    clean_title = str(title).strip()
    return SummaryDocument(
        title=clean_title,
        lines=tuple(str(line).rstrip() for line in lines if str(line).strip()),
        sections=tuple(section for section in sections if not section.is_empty()),
    )


def render_summary(document: SummaryDocument) -> str:
    """Render one structured summary document to public text."""
    lines = [f"[{document.title}] Summary:"]
    for line in document.lines:
        extend_user_text_lines(lines, f"  {line}")
    if document.lines and document.sections:
        lines.append("")
    for section_index, summary_section in enumerate(document.sections):
        if section_index:
            lines.append("")
        _append_section(lines=lines, summary_section=summary_section, indent=2)
    return "\n".join(lines)


def _append_section(
    *,
    lines: list[str],
    summary_section: SummarySection,
    indent: int,
) -> None:
    prefix = " " * indent
    lines.append(f"{prefix}{summary_section.title}:")
    child_prefix = " " * (indent + 2)
    for line in summary_section.lines:
        extend_user_text_lines(lines, f"{child_prefix}{line}")
    for item in summary_section.infos:
        for info_line in item.lines():
            extend_user_text_lines(lines, f"{child_prefix}{info_line}")
    for item in summary_section.warnings:
        for warning_line in item.lines():
            extend_user_text_lines(lines, f"{child_prefix}{warning_line}")
    if summary_section.children and (
        summary_section.lines or summary_section.infos or summary_section.warnings
    ):
        lines.append("")
    for child_index, child in enumerate(summary_section.children):
        if child_index:
            lines.append("")
        _append_section(lines=lines, summary_section=child, indent=indent + 2)
