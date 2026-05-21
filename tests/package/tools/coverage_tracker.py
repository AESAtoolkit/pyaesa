"""Coverage tracker utility for phased ratchet workflow.

Usage:
    python tests/package/tools/coverage_tracker.py --xml coverage.xml --top 30
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class FileCoverage:
    filename: str
    covered_lines: int
    total_lines: int
    total_branches: int
    covered_branches: int
    partial_branches: int

    @property
    def line_pct(self) -> float:
        if self.total_lines == 0:
            return 100.0
        return (self.covered_lines / self.total_lines) * 100.0

    @property
    def branch_pct(self) -> float:
        if self.total_branches == 0:
            return 100.0
        return (self.covered_branches / self.total_branches) * 100.0

    @property
    def uncovered_lines(self) -> int:
        return max(0, self.total_lines - self.covered_lines)

    @property
    def uncovered_branches(self) -> int:
        return max(0, self.total_branches - self.covered_branches)


def _file_rows(xml_path: Path) -> list[FileCoverage]:
    root = ET.parse(xml_path).getroot()
    rows: list[FileCoverage] = []
    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        line_nodes = cls.findall("./lines/line")
        if not line_nodes:
            continue
        covered_lines = 0
        total_lines = 0
        total_branches = 0
        covered_branches = 0
        partial_branches = 0
        for node in line_nodes:
            total_lines += 1
            hits = int(node.attrib.get("hits", "0"))
            if hits > 0:
                covered_lines += 1
            cond_cov = node.attrib.get("condition-coverage")
            if not cond_cov:
                continue
            # format: "50% (1/2)"
            inside = cond_cov.split("(")[-1].rstrip(")")
            covered_s, total_s = inside.split("/")
            covered = int(covered_s)
            total = int(total_s)
            total_branches += total
            covered_branches += covered
            if 0 < covered < total:
                partial_branches += 1
        rows.append(
            FileCoverage(
                filename=filename,
                covered_lines=covered_lines,
                total_lines=total_lines,
                total_branches=total_branches,
                covered_branches=covered_branches,
                partial_branches=partial_branches,
            )
        )
    return rows


def _print_summary(rows: list[FileCoverage], top: int) -> None:
    total_lines = sum(r.total_lines for r in rows)
    covered_lines = sum(r.covered_lines for r in rows)
    total_branches = sum(r.total_branches for r in rows)
    covered_branches = sum(r.covered_branches for r in rows)
    line_pct = (covered_lines / total_lines * 100.0) if total_lines else 100.0
    branch_pct = covered_branches / total_branches * 100.0 if total_branches else 100.0

    print("Coverage tracker summary")
    print(f"- files: {len(rows)}")
    print(f"- line coverage: {line_pct:.2f}% ({covered_lines}/{total_lines})")
    print(f"- branch coverage: {branch_pct:.2f}% ({covered_branches}/{total_branches})")
    print()

    by_uncovered_branches = sorted(
        rows,
        key=lambda r: (-r.uncovered_branches, -r.uncovered_lines, r.filename),
    )
    print(f"Top {min(top, len(by_uncovered_branches))} files by uncovered branches")
    for row in by_uncovered_branches[:top]:
        print(
            f"- pyaesa/{row.filename}: "
            f"branches {row.covered_branches}/{row.total_branches} "
            f"(uncovered={row.uncovered_branches}, partial={row.partial_branches}), "
            f"lines {row.covered_lines}/{row.total_lines} "
            f"(uncovered={row.uncovered_lines})"
        )

    print()
    by_uncovered_lines = sorted(
        rows,
        key=lambda r: (-r.uncovered_lines, -r.uncovered_branches, r.filename),
    )
    print(f"Top {min(top, len(by_uncovered_lines))} files by uncovered lines")
    for row in by_uncovered_lines[:top]:
        print(
            f"- pyaesa/{row.filename}: "
            f"lines {row.covered_lines}/{row.total_lines} "
            f"(uncovered={row.uncovered_lines}), "
            f"branches {row.covered_branches}/{row.total_branches} "
            f"(uncovered={row.uncovered_branches})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track uncovered coverage hotspots.")
    parser.add_argument(
        "--xml",
        default="coverage.xml",
        help="Path to coverage XML report (default: coverage.xml).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many rows to show in each ranking (default: 30).",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise SystemExit(f"Coverage XML not found: {xml_path}")
    rows = _file_rows(xml_path)
    _print_summary(rows, max(1, int(args.top)))


if __name__ == "__main__":
    main()
