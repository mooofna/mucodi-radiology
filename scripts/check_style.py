"""Fail if prose in the repository uses characters the house style forbids.

The house style writes ranges as "to" and separates items with commas, so the
em dash (U+2014), the en dash (U+2013) and the middle dot (U+00B7) never appear
in tracked text files. Usage: ``python scripts/check_style.py [FILE ...]``; with
no arguments every file tracked by git is checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FORBIDDEN = {
    chr(0x2014): "em dash",
    chr(0x2013): "en dash",
    chr(0x00B7): "middle dot",
}
SKIP_SUFFIXES = {
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bin",
    ".parquet",
    ".tif",
}
SKIP_NAMES = {"uv.lock", "package-lock.json", "check_style.py"}


def tracked_files() -> list[Path]:
    """Return all files tracked by git."""
    out = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True).stdout
    return [Path(line) for line in out.splitlines() if line]


def check(path: Path) -> list[str]:
    """Return one message per forbidden character found in ``path``."""
    if path.suffix in SKIP_SUFFIXES or path.name in SKIP_NAMES or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    problems = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for char, name in FORBIDDEN.items():
            col = line.find(char)
            if col >= 0:
                problems.append(f"{path}:{lineno}:{col + 1}: {name} (U+{ord(char):04X})")
    return problems


def main(argv: list[str]) -> int:
    """Check the given files, or all tracked files, and report violations."""
    files = [Path(a) for a in argv] if argv else tracked_files()
    problems = [msg for f in files for msg in check(f)]
    for msg in problems:
        print(msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
