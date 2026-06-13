"""Idempotently repair matplotlib support for active manuscript generation.

This helper is intentionally small and local to the academic release workflow:
it ensures the dependency manifest declares matplotlib and that the active
manuscript regeneration script selects a headless-safe backend before any
plotting imports. It can be re-run safely by future reviewers.
"""
from __future__ import annotations

from pathlib import Path
import ast
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_LINES = [
    "import os\n",
    "\n",
    "os.environ.setdefault(\"MPLBACKEND\", \"Agg\")\n",
    "\n",
]
BACKEND_BLOCK = "".join(BACKEND_LINES)


def ensure_requirement() -> bool:
    req = PROJECT_ROOT / "requirements.txt"
    if req.exists():
        text = req.read_text(encoding="utf-8")
        lines = text.splitlines()
    else:
        text = ""
        lines = []
    if any(re.match(r"\s*matplotlib\b", line, re.IGNORECASE) for line in lines):
        return False
    if text and not text.endswith(("\n", "\r")):
        text += "\n"
    text += "matplotlib>=3.8\n"
    req.write_text(text, encoding="utf-8")
    return True


def ensure_pyproject_dependency() -> bool:
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return False
    text = pyproject.read_text(encoding="utf-8")
    if "matplotlib" in text.lower():
        return False
    match = re.search(r"(dependencies\s*=\s*\[)(.*?)(\n\s*\])", text, flags=re.DOTALL)
    if not match:
        return False
    body = match.group(2)
    indent = "    "
    nonempty = [line for line in body.splitlines() if line.strip()]
    if nonempty:
        indent_match = re.match(r"(\s*)", nonempty[-1])
        if indent_match:
            indent = indent_match.group(1)
    updated = text[: match.end(2)] + f'\n{indent}"matplotlib>=3.8",' + text[match.end(2) :]
    pyproject.write_text(updated, encoding="utf-8")
    return True


def _insertion_line_after_docstring_and_future_imports(text: str) -> int:
    """Return a 0-based line insertion index that preserves future imports."""
    lines = text.splitlines(keepends=True)
    start = 0
    if lines and lines[0].startswith("#!"):
        start = 1
    if len(lines) > start and re.match(r"#.*coding[:=]", lines[start]):
        start += 1

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return start

    insert_line = start
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        insert_line = max(insert_line, getattr(body[0], "end_lineno", body[0].lineno))
        body = body[1:]
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_line = max(insert_line, getattr(node, "end_lineno", node.lineno))
            continue
        break
    return insert_line


def ensure_headless_backend() -> bool:
    script = PROJECT_ROOT / "scripts" / "regenerate_active_manuscript.py"
    if not script.exists():
        raise FileNotFoundError(script)
    text = script.read_text(encoding="utf-8")
    # Remove this helper's previous exact insertion, if present, so the backend
    # stanza can be placed after module docstrings and __future__ imports.
    normalized = text.replace(BACKEND_BLOCK, "")
    if "MPLBACKEND" in normalized:
        if normalized != text:
            script.write_text(normalized, encoding="utf-8")
            return True
        return False
    lines = normalized.splitlines(keepends=True)
    insert_at = _insertion_line_after_docstring_and_future_imports(normalized)
    lines[insert_at:insert_at] = BACKEND_LINES
    updated = "".join(lines)
    script.write_text(updated, encoding="utf-8")
    return updated != text


def main() -> None:
    changed = []
    if ensure_requirement():
        changed.append("requirements.txt")
    if ensure_pyproject_dependency():
        changed.append("pyproject.toml")
    if ensure_headless_backend():
        changed.append("scripts/regenerate_active_manuscript.py")
    print("changed_files=" + (", ".join(changed) if changed else "none"))


if __name__ == "__main__":
    main()
