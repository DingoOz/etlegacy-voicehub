"""Minimal in-place editing of config.toml values that keeps the file's comments and layout.

Only handles what config.toml uses: `[section]` tables with `key = value` lines where the
value is a number, bool or a basic string, optionally followed by a `# comment`."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_LINE = re.compile(r'^(\s*(?P<key>[A-Za-z0-9_\-]+)\s*=\s*)(?P<val>"(?:[^"\\]|\\.)*"|[^#\n]*?)(?P<rest>\s*(#.*)?)$')
_HEADER = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*(#.*)?$")


def format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        return s if s not in ("inf", "nan") else "0.0"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(format_value(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def set_values(text: str, section: str, values: dict[str, Any]) -> str:
    """Return `text` with each key in `values` set inside `[section]`. Existing lines keep
    their trailing comment; missing keys are appended at the end of the section; a missing
    section is appended to the file."""
    lines = text.split("\n")
    pending = dict(values)
    start = end = None
    for i, line in enumerate(lines):
        m = _HEADER.match(line)
        if m:
            if start is not None:
                end = i
                break
            if m.group("name").strip() == section:
                start = i
    if start is None:
        lines += ["", f"[{section}]"] + [f"{k} = {format_value(v)}" for k, v in pending.items()]
        return "\n".join(lines)
    if end is None:
        end = len(lines)
    for i in range(start + 1, end):
        m = _LINE.match(lines[i])
        if not m or m.group("key") not in pending:
            continue
        k = m.group("key")
        lines[i] = f"{m.group(1)}{format_value(pending.pop(k))}{m.group('rest')}"
    if pending:
        insert = end
        while insert > start + 1 and lines[insert - 1].strip() == "":
            insert -= 1
        lines[insert:insert] = [f"{k} = {format_value(v)}" for k, v in pending.items()]
    return "\n".join(lines)


def update_file(path: Path, changes: dict[str, dict[str, Any]]) -> None:
    """changes = {section: {key: value}}. Written atomically."""
    text = path.read_text(encoding="utf-8")
    for section, values in changes.items():
        if values:
            text = set_values(text, section, values)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
