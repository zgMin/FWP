from __future__ import annotations

import json
import re
from typing import Any


def _list_items(text: str) -> list[str]:
    items: list[str] = []
    for line in text.strip().splitlines():
        match = re.match(r"^\s*(?:[-*+] |\d+[.)、]\s*)(.+?)\s*$", line)
        if match:
            items.append(match.group(1).strip())
    return items


def validate_output(text: str, spec: dict[str, Any]) -> tuple[bool, str]:
    kind = spec.get("type")
    stripped = text.strip()

    if kind == "full_prompt_similarity":
        return True, "deferred: compare against the full-prompt output during evaluation"
    if kind == "answer_match":
        answers = [str(value).strip().lower() for value in spec.get("answers", [])]
        normalized = stripped.lower()
        ok = any(answer and answer in normalized for answer in answers)
        return ok, "matched a gold answer" if ok else "no gold answer found"
    if kind == "fixed_prefix":
        value = str(spec["value"])
        return stripped.startswith(value), f"response must begin with {value!r}"
    if kind == "fixed_suffix":
        value = str(spec["value"])
        return stripped.endswith(value), f"response must end with {value!r}"
    if kind == "list_item_suffix":
        value = str(spec["value"])
        items = _list_items(stripped)
        ok = bool(items) and all(item.endswith(value) for item in items) and stripped.count(value) == len(items)
        return ok, f"every parsed list item must end with {value!r} exactly once"
    if kind == "exact_item_count":
        expected = int(spec["count"])
        count = len(_list_items(stripped))
        return count == expected, f"expected {expected} list items, found {count}"
    if kind == "json_schema":
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"
        if not isinstance(value, dict):
            return False, "JSON root must be an object"
        keys = set(value)
        required = set(spec.get("required_keys", []))
        if not required.issubset(keys):
            return False, f"missing keys: {sorted(required - keys)}"
        if not spec.get("allow_extra_keys", True) and keys != required:
            return False, f"unexpected keys: {sorted(keys - required)}"
        if any(not isinstance(value[key], str) for key in required):
            return False, "required values must be strings"
        return True, "valid JSON schema"
    if kind == "xml_tag":
        tag = re.escape(str(spec["tag"]))
        pattern = rf"^\s*<{tag}>[\s\S]*</{tag}>\s*$"
        return bool(re.fullmatch(pattern, text)), f"response must contain exactly one outer <{spec['tag']}> tag"
    if kind == "numbered_list":
        items = _list_items(stripped)
        raw_numbers = [int(value) for value in re.findall(r"(?m)^\s*(\d+)[.)、]\s+", stripped)]
        expected = list(range(int(spec.get("start", 1)), int(spec.get("start", 1)) + len(raw_numbers)))
        return bool(items) and raw_numbers == expected, "numbered list must be consecutive"
    if kind == "markdown_table":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) < 3 or not all(line.startswith("|") and line.endswith("|") for line in lines):
            return False, "response is not a complete Markdown table"
        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
        return headers == spec.get("columns", []), f"expected columns {spec.get('columns', [])}"
    if kind == "key_value_lines":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        keys = [str(value) for value in spec.get("keys", [])]
        if len(lines) != len(keys):
            return False, f"expected {len(keys)} non-empty lines"
        separators = [":" if re.search(rf"^{re.escape(key)}:", line, re.I) else "：" for key, line in zip(keys, lines)]
        ok = all(re.search(rf"^{re.escape(key)}{re.escape(separator)}", line, re.I) for key, line, separator in zip(keys, lines, separators))
        return ok, "key-value line structure"
    if kind == "single_line_delimited":
        delimiter = str(spec["delimiter"])
        ok = "\n" not in stripped and delimiter in stripped
        return ok, f"response must be one line containing {delimiter!r}"
    return False, f"unknown validator type: {kind!r}"
