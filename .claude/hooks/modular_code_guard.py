#!/usr/bin/env python3
"""Modular code guard hook for rllm.

Reads a Claude Code hook payload on stdin and emits advisory (never blocking)
warnings when code drifts toward spaghetti:

  * source files longer than MAX_FILE_LINES (test files exempt)
  * functions longer than MAX_FN_LINES (.rs brace-matched, .py indent-scoped)
  * circular imports among project Python files (commit-time only)

Triggered on:
  * Write/Edit  -> checks the single touched .rs/.py file
  * Bash(git commit ...) -> re-scans changed .rs/.py files + python import cycles

Output contract: JSON on stdout. When clean, suppresses output. When dirty,
returns a systemMessage (shown to the user) plus additionalContext (fed back to
the model). Exit code stays 0 so nothing is ever blocked.
"""

import json
import os
import re
import subprocess
import sys

MAX_FILE_LINES = 600
MAX_FN_LINES = 100

TEST_FILE_RE = re.compile(r"(^|/)(tests\.rs$|tests_.*\.rs$)|/tests/")
RS_FN_RE = re.compile(r"^\s*(pub(\s*\([^)]*\))?\s+)?(async\s+)?(unsafe\s+)?(extern\s+\"[^\"]*\"\s+)?fn\s+(\w+)")
PY_FN_RE = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)")
PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([.\w]+)\s+import\b|import\s+([.\w]+))")


def read_payload():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def is_test_file(path):
    return bool(TEST_FILE_RE.search(path))


def longest_rs_fn(lines):
    """Return (fn_name, length) of the longest brace-matched Rust fn, or None."""
    worst = None
    i = 0
    n = len(lines)
    while i < n:
        m = RS_FN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(6)
        # find the opening brace of the body (could be on a later line; skip ; decls)
        depth = 0
        started = False
        start = i
        j = i
        while j < n:
            for ch in lines[j]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
            if started and depth <= 0:
                break
            if not started and ";" in lines[j]:
                # forward declaration / trait method without body
                break
            j += 1
        if started:
            length = j - start + 1
            if worst is None or length > worst[1]:
                worst = (name, length)
            i = j + 1
        else:
            i += 1
    return worst


def longest_py_fn(lines):
    """Return (fn_name, length) of the longest indent-scoped Python def, or None."""
    worst = None
    n = len(lines)
    for i in range(n):
        m = PY_FN_RE.match(lines[i])
        if not m:
            continue
        indent = len(m.group(1))
        name = m.group(3)
        end = i
        for j in range(i + 1, n):
            stripped = lines[j].strip()
            if not stripped or stripped.startswith("#"):
                continue
            cur_indent = len(lines[j]) - len(lines[j].lstrip())
            if cur_indent <= indent:
                break
            end = j
        length = end - i + 1
        if worst is None or length > worst[1]:
            worst = (name, length)
    return worst


def check_file(path):
    """Return a list of warning strings for one file."""
    warnings = []
    if not os.path.isfile(path):
        return warnings
    ext = os.path.splitext(path)[1]
    if ext not in (".rs", ".py"):
        return warnings
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception:
        return warnings

    nlines = len(lines)
    rel = os.path.relpath(path)
    if nlines > MAX_FILE_LINES and not is_test_file(path):
        warnings.append(
            f"{rel}: {nlines} lines (> {MAX_FILE_LINES}). Consider splitting into focused modules."
        )

    worst = longest_rs_fn(lines) if ext == ".rs" else longest_py_fn(lines)
    if worst and worst[1] > MAX_FN_LINES:
        warnings.append(
            f"{rel}: fn `{worst[0]}` is {worst[1]} lines (> {MAX_FN_LINES}). Extract cohesive helpers."
        )
    return warnings


def changed_files():
    """Files changed vs HEAD plus staged, limited to .rs/.py under the repo."""
    paths = set()
    for args in (["diff", "--name-only", "HEAD"], ["diff", "--name-only", "--cached"]):
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            ).stdout
        except Exception:
            continue
        for line in out.splitlines():
            if line.endswith((".rs", ".py")):
                paths.add(line.strip())
    return sorted(paths)


def python_import_cycles():
    """Detect circular imports among project .py files. Returns list of cycle strings."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.py"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return []
    files = [p for p in out.splitlines() if p.strip()]
    # module name (file stem) -> set of imported module stems that are also project files
    stems = {os.path.splitext(os.path.basename(p))[0]: p for p in files}
    graph = {s: set() for s in stems}
    for stem, path in stems.items():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = PY_IMPORT_RE.match(line)
                    if not m:
                        continue
                    mod = (m.group(1) or m.group(2) or "").lstrip(".").split(".")[0]
                    if mod in stems and mod != stem:
                        graph[stem].add(mod)
        except Exception:
            continue

    cycles = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s: WHITE for s in graph}

    def dfs(node, stack):
        color[node] = GRAY
        stack.append(node)
        for nxt in graph[node]:
            if color[nxt] == GRAY:
                idx = stack.index(nxt)
                cycles.append(" -> ".join(stack[idx:] + [nxt]))
            elif color[nxt] == WHITE:
                dfs(nxt, stack)
        stack.pop()
        color[node] = BLACK

    for s in graph:
        if color[s] == WHITE:
            dfs(s, [])
    # dedupe
    seen, uniq = set(), []
    for c in cycles:
        key = frozenset(c.split(" -> "))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def emit(warnings):
    if not warnings:
        print(json.dumps({"suppressOutput": True}))
        return
    body = "modular-code-guard:\n  - " + "\n  - ".join(warnings)
    print(
        json.dumps(
            {
                "systemMessage": body,
                "suppressOutput": False,
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": body
                    + "\n(Advisory only — split the code or note why the exception is justified. See the modular-code-guard skill.)",
                },
            }
        )
    )


def main():
    payload = read_payload()
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    warnings = []

    if tool in ("Write", "Edit"):
        path = tool_input.get("file_path", "")
        if path:
            warnings.extend(check_file(path))
    elif tool == "Bash":
        cmd = tool_input.get("command", "")
        if "git commit" in cmd:
            for rel in changed_files():
                warnings.extend(check_file(rel))
            for cycle in python_import_cycles():
                warnings.append(f"circular import: {cycle}")

    emit(warnings)


if __name__ == "__main__":
    main()
