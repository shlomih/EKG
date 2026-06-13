"""
show_session.py
===============
Convert a Claude Code session .jsonl transcript to a readable Markdown file.

Usage:
    python show_session.py <session.jsonl>           # writes <session>.md next to it
    python show_session.py <session.jsonl> -o out.md
    python show_session.py --latest                  # latest session for this cwd
    python show_session.py --list                    # list sessions for this cwd

Renders user/assistant turns as ## blocks. Tool calls show name + a one-line
summary of inputs; tool results are truncated to 40 lines. Thinking blocks and
queue/meta entries are skipped by default (use --thinking to include them).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
TOOL_RESULT_MAX_LINES = 40
TOOL_INPUT_MAX_CHARS = 200


def project_dir_for(cwd: Path) -> Path:
    # Claude Code encodes cwd as: drive + path with separators -> dashes,
    # prefixed with "c--" on Windows. Match by listing and finding suffix.
    cwd_str = str(cwd).lower().replace("\\", "/").replace("/", "-").replace(":", "-")
    for d in CLAUDE_PROJECTS.iterdir():
        if d.name.lower().endswith(cwd_str.lstrip("-")):
            return d
    raise SystemExit(f"No session dir found for cwd {cwd} under {CLAUDE_PROJECTS}")


def list_sessions(cwd: Path) -> list[Path]:
    pdir = project_dir_for(cwd)
    files = sorted(pdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def fmt_ts(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"... [+{len(s) - n} chars]"


def truncate_lines(s: str, n: int) -> str:
    lines = s.splitlines()
    if len(lines) <= n:
        return s
    return "\n".join(lines[:n]) + f"\n... [+{len(lines) - n} more lines]"


def render_user_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    out = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append(block.get("text", "").strip())
        elif btype == "tool_result":
            tid = block.get("tool_use_id", "")[:8]
            res = block.get("content", "")
            if isinstance(res, list):
                res = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in res)
            res = truncate_lines(str(res), TOOL_RESULT_MAX_LINES)
            err = " (error)" if block.get("is_error") else ""
            out.append(f"<details><summary>tool_result {tid}{err}</summary>\n\n```\n{res}\n```\n\n</details>")
        elif btype == "image":
            out.append("[image]")
        else:
            out.append(f"[{btype}]")
    return "\n\n".join(x for x in out if x)


def render_assistant_content(content, include_thinking: bool) -> str:
    out = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append(block.get("text", "").strip())
        elif btype == "thinking":
            if include_thinking:
                t = block.get("thinking", "").strip()
                if t:
                    out.append(f"<details><summary>thinking</summary>\n\n{t}\n\n</details>")
        elif btype == "tool_use":
            name = block.get("name", "?")
            tid = block.get("id", "")[:8]
            inp = block.get("input", {})
            try:
                summary = json.dumps(inp, ensure_ascii=False)
            except Exception:
                summary = str(inp)
            summary = truncate(summary, TOOL_INPUT_MAX_CHARS)
            out.append(f"**→ {name}** `{tid}` `{summary}`")
        else:
            out.append(f"[{btype}]")
    return "\n\n".join(x for x in out if x)


def convert(jsonl_path: Path, out_path: Path, include_thinking: bool = False, include_meta: bool = False) -> None:
    lines_written = 0
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh, out_path.open("w", encoding="utf-8") as out:
        out.write(f"# Session: {jsonl_path.name}\n\n")
        out.write(f"Source: `{jsonl_path}`\n\n---\n\n")
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            if rtype not in ("user", "assistant"):
                continue  # skip queue-operation, summary, etc.
            if rec.get("isMeta") and not include_meta:
                continue
            ts = fmt_ts(rec.get("timestamp"))
            msg = rec.get("message", {})
            if rtype == "user":
                body = render_user_content(msg.get("content", ""))
                if not body.strip():
                    continue
                out.write(f"## User — {ts}\n\n{body}\n\n---\n\n")
            else:
                body = render_assistant_content(msg.get("content", []), include_thinking)
                if not body.strip():
                    continue
                model = msg.get("model", "")
                out.write(f"## Assistant ({model}) — {ts}\n\n{body}\n\n---\n\n")
            lines_written += 1
    print(f"Wrote {lines_written} turns to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Claude Code session .jsonl as Markdown.")
    ap.add_argument("session", nargs="?", help="Path to .jsonl session file")
    ap.add_argument("-o", "--output", help="Output .md path (default: <session>.md)")
    ap.add_argument("--latest", action="store_true", help="Use the most recent session for this cwd")
    ap.add_argument("--list", action="store_true", help="List sessions for this cwd and exit")
    ap.add_argument("--thinking", action="store_true", help="Include <thinking> blocks (default: hide)")
    ap.add_argument("--meta", action="store_true", help="Include isMeta system entries (default: hide)")
    args = ap.parse_args()

    cwd = Path(os.getcwd())

    if args.list:
        for f in list_sessions(cwd):
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = f.stat().st_size // 1024
            print(f"{mtime}  {size_kb:>6} KB  {f.name}")
        return

    if args.latest:
        files = list_sessions(cwd)
        if not files:
            sys.exit("No sessions found for this cwd.")
        session_path = files[0]
    elif args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            sys.exit(f"Not found: {session_path}")
    else:
        ap.print_help()
        sys.exit(1)

    out_path = Path(args.output) if args.output else session_path.with_suffix(".md")
    convert(session_path, out_path, include_thinking=args.thinking, include_meta=args.meta)


if __name__ == "__main__":
    main()