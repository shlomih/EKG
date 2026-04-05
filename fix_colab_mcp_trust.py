"""
fix_colab_mcp_trust.py
======================
Sets hasTrustDialogAccepted: true for the colab-proxy MCP server
in the EKG project config inside ~/.claude.json.

Run this ONCE, then fully restart VS Code (File → Exit, not Reload Window).
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

CLAUDE_JSON = Path.home() / ".claude.json"
PROJECT_KEY = "C:/Users/osnat/Documents/Shlomi/EKG"
MCP_SERVER  = "colab-proxy"

def main():
    if not CLAUDE_JSON.exists():
        print(f"ERROR: {CLAUDE_JSON} not found.")
        return

    # ── Backup ────────────────────────────────────────────────────────────
    backup_path = CLAUDE_JSON.with_suffix(
        f".json.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(CLAUDE_JSON, backup_path)
    print(f"Backup saved → {backup_path}")

    # ── Load ──────────────────────────────────────────────────────────────
    with open(CLAUDE_JSON, encoding="utf-8") as f:
        config = json.load(f)

    # ── Find project ──────────────────────────────────────────────────────
    projects = config.get("projects", {})
    if PROJECT_KEY not in projects:
        # Try to find it case-insensitively
        match = next((k for k in projects if k.lower() == PROJECT_KEY.lower()), None)
        if not match:
            print(f"ERROR: Project '{PROJECT_KEY}' not found in {CLAUDE_JSON}")
            print(f"  Available projects: {list(projects.keys())}")
            return
        print(f"  (matched project key as: {match})")
        project_key = match
    else:
        project_key = PROJECT_KEY

    project = projects[project_key]

    # ── Check MCP server is present ───────────────────────────────────────
    mcp_servers = project.get("mcpServers", {})
    if MCP_SERVER not in mcp_servers:
        print(f"WARNING: MCP server '{MCP_SERVER}' not found in project config.")
        print(f"  Found: {list(mcp_servers.keys())}")
        print(f"  Proceeding to set hasTrustDialogAccepted anyway...")

    # ── Read current state ────────────────────────────────────────────────
    current_trust = project.get("hasTrustDialogAccepted")
    print(f"\nCurrent hasTrustDialogAccepted = {current_trust}")

    if current_trust is True:
        print("Already set to true — no change needed.")
        print("\nIf colab-proxy still isn't working, fully restart VS Code (File → Exit).")
        return

    # ── Apply fix ─────────────────────────────────────────────────────────
    config["projects"][project_key]["hasTrustDialogAccepted"] = True

    with open(CLAUDE_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("✓ hasTrustDialogAccepted set to true")
    print(f"✓ Saved → {CLAUDE_JSON}")

    # ── Next steps ────────────────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  NEXT STEPS                                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Fully close VS Code: File → Exit  (NOT just Reload Window)   ║
║  2. Reopen VS Code                                               ║
║  3. Open colab.new in your browser (keep it open)               ║
║  4. In Claude Code, type /mcp → colab-proxy should show as      ║
║     connected                                                    ║
║  5. Call open_colab_browser_connection tool                      ║
╠══════════════════════════════════════════════════════════════════╣
║  VERIFY: Check wrapper_debug.log for a new timestamp            ║
║  C:\\Users\\osnat\\.claude\\wrapper_debug.log                       ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # ── Check wrapper log ─────────────────────────────────────────────────
    wrapper_log = Path.home() / ".claude" / "wrapper_debug.log"
    if wrapper_log.exists():
        content = wrapper_log.read_text(encoding="utf-8", errors="ignore")
        print(f"Current wrapper_debug.log:\n  {content.strip()}")
        print("\n  (After VS Code restart, a new timestamp should appear here)")


if __name__ == "__main__":
    main()
