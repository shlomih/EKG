"""
setup_colab_mcp_desktop.py
==========================
Adds the colab-proxy MCP server to ~/.claude/mcp.json so it's available
in Claude Code Desktop (Cowork), not just the VS Code extension.

Run once, then restart Claude Desktop for the change to take effect.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

MCP_JSON = Path.home() / ".claude" / "mcp.json"

COLAB_SERVER = {
    "type": "stdio",
    "command": "python",
    "args": ["C:\\Users\\osnat\\.claude\\colab_mcp_wrapper.py"],
    "env": {}
}


def main():
    # ── Load or create mcp.json ───────────────────────────────────────────
    if MCP_JSON.exists():
        backup = MCP_JSON.with_suffix(
            f".json.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(MCP_JSON, backup)
        print(f"Backup saved → {backup}")

        with open(MCP_JSON, encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"Creating new {MCP_JSON}")
        config = {}

    # ── Add colab-proxy ───────────────────────────────────────────────────
    config.setdefault("mcpServers", {})

    if "colab-proxy" in config["mcpServers"]:
        print("colab-proxy already present — updating...")
    else:
        print("Adding colab-proxy...")

    config["mcpServers"]["colab-proxy"] = COLAB_SERVER

    # ── Save ──────────────────────────────────────────────────────────────
    MCP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MCP_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved → {MCP_JSON}")
    print(f"\nAll MCP servers now configured:")
    for name in config["mcpServers"]:
        print(f"  • {name}")

    # ── Verify wrapper exists ──────────────────────────────────────────────
    wrapper = Path("C:/Users/osnat/.claude/colab_mcp_wrapper.py")
    if wrapper.exists():
        print(f"\n✓ Wrapper script found at {wrapper}")
    else:
        print(f"\n⚠ WARNING: Wrapper not found at {wrapper}")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  NEXT STEPS                                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Fully quit Claude Desktop app                                ║
║  2. Open colab.new in your browser (keep it open)               ║
║  3. Reopen Claude Desktop / Cowork                              ║
║  4. In this chat, tell me to call open_colab_browser_connection  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
