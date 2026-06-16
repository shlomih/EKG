"""
Nightly agent summary email sender.
Sends a plain-text summary to shlomi.hazan@gmail.com after each nightly run.

FIRST-TIME SETUP (one time only):
1. Go to https://myaccount.google.com/apppasswords
2. Create an app password for "Mail" on "Windows Computer"
3. Copy the 16-character password
4. Create a file at C:\\Users\\osnat\\Documents\\Shlomi\\EKG\\.email_credentials
   with exactly two lines:
     sender@gmail.com
     your-16-char-app-password
   (The sender can be shlomi.hazan@gmail.com itself — sending to yourself is fine)

Usage (called by the nightly orchestrator):
    python send_summary_email.py --attempts 4 --before 3 --after 5 --log "Task A: bandpass filter improved HR84..."

If credentials file is missing, the script writes the summary to
NIGHTLY_SUMMARY.txt instead and prints a warning.
"""

import argparse
import smtplib
import sys
import traceback
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

RECIPIENT = "shlomi.hazan@gmail.com"
CREDS_FILE = Path(__file__).parent / ".email_credentials"
FALLBACK_FILE = Path(__file__).parent / "NIGHTLY_SUMMARY.txt"


def load_credentials():
    if not CREDS_FILE.exists():
        return None, None
    lines = CREDS_FILE.read_text().strip().splitlines()
    if len(lines) < 2:
        return None, None
    return lines[0].strip(), lines[1].strip()


def send_email(sender: str, password: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = RECIPIENT
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(sender, password)
            server.sendmail(sender, [RECIPIENT], msg.as_string())
        return True
    except Exception:
        traceback.print_exc()
        return False


def write_fallback(subject: str, body: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = f"=== {timestamp} ===\n{subject}\n\n{body}\n\n"
    with open(FALLBACK_FILE, "a", encoding="utf-8") as f:
        f.write(content)
    print(f"[email] Credentials not found — summary written to {FALLBACK_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Send nightly EKG agent summary email")
    parser.add_argument("--attempts", type=int, default=0, help="Number of fix attempts made")
    parser.add_argument("--before", type=int, default=3, help="Pass count before this run")
    parser.add_argument("--after", type=int, default=3, help="Pass count after this run")
    parser.add_argument("--log", type=str, default="", help="Brief log of what was tried")
    parser.add_argument("--duration", type=str, default="", help="Run duration e.g. '47 min'")
    args = parser.parse_args()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_str = datetime.now().strftime("%Y-%m-%d")

    improvement = args.after - args.before
    if improvement > 0:
        status_line = f"✅ IMPROVED: {args.before}/8 → {args.after}/8 (+{improvement} tests passing)"
    elif improvement == 0:
        status_line = f"➡️  NO CHANGE: {args.after}/8 passing (was {args.before}/8)"
    else:
        status_line = f"⚠️  REGRESSED: {args.before}/8 → {args.after}/8 — check SCAN_DEBUG_ANALYSIS.md"

    duration_line = f"Duration: {args.duration}" if args.duration else "Duration: unknown"

    subject = f"[EKG Nightly] {date_str} — {args.after}/8 tests passing"

    body = f"""EKG Nightly Agent Summary
Run completed at: {now}
{duration_line}

Result
------
{status_line}
Attempts made tonight: {args.attempts}

What was tried
--------------
{args.log if args.log else "(no log provided)"}

Next steps
----------
Check SCAN_DEBUG_ANALYSIS.md for full details and updated task queue.
File: C:\\Users\\osnat\\Documents\\Shlomi\\EKG\\SCAN_DEBUG_ANALYSIS.md

You can update task priorities in the ## Queued Tasks section
before tomorrow night's run.
"""

    sender, password = load_credentials()

    if sender and password:
        success = send_email(sender, password, subject, body)
        if success:
            print(f"[email] Summary sent to {RECIPIENT}")
        else:
            print("[email] Send failed — writing to fallback file")
            write_fallback(subject, body)
    else:
        write_fallback(subject, body)
        print(
            "\n[SETUP NEEDED] To enable email notifications:\n"
            "1. Go to https://myaccount.google.com/apppasswords\n"
            "2. Create an app password\n"
            f"3. Create file: {CREDS_FILE}\n"
            "   Line 1: your-gmail@gmail.com\n"
            "   Line 2: your-16-char-app-password\n"
        )


if __name__ == "__main__":
    main()
