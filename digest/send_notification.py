#!/usr/bin/env python3

import os
from pathlib import Path
from datetime import datetime
from typing import List, Union
from dotenv import load_dotenv

from config import csv_env

load_dotenv()

def recipients_for(full: bool) -> List[str]:
    """Who a run mails. Full runs (the nightly cron path) mail EMAIL_TO plus
    every address in EMAIL_TO_EXTRA (.env, comma-separated) -- both env vars,
    not code, since this is a public repo and an address in config.py would
    be committed for anyone to read. Anything else -- a --test run, or the
    manual "resend tonight's digest" path -- mails EMAIL_TO alone, so trying
    things out never spams the full list. A --dry-run never calls this at
    all: digest_run.py skips mailing before recipients matter.
    """
    primary = os.getenv("EMAIL_TO")
    if not primary:
        raise ValueError("Missing email config. Set EMAIL_TO in .env")
    if not full:
        return [primary]
    return [primary, *csv_env("EMAIL_TO_EXTRA")]

def send_email(subject: str, body: str, to_email: Union[str, List[str], None] = None):
    """Send email via mail command (for Linux VPS with sendmail/postfix)"""

    import subprocess

    # get recipient(s): a single address, a list, or fall back to EMAIL_TO
    to_email = to_email or os.getenv("EMAIL_TO")
    recipients = [to_email] if isinstance(to_email, str) else list(to_email or [])
    from_email = os.getenv("EMAIL_FROM", "AI Digest <noreply@kyro.local>")

    # debug print what was parsed
    print(f"📧 Debug - FROM: {repr(from_email)}")
    print(f"📧 Debug - TO: {recipients!r}")

    if not recipients:
        raise ValueError("Missing email config. Set EMAIL_TO in .env")

    # body is already HTML (produced by the summarizer's prompt)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3, h4 {{ color: #2c3e50; margin-top: 20px; }}
        h3 {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        ul, ol {{ padding-left: 30px; }}
        li {{ margin: 8px 0; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{body}
</body>
</html>"""
    
    # send using mail command with html content type
    print(f"📧 Debug - Sending HTML email to {', '.join(recipients)}")
    try:
        process = subprocess.Popen(
            ['mail', '-s', subject,
             '-a', f'From: {from_email}',
             '-a', 'Content-Type: text/html; charset=utf-8',
             *recipients],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=html_content)

        if process.returncode == 0:
            print(f"✅ Email sent to {', '.join(recipients)}")
            return True
        else:
            print(f"❌ Failed to send email: {stderr}")
            return False
    except FileNotFoundError:
        print("❌ 'mail' command not found. Install mailutils: apt install mailutils")
        return False
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def send_summary_email():
    """Send the latest summary via email"""
    
    # Latest AI digest specifically. The date-prefixed glob matters: the topic
    # digests write summary_<key>_<date>.html into the same directory, and a
    # bare summary_*.html would pick whichever topic ran most recently and mail
    # it under the AI digest's subject line. digest_run.py mails each topic the
    # file it just wrote, so this function is now only the manual "resend
    # tonight's AI digest" path.
    summaries_dir = Path("extracts/summaries")
    summary_files = list(summaries_dir.glob("summary_[0-9]*.html"))
    
    if not summary_files:
        print("No summaries found")
        return False
    
    latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
    
    # read summary
    with open(latest_summary, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # send it
    subject = f"AI Digest - {datetime.now().strftime('%Y-%m-%d')}"
    return send_email(subject, content)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # test mode - send a test email
        send_email(
            "Test from Digital Info Filter",
            "If you're reading this, email notifications are working! 🎉",
        )
    else:
        # send latest summary
        send_summary_email()