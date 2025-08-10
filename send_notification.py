#!/usr/bin/env python3

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def send_email(subject: str, body: str, to_email: str = None):
    """Send email via mail command (for Linux VPS with sendmail/postfix)"""
    
    import subprocess
    
    # get email from env
    to_email = to_email or os.getenv("EMAIL_TO")
    from_email = os.getenv("EMAIL_FROM", "AI Digest <noreply@kyro.local>")
    
    if not to_email:
        raise ValueError("Missing email config. Set EMAIL_TO in .env")
    
    # send using mail command with from address
    try:
        process = subprocess.Popen(
            ['mail', '-s', subject, '-a', f'From: {from_email}', to_email],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=body)
        
        if process.returncode == 0:
            print(f"✅ Email sent to {to_email}")
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
    
    # find latest summary
    summaries_dir = Path("extracts/summaries")
    summary_files = list(summaries_dir.glob("summary_*.md"))
    
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