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
    
    # debug print what was parsed
    print(f"📧 Debug - FROM: {repr(from_email)}")
    print(f"📧 Debug - TO: {repr(to_email)}")
    
    if not to_email:
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
    print(f"📧 Debug - Sending HTML email to {to_email}")
    try:
        process = subprocess.Popen(
            ['mail', '-s', subject, 
             '-a', f'From: {from_email}',
             '-a', 'Content-Type: text/html; charset=utf-8',
             to_email],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=html_content)
        
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
    summary_files = list(summaries_dir.glob("summary_*.html"))
    
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