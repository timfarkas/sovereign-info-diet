#!/usr/bin/env python3

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

import re

from x_scraper import is_blocked_link

load_dotenv(os.getenv("DOTENV_PATH") or None)


BLOCKED_HREF = re.compile(r'<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.S | re.I)


def strip_blocked_links(html: str) -> str:
    """Replace any <a> pointing at a blocked platform with its plain text.

    Belt to the prompt's braces: the model is told not to emit x.com/reddit.com
    links, and this guarantees it. Tim's devices block those domains, so one
    leaking through is a dead link in his inbox.
    """
    def repl(m):
        return m.group(2) if is_blocked_link(m.group(1)) else m.group(0)
    return BLOCKED_HREF.sub(repl, html)


def format_tweets(tweets: List[Dict], limit: int = None) -> str:
    """Render tweets for the prompt, highest-engagement first."""
    if limit is None:
        try:
            from config import X_MAX_TWEETS_IN_PROMPT as limit
        except ImportError:
            limit = 400
    ranked = sorted(tweets, key=lambda t: (t.get("likes") or 0) + (t.get("retweets") or 0),
                    reverse=True)[:limit]
    ranked.sort(key=lambda t: t.get("id") or "", reverse=True)
    out = ""
    for i, t in enumerate(ranked, 1):
        eng = f"{t.get('likes') or 0}♥ {t.get('retweets') or 0}RT"
        out += f"\n{i}. @{t.get('handle')} ({t.get('author_name')}) [{eng}] {t.get('created_at')}\n"
        out += f"   {(t.get('text') or '').strip()}\n"
        if t.get("quoted_text"):
            out += f"   quoting: {t['quoted_text'].strip()[:200]}\n"
        if t.get("external_links"):
            out += f"   external links: {', '.join(t['external_links'][:3])}\n"
    return out


def health_banner(path: str = "extracts/x_health.json") -> str:
    """An HTML notice when the X feed is sick, empty string when it is fine."""
    f = Path(path)
    if not f.exists():
        return ('<p><strong>⚠ X ingestion did not run</strong> — no health file at '
                f'{path}. The digest below is Reddit-only.</p>')
    h = json.loads(f.read_text())
    if h.get("status") == "healthy":
        return ""
    bits = [f"status <strong>{h.get('status')}</strong>",
            f"{h.get('batches_failed')}/{h.get('batches_attempted')} batches failed",
            f"{h.get('tweets_new')} new tweets"]
    if h.get("tweet_cap_hit"):
        bits.append("per-run tweet cap hit")
    for n in (h.get("notes") or []):
        bits.append(n)
    err = (h.get("errors") or [])
    tail = f"<br><em>last error: {err[-1][:300]}</em>" if err else ""
    return ("<p><strong>⚠ X ingestion needs a look</strong> — " + "; ".join(bits)
            + f". Checked {h.get('checked_at')}.{tail}</p>")


def report_cost(model: str, usage) -> float:
    """Print what this call cost and flag it if it blew the ceiling.

    Tim set the budget per digest, so the digest should say what it cost rather
    than leaving him to reconstruct it from an invoice a month later.
    """
    from config import MODEL_PRICING, SUMMARY_COST_CEILING_USD
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(usage, "completion_tokens_details", None),
                        "reasoning_tokens", 0) or 0
    if model not in MODEL_PRICING:
        print(f"Cost: unknown -- {model} has no entry in config.MODEL_PRICING "
              f"({inp} in / {out} out tokens)")
        return float("nan")
    p_in, p_cached, p_out = MODEL_PRICING[model]
    cost = ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6
    print(f"Cost: ${cost:.4f} on {model} "
          f"({inp} input [{cached} cached] / {out} output [{reasoning} reasoning] tokens)")
    if cost > SUMMARY_COST_CEILING_USD:
        print(f"⚠ OVER the ${SUMMARY_COST_CEILING_USD:.2f}/digest ceiling by "
              f"${cost - SUMMARY_COST_CEILING_USD:.4f} -- consider a cheaper model "
              f"or a smaller X_MAX_TWEETS_IN_PROMPT")
    return cost


class LLMSummarizer:
    def __init__(self, model_name: str = None):
        """Initialize the LLM summarizer with OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        if model_name is None:
            from config import SUMMARY_MODEL
            model_name = SUMMARY_MODEL
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.last_cost = None
        
    def create_summary_prompt(self, posts: List[Dict], tweets: List[Dict] = None) -> str:
        """Create the summarization prompt for reddit posts + X tweets"""
        
        from config import TIME_HORIZON_DAYS, SUMMARY_PROMPT_TEMPLATE
        
        # format all posts into a digest
        posts_content = ""
        for i, post in enumerate(posts, 1):
            posts_content += f"\n{i}. {post['title']} [{post['score']} pts]\n"
            posts_content += f"   r/{post['subreddit']}\n"
            
            if post.get('selftext'):
                preview = post['selftext'][:200] + "..." if len(post['selftext']) > 200 else post['selftext']
                posts_content += f"   Text: {preview}\n"
            
            # include top 3 comments for context
            if post.get('comments'):
                posts_content += "   Top comments:\n"
                for comment in post['comments'][:3]:
                    comment_preview = comment['body'][:150] + "..." if len(comment['body']) > 150 else comment['body']
                    posts_content += f"   - [{comment['score']}pts] {comment_preview}\n"
        
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            TIME_HORIZON_DAYS=TIME_HORIZON_DAYS,
            posts_content=posts_content or "(no reddit posts in this window)",
            tweets_content=format_tweets(tweets or []) or "(no X posts in this window)",
        )
        
        return prompt
    
    def summarize_posts(self, posts: List[Dict], tweets: List[Dict] = None) -> str:
        """Generate a summary of all posts"""
        
        tweets = tweets or []
        print(f"\nAnalyzing {len(posts)} reddit posts + {len(tweets)} X posts "
              f"with {self.model}...")
        
        prompt = self.create_summary_prompt(posts, tweets)
        
        try:
            # GPT-5 has different parameter requirements
            if self.model.startswith(('gpt-5', 'gpt-6', 'o1', 'o3', 'o4')):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a sharp, insightful AI/tech news analyst with a good sense of humor."},
                        {"role": "user", "content": prompt}
                    ]
                    # reasoning models reject temperature / max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a sharp, insightful AI/tech news analyst with a good sense of humor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
            summary = strip_blocked_links(response.choices[0].message.content.strip())
            self.last_cost = report_cost(self.model, response.usage)

            print("Summary generated successfully")
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Failed to generate summary: {str(e)}"
    
    def save_summary(self, summary: str, posts_analyzed: int, filename: str = None,
                     tweets_analyzed: int = 0, banner: str = "", footer: str = ""):
        """Save the summary to a file"""
        
        from config import TIME_HORIZON_DAYS
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"summary_{timestamp}.html"

        output_dir = Path("extracts/summaries")
        output_dir.mkdir(exist_ok=True, parents=True)

        filepath = output_dir / filename

        # add header
        full_content = f"""<h1>AI Digest - {datetime.now().strftime("%Y-%m-%d")}</h1>
<p><em>Analyzed {tweets_analyzed} X posts and {posts_analyzed} Reddit posts from the past {TIME_HORIZON_DAYS} days</em></p>
{banner}<hr>
{summary}
<hr>
<p><em>Generated at {datetime.now().strftime("%I:%M %p")} by {self.model}{footer}</em></p>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"Summary saved to {filepath}")
        return str(filepath)


if __name__ == "__main__":
    import glob
    import time

    MAX_AGE_HOURS = 20  # a daily pipeline: anything older than this is yesterday's

    def _latest_fresh(pattern):
        """Newest matching file, but only if it is from THIS run's window.

        Without this guard a failed leg silently falls back to yesterday's dump
        and Tim gets stale news presented as today's. Staler than no news.
        """
        hits = sorted(glob.glob(pattern), key=lambda f: Path(f).stat().st_mtime)
        if not hits:
            return None, "no file"
        newest = hits[-1]
        age_h = (time.time() - Path(newest).stat().st_mtime) / 3600
        if age_h > MAX_AGE_HOURS:
            return None, f"{newest} is {age_h:.1f}h old (stale, ignored)"
        return newest, f"{newest} ({age_h:.1f}h old)"

    reddit_file, reddit_why = _latest_fresh("extracts/reddit_data_*.json")
    x_file, x_why = _latest_fresh("extracts/x_data_*.json")

    posts = json.load(open(reddit_file)) if reddit_file else []
    tweets = json.load(open(x_file)) if x_file else []
    print(f"Loaded {len(posts)} reddit posts from {reddit_why}")
    print(f"Loaded {len(tweets)} X posts from {x_why}")

    if not posts and not tweets:
        raise SystemExit("no input from either source -- refusing to mail an empty digest")

    summarizer = LLMSummarizer()
    summary = summarizer.summarize_posts(posts, tweets)

    banner = health_banner()
    if not x_file:
        banner += (f'<p><strong>⚠ No fresh X data</strong> — {x_why}. '
                   f'The digest below is Reddit-only.</p>')
    if not reddit_file:
        banner += (f'<p><strong>⚠ No fresh Reddit data</strong> — {reddit_why}.</p>')

    output_file = summarizer.save_summary(summary, len(posts),
                                          tweets_analyzed=len(tweets),
                                          banner=banner,
                                          footer=(f" — ${summarizer.last_cost:.3f}"
                                                  if summarizer.last_cost else ""))

    print("\n" + "="*60)
    print(summary)
    print("="*60)
