#!/usr/bin/env python3
"""Fetch the pages that posts link to, so the summarizer can read the source.

Without this the model reasons about a headline and a handle, and honestly
reports that everything else is unavailable. With it, a claim in a tweet can be
checked against the blog post or paper the tweet is pointing at.

ASSUMPTIONS, stated so they fail loudly:
  1. Only links that survived `x_scraper.is_blocked_link` get fetched. We never
     hit x.com or reddit.com -- not for output, not for enrichment.
  2. Fetched pages are UNTRUSTED TEXT. Anything they say is a claim by that
     page, never an instruction to the model. The prompt labels them as such;
     this module strips scripts and returns text only.
  3. A page we cannot read is still worth LINKING. We skip reading it, record
     why, and hand it to the summarizer as a link the human should open --
     arguably the highest-value link in the digest, since it is the one whose
     content the pipeline could not extract for him. Never a reason to fail.
  4. Pages are cached on disk by URL, so regenerating a digest costs no refetch
     and we do not hammer anyone's server across runs.
"""

import concurrent.futures as cf
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from x_scraper import is_blocked_link


def has_no_text_value(url: str) -> bool:
    host = url.split("//", 1)[-1].split("/", 1)[0].split("@")[-1].split(":")[0].lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in NO_TEXT_VALUE_DOMAINS)

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

# Content we have no text extractor for. Skipped with a reason rather than
# silently mangled into byte soup.
BINARY_HINTS = (".pdf", ".zip", ".mp4", ".mp3", ".png", ".jpg", ".jpeg", ".gif", ".webp")

# Pages a plain HTTP fetch cannot get anything useful out of: video players
# return their chrome, not a transcript, and social mirrors return a permalink
# shell. Fetching them burns budget and pollutes the prompt with menu text.
NO_TEXT_VALUE_DOMAINS = ("youtube.com", "youtu.be", "m.youtube.com",
                         "firefly.social", "docs.google.com", "drive.google.com")

_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|svg|head|nav|footer|form|iframe)\b.*?</\1>",
    re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n\s*\n+")


_MAIN = re.compile(r"<(article|main)\b[^>]*>(.*?)</\1>", re.S | re.I)


def extract_text(body: str) -> Tuple[str, str]:
    """(title, readable text) from raw HTML. Crude on purpose -- no new deps."""
    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
    title = html.unescape(_TAG.sub("", m.group(1))).strip() if m else ""
    # prefer <article>/<main> when present: otherwise the first 500 chars of the
    # extract are a site's nav menu, which is prompt budget spent on nothing
    mains = _MAIN.findall(body)
    if mains:
        longest = max((c for _, c in mains), key=len)
        if len(longest) > 400:
            body = longest
    text = _DROP_BLOCKS.sub(" ", body)
    text = re.sub(r"<(p|div|br|li|h[1-6]|tr)\b[^>]*>", "\n", text, flags=re.I)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n", text)
    return title, "\n".join(l.strip() for l in text.split("\n") if l.strip())


class LinkFetcher:
    def __init__(self, cache_dir: str = "extracts/link_cache", timeout: int = 20,
                 max_bytes: int = 2_000_000, workers: int = 6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.workers = workers
        self.stats = {"cached": 0, "fetched": 0, "failed": 0, "skipped": 0}

    # --- link selection -------------------------------------------------
    @staticmethod
    def collect_links(tweets: List[Dict], posts: List[Dict],
                      limit: int = 20) -> List[str]:
        """Unique fetchable links, ranked by how much attention the post got.

        Engagement is a cheap proxy for "worth reading the source of", and it
        keeps the budget spent on the links the digest is most likely to cite.
        """
        scored: Dict[str, int] = {}
        seen_lower: Dict[str, str] = {}   # collapse case-variant duplicates

        def note(url: str, weight: int):
            canon = seen_lower.setdefault(url.lower(), url)
            scored[canon] = max(scored.get(canon, 0), weight)

        for t in tweets or []:
            weight = (t.get("likes") or 0) + (t.get("retweets") or 0)
            for u in t.get("external_links") or []:
                if not is_blocked_link(u) and not has_no_text_value(u):
                    note(u, weight)
        for p in posts or []:
            u = p.get("url")
            if u and not is_blocked_link(u) and not has_no_text_value(u):
                note(u, p.get("score") or 0)
        return [u for u, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:limit]

    # --- fetching -------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def fetch(self, url: str, max_chars: int = 2500) -> Dict[str, Any]:
        cache = self._cache_path(url)
        if cache.exists():
            self.stats["cached"] += 1
            rec = json.loads(cache.read_text())
            rec["text"] = rec["text"][:max_chars]
            return rec

        rec = {"url": url, "title": "", "text": "", "status": "ok",
               "fetched_at": datetime.now(timezone.utc).isoformat()}
        low = url.lower().split("?")[0]
        if low.endswith(BINARY_HINTS):
            rec.update(status="skipped: no text extractor for this file type")
            self.stats["skipped"] += 1
            return rec
        try:
            r = requests.get(url, timeout=self.timeout,
                             headers={"User-Agent": UA}, stream=True)
            ctype = r.headers.get("content-type", "")
            if r.status_code != 200:
                rec.update(status=f"skipped: HTTP {r.status_code}")
            elif "html" not in ctype and "text" not in ctype:
                rec.update(status=f"skipped: content-type {ctype.split(';')[0]}")
            else:
                body = r.raw.read(self.max_bytes, decode_content=True)
                body = body.decode(r.encoding or "utf-8", errors="replace")
                title, text = extract_text(body)
                rec.update(title=title, text=text)
                if not text:
                    rec.update(status="skipped: no extractable text")
        except requests.RequestException as e:
            rec.update(status=f"failed: {type(e).__name__}: {e}"[:200])
        finally:
            try:
                r.close()
            except Exception:
                pass

        if rec["status"] == "ok":
            self.stats["fetched"] += 1
        elif rec["status"].startswith("failed"):
            self.stats["failed"] += 1
        else:
            self.stats["skipped"] += 1

        cache.write_text(json.dumps(rec))
        rec = dict(rec, text=rec["text"][:max_chars])
        return rec

    def enrich(self, tweets: List[Dict], posts: List[Dict], limit: int = 20,
               max_chars: int = 2500) -> List[Dict[str, Any]]:
        urls = self.collect_links(tweets, posts, limit=limit)
        if not urls:
            print("  [links] no fetchable external links in this corpus")
            return []
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            pages = list(ex.map(lambda u: self.fetch(u, max_chars=max_chars), urls))
        usable = [p for p in pages if p["status"] == "ok"]
        print(f"  [links] {len(usable)}/{len(urls)} pages readable "
              f"({self.stats['fetched']} fetched, {self.stats['cached']} cached, "
              f"{self.stats['skipped']} skipped, {self.stats['failed']} failed)")
        for p in pages:
            if p["status"] != "ok":
                print(f"  [links] {p['status']} -- {p['url'][:90]}")
        # every record, readable or not -- an unreadable page is still a link
        return pages


def link_label(url: str, title: str = "") -> str:
    """A human-readable label for a page we may never have read.

    Walled pages have no title, so the URL is all we have -- but a raw URL
    truncated mid-querystring is unreadable. Use domain + the last path slug.
    """
    if title:
        return title[:90]
    bare = url.split("//", 1)[-1].split("?")[0].split("#")[0]
    host, _, path = bare.partition("/")
    host = host[4:] if host.lower().startswith("www.") else host
    # last meaningful path segment, skipping generic ones ("/fulltext", "/index")
    generic = {"fulltext", "index", "abstract", "abs", "view", "home", "en", "html",
               "article", "full", "content", "default"}
    segs = [seg for seg in path.rstrip("/").split("/") if seg]
    slug = next((seg for seg in reversed(segs) if seg.lower() not in generic),
                segs[-1] if segs else "")
    slug = slug.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").strip()
    return f"{host} — {slug[:70]}" if slug else host


def readable(pages: List[Dict]) -> List[Dict]:
    return [p for p in pages or [] if p.get("status") == "ok"]


def unreadable(pages: List[Dict]) -> List[Dict]:
    return [p for p in pages or [] if p.get("status") != "ok"]


def format_pages(pages: List[Dict]) -> str:
    """Render fetched pages for the prompt, clearly fenced as untrusted data.

    Unreadable pages are listed too, with the reason. They are still links worth
    putting in the digest -- a human with a browser and a subscription gets past
    walls the pipeline cannot.
    """
    if not pages:
        return "(no linked pages were available for this digest)"
    out = ""
    for i, p in enumerate(readable(pages), 1):
        out += (f"\n--- PAGE {i} (read) ---\nurl: {p['url']}\ntitle: {p['title']}\n"
                f"extract:\n{p['text']}\n")
    blocked = unreadable(pages)
    if blocked:
        out += ("\n--- PAGES I COULD NOT READ (still link them) ---\n"
                "A paywall or bot-wall stopped the fetch. The reader CAN usually open "
                "these, so they are worth linking -- often more worth it than a page I "
                "summarised for him. Link them by name, say what the linking post claims "
                "about them, and make clear you are relaying that claim rather than "
                "confirming it from the page itself.\n")
        for p in blocked:
            out += f"url: {p['url']}  [{p['status']}]\n"
    return out


if __name__ == "__main__":
    import glob
    import sys

    def _latest(pattern):
        hits = sorted(glob.glob(pattern))
        return json.load(open(hits[-1])) if hits else []

    tweets = _latest("extracts/x_data_*.json")
    posts = _latest("extracts/reddit_data_*.json")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pages = LinkFetcher().enrich(tweets, posts, limit=limit)
    for p in pages:
        print(f"\n=== {p['title'][:80]}\n{p['url']}\n{p['text'][:300]}...")
