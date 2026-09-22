#!/usr/bin/env python3
"""Contract tests for the X ingestion layer.

These test the behaviour Tim asked for -- survive transient downtime, cache
intermediate state, signal when the feed is sick, never surface a blocked link
-- against a fake twitterapi.io. No network, no credits burned.
"""

import json

import pytest

from llm_summarizer import format_tweets, health_banner, strip_blocked_links
from x_scraper import TwitterIOClient, XFeedDown, XScraper, is_blocked_link


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if payload is None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Replays a scripted list of responses and records the calls made."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        item = self.script.pop(0) if self.script else FakeResponse(200, {"tweets": []})
        if isinstance(item, Exception):
            raise item
        # a bare dict in the script is shorthand for "HTTP 200 with this body"
        return FakeResponse(200, item) if isinstance(item, dict) else item


def make_client(script, **kw):
    c = TwitterIOClient(api_key="test-key", min_interval=0.0, **kw)
    c.session = FakeSession(script)
    return c


def tweet(tid, handle="karpathy", text="hello", urls=()):
    return {
        "id": tid, "text": text, "createdAt": "Tue Sep 22 10:00:00 +0000 2026",
        "author": {"userName": handle, "name": handle, "followers": 1},
        "likeCount": 1, "retweetCount": 0,
        "entities": {"urls": [{"expanded_url": u} for u in urls]},
    }


# --- client resilience ------------------------------------------------------

def test_retries_transient_failure_then_succeeds():
    client = make_client([
        FakeResponse(503, text="upstream boom"),
        FakeResponse(429, text="slow down"),
        FakeResponse(200, {"tweets": [tweet("1")]}),
    ], max_retries=4)
    assert client.get("/twitter/tweet/advanced_search")["tweets"][0]["id"] == "1"
    assert len(client.session.calls) == 3
    assert len(client.errors) == 2          # survived failures are still recorded


def test_network_exception_is_retried():
    import requests
    client = make_client([
        requests.ConnectionError("dns died"),
        FakeResponse(200, {"tweets": []}),
    ], max_retries=3)
    assert client.get("/x") == {"tweets": []}


def test_gives_up_after_max_retries():
    client = make_client([FakeResponse(503, text="down")] * 3, max_retries=3)
    with pytest.raises(XFeedDown, match="failed after 3 attempts"):
        client.get("/x")
    assert len(client.session.calls) == 3


def test_auth_failure_fails_fast_without_retrying():
    """A bad key will never come good -- burning 4 retries on it is noise."""
    client = make_client([FakeResponse(401, text="bad key")] * 4, max_retries=4)
    with pytest.raises(XFeedDown, match="401"):
        client.get("/x")
    assert len(client.session.calls) == 1


def test_missing_api_key_raises_at_construction(monkeypatch):
    monkeypatch.delenv("TWITTER_IO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TWITTER_IO_API_KEY"):
        TwitterIOClient(api_key=None)


# --- run behaviour ----------------------------------------------------------

def scraper_with(tmp_path, script):
    return XScraper(base_dir=str(tmp_path), client=make_client(script))


def accounts_payload(names):
    return {"followings": [{"userName": n} for n in names], "has_next_page": False}


def test_partial_outage_keeps_what_it_got_and_reports_degraded(tmp_path):
    # 2 accounts/batch over 4 accounts = 2 batches; the second one dies for good
    s = scraper_with(tmp_path, [
        accounts_payload(["a", "b", "c", "d"]),
        {"tweets": [tweet("10"), tweet("11")], "has_next_page": False},
        FakeResponse(503, text="boom"), FakeResponse(503, text="boom"),
        FakeResponse(503, text="boom"), FakeResponse(503, text="boom"),
    ])
    got = s.scrape(seeds=["seed"], max_accounts_per_batch=2, max_query_chars=10_000)
    assert {t["id"] for t in got} == {"10", "11"}
    health = json.loads((tmp_path / "x_health.json").read_text())
    assert health["status"] == "degraded"
    assert health["batches_failed"] == 1 and health["batches_attempted"] == 2


def test_total_outage_raises_and_health_says_down(tmp_path):
    s = scraper_with(tmp_path, [accounts_payload(["a", "b"])]
                     + [FakeResponse(503, text="boom")] * 8)
    with pytest.raises(XFeedDown, match="looks down"):
        s.scrape(seeds=["seed"], max_accounts_per_batch=1, max_query_chars=10_000)
    assert json.loads((tmp_path / "x_health.json").read_text())["status"] == "down"


def test_cached_batches_are_reused_without_new_api_calls(tmp_path):
    """The point of intermediate caching: a re-run after an outage must not
    re-pay for the batches that already succeeded."""
    run_dir = tmp_path / "x_state" / "2026-09-22"
    run_dir.mkdir(parents=True)
    (run_dir / "batch_000.json").write_text(json.dumps(
        [{"id": "99", "handle": "a", "text": "cached", "external_links": []}]))
    s = scraper_with(tmp_path, [accounts_payload(["a", "b"]),
                               {"tweets": [tweet("100")], "has_next_page": False}])
    got = s.scrape(seeds=["seed"], max_accounts_per_batch=1, max_query_chars=10_000, run_id="2026-09-22")
    assert {t["id"] for t in got} == {"99", "100"}
    search_calls = [c for c in s.client.session.calls if "advanced_search" in c[0]]
    assert len(search_calls) == 1          # batch 0 came from disk, not the API


def test_already_seen_tweets_are_dropped(tmp_path):
    (tmp_path / "x_seen_ids.json").write_text(json.dumps({"ids": ["10"]}))
    s = scraper_with(tmp_path, [
        accounts_payload(["a"]),
        {"tweets": [tweet("10"), tweet("11")], "has_next_page": False},
    ])
    got = s.scrape(seeds=["seed"], max_accounts_per_batch=1, max_query_chars=10_000)
    assert [t["id"] for t in got] == ["11"]
    assert "11" in json.loads((tmp_path / "x_seen_ids.json").read_text())["ids"]


def test_stale_account_cache_beats_no_account_list(tmp_path):
    (tmp_path / "x_accounts.json").write_text(json.dumps(
        {"fetched_at": "2020-01-01T00:00:00+00:00", "seeds": ["seed"],
         "accounts": ["a", "b"]}))
    s = scraper_with(tmp_path, [FakeResponse(503, text="boom")] * 4
                     + [{"tweets": [tweet("1")], "has_next_page": False}])
    got = s.scrape(seeds=["seed"], max_accounts_per_batch=2, max_query_chars=10_000, ttl_days=1)
    assert [t["id"] for t in got] == ["1"]
    health = json.loads((tmp_path / "x_health.json").read_text())
    assert health["status"] == "degraded"
    assert any("stale cache" in n for n in health["notes"])


def test_seed_following_nobody_is_noted_but_does_not_raise_a_daily_alarm(tmp_path):
    s = scraper_with(tmp_path, [
        accounts_payload(["a"]),                      # seed1 follows a
        {"followings": [], "has_next_page": False},   # seed2 follows nobody
        {"tweets": [], "has_next_page": False},
    ])
    s.scrape(seeds=["seed1", "seed2"], max_accounts_per_batch=25, max_query_chars=10_000)
    health = json.loads((tmp_path / "x_health.json").read_text())
    assert any("@seed2" in n for n in health["notes"])
    # a permanent config truth must not flag the feed as sick every single day
    assert health["status"] == "healthy"


def test_tweet_cap_stops_the_run_and_is_flagged(tmp_path):
    s = scraper_with(tmp_path, [
        accounts_payload(["a", "b", "c", "d"]),
        {"tweets": [tweet("10"), tweet("11")], "has_next_page": False},
    ])
    s.scrape(seeds=["seed"], max_accounts_per_batch=1, max_query_chars=10_000, max_tweets=1)
    health = json.loads((tmp_path / "x_health.json").read_text())
    assert health["tweet_cap_hit"] is True


# --- link hygiene -----------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://x.com/karpathy/status/1", "http://twitter.com/a", "https://t.co/abc",
    "https://www.reddit.com/r/x/comments/y", "https://redd.it/abc",
    "https://mobile.twitter.com/a", "",
])
def test_blocked_links_detected(url):
    assert is_blocked_link(url)


@pytest.mark.parametrize("url", [
    "https://arxiv.org/abs/2601.00001", "https://github.com/org/repo",
    "https://openai.com/index/thing", "https://notreddit.com/x",
    "https://xcom.example.org/a",
])
def test_good_links_pass(url):
    assert not is_blocked_link(url)


def test_normalize_strips_blocked_links_from_tweets(tmp_path):
    s = scraper_with(tmp_path, [])
    out = s._normalize(tweet("1", urls=["https://x.com/a/status/2",
                                        "https://arxiv.org/abs/1"]))
    assert out["external_links"] == ["https://arxiv.org/abs/1"]


def test_summary_html_blocked_anchors_become_plain_text():
    html = ('<li><a href="https://x.com/a/status/1">the tweet</a> and '
            '<a href="https://arxiv.org/abs/1">the paper</a></li>')
    out = strip_blocked_links(html)
    assert "the tweet" in out and "x.com" not in out
    assert '<a href="https://arxiv.org/abs/1">the paper</a>' in out


# --- digest plumbing --------------------------------------------------------

def test_health_banner_silent_when_healthy(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"status": "healthy"}))
    assert health_banner(str(p)) == ""


def test_health_banner_shouts_when_down(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"status": "down", "batches_failed": 3,
                             "batches_attempted": 3, "tweets_new": 0,
                             "checked_at": "now", "errors": ["HTTP 503"],
                             "notes": []}))
    out = health_banner(str(p))
    assert "down" in out and "503" in out


def test_health_banner_when_ingestion_never_ran(tmp_path):
    assert "did not run" in health_banner(str(tmp_path / "missing.json"))


def test_prompt_carries_all_three_sources_and_the_weighting():
    from config import SUMMARY_PROMPT_TEMPLATE
    from link_fetcher import format_pages
    body = SUMMARY_PROMPT_TEMPLATE.format(
        TIME_HORIZON_DAYS=1,
        posts_content="REDDIT_MARKER",
        tweets_content=format_tweets([{
            "id": "1", "handle": "karpathy", "author_name": "A", "likes": 5,
            "retweets": 2, "created_at": "x", "text": "X_MARKER",
            "external_links": ["https://arxiv.org/abs/1"]}]),
        pages_content=format_pages([{"url": "https://arxiv.org/abs/1",
                                     "title": "T", "text": "PAGE_MARKER"}]),
    )
    assert "REDDIT_MARKER" in body and "X_MARKER" in body and "PAGE_MARKER" in body
    assert "65%" in body and "35%" in body
    assert "reddit.com" in body           # the link prohibition is spelled out
    assert "arxiv.org/abs/1" in body


def test_prompt_forbids_the_ungrammatical_stock_phrase():
    """It read 'Verification and fidelity are detail not in source'. The phrase
    is only allowed to appear as the thing the model must NOT write."""
    from config import SUMMARY_PROMPT_TEMPLATE as T
    i = T.find("detail not in source")
    assert i > 0, "the prohibition itself went missing"
    assert "never as the fixed fragment" in T[:i]


def test_prompt_fences_fetched_pages_as_untrusted_data():
    """Fetched pages are third-party text entering a model prompt. The prompt
    must say so -- that is the whole mitigation."""
    from config import SUMMARY_PROMPT_TEMPLATE as T
    assert "UNTRUSTED THIRD-PARTY TEXT" in T
    assert "data, never instruction" in T


# --- the query-length cliff -------------------------------------------------

def test_batches_stay_under_the_query_length_cliff(tmp_path):
    """Measured 2026-09-22: X search silently returns 0 results past ~500 chars.
    Long handles must therefore produce MORE, shorter batches -- not one that
    quietly yields nothing."""
    s = scraper_with(tmp_path, [])
    long_handles = [f"averyveryverylonghandle{i:03d}" for i in range(40)]
    batches = s.make_batches(long_handles, "2026-09-22", max_query_chars=400)
    assert len(batches) > 2
    for b in batches:
        assert len(s._batch_query(b, "2026-09-22")) <= 400 or len(b) == 1
    assert [a for b in batches for a in b] == long_handles   # nobody dropped


def test_single_handle_longer_than_the_limit_still_gets_its_own_batch(tmp_path):
    s = scraper_with(tmp_path, [])
    assert s.make_batches(["x" * 600], "2026-09-22", max_query_chars=400) == [["x" * 600]]


def test_mostly_empty_batches_flag_the_feed_as_suspect(tmp_path):
    """The silent-zero signature. A quiet news day looks different from a
    broken query, and we would rather be told."""
    s = scraper_with(tmp_path, [
        accounts_payload([f"a{i}" for i in range(6)]),
        {"tweets": [tweet("1")], "has_next_page": False},
        {"tweets": [], "has_next_page": False},
        {"tweets": [], "has_next_page": False},
        {"tweets": [], "has_next_page": False},
        {"tweets": [], "has_next_page": False},
        {"tweets": [], "has_next_page": False},
    ])
    s.scrape(seeds=["seed"], max_accounts_per_batch=1, max_query_chars=10_000)
    health = json.loads((tmp_path / "x_health.json").read_text())
    assert health["status"] == "degraded"
    assert any("zero tweets" in n for n in health["notes"])


def test_zero_new_tweets_does_not_shadow_the_previous_dump(tmp_path, monkeypatch, capsys):
    """A same-day re-run finds nothing new. It must not write an empty
    x_data_*.json -- that would become the 'newest' file and silently empty the
    digest."""
    import x_scraper
    monkeypatch.chdir(tmp_path)
    (tmp_path / "extracts").mkdir()
    good = tmp_path / "extracts" / "x_data_20260922_120000.json"
    good.write_text(json.dumps([{"id": "1"}]))

    monkeypatch.setattr(x_scraper, "XScraper",
                        lambda *a, **k: _NoNewScraper(base_dir="extracts"))
    assert x_scraper.main() == 0
    dumps = sorted((tmp_path / "extracts").glob("x_data_*.json"))
    assert dumps == [good]
    assert "no NEW tweets" in capsys.readouterr().out


class _NoNewScraper(XScraper):
    def __init__(self, **kw):
        super().__init__(client=make_client([]), **kw)

    def scrape(self, *a, **k):
        return []


# --- digest cost guard ------------------------------------------------------

class _Usage:
    def __init__(self, inp, out, cached=0, reasoning=0):
        self.prompt_tokens = inp
        self.completion_tokens = out
        self.prompt_tokens_details = type("d", (), {"cached_tokens": cached})
        self.completion_tokens_details = type("d", (), {"reasoning_tokens": reasoning})


def test_cost_is_computed_from_the_pricing_table():
    from llm_summarizer import report_cost
    # gpt-6-sol: $2.00/1M in, $10.00/1M out
    assert report_cost("gpt-6-sol", _Usage(1_000_000, 100_000)) == pytest.approx(3.0)


def test_cached_input_is_billed_at_the_cached_rate():
    from llm_summarizer import report_cost
    # 1M input of which all cached: $0.20 rather than $2.00
    assert report_cost("gpt-6-sol", _Usage(1_000_000, 0, cached=1_000_000)) \
        == pytest.approx(0.2)


def test_over_ceiling_is_flagged_loudly(capsys):
    from llm_summarizer import report_cost
    report_cost("gpt-6-astra", _Usage(1_000_000, 100_000))   # $15, way over
    assert "OVER the $0.20/digest ceiling" in capsys.readouterr().out


def test_configured_model_has_a_price_and_fits_the_ceiling_at_realistic_size():
    """If someone bumps SUMMARY_MODEL to something pricier, this should catch it
    before the invoice does. Sized on the measured 2026-09-22 run."""
    from config import MODEL_PRICING, SUMMARY_MODEL, SUMMARY_COST_CEILING_USD
    assert SUMMARY_MODEL in MODEL_PRICING, f"no price known for {SUMMARY_MODEL}"
    p_in, _, p_out = MODEL_PRICING[SUMMARY_MODEL]
    est = (60_000 * p_in + 4_000 * p_out) / 1e6
    assert est <= SUMMARY_COST_CEILING_USD, f"{SUMMARY_MODEL} ~${est:.3f}/digest"


def test_unknown_model_does_not_crash_the_digest(capsys):
    from llm_summarizer import report_cost
    import math
    assert math.isnan(report_cost("gpt-9-whatever", _Usage(10, 10)))
    assert "no entry in config.MODEL_PRICING" in capsys.readouterr().out


# --- linked-page enrichment -------------------------------------------------

from link_fetcher import LinkFetcher, extract_text, has_no_text_value


def test_never_fetches_a_blocked_platform_link():
    """Enrichment must not become a back door to x.com/reddit.com."""
    tweets = [{"likes": 999, "retweets": 9,
               "external_links": ["https://x.com/a/status/1",
                                  "https://arxiv.org/abs/1"]}]
    posts = [{"score": 500, "url": "https://www.reddit.com/r/x/y"},
             {"score": 10, "url": "https://github.com/o/r"}]
    assert LinkFetcher.collect_links(tweets, posts, limit=10) == [
        "https://arxiv.org/abs/1", "https://github.com/o/r"]


def test_links_are_ranked_by_attention():
    tweets = [{"likes": 1, "retweets": 0, "external_links": ["https://a.com/1"]},
              {"likes": 500, "retweets": 100, "external_links": ["https://b.com/2"]}]
    assert LinkFetcher.collect_links(tweets, [], limit=2) == [
        "https://b.com/2", "https://a.com/1"]


@pytest.mark.parametrize("url", ["https://youtu.be/x",
                                 "https://www.youtube.com/watch?v=1",
                                 "https://firefly.social/post/x/1"])
def test_pages_with_no_extractable_text_are_not_fetched(url):
    assert has_no_text_value(url)
    assert not has_no_text_value("https://arxiv.org/abs/1")


def test_binary_urls_are_skipped_without_a_request(tmp_path):
    f = LinkFetcher(cache_dir=str(tmp_path))
    rec = f.fetch("https://example.com/paper.pdf")
    assert rec["status"].startswith("skipped")
    assert f.stats["skipped"] == 1


def test_cached_page_is_not_refetched(tmp_path, monkeypatch):
    """Regenerating a digest must cost no refetch -- both for our latency and
    for the servers we are reading."""
    import link_fetcher
    calls = []

    class R:
        status_code = 200
        headers = {"content-type": "text/html"}
        encoding = "utf-8"
        raw = type("raw", (), {"read": staticmethod(
            lambda *a, **k: b"<html><title>T</title><p>hello world body text</p>")})

        def close(self): pass

    def fake_get(url, **kw):
        calls.append(url)
        return R()

    monkeypatch.setattr(link_fetcher.requests, "get", fake_get)
    f = LinkFetcher(cache_dir=str(tmp_path))
    first = f.fetch("https://example.com/a")
    second = f.fetch("https://example.com/a")
    assert len(calls) == 1
    assert first["text"] == second["text"] and "hello world" in first["text"]


def test_a_dead_page_does_not_break_enrichment(tmp_path, monkeypatch):
    import link_fetcher

    def boom(url, **kw):
        raise link_fetcher.requests.ConnectTimeout("nope")

    monkeypatch.setattr(link_fetcher.requests, "get", boom)
    f = LinkFetcher(cache_dir=str(tmp_path))
    pages = f.enrich([{"likes": 1, "retweets": 0,
                       "external_links": ["https://example.com/a"]}], [], limit=5)
    assert pages == []
    assert f.stats["failed"] == 1


def test_extraction_prefers_article_body_over_site_navigation():
    body = ("<html><title>Post</title><body><nav>Home Courses Pricing</nav>"
            "<article>" + "The actual measured result was 76.8 percent. " * 12 +
            "</article><footer>legal</footer></body></html>")
    title, text = extract_text(body)
    assert title == "Post"
    assert "76.8 percent" in text
    assert "Courses Pricing" not in text


def test_extraction_drops_scripts():
    title, text = extract_text(
        "<html><title>T</title><script>alert('x')</script><p>real text</p></html>")
    assert "alert" not in text and "real text" in text
