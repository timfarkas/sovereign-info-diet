#!/usr/bin/env python3
"""Tests for the multi-topic digest layer.

Two jobs, in order of how much they matter:

  1. GUARD THE AI DIGEST. It is in production and Tim notices breakage by a
     digest not arriving. The AI topic's row must keep reproducing the values
     the pipeline had hardcoded, and it must stay out of the feed corpus
     entirely. These are the tests that fail if a future edit to topics.py
     "tidies up" the AI row.
  2. Assert the invariants every topic shares -- no social-platform links, the
     untrusted-data fence, HTML-only output -- as PROPERTIES of each prompt
     rather than by comparing wording, since the AI prompt is deliberately a
     different text from the other three.

Behaviour and contracts, not implementation: nothing here reaches for a private
name or asserts on a log line.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

import config
import rss_scraper
import topics

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def feed_item(**kw):
    base = dict(title="t", summary="s", url="https://example.com/a",
                site="Example", feed="https://example.com/feed",
                published=NOW.isoformat(), author="", source="direct")
    base.update(kw)
    return base


def post(sub, hours_old=1.0, **kw):
    d = dict(subreddit=sub, title="p", url="https://example.com/p",
             created_utc=(NOW - timedelta(hours=hours_old)).timestamp())
    d.update(kw)
    return d


# =============================================================================
# 1. the AI digest must not move
# =============================================================================

class TestAIDigestUnchanged:
    """Every value here was hardcoded in the pipeline before topics.py existed."""

    def test_prompt_is_the_production_template_itself(self):
        # not a copy, not a reformat: the same object out of config
        assert topics.topic("ai").prompt is config.SUMMARY_PROMPT_TEMPLATE

    def test_window_and_lookback(self):
        ai = topics.topic("ai")
        assert ai.window_days == config.TIME_HORIZON_DAYS == 2
        assert ai.lookback_hours == 20
        assert ai.every_n_days == 1

    def test_subreddits_and_quota(self):
        ai = topics.topic("ai")
        assert list(ai.subreddits) == config.SUBREDDITS
        # 12, as POSTS_TO_ANALYZE // len(SUBREDDITS) gave before the union of
        # 19 subreddits would have turned that same expression into 3
        assert ai.posts_per_subreddit == 12

    def test_takes_the_whole_x_corpus_unfiltered(self):
        ai = topics.topic("ai")
        tweets = [{"text": "kubernetes yak shaving"}, {"text": "NATO in the Baltic"}]
        assert ai.x_all is True
        assert ai.select_tweets(tweets) == tweets

    def test_subscribes_to_no_feeds(self):
        ai = topics.topic("ai")
        assert ai.feeds == () and ai.feed_urls == ()

    def test_gets_zero_feed_items_from_a_full_pool(self):
        """The regression this guard exists for.

        `matches()` answers True for an empty keyword set, because for TWEETS
        that means "take everything". Routed through the feed selector that same
        answer handed the AI digest all 279 items of a live pool and filled the
        digest with Austrian domestic politics.
        """
        pool = [feed_item(title="EU fines Meta"), feed_item(title="H5N1 in dairy"),
                feed_item(title="GPT-6 released")]
        assert topics.topic("ai").select_feed_items(pool) == []

    def test_history_and_page_names_are_the_pre_existing_ones(self):
        ai = topics.topic("ai")
        assert ai.stats_kind == "digest"       # months of existing history
        assert ai.page == "ai-digest"          # the existing served directory
        assert ai.subject == "AI Digest"       # the existing mail subject

    def test_is_always_due_with_no_state_at_all(self):
        is_due, _ = topics.due(topics.topic("ai"), {}, now=NOW)
        assert is_due
        # and stays due the same night it already ran
        state = topics.record_run({}, topics.topic("ai"), ok=True, now=NOW)
        assert topics.due(topics.topic("ai"), state, now=NOW)[0]

    def test_is_the_first_topic(self):
        """Ordering is load-bearing: digest_run runs AI first so a later topic
        blowing up cannot consume the run's time budget before AI gets one."""
        assert topics.all_topics()[0].key == "ai"


# =============================================================================
# 2. invariants that must hold for every topic, AI included
# =============================================================================

BLOCKED_HOSTS = ("x.com", "twitter.com", "t.co", "reddit.com", "redd.it")


@pytest.mark.parametrize("t", topics.all_topics(), ids=lambda t: t.key)
class TestEveryPromptHonoursTheInvariants:

    def test_forbids_social_platform_links(self, t):
        p = t.prompt.lower()
        assert "never" in p
        for host in BLOCKED_HOSTS:
            assert host in p, f"{t.key} prompt does not name {host} as blocked"

    def test_fences_fetched_pages_as_untrusted(self, t):
        p = t.prompt
        assert "SECURITY" in p
        assert "instruction" in p.lower()

    def test_asks_for_an_html_fragment_only(self, t):
        """A full document would nest inside send_notification's own wrapper.

        Asserted as "the prompt forbids the document tags", not as "the prompt
        never contains them" -- naming <body> in order to rule it out is exactly
        what it should be doing.
        """
        p = t.prompt
        assert "<h3>" in p and "<ul>" in p
        assert "fragment" in p.lower()
        assert "no <html>/<head>/<body>" in p
        assert "no code fence" in p.lower()

    def test_organises_by_topic_not_by_source(self, t):
        assert "by topic" in t.prompt.lower() or "by TOPIC" in t.prompt

    def test_demands_a_high_discard_rate(self, t):
        assert "discard" in t.prompt.lower()

    def test_formats_without_leftover_slots(self, t):
        """Every run-time slot must be one this topic's template actually fills.

        str.format ignores kwargs a template does not mention, which is what
        lets one build() serve both the old AI template and the new skeleton --
        but it raises KeyError on a slot with no kwarg, so this catches a typo
        in a placeholder name at test time instead of at 01:00.
        """
        out = t.prompt.format(TIME_HORIZON_DAYS=t.window_days,
                              window_days=t.window_days,
                              pages_content="P", feed_content="F",
                              tweets_content="T", posts_content="R")
        assert "{" not in out.replace("{{", "").replace("}}", "")

    def test_has_a_system_message_and_a_blurb(self, t):
        assert t.system.strip() and t.priorities_blurb.strip()

    def test_cost_is_capped(self, t):
        assert 0 < t.cost_ceiling_usd <= 0.25


class TestTopicSetIsWellFormed:

    def test_keys_pages_and_history_files_are_unique(self):
        ts = topics.all_topics()
        for attr in ("key", "page", "stats_kind", "subject"):
            vals = [getattr(t, attr) for t in ts]
            assert len(set(vals)) == len(vals), f"duplicate {attr}: {vals}"

    def test_a_multi_day_topic_can_reach_back_over_its_whole_window(self):
        """A prompt that says "the past 3 days" must be allowed to read 3 days
        of dumps, or it describes material it was never given.

        The AI topic is exempt and must stay exempt: lookback_hours is a limit
        on how OLD a dump file may be, not on the age of the posts inside it.
        Its 20h keeps it on a single dump -- one night, one digest, as before --
        while x_scraper already fetches 2 days of tweets into that dump. The
        multi-day topics are the ones that union several nights' dumps, so they
        are the ones whose lookback has to span the window.
        """
        for t in topics.all_topics():
            if t.every_n_days == 1:
                continue
            assert t.lookback_hours >= t.window_days * 24 - 4, t.key

    def test_a_nightly_topic_reads_exactly_one_dump(self):
        for t in topics.all_topics():
            if t.every_n_days == 1:
                assert t.lookback_hours <= 24, t.key

    def test_a_subreddit_belongs_to_one_topic(self):
        """Subreddit routing IS the classifier, so an overlap would silently
        put the same thread in two digests."""
        seen = {}
        for t in topics.all_topics():
            for s in t.subreddits:
                assert s.lower() not in seen, f"{s} in {seen.get(s.lower())} and {t.key}"
                seen[s.lower()] = t.key

    def test_shared_scrape_lists_are_deduplicated_unions(self):
        assert len(topics.all_subreddits()) == len(set(topics.all_subreddits()))
        assert len(topics.all_feed_urls()) == len(set(topics.all_feed_urls()))
        for t in topics.all_topics():
            assert set(t.subreddits) <= set(topics.all_subreddits())
            assert set(t.feed_urls) <= set(topics.all_feed_urls())

    def test_topic_lookup_rejects_an_unknown_key(self):
        with pytest.raises(KeyError):
            topics.topic("nonexistent")

    def test_a_topic_cannot_edit_its_own_config(self):
        with pytest.raises(Exception):
            topics.topic("ai").window_days = 99


# =============================================================================
# 3. routing
# =============================================================================

class TestKeywordRouting:

    def test_keywords_are_word_bounded(self):
        """"eu" must match "the EU" and not reach inside "euler"."""
        eu = topics.topic("europe")
        assert eu.matches("the EU fined Meta EUR 800m")
        assert not eu.matches("euler angles and federated learning")
        assert not eu.matches("a eucalyptus in the garden")

    def test_a_phrase_matches_across_a_line_break(self):
        g = topics.topic("geopolitics")
        assert g.matches("tensions in the\nSouth China Sea are rising")

    def test_matching_is_case_insensitive(self):
        p = topics.topic("pandemic")
        assert p.matches("H5N1 detected") and p.matches("h5n1 detected")

    def test_a_topic_only_takes_tweets_it_matches(self):
        g = topics.topic("geopolitics")
        tweets = [{"text": "new tariffs on semiconductors"},
                  {"text": "my cat sat on the keyboard"}]
        assert g.select_tweets(tweets) == tweets[:1]

    def test_summary_text_can_route_an_item_the_title_does_not(self):
        p = topics.topic("pandemic")
        assert p.select_feed_items([feed_item(title="Quiet week in the lab",
                                             summary="an H5N1 spillover event")])


class TestPublicationRouting:

    def test_a_routed_publication_needs_no_keyword(self):
        g = topics.topic("geopolitics")
        item = feed_item(title="Nothing topical here", summary="",
                         site="War on the Rocks")
        assert g.select_feed_items([item]) == [item]

    def test_routing_reads_site_author_and_feed(self):
        g = topics.topic("geopolitics")
        for field in ("site", "author", "feed"):
            item = feed_item(title="x", summary="", site="", author="", feed="")
            item[field] = "Lawfare"
            assert g.from_feed(item), field

    def test_feed_urls_alone_do_not_route(self):
        """Adding a URL to feed_urls says FETCH it, not "send it to me". WHO is
        fetched by the pandemic topic but reaches it by keyword, so WHO news
        about staffing does not land in a bio-risk digest."""
        p = topics.topic("pandemic")
        off_topic = feed_item(title="WHO Director-General visits Jordan",
                             summary="recognizing hospitality to refugees",
                             site="World Health Organization")
        assert p.select_feed_items([off_topic]) == []


class TestFeedBudget:

    def test_cap_is_shared_round_robin_across_publications(self):
        """A daily paper must not eat a weekly blog's slot.

        Live run before this fix: `europe` was ~60% one newspaper, with
        Verfassungsblog and EDRi pushed off the end of the cap.
        """
        eu = topics.topic("europe")
        loud = [feed_item(site="Daily Paper", title="EU summit",
                          published=f"2026-09-29T{h:02d}:00:00+00:00")
                for h in range(23)]
        quiet = [feed_item(site="Weekly Blog", title="EU rule of law",
                           published="2026-09-20T00:00:00+00:00")]
        sel = eu.select_feed_items(loud + quiet)
        assert quiet[0] in sel, "the low-volume publication was starved"

    def test_cap_is_enforced(self):
        eu = topics.topic("europe")
        many = [feed_item(site=f"Pub {i}", title="EU budget") for i in range(200)]
        assert len(eu.select_feed_items(many)) == eu.max_feed_items

    def test_newest_first_within_a_publication(self):
        eu = topics.topic("europe")
        old = feed_item(title="EU old", published="2026-09-01T00:00:00+00:00")
        new = feed_item(title="EU new", published="2026-09-28T00:00:00+00:00")
        assert eu.select_feed_items([old, new])[0]["title"] == "EU new"


class TestRedditWindowNarrowing:
    """The reddit leg scrapes at the WIDEST window any topic asks for, so each
    topic has to re-apply its own or the AI digest silently starts seeing
    3-day-old posts it never used to see."""

    def test_other_topics_subreddits_are_excluded(self):
        ai = topics.topic("ai")
        sel = ai.select_posts([post("singularity"), post("geopolitics"),
                               post("europe")], now=NOW)
        assert [p["subreddit"] for p in sel] == ["singularity"]

    def test_subreddit_match_is_case_insensitive(self):
        """config spells it DeepLearning; the API returns deeplearning."""
        ai = topics.topic("ai")
        assert len(ai.select_posts([post("deeplearning")], now=NOW)) == 1

    def test_a_post_older_than_this_topics_window_is_dropped(self):
        ai = topics.topic("ai")              # window_days=2
        sel = ai.select_posts([post("singularity", hours_old=12),
                               post("singularity", hours_old=70)], now=NOW)
        assert len(sel) == 1

    def test_a_wider_topic_keeps_what_the_ai_topic_drops(self):
        g = topics.topic("geopolitics")      # window_days=3
        assert len(g.select_posts([post("geopolitics", hours_old=70)], now=NOW)) == 1

    def test_a_post_with_no_timestamp_is_kept(self):
        """Dumps written before created_utc existed were already window-filtered
        at scrape time, so dropping them would lose real material."""
        p = post("singularity")
        del p["created_utc"]
        assert len(topics.topic("ai").select_posts([p], now=NOW)) == 1


# =============================================================================
# 4. cadence
# =============================================================================

class TestDueLogic:

    def test_a_multi_day_topic_is_due_when_it_has_never_run(self):
        assert topics.due(topics.topic("europe"), {}, now=NOW)[0]

    def test_and_not_due_the_day_after_it_ran(self):
        eu = topics.topic("europe")
        state = topics.record_run({}, eu, ok=True, now=NOW - timedelta(days=1))
        assert not topics.due(eu, state, now=NOW)[0]

    def test_and_due_again_after_every_n_days(self):
        eu = topics.topic("europe")
        state = topics.record_run({}, eu, ok=True,
                                  now=NOW - timedelta(days=eu.every_n_days))
        assert topics.due(eu, state, now=NOW)[0]

    def test_fires_slightly_early_rather_than_drifting_later(self):
        """Cron jitter must not push an every-3-days topic to a 4-day cadence."""
        eu = topics.topic("europe")
        state = topics.record_run({}, eu, ok=True,
                                  now=NOW - timedelta(days=eu.every_n_days,
                                                      minutes=-30))
        assert topics.due(eu, state, now=NOW)[0]

    def test_a_failed_run_leaves_the_topic_due(self):
        """The asymmetry that matters: a failed model call must retry tomorrow,
        not bench the topic for another every_n_days."""
        eu = topics.topic("europe")
        state = topics.record_run({}, eu, ok=False, now=NOW)
        assert state["europe"]["last_attempt"]
        assert "last_success" not in state["europe"]
        assert topics.due(eu, state, now=NOW)[0]

    def test_a_corrupt_timestamp_runs_the_topic(self):
        """Fail loud and forward: a digest arriving is better than state silently
        benching a topic forever."""
        eu = topics.topic("europe")
        assert topics.due(eu, {"europe": {"last_success": "not a date"}}, now=NOW)[0]

    def test_state_survives_a_round_trip(self, tmp_path):
        p = tmp_path / "state.json"
        eu = topics.topic("europe")
        topics.save_state(topics.record_run({}, eu, ok=True, now=NOW), str(p))
        assert topics.load_state(str(p))["europe"]["successes"] == 1

    def test_a_corrupt_state_file_is_not_fatal(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("{not json")
        assert topics.load_state(str(p)) == {}

    def test_a_missing_state_file_is_not_fatal(self, tmp_path):
        assert topics.load_state(str(tmp_path / "nope.json")) == {}


# =============================================================================
# 5. feed ingestion
# =============================================================================

class TestSelfDigestExclusion:
    """His own digests are auto-forwarded into his Readwise feed, so without
    this the pipeline summarises itself -- and each pass compounds."""

    def test_the_forwarding_author_is_excluded(self):
        assert rss_scraper.is_self_digest(
            feed_item(title="Anything at all", author="Meta Minsky"))

    def test_every_topics_own_subject_line_is_excluded(self):
        for t in topics.all_topics():
            title = f"{t.subject} - 2026-09-28"
            assert rss_scraper.is_self_digest(feed_item(title=title, author="")), title

    def test_an_ordinary_article_is_kept(self):
        assert not rss_scraper.is_self_digest(
            feed_item(title="Europe's AI Act enters force", author="Jane Doe"))


class TestWindowing:

    def test_an_item_inside_the_window_is_kept(self):
        assert rss_scraper.within_window(
            feed_item(published=datetime.now(timezone.utc).isoformat()), 4)

    def test_an_old_item_is_dropped(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        assert not rss_scraper.within_window(feed_item(published=old), 4)

    def test_an_item_with_no_usable_date_is_kept(self):
        """Deliberate: some feeds omit dates entirely, and dropping them would
        silently lose a whole publication. A stale item costs a few tokens."""
        assert rss_scraper.within_window(feed_item(published=""), 4)
        assert rss_scraper.within_window(feed_item(published="tuesday-ish"), 4)

    @pytest.mark.parametrize("raw", [
        "2026-09-29T00:00:00+00:00", "2026-09-29T00:00:00Z",
        1790000000, 1790000000000, "1790000000",
    ])
    def test_date_formats_that_feeds_actually_emit(self, raw):
        assert rss_scraper.parse_date(raw) is not None


class TestUrlHandling:

    def test_campaign_tracking_is_stripped(self):
        assert rss_scraper.clean_url(
            "https://example.com/a?utm_source=newsletter&ref=x#top"
        ) == "https://example.com/a"

    def test_the_same_article_on_both_paths_is_kept_once(self):
        """The direct path sees the bare permalink, Readwise sees it decorated."""
        a = feed_item(url="https://example.com/a?utm_source=rw", source="readwise")
        b = feed_item(url="https://example.com/a", source="direct")
        assert len(rss_scraper.dedupe([a, b])) == 1

    def test_two_articles_sharing_a_cleaned_url_both_survive(self):
        """Some sites identify an article entirely by query string. Keeping a
        duplicate costs tokens and is visible; dropping an article is silent."""
        a = feed_item(url="https://example.com/news.php?id=1", title="First")
        b = feed_item(url="https://example.com/news.php?id=2", title="Second")
        assert len(rss_scraper.dedupe([a, b])) == 2

    def test_readwise_wins_a_duplicate(self):
        """Its copy has the better parse and a word count, so it is the one to
        keep -- which is why the caller puts Readwise items first."""
        a = feed_item(url="https://example.com/a", source="readwise")
        b = feed_item(url="https://example.com/a", source="direct")
        assert rss_scraper.dedupe([a, b])[0]["source"] == "readwise"

    def test_distinct_articles_both_survive(self):
        a = feed_item(url="https://example.com/a")
        b = feed_item(url="https://example.com/b")
        assert len(rss_scraper.dedupe([a, b])) == 2


class TestFeedFetchFailsSoft:
    """A broken feed must cost its own items and nothing else -- one publisher
    404ing cannot take the night's digest with it."""

    def test_an_unreachable_feed_returns_an_error_not_an_exception(self):
        items, err = rss_scraper.fetch_feed(
            "https://localhost:1/definitely-not-a-feed.xml", limit=5)
        assert items == [] and err

    def test_garbage_in_place_of_a_feed_is_survivable(self, tmp_path):
        items, err = rss_scraper.fetch_feed("file:///dev/null", limit=5)
        assert items == []
