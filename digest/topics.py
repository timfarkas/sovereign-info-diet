#!/usr/bin/env python3
"""Topics as data, so a digest is a configuration rather than a codebase.

The AI digest used to BE the pipeline: one prompt, one subreddit list, one
email, all hardcoded. This module turns "which digest" into a value, and the AI
digest is simply the first element of `TOPICS`. Everything downstream --
`digest_run.py`, the summarizer, the status page -- takes a Topic and has no
idea which one it got. That is what makes adding a fourth topic a config change
and what makes "no regression on the AI digest" a structural fact rather than
something to re-test by eye: the AI topic's row below reproduces the old
hardcoded values exactly, and `test_topics.py` asserts each of them.

ROUTING, i.e. which material reaches which topic:

  X posts    -- `x_all=True` takes the whole corpus (the AI topic, unchanged).
                Otherwise a topic gets the tweets whose text matches its
                keywords.
  reddit     -- by subreddit. A subreddit belongs to exactly one topic in
                practice, and that is the whole classifier.
  feed items -- by publication first (`feeds`: a site or newsletter routed
                wholesale, e.g. ChinaTalk -> geopolitics), then by keyword for
                the general-interest feeds that span topics (Reuters, WIRED,
                the Guardian).

That is deliberately the low-bit version. Tim's rule is to try the small-brain
thing first, and a keyword regex over ~150 items a night is auditable, free,
and instant, where a classifier would be none of those. The summarizer is told
to discard >80% of what it is handed, so the cost of a false positive is a few
hundred tokens and the cost of a false negative is a missed item -- which is
why the keyword lists lean inclusive.
"""

import json
import re

import prompts
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_PATH = "extracts/topic_state.json"


@dataclass(frozen=True)
class Topic:
    """One digest. Frozen because a run must not be able to edit its own config."""

    key: str                      # stable id: filenames, state, page dir, CLI
    name: str                     # human name, used in the email subject
    prompt: str                   # the template digest_run fills
    system: str                   # the model's system message
    priorities_blurb: str         # one line for the status page

    every_n_days: int = 1         # how often this topic is allowed to run
    window_days: int = 2          # what the prompt claims the window is
    lookback_hours: int = 20      # how far back X/reddit dumps may be read

    subreddits: Tuple[str, ...] = ()
    posts_per_subreddit: int = 12
    x_all: bool = False           # True = the entire X corpus, no filtering
    keywords: Tuple[str, ...] = ()
    feeds: Tuple[str, ...] = ()   # site_name/author substrings, routed wholesale
    feed_urls: Tuple[str, ...] = ()   # direct-fetch feeds for the standalone path
    max_feed_items: int = 60      # circuit breaker on prompt size

    cost_ceiling_usd: float = 0.25
    link_fetch_max_pages: int = 30

    # -- derived ------------------------------------------------------------
    @property
    def stats_kind(self) -> str:
        """History file under data/run_stats/, e.g. `ai-digest.jsonl`.

        Streamlined 2026-09-30 to a uniform `<key>-digest` for every topic,
        AI included. AI previously kept the bare `digest` name for backward
        compatibility; that history file was renamed on disk to match.
        """
        return f"{self.key}-digest"

    @property
    def page(self) -> str:
        return f"{self.key}-digest"

    @property
    def subject(self) -> str:
        return f"{self.name} Digest"

    @property
    def pattern(self) -> Optional[re.Pattern]:
        """Compiled keyword matcher, or None for a topic that takes everything.

        Word-bounded so "eu" cannot match "euler" and "fed" cannot match
        "federated". Multi-word phrases are matched with flexible whitespace so
        a line break inside a tweet does not defeat them.
        """
        if not self.keywords:
            return None
        return _compile(self.keywords)

    def matches(self, *texts: Optional[str]) -> bool:
        pat = self.pattern
        if pat is None:
            return True
        blob = " ".join(t for t in texts if t)
        return bool(pat.search(blob))

    def from_feed(self, item: Dict[str, Any]) -> bool:
        """True when this item's publication is routed wholesale to this topic."""
        haystack = " ".join(str(item.get(k) or "")
                            for k in ("site", "author", "feed")).lower()
        return any(f.lower() in haystack for f in self.feeds)

    # -- selection ----------------------------------------------------------
    def select_tweets(self, tweets: List[Dict]) -> List[Dict]:
        if self.x_all:
            return list(tweets or [])
        return [t for t in tweets or []
                if self.matches(t.get("text"), t.get("quoted_text"))]

    def select_posts(self, posts: List[Dict], now: datetime = None) -> List[Dict]:
        """This topic's subreddits, narrowed to this topic's own window.

        The reddit leg scrapes at the WIDEST window any topic asks for, so the
        window has to be re-applied here or the AI digest would start seeing
        3-day-old posts it never used to see. A post with no `created_utc` --
        i.e. one scraped before that field existed -- is kept, since the old
        dumps were already window-filtered at scrape time.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=self.window_days)).timestamp()
        wanted = {s.lower() for s in self.subreddits}
        out = []
        for p in posts or []:
            if str(p.get("subreddit") or "").lower() not in wanted:
                continue
            created = p.get("created_utc")
            if created is not None and float(created) < cutoff:
                continue
            out.append(p)
        return out

    def select_feed_items(self, items: List[Dict]) -> List[Dict]:
        """Feed items for this topic: publication match, else keyword match.

        A topic with neither `feeds` nor `keywords` has expressed no way to
        select from the shared feed pool, so it gets NOTHING -- explicitly, and
        not via `matches()`, which answers True for an empty keyword set because
        for TWEETS that means "take the whole corpus". The AI topic is exactly
        this case, and without the guard it inherited all 104 feed items and the
        digest filled up with Austrian domestic politics. Its contract is the
        pre-existing one: X + Reddit, no feeds.

        Note the asymmetry with `feed_urls`: that field says which feeds to
        FETCH, not which items land here. A feed you add there still needs its
        publication in `feeds`, or a keyword hit, to reach any topic -- which is
        what routes who.int by keyword instead of wholesale.

        Newest first, then capped -- a topic that suddenly matches 400 items
        (a keyword that turned out to be too broad) must cost a bounded number
        of tokens, not an unbounded one. The cap is reached in practice, not in
        theory: `europe` matched 60/60 on the first live run.

        Which is why the cap is spent ROUND-ROBIN over publications rather than
        strictly newest-first. A daily paper publishes ~35 items per window and
        a weekly law blog publishes ~4, so newest-first hands the entire budget
        to the highest-volume feed -- the first live run would have been ~60%
        derStandard, with Verfassungsblog and EDRi pushed off the end. He
        subscribed to each of these separately, so each gets a turn, and the
        publications he reads slowly are the ones this protects. Within a
        publication it is still newest-first.
        """
        if not self.feeds and not self.keywords:
            return []
        hits = [i for i in items or []
                if self.from_feed(i) or self.matches(i.get("title"),
                                                     i.get("summary"))]
        hits.sort(key=lambda i: str(i.get("published") or ""), reverse=True)

        by_pub: Dict[str, List[Dict]] = {}
        for i in hits:
            by_pub.setdefault(str(i.get("site") or i.get("feed") or "?"), []).append(i)
        out: List[Dict] = []
        queues = list(by_pub.values())        # insertion order == newest-publication first
        while len(out) < self.max_feed_items and queues:
            queues = [q for q in queues if q]
            for q in queues:
                if len(out) >= self.max_feed_items:
                    break
                out.append(q.pop(0))
        return out


def _compile(keywords) -> re.Pattern:
    parts = []
    for kw in keywords:
        # a phrase matches across any whitespace; a bare word is bounded
        body = r"\s+".join(re.escape(w) for w in kw.split())
        parts.append(rf"\b{body}\b" if kw[:1].isalnum() else body)
    return re.compile("|".join(parts), re.I)


# =============================================================================
# The topics
# =============================================================================

# -- AI: the original digest, reproduced exactly -------------------------------
# Every value here is what the pipeline hardcoded before this module existed:
# TIME_HORIZON_DAYS=2, MAX_AGE_HOURS=20, SUBREDDITS, POSTS_TO_ANALYZE//5 == 12,
# SUMMARY_COST_CEILING_USD, LINK_FETCH_MAX_PAGES and the whole X corpus. Its
# prompt now lives in prompts.py with the others; it is the same text as before
# plus the scratch-notes and long-running-board trailer the other topics have. Changing one of them is a change to
# the digest Tim reads every morning, so do it on purpose.
def _ai_topic() -> Topic:
    from config import (LINK_FETCH_MAX_PAGES, POSTS_TO_ANALYZE, SUBREDDITS,
                        SUMMARY_COST_CEILING_USD, TIME_HORIZON_DAYS)
    from prompts import AI
    return Topic(
        key="ai",
        name="AI",
        prompt=AI,
        system=("You are a sharp, insightful AI/tech news analyst with a good "
                "sense of humor."),
        priorities_blurb="frontier capability, alignment and x-risk, sentiment, "
                         "the absurd",
        every_n_days=1,
        window_days=TIME_HORIZON_DAYS,
        lookback_hours=20,
        subreddits=tuple(SUBREDDITS),
        posts_per_subreddit=POSTS_TO_ANALYZE // len(SUBREDDITS),
        x_all=True,          # the AI digest reads every tweet, as it always has
        keywords=(),         # nothing to filter: x_all short-circuits it
        feeds=(),            # and no feed items, so its prompt is unchanged
        feed_urls=(),
        cost_ceiling_usd=SUMMARY_COST_CEILING_USD,
        link_fetch_max_pages=LINK_FETCH_MAX_PAGES,
    )


GEOPOLITICS_KEYWORDS = (
    # state behaviour
    "china", "taiwan", "russia", "russian", "ukraine", "iran", "israel",
    "north korea", "nato", "pentagon", "kremlin", "beijing", "geopolitic",
    "ceasefire", "escalation", "escalate", "mobilisation", "mobilization",
    "airstrike", "air strike", "missile", "drone strike", "nuclear weapon",
    "nuclear test", "treaty", "sanction", "sanctions", "tariff", "tariffs",
    "trade war", "export control", "export controls", "entity list",
    "blockade", "annex", "invasion", "war in", "defence spending",
    "defense spending", "arms deal", "military exercise",
    # the industrial layer
    "semiconductor", "semiconductors", "tsmc", "asml", "euv", "fab ",
    "foundry", "chip export", "chip ban", "rare earth", "rare earths",
    "critical minerals", "lithium", "cobalt", "gallium", "germanium",
    "supply chain", "supply chains", "shipping rate", "freight rate",
    "container ship", "suez", "strait of hormuz", "strait of malacca",
    "panama canal", "red sea", "port strike", "opec", "lng", "natural gas",
    "crude oil", "uranium", "enrichment", "grid capacity",
    # the financial layer
    "central bank", "federal reserve", "ecb rate", "interest rate",
    "sovereign debt", "bond yield", "treasury yield", "currency",
    "devaluation", "inflation", "recession", "stock market", "equities",
    "commodity price", "default risk", "capital controls", "imf", "world bank",
)

GEOPOLITICS_FEEDS = (
    # publications routed wholesale: everything they publish is on-topic enough
    # that keyword-gating them would lose more than it saves
    "Reuters", "Bloomberg", "Financial Times", "WSJ", "The Economist",
    "Foreign Affairs", "Foreign Policy", "War on the Rocks", "ChinaTalk",
    "Noahpinion", "SemiAnalysis", "Matt Levine", "Money Stuff",
    "Sentinel Global Risks Watch", "Bruegel", "Lawfare",
    "Responsible Statecraft", "Supply Chain Dive", "Federal Reserve",
)

GEOPOLITICS = Topic(
    key="geopolitics",
    name="Geopolitics, Markets & Supply Chains",
    prompt=prompts.GEOPOLITICS,
    system=("You are a rigorous geopolitical and macro analyst. You connect "
            "military, diplomatic, industrial and financial signals into one "
            "picture, you prefer a number to an adjective, and you never "
            "predict past your evidence."),
    priorities_blurb="great-power escalation, the industrial and financial "
                     "layer under it, tail risk",
    every_n_days=3,
    window_days=3,
    lookback_hours=76,          # 3 days plus cron slack
    # LessCredibleDefence dropped 2026-09-30: it's CredibleDefense's own
    # meme/shitpost spinoff sub (the name is the joke), not a second source of
    # analysis -- same failure mode as AskEurope below, just for this topic.
    subreddits=("geopolitics", "CredibleDefense", "Economics", "supplychain"),
    keywords=GEOPOLITICS_KEYWORDS,
    feeds=GEOPOLITICS_FEEDS,
    # Verified 2026-09-29 on BOTH axes: reachable AND actually publishing.
    # Status 200 is not enough -- csis.org/rss.xml returns 200 and its newest
    # entry is from 2016, so it was dropped after being caught by the freshness
    # probe rather than the reachability one. bruegel.org, ecfr.eu and
    # brookings.edu 403/302 a scripted fetch and euobserver's feed is 410 gone:
    # dropped rather than worked around, since a feed that needs a
    # browser-shaped request is a feed whose owner does not want one.
    # Deliberately short and high-signal. thediplomat.com and freightwaves.com
    # are both live and both publish ~45 items per 4-day window, which would
    # push this topic permanently against max_feed_items and make selection
    # "whatever was newest" instead of "what matters". This list is the
    # INDEPENDENT path anyway; the primary one is whatever Tim subscribes to in
    # Readwise, so breadth belongs there, not here.
    feed_urls=(
        "https://warontherocks.com/feed/",
        "https://www.lawfaremedia.org/feeds/articles",
        "https://www.chinatalk.media/feed",
        "https://responsiblestatecraft.org/feed/",
        "https://www.supplychaindive.com/feeds/news/",
        "https://www.federalreserve.gov/feeds/press_all.xml",
    ),
    cost_ceiling_usd=0.15,
    link_fetch_max_pages=25,
)


PANDEMIC_KEYWORDS = (
    "pandemic", "epidemic", "outbreak", "spillover", "zoonotic", "zoonosis",
    "h5n1", "h5n5", "h7n9", "avian influenza", "avian flu", "bird flu",
    "influenza", "sars-cov", "covid", "coronavirus", "measles", "polio",
    "cholera", "ebola", "marburg", "nipah", "lassa", "mpox", "monkeypox",
    "dengue", "zika", "chikungunya", "tuberculosis", "malaria", "rabies",
    "prion", "antimicrobial resistance", "antibiotic resistance",
    "drug resistant", "case fatality", "attack rate", "seroprevalence",
    "reassortment", "reassortant", "virulence", "transmissibility",
    "human-to-human", "human to human", "quarantine", "contact tracing",
    "wastewater surveillance", "genomic surveillance", "metagenomic",
    "vaccine", "vaccination", "antiviral", "monoclonal", "mrna",
    "clinical trial", "phase 3", "phase iii",
    # biosecurity / dual use
    "biosecurity", "biosafety", "bsl-3", "bsl-4", "gain of function",
    "gain-of-function", "dual use research", "dual-use research",
    "enhanced potential pandemic", "bioweapon", "biological weapon",
    "biological weapons convention", "select agent", "dna synthesis",
    "benchtop synthesizer", "nucleic acid synthesis", "biorisk",
    "bioterror", "pathogen", "lab leak", "biodefense", "biodefence",
    "stockpile", "cdc", "ecdc", "who declares", "public health emergency",
)

PANDEMIC_FEEDS = (
    "CIDRAP", "Your Local Epidemiologist", "Asimov Press", "Health Security",
    "Sentinel Global Risks Watch", "Outbreak News", "Contagion",
    "Johns Hopkins Center for Health Security", "Health Policy Watch",
    "Lancet", "Global Biodefense",
)

PANDEMIC = Topic(
    key="pandemic",
    name="Pandemic Preparedness & Bio-risk",
    prompt=prompts.PANDEMIC,
    system=("You are a careful epidemiologist and biosecurity analyst. You "
            "separate confirmed from reported from projected, you never "
            "inflate a signal, and you say plainly when a week was quiet."),
    priorities_blurb="outbreak trajectory, biosecurity and dual-use policy, "
                     "countermeasures, calibration",
    every_n_days=3,
    window_days=3,
    lookback_hours=76,
    subreddits=("H5N1_AvianFlu", "ContagionCuriosity", "epidemiology",
                "virology", "ID_News"),
    keywords=PANDEMIC_KEYWORDS,
    feeds=PANDEMIC_FEEDS,
    # Verified live 2026-09-29. Both of the obvious feeds here are traps that
    # return 200: cidrap.umn.edu/rss.xml is an archive whose newest entry is
    # from 2022, and who.int/rss-feeds/news-english.xml had not published in 7
    # months. WHO's Disease Outbreak News feed, which is the one actually worth
    # having, is 404 at every documented URL -- so WHO reaches this topic only
    # via keyword now, which is why "World Health Organization" is deliberately
    # absent from PANDEMIC_FEEDS above. STAT is paywalled: titles and summaries
    # still route correctly and link_fetcher hands a walled page to the reader
    # as a link rather than dropping it.
    feed_urls=(
        "https://yourlocalepidemiologist.substack.com/feed",
        "https://healthpolicy-watch.news/feed/",
        "https://www.thelancet.com/rssfeed/laninf_current.xml",
        "https://www.statnews.com/feed/",
        "https://www.globalbiodefense.com/feed/",
    ),
    cost_ceiling_usd=0.15,
    link_fetch_max_pages=25,
)


EUROPE_KEYWORDS = (
    # "eu" bare, word-bounded, was missing until a test asked whether "EU fines
    # Meta" routes -- it did not, and that is how most EU stories are actually
    # written. \beu\b cannot reach inside "euler" or "Europe".
    "eu", "dsa", "dma",
    "european union", "european commission", "european parliament",
    "european council", "brussels", "eurozone", "euro area", "schengen",
    "cjeu", "court of justice", "echr", "european court", "meps", "mep ",
    "article 7", "rule of law", "infringement procedure", "directive",
    "regulation eu", "gdpr", "digital services act", "digital markets act",
    "ai act", "chat control", "chatcontrol", "client-side scanning",
    "data retention", "e-evidence", "eidas", "age verification",
    "encryption ban", "backdoor encryption", "spyware", "pegasus",
    "predator spyware", "press freedom", "media freedom act",
    "academic freedom", "judicial independence", "free movement",
    "asylum", "migration pact", "enlargement", "accession",
    "ecb", "eurostat", "frontex", "europol", "edri", "edps",
    # member states and the ones whose politics move EU-level outcomes
    "austria", "vienna", "germany", "german government", "france",
    "french government", "poland", "hungary", "orban", "orbán",
    "netherlands", "belgium", "italy", "spain", "portugal", "sweden",
    "denmark", "finland", "ireland", "czech", "slovakia", "romania",
    "bulgaria", "greece", "croatia", "slovenia", "baltic", "estonia",
    "latvia", "lithuania", "ukraine accession", "brexit",
)

EUROPE_FEEDS = (
    "Euractiv", "Politico", "netzpolitik", "EDRi", "Verfassungsblog",
    "Electronic Frontier Foundation", "Der Standard", "derstandard",
    "the Guardian", "Le Monde", "Der Spiegel", "Financial Times",
)

EUROPE = Topic(
    key="europe",
    name="Europe, the EU & Liberal Values",
    prompt=prompts.EUROPE,
    system=("You are a precise European affairs and digital-rights analyst. "
            "You report mechanism -- which instrument, which court, which "
            "vote -- and you give the illiberal turn and the pushback against "
            "it equal rigour."),
    priorities_blurb="rights and rule of law in mechanism, political "
                     "direction, sovereignty, civil society",
    every_n_days=3,
    window_days=3,
    lookback_hours=76,
    # YUROP and AskEurope dropped 2026-09-30 on signal/noise review: YUROP is
    # a meme sub (the misspelling is the joke, content is nationalist image
    # macros) and AskEurope is casual "what's your country like" chitchat --
    # neither carries news or analysis. europe and EuropeanUnion are real
    # discussion/news subs and stay.
    subreddits=("europe", "EuropeanUnion"),
    keywords=EUROPE_KEYWORDS,
    feeds=EUROPE_FEEDS,
    # euractiv.com/feed/ 302s to a consent wall -- dropped. derStandard is
    # German-language and that is on purpose: Austrian domestic politics is
    # where "risks to my existence and freedom" actually lands for him, and
    # the summarizer is told to read non-English items and write in English.
    feed_urls=(
        "https://edri.org/feed/",
        "https://netzpolitik.org/feed/",
        "https://verfassungsblog.de/feed/",
        "https://www.politico.eu/feed/",
        "https://www.derstandard.at/rss/inland",
        "https://www.derstandard.at/rss/international",
    ),
    cost_ceiling_usd=0.15,
    link_fetch_max_pages=25,
)


def all_topics() -> Tuple[Topic, ...]:
    """Every configured topic, AI first.

    A function rather than a module constant because the AI topic reads its
    values out of config.py at call time -- so a test that monkeypatches a
    config knob sees the effect, instead of whatever was true at import.
    """
    return (_ai_topic(), GEOPOLITICS, PANDEMIC, EUROPE)


def topic(key: str) -> Topic:
    for t in all_topics():
        if t.key == key:
            return t
    raise KeyError(f"no topic {key!r}; have {[t.key for t in all_topics()]}")


def all_subreddits() -> List[str]:
    """Union of every topic's subreddits, in topic order, deduplicated.

    This is what the reddit leg scrapes. Each topic still gets only its own,
    and the per-subreddit quota is per topic -- so adding a topic never shrinks
    another topic's reddit material.
    """
    seen: Dict[str, None] = {}
    for t in all_topics():
        for s in t.subreddits:
            seen.setdefault(s, None)
    return list(seen)


def all_feed_urls() -> List[str]:
    seen: Dict[str, None] = {}
    for t in all_topics():
        for u in t.feed_urls:
            seen.setdefault(u, None)
    return list(seen)


# =============================================================================
# "every n days": which topics are due
# =============================================================================

def load_state(path: str = STATE_PATH) -> Dict[str, Any]:
    """Per-topic run history. A corrupt file is replaced, never fatal."""
    f = Path(path)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"[topics] {path} is unreadable -- starting fresh")
        return {}


def save_state(state: Dict[str, Any], path: str = STATE_PATH) -> None:
    f = Path(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=2, sort_keys=True))


def record_run(state: Dict[str, Any], t: Topic, ok: bool,
               now: datetime = None) -> Dict[str, Any]:
    """Note that a topic ran. Only a SUCCESS moves `last_success`.

    That asymmetry is the point: a topic whose model call failed must be due
    again tomorrow, not silently benched for another every_n_days.
    """
    now = now or datetime.now(timezone.utc)
    entry = dict(state.get(t.key) or {})
    entry["last_attempt"] = now.isoformat()
    entry["attempts"] = (entry.get("attempts") or 0) + 1
    if ok:
        entry["last_success"] = now.isoformat()
        entry["successes"] = (entry.get("successes") or 0) + 1
    state[t.key] = entry
    return state


def due(t: Topic, state: Dict[str, Any], now: datetime = None) -> Tuple[bool, str]:
    """(is it due, why). The `why` is logged and shown on the status page.

    Compared against `last_success`, not `last_attempt`, for the reason above.
    A topic with every_n_days=1 is always due, which is how the AI digest keeps
    running nightly with no state at all.
    """
    now = now or datetime.now(timezone.utc)
    if t.every_n_days <= 1:
        return True, "runs every night"
    last = (state.get(t.key) or {}).get("last_success")
    if not last:
        return True, "never run before"
    try:
        when = datetime.fromisoformat(last)
    except ValueError:
        return True, f"unparseable last_success {last!r} -- running"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = now - when
    # Fire a few hours early rather than late: a cron that drifts by a minute
    # must not push an every-3-days topic onto a 4-day cadence forever.
    if age >= timedelta(days=t.every_n_days) - timedelta(hours=6):
        return True, f"last ran {age.days}d{age.seconds // 3600}h ago"
    remaining = timedelta(days=t.every_n_days) - age
    return False, (f"ran {age.days}d{age.seconds // 3600}h ago, next in "
                   f"~{remaining.days}d{remaining.seconds // 3600}h")
