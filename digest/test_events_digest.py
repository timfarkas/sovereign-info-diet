"""events_digest.py: cadence, cleaning, rendering, and the run's failure modes.

The model is replaced by a fake client that answers research calls with canned
events per category and the scoring call with a fixed score per id, so the run
is exercised end to end without the network.
"""

import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import events_digest as ev

EXAMPLE = Path(__file__).resolve().parent.parent / "events_config.example.json"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    """Every test runs on the committed example, never on the private file."""
    monkeypatch.setattr(config, "EVENTS_CONFIG_PATH", EXAMPLE)
    ev.SETTINGS.clear()
    ev.SETTINGS.update(ev.load_settings(EXAMPLE))


# -- cadence --------------------------------------------------------------------

SAT, SUN, MON = date(2026, 10, 31), date(2026, 11, 1), date(2026, 11, 2)


def test_due_on_sunday_until_sent_that_sunday():
    assert ev.due("2026-10-25", SUN)[0]
    assert not ev.due("2026-11-01", SUN)[0]


def test_a_failed_sunday_is_retried_on_monday_not_skipped_a_week():
    assert ev.due("2026-10-25", MON)[0]
    assert not ev.due("2026-11-01", MON)[0]


def test_not_due_midweek_once_this_weeks_digest_went_out():
    assert not ev.due("2026-10-25", SAT)[0]
    assert ev.due(None, SAT)[0]          # never sent: the first run goes out


def test_window_is_seven_days_inclusive():
    assert ev.window(SUN) == (SUN, date(2026, 11, 7))


# -- cleaning -------------------------------------------------------------------

def event(title="Talk", start="2026-11-03T19:00", url="https://luma.com/abc", **kw):
    return {"title": title, "start": start, "end": None, "venue": "c-base",
            "organizer": None, "url": url, "description": "A talk.",
            "category": "AI safety, EA & rationalism", **kw}


def test_keep_drops_out_of_window_bad_dates_social_links_and_duplicates():
    raw = [event(),
           event(url="https://luma.com/abc?utm_source=x#top"),   # same, tracked
           event(url="https://meetup.com/other"),                # same title+day
           event("Late", start="2026-11-09"),
           event("Early", start="2026-10-30T10:00"),
           event("Vague", start="sometime in November"),
           event("Tweet", url="https://x.com/someone/status/1"),
           event("Thread", url="https://old.reddit.com/r/somecity/1"),
           event("Last day", start="2026-11-07", url="https://luma.com/last")]
    kept, dropped = ev.keep(raw, SUN, date(2026, 11, 7))
    assert [e["title"] for e in kept] == ["Talk", "Last day"]
    assert [e["id"] for e in kept] == [0, 1]
    assert dropped == {"bad date": 1, "outside window": 2, "blocked link": 2,
                       "duplicate": 2}


# -- rendering ------------------------------------------------------------------

def scored(i, s, **kw):
    e = event(f"Event {i}", start=f"2026-11-0{2 + i % 5}T19:00",
              url=f"https://example.org/{i}", **kw)
    return {**e, "id": i, "when": datetime.fromisoformat(e["start"]),
            "score": s, "why": f"why {i}"}


def test_render_splits_top_picks_from_the_rest_and_hides_low_scores():
    events = [scored(0, 9), scored(1, config.EVENTS_TOP_SCORE),
              scored(2, config.EVENTS_MIN_SCORE), scored(3, config.EVENTS_MIN_SCORE - 1)]
    page = ev.render(events, SUN, date(2026, 11, 7), footer="f")
    top, rest = page.split("<h3>Also this week</h3>")
    assert "Event 0" in top and "Event 1" in top and "why 0" in top
    assert "Event 2" in rest and "A talk." in rest
    assert "<strong>Mon 2 Nov, 19:00</strong>" in top        # top picks carry the day
    assert "<strong>19:00</strong>" in rest                   # the day is the heading
    assert "Event 3" not in page


def test_top_picks_are_capped_and_the_overflow_still_appears():
    events = [scored(i, 9) for i in range(config.EVENTS_TOP_PICKS + 2)]
    top, rest = ev.render(events, SUN, date(2026, 11, 7), "f").split(
        "<h3>Also this week</h3>")
    assert top.count("<li>") == config.EVENTS_TOP_PICKS
    assert rest.count("<li>") == 2


def test_third_party_text_is_escaped():
    e = scored(0, 9, venue="<script>x</script>")
    e["title"] = 'Hack "night" <b>'
    page = ev.render([e], SUN, date(2026, 11, 7), "f")
    assert "<script>" not in page and "<b>" not in page
    assert "&lt;script&gt;" in page


# -- the run --------------------------------------------------------------------

class FakeClient:
    """Answers research calls with `by_category[name]`, scoring with `score`.

    A category mapped to an Exception raises it, like a failed API call.
    """

    def __init__(self, by_category, score=8, skip_ids=()):
        self.by_category, self.score_value, self.skip_ids = by_category, score, skip_ids
        self.responses = SimpleNamespace(create=self.create)

    def create(self, *, input, tools=None, **kw):
        usage = SimpleNamespace(input_tokens=1000, output_tokens=100,
                                input_tokens_details=SimpleNamespace(cached_tokens=0))
        if tools:
            name = next(c["name"] for c in ev.SETTINGS["categories"]
                        if f"category: {c['name']}." in input)
            found = self.by_category.get(name, [])
            if isinstance(found, Exception):
                raise found
            out = {"events": found}
            output = [SimpleNamespace(type="web_search_call")] * 3
        else:
            ids = [json.loads(line)["id"] for line in input.splitlines()
                   if line.startswith('{"id"')]
            out = {"scores": [{"id": i, "score": self.score_value, "why": "fits"}
                              for i in ids if i not in self.skip_ids]}
            output = []
        return SimpleNamespace(output_text=json.dumps(out), usage=usage,
                               output=output)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """main() in a scratch dir, on Sunday 2026-11-01, with mail captured."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ev, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(ev, "today_local", lambda now: SUN)
    monkeypatch.setenv("EMAIL_TO", "tim@example.org")
    monkeypatch.setenv("EMAIL_TO_EXTRA", "friend@example.org")
    sent = []

    import send_notification
    monkeypatch.setattr(send_notification, "send_email",
                        lambda subject, body, to_email: sent.append(
                            (subject, body, to_email)) or True)

    def go(client, *argv):
        monkeypatch.setattr(ev, "_client", lambda: client)
        return ev.main(list(argv))
    go.sent = sent
    return go


CATS = [c["name"] for c in ev.load_settings(EXAMPLE)["categories"]]


def test_happy_path_mails_tim_alone_under_a_digest_series_subject(run):
    client = FakeClient({CATS[0]: [event("A", category=None)],
                         CATS[1]: [event("B", url="https://luma.com/b")]})
    assert run(client) == 0
    (subject, body, to), = run.sent
    assert subject == "Amsterdam Events Digest - 2026-11-01"
    assert to == ["tim@example.org"]          # never EMAIL_TO_EXTRA
    assert "Top picks" in body and ">A</a>" in body and ">B</a>" in body
    assert json.loads(ev.STATE_PATH.read_text()) == {"last_success": "2026-11-01"}
    assert not ev.due("2026-11-01", SUN)[0]   # so tomorrow's 01:00 run skips


def test_not_due_means_no_model_call_at_all(run):
    ev.STATE_PATH.write_text(json.dumps({"last_success": "2026-11-01"}))
    assert run(None) == 0                     # None client: any call would crash
    assert run.sent == []


def test_dry_run_saves_the_page_but_mails_nothing_and_stays_due(run):
    assert run(FakeClient({CATS[0]: [event()]}), "--dry-run") == 0
    assert run.sent == []
    assert not ev.STATE_PATH.exists()
    assert list(ev.OUT_DIR.glob("summary_events_*.html"))


def test_one_failed_category_is_flagged_in_the_mail_not_fatal(run):
    client = FakeClient({CATS[0]: RuntimeError("search 500"),
                         CATS[1]: [event()]})
    assert run(client) == 0
    body = run.sent[0][1]
    assert "Research failed for" in body and CATS[0].split(" ")[0] in body


def test_nothing_found_refuses_to_mail_and_stays_due(run):
    client = FakeClient({c: RuntimeError("down") for c in CATS})
    assert run(client) == 1
    assert run.sent == [] and not ev.STATE_PATH.exists()


def test_events_the_scorer_skipped_are_dropped_not_shown_unscored(run):
    client = FakeClient({CATS[0]: [event("A"), event("B", url="https://luma.com/b")]},
                        skip_ids={1})
    assert run(client) == 0
    body = run.sent[0][1]
    assert ">A</a>" in body and ">B</a>" not in body


def test_a_failed_send_leaves_it_due(run, monkeypatch):
    import send_notification
    monkeypatch.setattr(send_notification, "send_email", lambda *a, **k: False)
    assert run(FakeClient({CATS[0]: [event()]})) == 1
    assert not ev.STATE_PATH.exists()


def test_a_missing_private_config_is_fatal_not_a_fallback(run, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVENTS_CONFIG_PATH", tmp_path / "nope.json")
    with pytest.raises(SystemExit, match="events_config.example.json"):
        run(None)
    assert run.sent == []


def test_a_category_without_interests_is_rejected(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"city": "X", "timezone": "Europe/Berlin",
                               "categories": [{"name": "a", "hints": ""}]}))
    with pytest.raises(AssertionError, match="interests"):
        ev.load_settings(bad)


def test_the_private_config_is_gitignored_and_the_example_is_not():
    import subprocess
    root = EXAMPLE.parent
    check = lambda name: subprocess.run(["git", "check-ignore", "-q", name],
                                        cwd=root).returncode == 0
    assert check("events_config.json") and not check("events_config.example.json")
