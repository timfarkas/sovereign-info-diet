"""Research digest: config validation, cadence/expiry, link checking, the X tool's
budget, and the agent loop against a fake OpenAI client (no network)."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace as NS

import pytest

import config
import research_digest as rd

ENTRY = {"slug": "foo", "name": "Foo Lab", "question": "what is foo up to?",
         "every_n_days": 14, "keywords": ["foo"]}


def write(tmp_path, entries):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(entries))
    return p


def test_valid_config_loads(tmp_path):
    assert rd.load_research_topics(write(tmp_path, [ENTRY]))[0]["slug"] == "foo"


@pytest.mark.parametrize("bad", [
    {k: v for k, v in ENTRY.items() if k != "keywords"},     # missing field
    {**ENTRY, "keywords": []},                                # nothing narrow
    {**ENTRY, "typo_field": 1},                               # unknown key
    {**ENTRY, "expires": "next week"},                        # bad date
    {**ENTRY, "slug": "Bad Slug"},
])
def test_bad_config_fails_loudly(tmp_path, bad):
    with pytest.raises((AssertionError, ValueError)):
        rd.load_research_topics(write(tmp_path, [bad]))


def test_missing_config_exits(tmp_path):
    with pytest.raises(SystemExit):
        rd.load_research_topics(tmp_path / "nope.json")


def test_expiry_is_inclusive_of_the_last_day():
    e = {**ENTRY, "expires": "2026-12-01"}
    assert not rd.expired(e, date(2026, 12, 1))
    assert rd.expired(e, date(2026, 12, 2))
    assert not rd.expired(ENTRY, date(2099, 1, 1))


def test_subject_is_a_digest_series_the_recommender_recognises():
    import re
    s = rd.subject_for(ENTRY, date(2026, 11, 17))
    assert re.sub(r"\s*-\s*\d{4}-\d{2}-\d{2}\s*$", "", s) == "Foo Lab Digest"


def test_cadence_is_from_last_success_only():
    t = rd.as_topic(ENTRY)
    now = datetime(2026, 11, 17, tzinfo=timezone.utc)
    assert rd.topics.due(t, {}, now)[0]
    ok = rd.topics.record_run({}, t, ok=True, now=datetime(2026, 11, 10, tzinfo=timezone.utc))
    assert not rd.topics.due(t, ok, now)[0]
    failed = rd.topics.record_run({}, t, ok=False, now=datetime(2026, 11, 16, tzinfo=timezone.utc))
    assert rd.topics.due(t, failed, now)[0]


def test_only_tool_returned_links_survive():
    page = ('<p><a href="https://a.org/x?utm=1">ok</a> <a href="https://made.up/y">fake</a></p>')
    out, n = rd.unlink_unverified(page, {rd.clean_url("https://a.org/x")})
    assert 'href="https://a.org/x?utm=1"' in out and "made.up" not in out
    assert n == 1 and ">fake<" not in out and "fake" in out


class FakeX:
    def __init__(self):
        self.queries = []

    def get(self, path, **kw):
        self.queries.append(kw["query"])
        return {"tweets": [{"id": "1", "text": "hi", "author": {"userName": "u"},
                            "entities": {"urls": [{"expanded_url": "https://a.org/p"}]}}]}


def test_x_search_budget_and_window():
    x = rd.XSearch(FakeX(), date(2026, 11, 3), max_calls=1)
    first = json.loads(x("mindspan", "Latest"))
    assert first[0]["handle"] == "u" and "since:2026-11-03" in x.client.queries[0]
    assert rd.clean_url("https://a.org/p") in x.links
    assert "budget" in json.loads(x("again", "Top"))["error"]
    assert len(x.client.queries) == 1


def fake_response(output, text=""):
    u = NS(input_tokens=1000, output_tokens=100, input_tokens_details=NS(cached_tokens=0))
    return NS(output=output, output_text=text, usage=u)


def test_agent_loop_runs_tool_then_returns_digest(monkeypatch):
    call = NS(type="function_call", call_id="c1", name="x_search",
              arguments=json.dumps({"query": "foo", "sort": "Top"}))
    msg = NS(type="message", content=[NS(annotations=[NS(type="url_citation",
                                                          url="https://a.org/p")])])
    src = NS(type="web_search_call", action=NS(sources=[{"url": "https://b.org/q"}]))
    replies = iter([fake_response([call]), fake_response([src, msg], "<p>done</p>")])
    monkeypatch.setattr(rd, "_respond", lambda client, **kw: next(replies))
    x = rd.XSearch(FakeX(), date(2026, 11, 3), 5)
    res = rd.run_agent(None, "prompt", x)
    assert res["html"] == "<p>done</p>" and res["rounds"] == 2 and not res["truncated"]
    assert x.calls == 1 and rd.clean_url("https://a.org/p") in res["links"]
    assert rd.clean_url("https://b.org/q") in res["links"]      # dict-shaped source


def test_agent_loop_that_never_stops_is_cut_off_and_flagged(monkeypatch):
    call = NS(type="function_call", call_id="c", name="x_search",
              arguments=json.dumps({"query": "foo", "sort": "Top"}))
    monkeypatch.setattr(rd, "_respond", lambda client, **kw: fake_response(
        [] if "tools" not in kw else [call], "<p>forced</p>"))
    monkeypatch.setattr(config, "RESEARCH_MAX_TOOL_ROUNDS", 3)
    res = rd.run_agent(None, "p", rd.XSearch(FakeX(), date(2026, 11, 3), 99))
    assert res["truncated"] and res["html"] == "<p>forced</p>"
