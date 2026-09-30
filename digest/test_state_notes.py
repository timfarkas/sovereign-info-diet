#!/usr/bin/env python3
"""Tests for the cross-run scratchpad notes mechanism.

Behaviour and contracts: append/load round-tripping, the 5-entry FIFO cap,
and trailer extraction -- including the AI digest's case, which never emits
a trailer and must round-trip untouched.
"""

from datetime import datetime

import state_notes


def test_load_with_no_prior_file_returns_the_placeholder(tmp_path, monkeypatch):
    monkeypatch.setattr(state_notes, "NOTES_DIR", tmp_path)
    assert "no notes from prior runs" in state_notes.load("pandemic")


def test_append_then_load_round_trips_the_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(state_notes, "NOTES_DIR", tmp_path)
    state_notes.append("pandemic", "Covered H5N1 spillover in region X.",
                        now=datetime(2026, 9, 30))
    out = state_notes.load("pandemic")
    assert "## 2026-09-30" in out
    assert "Covered H5N1 spillover in region X." in out


def test_append_keeps_only_the_last_five_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(state_notes, "NOTES_DIR", tmp_path)
    for day in range(1, 8):
        state_notes.append("pandemic", f"run {day}",
                            now=datetime(2026, 9, day))
    out = state_notes.load("pandemic")
    assert out.count("## ") == 5
    assert "run 1" not in out and "run 2" not in out          # evicted
    assert "run 7" in out and "run 3" in out                  # kept


def test_append_with_blank_entry_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(state_notes, "NOTES_DIR", tmp_path)
    state_notes.append("pandemic", "   ", now=datetime(2026, 9, 30))
    assert not (tmp_path / "pandemic.md").exists()


def test_topics_get_separate_note_files(tmp_path, monkeypatch):
    monkeypatch.setattr(state_notes, "NOTES_DIR", tmp_path)
    state_notes.append("pandemic", "bio note", now=datetime(2026, 9, 30))
    state_notes.append("europe", "eu note", now=datetime(2026, 9, 30))
    assert "bio note" in state_notes.load("pandemic")
    assert "bio note" not in state_notes.load("europe")
    assert "eu note" in state_notes.load("europe")


def test_extract_splits_trailer_from_the_mailed_html():
    html = ('<h3>Signals</h3><ul><li>thing</li></ul>'
            '<!-- STATE-NOTES-START -->\n'
            '<p>Covered <b>H5N1</b> spillover.</p>\n'
            '<!-- STATE-NOTES-END -->')
    clean, notes = state_notes.extract(html)
    assert "STATE-NOTES" not in clean
    assert "<h3>Signals</h3><ul><li>thing</li></ul>" in clean
    assert notes == "Covered H5N1 spillover."


def test_extract_with_no_trailer_round_trips_unchanged():
    """The AI digest's template never asks for a trailer -- this is the whole
    mechanism by which it is unaffected without any per-topic branching."""
    html = "<h3>AI News</h3><ul><li>thing</li></ul>"
    clean, notes = state_notes.extract(html)
    assert clean == html
    assert notes == ""
