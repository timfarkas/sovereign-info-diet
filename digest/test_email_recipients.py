#!/usr/bin/env python3
"""Tests for the multi-recipient mail flow: who a run mails, and why.

Behaviour and contracts, not implementation: nothing here reaches into the
`mail` subprocess call except to check the recipient list it was handed.

Recipients come from .env (EMAIL_TO, EMAIL_TO_EXTRA), never from committed
code -- this is a public repo, and a real address in config.py is committed
for anyone to read the moment it lands.
"""

import re
import subprocess
from pathlib import Path

import pytest

import digest_run
import send_notification


@pytest.fixture(autouse=True)
def email_env(monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "tim@example.com")
    monkeypatch.delenv("EMAIL_TO_EXTRA", raising=False)


class TestNoAddressInCode:
    """Regression guard for the thing this file is actually about."""

    def test_config_has_no_email_address(self):
        text = Path(__file__).with_name("config.py").read_text()
        assert not re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text)


class TestRecipientsFor:
    def test_full_run_is_primary_plus_env_extras(self, monkeypatch):
        monkeypatch.setenv("EMAIL_TO_EXTRA", "a@example.com, b@example.com")
        assert send_notification.recipients_for(full=True) == [
            "tim@example.com", "a@example.com", "b@example.com"]

    def test_full_run_with_no_extras_set_is_primary_only(self):
        assert send_notification.recipients_for(full=True) == ["tim@example.com"]

    def test_test_run_ignores_extras(self, monkeypatch):
        monkeypatch.setenv("EMAIL_TO_EXTRA", "a@example.com")
        assert send_notification.recipients_for(full=False) == ["tim@example.com"]

    def test_missing_email_to_raises_either_way(self, monkeypatch):
        monkeypatch.delenv("EMAIL_TO", raising=False)
        with pytest.raises(ValueError):
            send_notification.recipients_for(full=True)
        with pytest.raises(ValueError):
            send_notification.recipients_for(full=False)


class TestSendEmailRecipients:
    """send_email must hand the mail command every recipient it was given."""

    def _fake_popen(self, monkeypatch, captured):
        class FakeProcess:
            returncode = 0
            def communicate(self, input=None):
                return ("", "")

        def fake(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeProcess()

        monkeypatch.setattr(subprocess, "Popen", fake)

    def test_mails_every_recipient_in_a_list(self, monkeypatch):
        captured = {}
        self._fake_popen(monkeypatch, captured)
        ok = send_notification.send_email(
            "subject", "<p>body</p>",
            to_email=["a@example.com", "b@example.com"])
        assert ok is True
        assert captured["cmd"][-2:] == ["a@example.com", "b@example.com"]

    def test_string_recipient_still_works(self, monkeypatch):
        captured = {}
        self._fake_popen(monkeypatch, captured)
        send_notification.send_email("s", "b", to_email="solo@example.com")
        assert captured["cmd"][-1] == "solo@example.com"

    def test_falls_back_to_email_to_env_var(self, monkeypatch):
        captured = {}
        self._fake_popen(monkeypatch, captured)
        send_notification.send_email("s", "b")
        assert captured["cmd"][-1] == "tim@example.com"

    def test_empty_recipient_list_raises_when_no_fallback(self, monkeypatch):
        monkeypatch.delenv("EMAIL_TO", raising=False)
        with pytest.raises(ValueError):
            send_notification.send_email("s", "b", to_email=[])


class TestMailMode:
    """digest_run.mail_mode: the three run tiers, checked in priority order."""

    def test_default_is_a_full_mailed_run(self):
        assert digest_run.mail_mode([]) == (True, True)

    def test_test_flag_mails_but_not_full(self):
        assert digest_run.mail_mode(["--test"]) == (True, False)

    def test_dry_run_overrides_test(self):
        assert digest_run.mail_mode(["--dry-run", "--test"]) == (False, False)

    def test_no_mail_sends_nothing_regardless_of_test(self):
        assert digest_run.mail_mode(["--no-mail", "--test"]) == (False, False)
