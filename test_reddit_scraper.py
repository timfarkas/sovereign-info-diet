#!/usr/bin/env python3

import pytest
from unittest.mock import MagicMock

import reddit_scraper


class FakeComments(list):
    def replace_more(self, limit=0):
        pass


class FakeSubmission:
    """Duck-types just what process_post touches on a praw Submission."""
    def __init__(self, media_metadata):
        self.id = "abc123"
        self.title = "a gallery post"
        self.score = 42
        self.selftext = None
        self.is_self = True
        self.url = "https://reddit.com/r/test/comments/abc123"
        self.subreddit = MagicMock(display_name="test")
        self.is_gallery = True
        self.media_metadata = media_metadata
        self.comments = FakeComments()


@pytest.fixture
def scraper(monkeypatch):
    monkeypatch.setattr(reddit_scraper.praw, "Reddit", lambda **kw: MagicMock())
    return reddit_scraper.RedditScraper()


def test_gallery_with_a_gif_item_does_not_crash(scraper, monkeypatch):
    # a gallery mixing a static image with an animated gif/video: the gif's 's'
    # preview dict has no 'u' key, only y/x/gif/mp4 -- this used to raise KeyError
    # and take the whole scrape (all subreddits already processed) down with it
    monkeypatch.setattr(scraper, "download_image", lambda url, post_id, idx=0: f"local/{post_id}_{idx}.jpg")
    media_metadata = {
        "img1": {"s": {"u": "https://preview.redd.it/img1.jpg?width=100"}},
        "gif1": {"s": {"y": 480, "x": 640, "gif": "https://preview.redd.it/gif1.gif", "mp4": "https://preview.redd.it/gif1.mp4"}},
    }
    post = scraper.process_post(FakeSubmission(media_metadata), condensed=True)
    assert post["images"] == ["local/abc123_0.jpg"]


def test_gallery_with_only_non_image_items_yields_no_images(scraper, monkeypatch):
    monkeypatch.setattr(scraper, "download_image", lambda url, post_id, idx=0: f"local/{post_id}_{idx}.jpg")
    media_metadata = {
        "gif1": {"s": {"gif": "https://preview.redd.it/gif1.gif"}},
    }
    post = scraper.process_post(FakeSubmission(media_metadata), condensed=True)
    assert "images" not in post
