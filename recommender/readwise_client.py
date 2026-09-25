#!/usr/bin/env python3
"""Readwise Reader API client.

Only the parts the shortlist recommender needs: read documents, read/write tags.

Rate limits are per-endpoint and low (LIST and BULK_UPDATE are 20/min, UPDATE is
50/min), so every call goes through a per-endpoint minimum interval. A 429 is
still handled -- the interval keeps us under the limit, the 429 handler is there
for when Readwise disagrees about what the limit is.
"""

import os
import time

import requests

BASE = "https://readwise.io/api/v3"
AUTH_CHECK = "https://readwise.io/api/v2/auth/"

# Seconds between calls to each endpoint, from the documented per-minute limits
# plus headroom. LIST/BULK are 20/min -> 3s; UPDATE is 50/min -> 1.2s.
_MIN_INTERVAL = {"list": 3.2, "update": 1.3, "bulk_update": 3.2, "tags": 3.2}

# Readwise caps a LIST page at 100 documents.
PAGE_LIMIT = 100


class ReadwiseError(RuntimeError):
    pass


class ReadwiseClient:
    def __init__(self, token=None, session=None, sleep=time.sleep):
        self.token = token or os.environ["READWISE_API_KEY"]
        self.session = session or requests.Session()
        self._sleep = sleep
        self._last_call = {}

    # -- plumbing ------------------------------------------------------------

    def _throttle(self, endpoint):
        gap = _MIN_INTERVAL[endpoint]
        last = self._last_call.get(endpoint)
        if last is not None:
            wait = gap - (time.monotonic() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_call[endpoint] = time.monotonic()

    def _request(self, method, url, endpoint, **kwargs):
        headers = {"Authorization": f"Token {self.token}"}
        for attempt in range(5):
            self._throttle(endpoint)
            response = self.session.request(
                method, url, headers=headers, timeout=30, **kwargs
            )
            if response.status_code == 429:
                self._sleep(int(response.headers.get("Retry-After", 60)) + 1)
                continue
            if response.status_code >= 400:
                raise ReadwiseError(
                    f"{method} {url} -> {response.status_code}: {response.text[:300]}"
                )
            return response
        raise ReadwiseError(f"{method} {url} still rate-limited after 5 attempts")

    # -- reads ---------------------------------------------------------------

    def check_auth(self):
        """204 means the token is good. Anything else raises."""
        headers = {"Authorization": f"Token {self.token}"}
        response = self.session.get(AUTH_CHECK, headers=headers, timeout=30)
        if response.status_code != 204:
            raise ReadwiseError(f"auth check returned {response.status_code}")
        return True

    def documents(self, location=None, updated_after=None, tag=None, category=None):
        """Yield documents, following the cursor until the last page."""
        cursor = None
        while True:
            params = {}
            if location:
                params["location"] = location
            if updated_after:
                params["updatedAfter"] = updated_after
            if tag:
                params["tag"] = tag
            if category:
                params["category"] = category
            if cursor:
                params["pageCursor"] = cursor
            body = self._request("GET", f"{BASE}/list/", "list", params=params).json()
            yield from body.get("results", [])
            cursor = body.get("nextPageCursor")
            if not cursor:
                return

    def tags(self):
        return self._request("GET", f"{BASE}/tags/", "tags").json().get("results", [])

    # -- writes --------------------------------------------------------------

    def set_tags(self, doc_id, tags):
        """Replace a document's tags. Readwise treats `tags` as the full set."""
        return self._request(
            "PATCH",
            f"{BASE}/update/{doc_id}/",
            "update",
            json={"tags": list(tags)},
        ).json()

    def bulk_set_tags(self, doc_tags):
        """doc_tags: {doc_id: [tag, ...]}. Chunked to the documented 50/request."""
        items = list(doc_tags.items())
        results = []
        for start in range(0, len(items), 50):
            updates = [
                {"id": doc_id, "tags": list(tags)}
                for doc_id, tags in items[start : start + 50]
            ]
            response = self._request(
                "PATCH",
                f"{BASE}/bulk_update/",
                "bulk_update",
                json={"updates": updates},
            )
            results.append(response.json() if response.content else {})
        return results


def tag_names(document):
    """Reader returns tags as {name: {...}} on reads but takes a list on writes."""
    tags = document.get("tags") or {}
    return sorted(tags.keys() if isinstance(tags, dict) else tags)
