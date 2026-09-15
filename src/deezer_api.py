"""Deezer API client with rate limiting, retry logic, and thread safety.

No authentication required — Deezer search and track endpoints are public.
"""

import re
import time
import logging
import threading
import requests

from config import DEEZER_API_BASE, DEEZER_RATE_LIMIT, DEEZER_MAX_RETRIES
from src.lastfm_api import TokenBucket

logger = logging.getLogger(__name__)


class DeezerRetryableError(Exception):
    """Raised when retries are exhausted due to rate limiting or server errors.

    The request should be retried later — do NOT treat as 'not found'.
    """
    pass


def _normalize_for_search(text):
    """Normalize artist/track name for better Deezer matching.

    Strips parenthetical info like (feat. X), (Remix), (Deluxe), etc.
    Normalizes common abbreviations.
    """
    # Remove parenthetical suffixes: (feat. ...), (Remix), (Live), etc.
    text = re.sub(r"\s*\(.*?\)\s*", " ", text)
    # Remove bracket suffixes: [feat. ...], [Remix], etc.
    text = re.sub(r"\s*\[.*?\]\s*", " ", text)
    # Normalize "feat."/"ft." variations
    text = re.sub(r"\s+(feat\.?|ft\.?|featuring)\s+.*", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DeezerClient:
    """Deezer API client with built-in rate limiting, retry, and thread safety.

    Uses thread-local HTTP sessions so multiple threads can make concurrent
    requests while sharing a single token bucket rate limiter. When any thread
    hits a rate limit (Deezer code 4 or HTTP 429), a global backoff pauses
    all threads to avoid wasting requests on doomed retries.
    """

    def __init__(self):
        self.base_url = DEEZER_API_BASE
        self.bucket = TokenBucket(DEEZER_RATE_LIMIT, max_burst=1)
        self._local = threading.local()
        self._request_count = 0
        self._count_lock = threading.Lock()
        self._backoff_until = 0.0
        self._backoff_lock = threading.Lock()

    def _get_session(self):
        """Get or create a thread-local requests session."""
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                "User-Agent": "MelodyMatcher/1.0 (music research project)"
            })
        return self._local.session

    @property
    def request_count(self):
        with self._count_lock:
            return self._request_count

    def _increment_count(self):
        with self._count_lock:
            self._request_count += 1

    def _set_backoff(self, seconds):
        """Set global backoff — all threads pause before their next request."""
        with self._backoff_lock:
            target = time.monotonic() + seconds
            if target > self._backoff_until:
                self._backoff_until = target

    def _wait_for_backoff(self):
        """Sleep until any global backoff has elapsed."""
        with self._backoff_lock:
            target = self._backoff_until
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _api_request(self, endpoint, params=None):
        """Make a rate-limited API request with retry logic.

        Returns response data dict, or None for genuine 'no results'.
        Raises DeezerRetryableError if retries are exhausted due to rate
        limiting or server errors — the caller should NOT treat this as
        'not found'.
        """
        url = f"{self.base_url}/{endpoint}"
        session = self._get_session()
        retryable_failure = False

        for attempt in range(DEEZER_MAX_RETRIES + 1):
            self._wait_for_backoff()
            self.bucket.wait()
            timeout = 10 * (1 + attempt)

            try:
                self._increment_count()
                resp = session.get(url, params=params, timeout=timeout)

                if resp.status_code == 200:
                    data = resp.json()
                    # Deezer returns errors inside 200 responses
                    if "error" in data:
                        error = data["error"]
                        code = error.get("code", 0)
                        msg = error.get("message", "Unknown error")
                        # Code 4 = quota exceeded — retryable
                        if code == 4:
                            retryable_failure = True
                            wait = 2 ** (attempt + 1)
                            self._set_backoff(30)
                            logger.warning(
                                "Deezer quota exceeded, retry %d/%d in %ds "
                                "(global backoff 30s)",
                                attempt + 1, DEEZER_MAX_RETRIES, wait,
                            )
                            time.sleep(wait)
                            continue
                        logger.debug("Deezer API error %d: %s", code, msg)
                        return None
                    return data

                if resp.status_code == 429 or resp.status_code >= 500:
                    retryable_failure = True
                    wait = 2 ** (attempt + 1)
                    if resp.status_code == 429:
                        self._set_backoff(30)
                    logger.warning(
                        "HTTP %d, retry %d/%d in %ds",
                        resp.status_code, attempt + 1, DEEZER_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue

                logger.warning("Unexpected HTTP %d from Deezer", resp.status_code)
                return None

            except requests.exceptions.Timeout:
                retryable_failure = True
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Timeout, retry %d/%d in %ds",
                    attempt + 1, DEEZER_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            except requests.exceptions.ConnectionError:
                retryable_failure = True
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Connection error, retry %d/%d in %ds",
                    attempt + 1, DEEZER_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            except requests.exceptions.RequestException as e:
                logger.error("Request failed: %s", e)
                return None

        # All retries exhausted
        if retryable_failure:
            raise DeezerRetryableError(
                f"All {DEEZER_MAX_RETRIES} retries exhausted for {endpoint} "
                f"(rate limit or server error)"
            )
        logger.error("All retries exhausted for Deezer %s", endpoint)
        return None

    def search_track(self, artist_name, track_name):
        """Search for a track on Deezer.

        Returns dict with {deezer_id, title, artist_name, preview_url} or None.
        Raises DeezerRetryableError if the search failed due to rate limiting.
        """
        artist_clean = _normalize_for_search(artist_name)
        track_clean = _normalize_for_search(track_name)
        query = f'artist:"{artist_clean}" track:"{track_clean}"'

        data = self._api_request("search", params={"q": query})
        if not data or "data" not in data or not data["data"]:
            return None

        result = data["data"][0]
        preview_url = result.get("preview")

        return {
            "deezer_id": str(result["id"]),
            "title": result.get("title", ""),
            "artist_name": result.get("artist", {}).get("name", ""),
            "preview_url": preview_url if preview_url else None,
        }

    def get_fresh_preview_url(self, deezer_id):
        """Fetch a fresh (non-expired) preview URL for a track by its Deezer ID.

        The URLs stored from search results contain short-lived HMAC tokens
        that expire within days. This fetches a current one via the track endpoint.

        Returns preview URL string, or None if unavailable.
        Raises DeezerRetryableError on rate-limit / server failure.
        """
        data = self._api_request(f"track/{deezer_id}")
        if not data:
            return None
        url = data.get("preview")
        return url if url else None

    def download_preview(self, preview_url):
        """Download a preview MP3 from Deezer CDN. Returns raw bytes or None."""
        try:
            resp = requests.get(
                preview_url,
                timeout=15,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://www.deezer.com/",
                    "Accept": "audio/mpeg, audio/*, */*",
                },
            )
            if resp.status_code == 200:
                return resp.content
            logger.warning("Preview download failed: HTTP %d", resp.status_code)
            return None
        except requests.exceptions.RequestException as e:
            logger.warning("Preview download error: %s", e)
            return None
