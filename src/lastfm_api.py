"""Last.fm API client with token bucket rate limiting and retry logic."""

import time
import logging
import threading
import requests

from config import (
    LASTFM_API_KEY,
    LASTFM_API_BASE,
    API_RATE_LIMIT,
    API_MAX_RETRIES,
    API_BASE_TIMEOUT,
)

logger = logging.getLogger(__name__)


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Args:
        rate: Tokens per second (sustained throughput).
        max_burst: Maximum tokens that can accumulate. Defaults to rate.
                   Set to 1 to prevent bursts entirely — requests are
                   strictly spaced at 1/rate intervals.
    """

    def __init__(self, rate, max_burst=None):
        self.rate = rate
        self.max_burst = max_burst if max_burst is not None else rate
        self.tokens = min(1, self.max_burst)  # start with 1 to avoid initial burst
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def wait(self):
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.max_burst, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                sleep_time = (1 - self.tokens) / self.rate
            time.sleep(sleep_time)


class LastFMClient:
    """Last.fm API client with built-in rate limiting and retry."""

    def __init__(self, api_key=None):
        self.api_key = api_key or LASTFM_API_KEY
        if not self.api_key:
            raise ValueError(
                "LASTFM_API_KEY not set. Copy .env.example to .env and fill in your key."
            )
        self.base_url = LASTFM_API_BASE
        self.bucket = TokenBucket(API_RATE_LIMIT)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MelodyMatcher/1.0 (music research project)"
        })
        self._request_count = 0

    @property
    def request_count(self):
        return self._request_count

    def _request(self, params):
        """Make a rate-limited API request with retry logic."""
        params["api_key"] = self.api_key
        params["format"] = "json"

        for attempt in range(API_MAX_RETRIES + 1):
            self.bucket.wait()
            timeout = API_BASE_TIMEOUT * (1 + attempt)

            try:
                self._request_count += 1
                resp = self.session.get(
                    self.base_url, params=params, timeout=timeout
                )

                if resp.status_code == 200:
                    data = resp.json()
                    # Last.fm returns errors inside 200 responses
                    if "error" in data:
                        error_code = data["error"]
                        error_msg = data.get("message", "Unknown error")
                        # 6 = user/artist not found, 17 = login required
                        if error_code in (6, 17):
                            logger.debug("API error %d: %s", error_code, error_msg)
                            return None
                        logger.warning("API error %d: %s", error_code, error_msg)
                        return None
                    return data

                if resp.status_code == 403:
                    logger.debug("403 Forbidden (private profile), skipping")
                    return None

                if resp.status_code == 404:
                    logger.debug("404 Not Found, skipping")
                    return None

                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "HTTP %d, retry %d/%d in %ds",
                        resp.status_code, attempt + 1, API_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue

                # Other errors
                logger.warning("Unexpected HTTP %d", resp.status_code)
                return None

            except requests.exceptions.Timeout:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Timeout, retry %d/%d in %ds",
                    attempt + 1, API_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            except requests.exceptions.ConnectionError:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Connection error, retry %d/%d in %ds",
                    attempt + 1, API_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            except requests.exceptions.RequestException as e:
                logger.error("Request failed: %s", e)
                return None

        logger.error("All retries exhausted for params: %s", params.get("method"))
        return None

    def get_user_info(self, username):
        """Get user info including playcount and registration date.
        Returns dict with 'playcount', 'registered' or None.
        """
        data = self._request({
            "method": "user.getInfo",
            "user": username,
        })
        if data and "user" in data:
            user = data["user"]
            return {
                "playcount": int(user.get("playcount", 0)),
                "registered": user.get("registered", {}).get("unixtime"),
                "name": user.get("name", username),
            }
        return None

    def get_user_friends(self, username, page=1, limit=50):
        """Get a user's friends list. Returns list of usernames or None."""
        data = self._request({
            "method": "user.getFriends",
            "user": username,
            "page": page,
            "limit": limit,
        })
        if data and "friends" in data:
            friends = data["friends"].get("user", [])
            if isinstance(friends, dict):
                friends = [friends]
            return [f["name"] for f in friends if "name" in f]
        return None

    def get_user_top_tracks(self, username, period="overall", page=1, limit=200):
        """Get a user's top tracks. Returns list of track dicts or None.
        Each track has: 'name', 'artist' (name), 'playcount'.
        """
        data = self._request({
            "method": "user.getTopTracks",
            "user": username,
            "period": period,
            "page": page,
            "limit": limit,
        })
        if data and "toptracks" in data:
            tracks = data["toptracks"].get("track", [])
            if isinstance(tracks, dict):
                tracks = [tracks]
            result = []
            for t in tracks:
                artist_name = t.get("artist", {}).get("name", "")
                if isinstance(t.get("artist"), str):
                    artist_name = t["artist"]
                result.append({
                    "name": t.get("name", ""),
                    "artist": artist_name,
                    "playcount": int(t.get("playcount", 0)),
                })
            return result
        return None

    def get_artist_top_tags(self, artist):
        """Get top tags for an artist. Returns list of (tag_name, count) or None."""
        data = self._request({
            "method": "artist.getTopTags",
            "artist": artist,
        })
        if data and "toptags" in data:
            tags = data["toptags"].get("tag", [])
            if isinstance(tags, dict):
                tags = [tags]
            return [
                (t.get("name", "").lower(), int(t.get("count", 0)))
                for t in tags
                if t.get("name")
            ]
        return None

    def get_artist_info(self, artist):
        """Get artist info including listener count. Returns dict or None."""
        data = self._request({
            "method": "artist.getInfo",
            "artist": artist,
        })
        if data and "artist" in data:
            a = data["artist"]
            stats = a.get("stats", {})
            return {
                "name": a.get("name", artist),
                "listeners": int(stats.get("listeners", 0)),
                "playcount": int(stats.get("playcount", 0)),
            }
        return None

    def get_user_recent_tracks(self, username, page=1, limit=200):
        """Get a user's recent scrobbles. Returns (tracks, total_pages) or (None, 0).
        Each track has: 'name', 'artist', 'listened_at' (unix timestamp).
        Skips "now playing" entries (no timestamp).
        """
        data = self._request({
            "method": "user.getRecentTracks",
            "user": username,
            "page": page,
            "limit": limit,
            "extended": 0,
        })
        if data and "recenttracks" in data:
            attr = data["recenttracks"].get("@attr", {})
            total_pages = int(attr.get("totalPages", 0))

            raw_tracks = data["recenttracks"].get("track", [])
            if isinstance(raw_tracks, dict):
                raw_tracks = [raw_tracks]

            result = []
            for t in raw_tracks:
                # Skip "now playing" — these have @attr.nowplaying="true" and no date
                if t.get("@attr", {}).get("nowplaying") == "true":
                    continue

                date_info = t.get("date", {})
                uts = date_info.get("uts")
                if not uts:
                    continue

                artist_name = t.get("artist", {}).get("#text", "")
                if isinstance(t.get("artist"), str):
                    artist_name = t["artist"]

                result.append({
                    "name": t.get("name", ""),
                    "artist": artist_name,
                    "listened_at": int(uts),
                })
            return result, total_pages
        return None, 0

    def get_chart_top_artists(self, page=1, limit=50):
        """Get chart top artists. Returns list of artist names or None."""
        data = self._request({
            "method": "chart.getTopArtists",
            "page": page,
            "limit": limit,
        })
        if data and "artists" in data:
            artists = data["artists"].get("artist", [])
            if isinstance(artists, dict):
                artists = [artists]
            return [a["name"] for a in artists if "name" in a]
        return None
