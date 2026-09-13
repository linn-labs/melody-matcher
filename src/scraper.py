"""Web scraper for Last.fm artist listener pages."""

import re
import time
import logging
import requests
from bs4 import BeautifulSoup

from config import SCRAPE_RATE_LIMIT

logger = logging.getLogger(__name__)

# Match /user/USERNAME links, avoiding subpaths like /user/USERNAME/library
USER_LINK_PATTERN = re.compile(r"^/user/([^/]+)$")

LASTFM_BASE = "https://www.last.fm"


class ListenerScraper:
    """Scrapes Last.fm artist listener pages to discover usernames."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MelodyMatcher/1.0 (music research project)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.min_interval = 1.0 / SCRAPE_RATE_LIMIT
        self.last_request_time = 0
        # Set by scrape_artist_listeners — actual pages fetched (counts early-stops).
        self.last_pages_scraped = 0

    def _wait(self):
        """Enforce rate limit between requests."""
        now = time.monotonic()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.monotonic()

    def _get_page(self, url, max_retries=3):
        """Fetch a page with rate limiting and error handling."""
        for attempt in range(max_retries + 1):
            self._wait()
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code in (403, 404):
                    logger.debug("HTTP %d for %s", resp.status_code, url)
                    return None
                if resp.status_code == 429:
                    logger.warning("Rate limited on scraping, backing off 30s")
                    time.sleep(30)
                    continue
                if resp.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "HTTP %d for %s, retry %d/%d in %ds",
                        resp.status_code, url, attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.warning("HTTP %d for %s", resp.status_code, url)
                return None
            except requests.exceptions.RequestException as e:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Scrape request failed for %s: %s, retry %d/%d in %ds",
                    url, e, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
        logger.warning("All retries exhausted for %s", url)
        return None

    def _extract_usernames(self, html):
        """Extract usernames from a listener page's HTML."""
        soup = BeautifulSoup(html, "html.parser")
        usernames = set()
        for link in soup.find_all("a", href=True):
            match = USER_LINK_PATTERN.match(link["href"])
            if match:
                username = match.group(1)
                # Skip common non-user links
                if username.lower() not in ("", "join", "login", "signup"):
                    usernames.add(username)
        return usernames

    def scrape_artist_listeners(self, artist_name, max_pages=9):
        """Scrape listener pages for an artist.

        Args:
            artist_name: The artist name as it appears on Last.fm
            max_pages: Maximum number of pages to scrape (each ~28 users)

        Returns:
            set of discovered usernames
        """
        # URL-encode the artist name for the URL path
        encoded = requests.utils.quote(artist_name, safe="")
        all_usernames = set()

        for page in range(1, max_pages + 1):
            url = f"{LASTFM_BASE}/music/{encoded}/+listeners?page={page}"
            html = self._get_page(url)

            if html is None:
                logger.info(
                    "Stopped scraping %s at page %d (no response)",
                    artist_name, page,
                )
                break

            usernames = self._extract_usernames(html)
            if not usernames:
                logger.info(
                    "Stopped scraping %s at page %d (no users found)",
                    artist_name, page,
                )
                break

            all_usernames.update(usernames)
            logger.debug(
                "Scraped %s page %d: %d users (total: %d)",
                artist_name, page, len(usernames), len(all_usernames),
            )

        pages_done = min(page, max_pages)
        self.last_pages_scraped = pages_done
        logger.info(
            "Scraped %s: %d unique users from %d pages",
            artist_name, len(all_usernames), pages_done,
        )
        return all_usernames
