"""Parse an Apple Music / iTunes Library XML export into a list of tracks.

Apple Music (desktop, macOS): File -> Library -> Export Library...
produces a plist XML with a top-level `Tracks` dict keyed by persistent ID.
Fields we care about: Name, Artist, Album, Year, Genre, Total Time, Play Count.
"""

from __future__ import annotations

import logging
import plistlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImportedTrack:
    apple_track_id: int
    title: str
    artist: str
    album: Optional[str]
    year: Optional[int]
    genre: Optional[str]
    duration_ms: Optional[int]
    play_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_apple_library(path: Path) -> List[ImportedTrack]:
    """Read an Apple Music Library XML file. Raises on malformed input."""
    with open(path, "rb") as f:
        plist = plistlib.load(f, fmt=plistlib.FMT_XML)

    raw_tracks = plist.get("Tracks") or {}
    out: List[ImportedTrack] = []

    for tid_str, rec in raw_tracks.items():
        title = rec.get("Name")
        artist = rec.get("Artist") or rec.get("Album Artist")
        if not title or not artist:
            continue  # unusable without both

        try:
            apple_id = int(tid_str)
        except (TypeError, ValueError):
            apple_id = int(rec.get("Track ID") or 0)

        out.append(ImportedTrack(
            apple_track_id=apple_id,
            title=str(title),
            artist=str(artist),
            album=_as_str(rec.get("Album")),
            year=_as_int(rec.get("Year")),
            genre=_as_str(rec.get("Genre")),
            duration_ms=_as_int(rec.get("Total Time")),
            play_count=int(rec.get("Play Count") or 0),
        ))

    logger.info("Parsed %d tracks from %s", len(out), path)
    return out


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None
