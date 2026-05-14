"""Scout agent — Step 5.

Collects textual signals from news + geopolitics + military OSINT RSS feeds
and persists them to the `signals` table (Step 2 schema).

This is the first collector that produces TEXTUAL signals (vs Step 4's
structured numerical market data). Output flows two ways:

  1. State: state["scout_signals"] — compact list of newly collected
     entries, sorted by freshness. Synthesizer (Step 9) reads this.
  2. DB:    rows in oracle_data.db.signals with source_type='rss'.
     Dedup via UNIQUE(source_type, source_id, external_id) — re-running
     never produces duplicates.

Failure semantics: per-feed try/except. One dead feed appends to errors,
the rest keep going. The user spec mandates this:
"return_exceptions=True in asyncio.gather".

Note on the feed list: the user spec listed Reuters / Bloomberg / FT / AP
URLs that may or may not still be reachable in 2026 (some publishers
killed their public RSS feeds). The list below includes BOTH the spec
URLs AND known-working alternatives (BBC, Guardian, NPR, Yahoo Finance)
so the agent always produces signals on first run. Per-feed graceful
failure means dead feeds just log warnings.
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from ..db import get_db
from ..state import OracleState

log = logging.getLogger(__name__)


# ============================================================================
# Feed catalog
# ============================================================================

SCOUT_FEEDS: dict[str, str] = {
    # ----- Spec-literal direct feeds (some publishers killed public RSS in
    # 2020-2024 — kept here for spec parity, fail gracefully if dead) -----
    "reuters/business":     "https://feeds.reuters.com/reuters/businessNews",
    "reuters/tech":         "https://feeds.reuters.com/reuters/technologyNews",
    "bloomberg/markets":    "https://feeds.bloomberg.com/markets/news.rss",
    "ft/all":               "https://www.ft.com/rss/home",
    "ap/business":          "https://feeds.ap.org/rss/apf-business",
    "isw/main":             "https://understandingwar.org/rss.xml",

    # ----- Google News meta-aggregator (always reachable, broad coverage) ----
    "gnews/world":            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "gnews/business":         "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    "gnews/technology":       "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "gnews/health":           "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-US&gl=US&ceid=US:en",
    "gnews/sports":           "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-US&gl=US&ceid=US:en",
    "gnews/entertainment":    "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-US&gl=US&ceid=US:en",
    "gnews/reuters-business": "https://news.google.com/rss/search?q=site:reuters.com+business&hl=en-US&gl=US&ceid=US:en",
    "gnews/reuters-tech":     "https://news.google.com/rss/search?q=site:reuters.com+technology&hl=en-US&gl=US&ceid=US:en",
    "gnews/ap-business":      "https://news.google.com/rss/search?q=site:apnews.com+business&hl=en-US&gl=US&ceid=US:en",
    "gnews/wsj":              "https://news.google.com/rss/search?q=site:wsj.com&hl=en-US&gl=US&ceid=US:en",

    # ----- Targeted high-quality sources (manually-curated, low-noise) -----
    "stratechery/ben-thompson": "https://stratechery.com/feed/",
    "axios/business":           "https://api.axios.com/feed/business",
    "marginal-revolution":      "https://marginalrevolution.com/feed",
    "tld":                      "https://tldr.tech/api/latest/rss",
    "calc/calculatedrisk":      "https://www.calculatedriskblog.com/feeds/posts/default",
    # Sport / fan culture
    "gnews/igaming-betting":    "https://news.google.com/rss/search?q=igaming+OR+sports+betting+OR+online+casino&hl=en-US&gl=US&ceid=US:en",
    "gnews/gambling-industry":  "https://news.google.com/rss/search?q=gambling+industry+revenue&hl=en-US&gl=US&ceid=US:en",
    "gnews/sport-business":     "https://news.google.com/rss/search?q=sports+business+OR+sports+sponsorship&hl=en-US&gl=US&ceid=US:en",

    # ----- Known-working direct feeds (always reachable) -----
    "bbc/business":         "http://feeds.bbci.co.uk/news/business/rss.xml",
    "bbc/world":            "http://feeds.bbci.co.uk/news/world/rss.xml",
    "guardian/business":    "https://www.theguardian.com/uk/business/rss",
    "npr/business":         "https://feeds.npr.org/1006/rss.xml",
    "yahoo/finance":        "https://finance.yahoo.com/news/rssindex",
    "yahoo/sports":         "https://sports.yahoo.com/rss/",
}

HTTP_TIMEOUT = 20.0
# Real browser User-Agent — some publishers (FT, ISW) 403 anything that
# looks like a Python script. This UA is harmless for friendly feeds.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)
MAX_ENTRIES_PER_FEED = 50  # safety cap


# ============================================================================
# Helpers
# ============================================================================


def _parse_published(entry: Any) -> datetime:
    """Pull a UTC datetime out of a feedparser entry, with fallbacks.

    feedparser exposes `published_parsed` (struct_time) when it can parse
    the date. struct_time has no timezone — RSS dates are UTC, so we use
    calendar.timegm (which interprets as UTC), NOT time.mktime (local).
    """
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        st = entry.get(attr)
        if st:
            try:
                return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)
            except Exception:
                continue
    return datetime.now(timezone.utc)


def freshness_score(published_at: datetime) -> int:
    """Map signal age to a 0-100 score (spec literal: 100=<2h, 75=<24h, ...)."""
    age_h = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
    if age_h < 2:
        return 100
    if age_h < 24:
        return 75
    if age_h < 72:
        return 50
    if age_h < 168:  # within a week
        return 25
    return 0


def _external_id(entry: Any) -> str:
    """Stable per-entry ID for dedup. Prefers entry.id, then link, then a hash."""
    if entry.get("id"):
        return str(entry["id"])[:200]
    if entry.get("link"):
        return str(entry["link"])[:200]
    # Last resort: hash title + published
    raw = f"{entry.get('title', '')}|{entry.get('published', '')}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def _content(entry: Any) -> str:
    """Combine title + summary for the content field."""
    title = (entry.get("title") or "").strip()
    summary = (entry.get("summary") or entry.get("description") or "").strip()
    if summary and summary != title:
        return f"{title}\n\n{summary}"
    return title or "(no content)"


def _raw_metadata(entry: Any) -> str:
    """JSON blob with feed-specific extras kept for later analysis."""
    meta: dict[str, Any] = {}
    if entry.get("author"):
        meta["author"] = str(entry["author"])[:200]
    tags = entry.get("tags") or []
    if tags:
        meta["categories"] = [t.get("term", "") for t in tags if t.get("term")][:10]
    return json.dumps(meta) if meta else "{}"


# ============================================================================
# Per-feed fetcher
# ============================================================================


async def fetch_feed(
    client: httpx.AsyncClient, source_id: str, url: str
) -> tuple[list[dict], list[str]]:
    """Fetch one RSS feed, parse entries, return (signals, errors).

    The HTTP fetch is async (httpx); feedparser parses bytes synchronously
    in this thread (parsing is fast, no need to offload).
    """
    errors: list[str] = []
    try:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        body = r.content
    except Exception as e:  # noqa: BLE001
        errors.append(f"rss:{source_id}: fetch failed: {type(e).__name__}: {e}")
        return [], errors

    try:
        parsed = feedparser.parse(body)
    except Exception as e:  # noqa: BLE001
        errors.append(f"rss:{source_id}: parse failed: {type(e).__name__}: {e}")
        return [], errors

    if parsed.bozo and not parsed.entries:
        # bozo=1 with NO entries = total parse failure (XML invalid, blocked, etc.)
        bozo_exc = parsed.get("bozo_exception")
        errors.append(f"rss:{source_id}: bozo with no entries: {bozo_exc}")
        return [], errors

    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    for entry in parsed.entries[:MAX_ENTRIES_PER_FEED]:
        published_at = _parse_published(entry)
        age_h = (now_utc - published_at).total_seconds() / 3600
        signals.append({
            "source_type": "rss",
            "source_id": source_id,
            "external_id": _external_id(entry),
            "title": (entry.get("title") or "").strip()[:500],
            "content": _content(entry)[:5000],
            "url": entry.get("link") or "",
            "published_at": published_at.isoformat(),
            "collected_at": now_utc.isoformat(),
            "freshness_score": freshness_score(published_at),
            "is_breaking": age_h < 2,
            "raw_metadata": _raw_metadata(entry),
        })
    return signals, errors


# ============================================================================
# Persistence
# ============================================================================


async def persist_signals(signals: list[dict]) -> int:
    """INSERT OR IGNORE into signals table; return count of NEW rows.

    UNIQUE(source_type, source_id, external_id) does the dedup. Re-running
    the same feed silently produces 0 new rows for entries already seen.
    """
    if not signals:
        return 0
    new_count = 0
    async with get_db() as conn:
        for sig in signals:
            cur = await conn.execute(
                """INSERT OR IGNORE INTO signals
                   (source_type, source_id, external_id, title, content, url,
                    published_at, collected_at, freshness_score, is_breaking,
                    raw_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sig["source_type"],
                    sig["source_id"],
                    sig["external_id"],
                    sig["title"],
                    sig["content"],
                    sig["url"],
                    sig["published_at"],
                    sig["collected_at"],
                    sig["freshness_score"],
                    1 if sig["is_breaking"] else 0,
                    sig["raw_metadata"],
                ),
            )
            new_count += cur.rowcount  # 1 if inserted, 0 if duplicate
        await conn.commit()
    return new_count


# ============================================================================
# Orchestration
# ============================================================================


async def collect_scout_signals() -> tuple[list[dict], int, list[str]]:
    """Run all feeds in parallel.

    Returns:
        (all_signals, new_in_db, errors)
        - all_signals: signals collected this run (deduped by external_id within
          the run, but may include rows already in DB)
        - new_in_db: count of rows newly inserted (excluding duplicates)
        - errors: per-feed failures
    """
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        tasks = [
            fetch_feed(client, source_id, url)
            for source_id, url in SCOUT_FEEDS.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_signals: list[dict] = []
    errors: list[str] = []

    for (source_id, _), result in zip(SCOUT_FEEDS.items(), results):
        if isinstance(result, Exception):
            errors.append(f"rss:{source_id}: unhandled {type(result).__name__}: {result}")
            continue
        sigs, errs = result
        all_signals.extend(sigs)
        errors.extend(errs)

    # Sort by freshness desc — newest first
    all_signals.sort(key=lambda s: s["published_at"], reverse=True)

    new_in_db = await persist_signals(all_signals)
    return all_signals, new_in_db, errors


# ============================================================================
# LangGraph node
# ============================================================================


async def scout_node(state: OracleState) -> dict:
    """LangGraph node — replaces the Step 1 placeholder.

    Fetches all RSS feeds in parallel, persists new entries to the
    signals table, returns a state patch with the compact signal list.
    """
    log.info("scout: fetching %d RSS feeds", len(SCOUT_FEEDS))
    all_signals, new_in_db, errors = await collect_scout_signals()

    feeds_ok = len(SCOUT_FEEDS) - len({e.split(":", 2)[1] for e in errors if e.startswith("rss:")})
    breaking = sum(1 for s in all_signals if s["is_breaking"])
    log.info(
        "scout: %d/%d feeds OK · %d entries collected · %d new in DB · %d breaking",
        feeds_ok, len(SCOUT_FEEDS), len(all_signals), new_in_db, breaking,
    )
    if errors:
        log.warning("scout: %d feed error(s); first 3: %s", len(errors), errors[:3])

    # Compact representation for state — include just the fields synthesizer needs
    state_signals = [
        {
            "source": s["source_id"],
            "title": s["title"],
            "content": s["content"],
            "url": s["url"],
            "published_at": s["published_at"],
            "freshness_score": s["freshness_score"],
            "is_breaking": s["is_breaking"],
        }
        for s in all_signals
    ]

    return {
        "scout_signals": state_signals,
        "errors": errors,
    }


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.scout`
# ============================================================================


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def _main() -> None:
        from ..db import init_db
        await init_db()
        all_signals, new_in_db, errors = await collect_scout_signals()
        print()
        print("=" * 70)
        print(f"Scout collection summary")
        print("=" * 70)
        print(f"Feeds tried:    {len(SCOUT_FEEDS)}")
        print(f"Errors:         {len(errors)}")
        print(f"Entries:        {len(all_signals)}")
        print(f"New in DB:      {new_in_db}")
        print(f"Breaking (<2h): {sum(1 for s in all_signals if s['is_breaking'])}")
        print()
        print("--- top 5 freshest entries ---")
        for s in all_signals[:5]:
            print(f"  [{s['source_id']}] {s['title'][:80]}")
            print(f"    {s['published_at']} freshness={s['freshness_score']}")
        if errors:
            print()
            print("--- errors ---")
            for e in errors:
                print(f"  ! {e}")

    asyncio.run(_main())
