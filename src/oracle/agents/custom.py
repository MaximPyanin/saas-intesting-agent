"""Custom sources collector — Step 7.

Reads `user_sources` table (Step 2 schema) and fetches each active source
according to its type. Persists results to the `signals` table and updates
`last_seen_id` / `last_fetched_at` for incremental fetching.

Supported source types:

  website_blog       — auto-detect RSS at /feed, /rss, /atom.xml or via
                       <link rel="alternate"> in HTML; fall back to plain
                       HTML scraping (Step 7 minimal — RSS path preferred)
  rss_custom         — feedparser direct (user pasted an RSS URL)
  telegram_channel   — Telethon MTProto, requires TELEGRAM_API_ID/HASH +
                       a one-time interactive auth (creates a session file)
  youtube_channel    — YouTube Data API v3 + youtube-transcript-api,
                       requires YOUTUBE_API_KEY
  reddit_custom      — additional subreddit beyond the built-in trend list
  twitter_x          — nitter.net (mostly defunct in 2026, kept as stub)

Each fetcher gracefully no-ops if its dependency or credential is missing,
so the agent always runs and never crashes the parent graph.

This file ALSO ships a CLI for managing sources directly via SQL until
Step 8's Telegram /add_source flow lands:

    uv run python -m oracle.agents.custom add  <type> <url> <name> <category>
    uv run python -m oracle.agents.custom list
    uv run python -m oracle.agents.custom remove <id>
    uv run python -m oracle.agents.custom fetch
    uv run python -m oracle.agents.custom auth-tg     # one-time TG login
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup

from ..config import get_settings
from ..db import get_db, init_db
from ..state import OracleState
from .scout import fetch_feed, freshness_score, persist_signals
from .trend import BROWSER_USER_AGENT, REDDIT_USER_AGENT, fetch_reddit_subreddit

log = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

HTTP_TIMEOUT = 30.0
TG_SESSION_PATH = "oracle_reader"  # creates ./oracle_reader.session
TG_RATE_LIMIT_SEC = 2.5            # spec literal: "2-3 sec between channels"
TG_MESSAGES_PER_FETCH = 20
YT_VIDEOS_PER_FETCH = 5
YT_TRANSCRIPT_MAX_CHARS = 3000

VALID_SOURCE_TYPES = {
    "website_blog",
    "rss_custom",
    "telegram_channel",
    "youtube_channel",
    "reddit_custom",
    "twitter_x",
}

VALID_CATEGORIES = {
    "business_ideas",
    "investments",
    "geopolitics",
    "ai_tech",
    "other",
}


# ============================================================================
# DB helpers — read/update user_sources
# ============================================================================


async def load_active_sources() -> list[dict]:
    """Read all active user sources from oracle_data.db."""
    async with get_db() as conn:
        async with conn.execute(
            "SELECT * FROM user_sources WHERE is_active = 1 ORDER BY id"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def update_source_cursor(source_pk: int, last_seen_id: str) -> None:
    """Update last_seen_id + last_fetched_at after a successful fetch."""
    async with get_db() as conn:
        await conn.execute(
            "UPDATE user_sources SET last_seen_id = ?, last_fetched_at = ? WHERE id = ?",
            (last_seen_id, datetime.now(timezone.utc).isoformat(), source_pk),
        )
        await conn.commit()


async def add_user_source(
    source_type: str,
    source_url: str,
    display_name: str,
    category: str,
) -> int:
    """Insert a new user source. Returns the row ID."""
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"invalid source_type {source_type!r} (valid: {sorted(VALID_SOURCE_TYPES)})")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"invalid category {category!r} (valid: {sorted(VALID_CATEGORIES)})")
    async with get_db() as conn:
        cur = await conn.execute(
            """INSERT INTO user_sources
               (source_type, source_url, display_name, category, added_at, is_active)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (source_type, source_url, display_name, category,
             datetime.now(timezone.utc).isoformat()),
        )
        await conn.commit()
        return cur.lastrowid or 0


async def remove_user_source(source_pk: int) -> int:
    """Delete a user source by ID. Returns rowcount."""
    async with get_db() as conn:
        cur = await conn.execute("DELETE FROM user_sources WHERE id = ?", (source_pk,))
        await conn.commit()
        return cur.rowcount


# ============================================================================
# 1. Website fetcher — RSS auto-discovery
# ============================================================================


COMMON_RSS_PATHS = [
    "/feed", "/feed.xml", "/feed/", "/rss", "/rss.xml",
    "/atom.xml", "/index.xml", "/feeds/posts/default",
]


async def discover_rss(client: httpx.AsyncClient, base_url: str) -> str | None:
    """Try to find an RSS/Atom feed for a given website URL.

    Strategy:
      1. Try common paths (/feed, /rss, /atom.xml, etc.)
      2. Parse the homepage HTML for <link rel="alternate" type=...rss/atom>
    Returns the discovered feed URL or None.
    """
    parsed = urlparse(base_url)
    if not parsed.scheme:
        base_url = "https://" + base_url
        parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    # 1. Common paths
    for path in COMMON_RSS_PATHS:
        candidate = base + path
        try:
            r = await client.get(candidate, timeout=8.0)
            if r.status_code == 200:
                ct = (r.headers.get("content-type") or "").lower()
                if "xml" in ct or "rss" in ct or "atom" in ct:
                    return candidate
                # Some servers send text/html for the feed — sniff content
                if r.content[:200].lower().lstrip().startswith((b"<?xml", b"<rss", b"<feed")):
                    return candidate
        except Exception:
            continue

    # 2. Homepage HTML <link rel="alternate">
    try:
        r = await client.get(base_url, timeout=10.0)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            for link in soup.find_all("link", rel="alternate"):
                type_attr = (link.get("type") or "").lower()
                href = link.get("href") or ""
                if href and ("rss" in type_attr or "atom" in type_attr or "xml" in type_attr):
                    return urljoin(base_url, href)
    except Exception:
        pass

    return None


async def fetch_website(
    client: httpx.AsyncClient, source: dict
) -> tuple[list[dict], list[str]]:
    """Fetch a website source by RSS auto-discovery (preferred) or scrape fallback."""
    url = source["source_url"]
    pk = source["id"]
    source_id = f"website/{source['display_name']}"
    errors: list[str] = []

    # 1. Try RSS discovery
    rss_url = await discover_rss(client, url)
    if rss_url:
        log.info("website[%s]: discovered RSS at %s", source["display_name"], rss_url)
        sigs, errs = await fetch_feed(client, source_id, rss_url)
        for s in sigs:
            s["source_type"] = "website"
            s["source_id"] = source_id
        # Update last-seen marker so subsequent runs use it (for analytics/UX)
        if sigs:
            await update_source_cursor(pk, sigs[0]["external_id"])
        return sigs, errs + errors

    # 2. Fallback: basic HTML scrape
    log.info("website[%s]: no RSS, falling back to HTML scrape", source["display_name"])
    try:
        r = await client.get(url, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return [], [f"website:{url}: {type(e).__name__}: {e}"]

    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:  # noqa: BLE001
        return [], [f"website:{url}: parse failed: {type(e).__name__}: {e}"]

    # Look for article-like containers and pull h1/h2 titles + first paragraph.
    # This is intentionally simple — works for most blogs, will be replaced
    # with Playwright in a future polish step for JS-heavy sites.
    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    seen_titles: set[str] = set()

    for art in soup.find_all(["article", "section", "div"], class_=True)[:30]:
        h = art.find(["h1", "h2", "h3"])
        if not h:
            continue
        title = h.get_text(strip=True)
        if not title or len(title) < 15 or title in seen_titles:
            continue
        seen_titles.add(title)

        link = h.find("a") or art.find("a", href=True)
        href = link.get("href") if link else ""
        article_url = urljoin(url, href) if href else url

        para = art.find("p")
        summary = para.get_text(strip=True) if para else ""
        content = f"{title}\n\n{summary}"[:5000]
        ext_id = hashlib.sha1(article_url.encode("utf-8", errors="replace")).hexdigest()

        signals.append({
            "source_type": "website",
            "source_id": source_id,
            "external_id": ext_id,
            "title": title[:500],
            "content": content,
            "url": article_url,
            "published_at": now_utc.isoformat(),  # unknown — use collection time
            "collected_at": now_utc.isoformat(),
            "freshness_score": 50,  # neutral — we don't know real publish time
            "is_breaking": False,
            "raw_metadata": json.dumps({"scraped": True, "host": parsed.netloc if (parsed := urlparse(url)) else ""}),
        })
        if len(signals) >= 10:  # spec rate limit: max 10 articles per site per run
            break

    if signals:
        await update_source_cursor(pk, signals[0]["external_id"])

    return signals, errors


# ============================================================================
# 2. Custom RSS — feedparser direct
# ============================================================================


async def fetch_rss_custom_one(
    client: httpx.AsyncClient, source: dict
) -> tuple[list[dict], list[str]]:
    """Fetch a user-pasted RSS URL via the shared scout.fetch_feed."""
    pk = source["id"]
    source_id = f"rss_custom/{source['display_name']}"
    sigs, errs = await fetch_feed(client, source_id, source["source_url"])
    for s in sigs:
        s["source_type"] = "rss_custom"
        s["source_id"] = source_id
    if sigs:
        await update_source_cursor(pk, sigs[0]["external_id"])
    return sigs, errs


# ============================================================================
# 3. Telegram channel fetcher (Telethon)
# ============================================================================


async def fetch_telegram_channels(sources: list[dict]) -> tuple[list[dict], list[str]]:
    """Fetch all TG sources serially with rate limit between channels.

    Telethon is sync-style under the hood but exposes async methods. The
    session file (`oracle_reader.session`) must exist — run `auth-tg` once
    to create it.
    """
    if not sources:
        return [], []

    settings = get_settings()
    if not (settings.telegram_api_id and settings.telegram_api_hash):
        return [], ["tg: TELEGRAM_API_ID/HASH not set in .env — TG channels skipped"]

    try:
        from telethon import TelegramClient  # noqa: PLC0415 — lazy import
        from telethon.errors import (  # noqa: PLC0415
            ChannelPrivateError,
            FloodWaitError,
        )
    except ImportError:
        return [], ["tg: telethon not installed — `uv add telethon`"]

    try:
        api_id = int(settings.telegram_api_id)
    except ValueError:
        return [], [f"tg: TELEGRAM_API_ID is not a number: {settings.telegram_api_id!r}"]

    client = TelegramClient(TG_SESSION_PATH, api_id, settings.telegram_api_hash)

    try:
        await client.connect()
    except Exception as e:  # noqa: BLE001
        return [], [f"tg: connect failed: {type(e).__name__}: {e}"]

    if not await client.is_user_authorized():
        await client.disconnect()
        return [], [
            "tg: session not authorized — run `uv run python -m oracle.agents.custom auth-tg`"
        ]

    signals: list[dict] = []
    errors: list[str] = []

    try:
        for source in sources:
            channel = source["source_url"]
            pk = source["id"]
            try:
                last_seen = int(source.get("last_seen_id") or 0)
            except (TypeError, ValueError):
                last_seen = 0

            try:
                messages = []
                async for msg in client.iter_messages(
                    channel,
                    limit=TG_MESSAGES_PER_FETCH,
                    min_id=last_seen,
                ):
                    if msg.text:
                        messages.append(msg)

                if not messages:
                    log.info("tg[%s]: no new messages since id=%s", channel, last_seen)
                    await asyncio.sleep(TG_RATE_LIMIT_SEC)
                    continue

                now_utc = datetime.now(timezone.utc)
                channel_clean = channel.lstrip("@")
                for msg in messages:
                    text = (msg.text or "").strip()
                    title = text.split("\n", 1)[0][:200] if text else "(empty)"
                    pub_dt = msg.date.astimezone(timezone.utc) if msg.date else now_utc
                    age_h = (now_utc - pub_dt).total_seconds() / 3600
                    signals.append({
                        "source_type": "telegram",
                        "source_id": f"telegram/{channel_clean}",
                        "external_id": str(msg.id),
                        "title": title,
                        "content": text[:5000] or "(empty)",
                        "url": f"https://t.me/{channel_clean}/{msg.id}",
                        "published_at": pub_dt.isoformat(),
                        "collected_at": now_utc.isoformat(),
                        "freshness_score": freshness_score(pub_dt),
                        "is_breaking": age_h < 2,
                        "raw_metadata": json.dumps({
                            "tg_msg_id": msg.id,
                            "channel": channel_clean,
                            "views": getattr(msg, "views", 0) or 0,
                            "forwards": getattr(msg, "forwards", 0) or 0,
                        }),
                    })

                # Update cursor to highest message ID seen
                max_id = str(max(m.id for m in messages))
                await update_source_cursor(pk, max_id)
                log.info("tg[%s]: %d new messages, cursor → %s", channel, len(messages), max_id)

            except FloodWaitError as e:
                errors.append(f"tg:{channel}: flood wait {e.seconds}s")
                await asyncio.sleep(e.seconds)
            except ChannelPrivateError:
                errors.append(f"tg:{channel}: channel is private or not joined")
            except Exception as e:  # noqa: BLE001
                errors.append(f"tg:{channel}: {type(e).__name__}: {e}")

            # Politeness — be a good API citizen
            await asyncio.sleep(TG_RATE_LIMIT_SEC)
    finally:
        await client.disconnect()

    return signals, errors


async def auth_telegram_interactive() -> None:
    """One-shot interactive Telegram auth. Creates oracle_reader.session.

    Run via: `uv run python -m oracle.agents.custom auth-tg`
    """
    settings = get_settings()
    if not (settings.telegram_api_id and settings.telegram_api_hash):
        print("ERROR: Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env first.")
        print("       Get them from https://my.telegram.org → API development tools.")
        return

    try:
        from telethon import TelegramClient  # noqa: PLC0415
    except ImportError:
        print("ERROR: telethon not installed. Run `uv sync`.")
        return

    try:
        api_id = int(settings.telegram_api_id)
    except ValueError:
        print(f"ERROR: TELEGRAM_API_ID is not a number: {settings.telegram_api_id!r}")
        return

    client = TelegramClient(TG_SESSION_PATH, api_id, settings.telegram_api_hash)
    print("Starting Telegram auth — you'll receive a code via Telegram (or SMS).")
    print(f"Session file will be saved at: {TG_SESSION_PATH}.session")
    print()
    await client.start()
    me = await client.get_me()
    print()
    print(f"✓ Authorized as: {me.first_name} (@{me.username or '—'})")
    print(f"  Session saved. Future fetches will be non-interactive.")
    await client.disconnect()


# ============================================================================
# 4. YouTube channel fetcher
# ============================================================================


YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


async def fetch_youtube_channel(
    client: httpx.AsyncClient, source: dict, api_key: str
) -> tuple[list[dict], list[str]]:
    """Fetch recent videos from a YT channel + transcripts.

    `source["source_url"]` should be a channel ID (UC...) — Step 8 will add
    username → channel ID resolution. For now, user must paste the UC ID.
    """
    if not api_key:
        return [], []  # silent skip — outer collector logs once

    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # noqa: PLC0415
        has_transcript = True
    except ImportError:
        has_transcript = False

    channel_id = source["source_url"]
    pk = source["id"]
    source_id = f"youtube/{source['display_name']}"
    last_seen = source.get("last_seen_id") or ""
    errors: list[str] = []

    # 1. List recent videos
    try:
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "maxResults": YT_VIDEOS_PER_FETCH,
            "key": api_key,
        }
        r = await client.get(YT_SEARCH_URL, params=params, timeout=15.0)
        r.raise_for_status()
        videos = r.json().get("items", [])
    except Exception as e:  # noqa: BLE001
        return [], [f"yt:{channel_id}: list failed: {type(e).__name__}: {e}"]

    if not videos:
        return [], errors

    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    max_id_seen = last_seen

    for video in videos:
        video_id = (video.get("id") or {}).get("videoId", "")
        if not video_id:
            continue
        if last_seen and video_id == last_seen:
            break  # already seen this point in time

        snippet = video.get("snippet", {}) or {}
        title = (snippet.get("title") or "").strip()
        description = (snippet.get("description") or "").strip()
        published = snippet.get("publishedAt", "")
        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            pub_dt = now_utc

        # 2. Fetch transcript (optional, sync → to_thread)
        transcript_text = ""
        if has_transcript:
            try:
                transcript = await asyncio.to_thread(
                    YouTubeTranscriptApi.get_transcript, video_id, ["en"]
                )
                transcript_text = " ".join(seg["text"] for seg in transcript)[:YT_TRANSCRIPT_MAX_CHARS]
            except Exception as e:  # noqa: BLE001 — many failure modes
                errors.append(f"yt:{video_id}: transcript: {type(e).__name__}")

        content_parts = [title, description[:600]]
        if transcript_text:
            content_parts.append(f"[transcript]\n{transcript_text}")
        content = "\n\n".join(p for p in content_parts if p)[:5000]

        signals.append({
            "source_type": "youtube",
            "source_id": source_id,
            "external_id": video_id,
            "title": title[:500],
            "content": content,
            "url": f"https://youtube.com/watch?v={video_id}",
            "published_at": pub_dt.isoformat(),
            "collected_at": now_utc.isoformat(),
            "freshness_score": freshness_score(pub_dt),
            "is_breaking": (now_utc - pub_dt).total_seconds() / 3600 < 2,
            "raw_metadata": json.dumps({
                "channel_id": channel_id,
                "channel_title": snippet.get("channelTitle", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                "has_transcript": bool(transcript_text),
            }),
        })
        if not max_id_seen or video_id > max_id_seen:
            max_id_seen = video_id

    if max_id_seen and max_id_seen != last_seen:
        await update_source_cursor(pk, max_id_seen)

    return signals, errors


# ============================================================================
# 5. Reddit custom — additional subreddits beyond the trend builtins
# ============================================================================


async def fetch_reddit_customs(
    client: httpx.AsyncClient, sources: list[dict]
) -> tuple[list[dict], list[str]]:
    """Fetch user-added subreddits in parallel via trend.fetch_reddit_subreddit."""
    if not sources:
        return [], []
    results = await asyncio.gather(
        *[
            fetch_reddit_subreddit(
                client,
                source["source_url"].lstrip("r/").lstrip("/"),
            )
            for source in sources
        ],
        return_exceptions=True,
    )
    signals: list[dict] = []
    errors: list[str] = []
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            errors.append(f"reddit_custom:{source['source_url']}: {type(result).__name__}: {result}")
            continue
        sigs, errs = result
        sub_clean = source["source_url"].lstrip("r/").lstrip("/")
        for s in sigs:
            s["source_type"] = "reddit_custom"
            s["source_id"] = f"reddit_custom/r/{sub_clean}"
        signals.extend(sigs)
        errors.extend(errs)
        if sigs:
            await update_source_cursor(source["id"], sigs[0]["external_id"])
    return signals, errors


# ============================================================================
# 6. Twitter via nitter (mostly defunct in 2026 — graceful stub)
# ============================================================================


async def fetch_twitter_x(
    client: httpx.AsyncClient, source: dict
) -> tuple[list[dict], list[str]]:
    """Stub — most public nitter instances are down or rate-limited.

    Step 7+ polish will revisit if a working nitter mirror or alternative
    free Twitter API exists.
    """
    return [], [
        f"twitter_x:{source['source_url']}: nitter mirrors are unreliable in 2026; skipping"
    ]


# ============================================================================
# Orchestration
# ============================================================================


async def _gather_per_source(
    label: str,
    coros: list,
) -> tuple[list[dict], list[str]]:
    """Helper: gather a list of (sigs, errs)-returning coroutines, flatten."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    sigs: list[dict] = []
    errs: list[str] = []
    for r in results:
        if isinstance(r, Exception):
            errs.append(f"{label}: unhandled {type(r).__name__}: {r}")
            continue
        s_list, e_list = r
        sigs.extend(s_list)
        errs.extend(e_list)
    return sigs, errs


async def collect_custom_signals() -> tuple[list[dict], int, list[str], dict[str, int]]:
    """Read user_sources, fetch each, persist new entries.

    Returns:
        (all_signals, new_in_db, errors, per_type_counts)
    """
    settings = get_settings()
    sources = await load_active_sources()

    if not sources:
        log.info("custom: no active user sources — add some via `python -m oracle.agents.custom add`")
        return [], 0, [], {}

    by_type: dict[str, list[dict]] = {}
    for s in sources:
        by_type.setdefault(s["source_type"], []).append(s)

    log.info(
        "custom: %d active sources (%s)",
        len(sources),
        ", ".join(f"{k}={len(v)}" for k, v in by_type.items()),
    )

    headers = {"User-Agent": BROWSER_USER_AGENT}
    all_signals: list[dict] = []
    errors: list[str] = []
    counts: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
        # Each bucket is a single coroutine that returns (sigs, errs).
        # We then gather buckets in parallel at the orchestrator level.
        bucket_coros: dict[str, Any] = {}

        if "website_blog" in by_type:
            bucket_coros["website_blog"] = _gather_per_source(
                "website_blog",
                [fetch_website(client, s) for s in by_type["website_blog"]],
            )
        if "rss_custom" in by_type:
            bucket_coros["rss_custom"] = _gather_per_source(
                "rss_custom",
                [fetch_rss_custom_one(client, s) for s in by_type["rss_custom"]],
            )
        if "youtube_channel" in by_type:
            bucket_coros["youtube_channel"] = _gather_per_source(
                "youtube_channel",
                [
                    fetch_youtube_channel(client, s, settings.youtube_api_key)
                    for s in by_type["youtube_channel"]
                ],
            )
        if "reddit_custom" in by_type:
            # Already returns (sigs, errs) — wrap for uniformity
            bucket_coros["reddit_custom"] = fetch_reddit_customs(client, by_type["reddit_custom"])
        if "twitter_x" in by_type:
            bucket_coros["twitter_x"] = _gather_per_source(
                "twitter_x",
                [fetch_twitter_x(client, s) for s in by_type["twitter_x"]],
            )
        if "telegram_channel" in by_type:
            # Telethon owns its own connection — already returns (sigs, errs)
            bucket_coros["telegram_channel"] = fetch_telegram_channels(by_type["telegram_channel"])

        # Run all buckets in parallel
        bucket_results = await asyncio.gather(
            *bucket_coros.values(), return_exceptions=True
        )

        for stype, result in zip(bucket_coros.keys(), bucket_results):
            if isinstance(result, Exception):
                errors.append(f"{stype}: unhandled {type(result).__name__}: {result}")
                counts[stype] = 0
                continue
            sigs, errs = result
            counts[stype] = len(sigs)
            all_signals.extend(sigs)
            errors.extend(errs)

    # Sort newest first
    all_signals.sort(key=lambda s: s["published_at"], reverse=True)

    new_in_db = await persist_signals(all_signals)
    return all_signals, new_in_db, errors, counts


# ============================================================================
# LangGraph node
# ============================================================================


async def custom_node(state: OracleState) -> dict:
    """LangGraph node — replaces the Step 1 placeholder."""
    log.info("custom: collecting from user_sources table")
    all_signals, new_in_db, errors, counts = await collect_custom_signals()

    breaking = sum(1 for s in all_signals if s["is_breaking"])
    type_summary = " ".join(f"{k}={v}" for k, v in counts.items()) if counts else "(no sources)"
    log.info(
        "custom: %d entries · %d new in DB · %d breaking · %s",
        len(all_signals), new_in_db, breaking, type_summary,
    )
    if errors:
        log.warning("custom: %d source error(s); first 3: %s", len(errors), errors[:3])

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
        "custom_signals": state_signals,
        "errors": errors,
    }


# ============================================================================
# CLI subcommands
# ============================================================================


async def cmd_add(args: argparse.Namespace) -> None:
    await init_db()
    try:
        new_id = await add_user_source(
            source_type=args.type,
            source_url=args.url,
            display_name=args.name,
            category=args.category,
        )
        print(f"✓ Added source #{new_id}: [{args.type}] {args.name} ({args.category})")
        print(f"  URL: {args.url}")
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {e}")


async def cmd_list(args: argparse.Namespace) -> None:
    await init_db()
    sources = await load_active_sources()
    if not sources:
        print("(no active sources — add one with `add <type> <url> <name> <category>`)")
        return
    print(f"{'ID':<4} {'TYPE':<18} {'CATEGORY':<15} {'NAME':<30} URL")
    print("-" * 100)
    for s in sources:
        name = (s["display_name"] or "")[:28]
        url = (s["source_url"] or "")[:50]
        print(f"{s['id']:<4} {s['source_type']:<18} {s['category']:<15} {name:<30} {url}")


async def cmd_remove(args: argparse.Namespace) -> None:
    await init_db()
    n = await remove_user_source(args.id)
    print(f"Removed {n} source(s)" if n else f"No source with id={args.id}")


async def cmd_fetch(args: argparse.Namespace) -> None:
    await init_db()
    all_signals, new_in_db, errors, counts = await collect_custom_signals()
    print()
    print("=" * 70)
    print("Custom collection summary")
    print("=" * 70)
    print(f"Entries: {len(all_signals)}  ·  New in DB: {new_in_db}  ·  Errors: {len(errors)}")
    print(f"Per-type: {dict(counts)}")
    print()
    print("--- top 10 freshest ---")
    for s in all_signals[:10]:
        print(f"  [{s['source_id'][:30]:30s}] {s['title'][:80]}")
    if errors:
        print()
        print("--- errors ---")
        for e in errors[:10]:
            print(f"  ! {e}")


async def cmd_auth_tg(args: argparse.Namespace) -> None:
    await auth_telegram_interactive()


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle.agents.custom",
        description="ORACLE custom source collector + manager (Step 7).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # add
    p_add = sub.add_parser("add", help="add a new source")
    p_add.add_argument("type", choices=sorted(VALID_SOURCE_TYPES))
    p_add.add_argument("url", help="source URL or identifier (e.g. @bensbites, UCabc123, https://...)")
    p_add.add_argument("name", help="human-friendly display name")
    p_add.add_argument("category", choices=sorted(VALID_CATEGORIES))

    # list
    sub.add_parser("list", help="list all active sources")

    # remove
    p_rm = sub.add_parser("remove", help="remove a source by ID")
    p_rm.add_argument("id", type=int)

    # fetch
    sub.add_parser("fetch", help="run the custom collector once")

    # auth-tg
    sub.add_parser("auth-tg", help="one-time interactive Telegram auth (creates session file)")

    return parser


# ============================================================================
# CLI entry: `uv run python -m oracle.agents.custom <cmd>`
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

    parser = _build_cli()
    args = parser.parse_args()

    handlers = {
        "add":     cmd_add,
        "list":    cmd_list,
        "remove":  cmd_remove,
        "fetch":   cmd_fetch,
        "auth-tg": cmd_auth_tg,
    }
    asyncio.run(handlers[args.cmd](args))
