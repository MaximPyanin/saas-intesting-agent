"""Trend agent — Step 6.

Collects business / tech / startup trend signals from 6 sources, all in
parallel. Writes textual signals to oracle_data.db.signals (Step 2 schema)
and returns a compact list in state["trend_signals"].

Sources (per user spec — listed in priority order for the BUSINESS IDEA
pathway, which is ORACLE's primary goal):

  1. HackerNews   — Firebase API, no auth — top stories (highest signal/noise)
  2. ProductHunt  — GraphQL, optional token — what's launching today
  3. Reddit       — public JSON, no auth — r/SaaS, r/indiehackers, r/startups,
                                            r/microsaas, r/MachineLearning,
                                            r/LocalLLaMA + secondary investing
  4. GitHub       — HTML scrape — trending Python/TS/Rust repos
  5. TechCrunch   — RSS — startup funding announcements
  6. Crunchbase   — RSS — VC deal flow

Per the user's repeated emphasis: idea sources get more SLOTS than investment
sources. Reddit subreddit list is ~80% idea-relevant, ~20% investing.
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from ..config import get_settings
from ..db import get_db
from ..state import OracleState
from .scout import (
    fetch_feed,
    freshness_score,
    persist_signals,
)

log = logging.getLogger(__name__)


# ============================================================================
# Source catalogs
# ============================================================================

# Reddit subreddits — heavily biased toward business idea sources (per user
# priority: 70% ideas / 30% investments). r/MachineLearning + r/LocalLLaMA
# match Maksim's RAG/LLM expertise edge.
SUBREDDITS_PRIMARY = [
    # Direct idea sources (highest signal)
    "SaaS",
    "indiehackers",
    "startups",
    "microsaas",
    # AI / ML — Maksim's competitive edge
    "MachineLearning",
    "LocalLLaMA",
    "artificial",
]

SUBREDDITS_SECONDARY = [
    # Investment side — secondary per user spec
    "investing",
    "stocks",
]

# GitHub trending — focus on languages relevant to AI/SaaS dev
GITHUB_LANGUAGES = ["python", "typescript", "rust"]

# RSS feeds for VC deal flow (TechCrunch + Crunchbase)
TREND_RSS_FEEDS: dict[str, str] = {
    "rss/techcrunch":  "https://techcrunch.com/category/startups/feed/",
    "rss/crunchbase":  "https://news.crunchbase.com/feed/",
    # Bonus AI-focused tech feeds (high idea signal):
    "rss/the-verge-tech": "https://www.theverge.com/rss/index.xml",
    "rss/ars-technica":   "https://feeds.arstechnica.com/arstechnica/index",
}

# HTTP config
HTTP_TIMEOUT = 20.0
HN_TOP_LIMIT = 30        # how many top stories to fetch from HN
HN_CONCURRENCY = 10      # max parallel HTTP to HN
PH_LIMIT = 20            # how many top posts from PH
REDDIT_LIMIT = 15        # posts per subreddit
GITHUB_REPO_LIMIT = 25   # repos per language

# Reddit requires a unique, descriptive User-Agent (per their API rules).
# Generic Python UAs get 429'd. PRAW uses one like this internally.
REDDIT_USER_AGENT = "ORACLE-bot/0.1 (personal intelligence collector)"

# General UA for sites that 403 the default httpx UA (GitHub, ProductHunt OK
# with anything, but doesn't hurt).
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


# ============================================================================
# 1. HackerNews — Firebase API
# ============================================================================

HN_BASE = "https://hacker-news.firebaseio.com/v0"


async def fetch_hackernews(client: httpx.AsyncClient) -> tuple[list[dict], list[str]]:
    """Top N HN stories with title, score, comments, URL."""
    errors: list[str] = []
    try:
        r = await client.get(f"{HN_BASE}/topstories.json")
        r.raise_for_status()
        ids = r.json()[:HN_TOP_LIMIT]
    except Exception as e:  # noqa: BLE001
        errors.append(f"hn:topstories: {type(e).__name__}: {e}")
        return [], errors

    sem = asyncio.Semaphore(HN_CONCURRENCY)

    async def _fetch_item(item_id: int) -> dict | None:
        async with sem:
            try:
                r = await client.get(f"{HN_BASE}/item/{item_id}.json")
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                errors.append(f"hn:item/{item_id}: {type(e).__name__}: {e}")
                return None

    items = await asyncio.gather(*[_fetch_item(i) for i in ids])

    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    for item in items:
        if not item or item.get("type") != "story":
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        published_at = datetime.fromtimestamp(item.get("time", 0), tz=timezone.utc)
        score = item.get("score", 0)
        descendants = item.get("descendants", 0)
        ext_url = item.get("url", "")
        item_url = f"https://news.ycombinator.com/item?id={item['id']}"

        # Self-post (text post) — include the text body in content
        text = (item.get("text") or "").strip()
        body_preview = ""
        if text:
            # Strip HTML tags from text
            try:
                body_preview = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)[:1500]
            except Exception:
                body_preview = text[:1500]

        content_parts = [title, f"[HN: {score} points · {descendants} comments]"]
        if body_preview:
            content_parts.append(body_preview)
        content = "\n".join(content_parts)

        signals.append({
            "source_type": "hn",
            "source_id": "hn/topstories",
            "external_id": str(item["id"]),
            "title": title[:500],
            "content": content[:5000],
            "url": ext_url or item_url,
            "published_at": published_at.isoformat(),
            "collected_at": now_utc.isoformat(),
            "freshness_score": freshness_score(published_at),
            "is_breaking": (now_utc - published_at).total_seconds() / 3600 < 2,
            "raw_metadata": json.dumps({
                "score": score,
                "comments": descendants,
                "author": item.get("by", ""),
                "hn_url": item_url,
            }),
        })
    return signals, errors


# ============================================================================
# 2. ProductHunt — GraphQL (optional token)
# ============================================================================

PH_GRAPHQL = "https://api.producthunt.com/v2/api/graphql"

PH_QUERY = """
query TopPosts($first: Int!) {
  posts(order: VOTES, first: $first) {
    edges {
      node {
        id
        name
        tagline
        slug
        url
        votesCount
        commentsCount
        createdAt
      }
    }
  }
}
"""


async def fetch_producthunt(
    client: httpx.AsyncClient, token: str
) -> tuple[list[dict], list[str]]:
    """Top ProductHunt posts via GraphQL. Gracefully no-op without token."""
    if not token:
        log.info("ph: PRODUCTHUNT_TOKEN not set — ProductHunt skipped (set in .env to enable)")
        return [], []

    errors: list[str] = []
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        r = await client.post(
            PH_GRAPHQL,
            headers=headers,
            json={"query": PH_QUERY, "variables": {"first": PH_LIMIT}},
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as e:  # noqa: BLE001
        errors.append(f"producthunt: {type(e).__name__}: {e}")
        return [], errors

    if "errors" in payload:
        errors.append(f"producthunt: graphql errors: {payload['errors']}")
        return [], errors

    edges = payload.get("data", {}).get("posts", {}).get("edges", []) or []
    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    for edge in edges:
        node = edge.get("node") or {}
        try:
            published_at = datetime.fromisoformat(node["createdAt"].replace("Z", "+00:00"))
        except Exception:
            published_at = now_utc
        name = (node.get("name") or "").strip()
        tagline = (node.get("tagline") or "").strip()
        if not name:
            continue
        signals.append({
            "source_type": "producthunt",
            "source_id": "producthunt/top",
            "external_id": str(node["id"]),
            "title": name[:500],
            "content": f"{name}: {tagline}"[:5000],
            "url": node.get("url", ""),
            "published_at": published_at.isoformat(),
            "collected_at": now_utc.isoformat(),
            "freshness_score": freshness_score(published_at),
            "is_breaking": (now_utc - published_at).total_seconds() / 3600 < 2,
            "raw_metadata": json.dumps({
                "votes": node.get("votesCount", 0),
                "comments": node.get("commentsCount", 0),
                "slug": node.get("slug", ""),
            }),
        })
    return signals, errors


# ============================================================================
# 3. Reddit — public JSON endpoints (unauth, public read access)
# ============================================================================


async def fetch_reddit_subreddit(
    client: httpx.AsyncClient, subreddit: str, sort: str = "hot"
) -> tuple[list[dict], list[str]]:
    """Fetch top posts from one subreddit via the public JSON endpoint."""
    errors: list[str] = []
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={REDDIT_LIMIT}"
    try:
        r = await client.get(url, headers={"User-Agent": REDDIT_USER_AGENT})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:  # noqa: BLE001
        errors.append(f"reddit:r/{subreddit}: {type(e).__name__}: {e}")
        return [], errors

    children = payload.get("data", {}).get("children", []) or []
    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []
    for child in children:
        d = child.get("data") or {}
        if d.get("stickied"):  # skip pinned mod posts
            continue
        title = (d.get("title") or "").strip()
        if not title:
            continue
        try:
            published_at = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc)
        except Exception:
            published_at = now_utc
        score = d.get("score", 0)
        num_comments = d.get("num_comments", 0)
        selftext = (d.get("selftext") or "").strip()
        permalink = d.get("permalink", "")
        ext_url = d.get("url", "")

        content_parts = [title, f"[r/{subreddit}: {score} upvotes · {num_comments} comments]"]
        if selftext:
            content_parts.append(selftext[:1500])
        content = "\n".join(content_parts)

        signals.append({
            "source_type": "reddit",
            "source_id": f"reddit/r/{subreddit}",
            "external_id": d.get("id", "") or d.get("name", ""),
            "title": title[:500],
            "content": content[:5000],
            "url": f"https://www.reddit.com{permalink}" if permalink else ext_url,
            "published_at": published_at.isoformat(),
            "collected_at": now_utc.isoformat(),
            "freshness_score": freshness_score(published_at),
            "is_breaking": (now_utc - published_at).total_seconds() / 3600 < 2,
            "raw_metadata": json.dumps({
                "score": score,
                "comments": num_comments,
                "author": d.get("author", ""),
                "flair": d.get("link_flair_text", ""),
                "external_url": ext_url,
            }),
        })
    return signals, errors


async def fetch_reddit_all(client: httpx.AsyncClient) -> tuple[list[dict], list[str]]:
    """Fetch all configured subreddits in parallel."""
    all_subs = SUBREDDITS_PRIMARY + SUBREDDITS_SECONDARY
    results = await asyncio.gather(
        *[fetch_reddit_subreddit(client, sub) for sub in all_subs],
        return_exceptions=True,
    )
    signals: list[dict] = []
    errors: list[str] = []
    for sub, result in zip(all_subs, results):
        if isinstance(result, Exception):
            errors.append(f"reddit:r/{sub}: unhandled {type(result).__name__}: {result}")
            continue
        sigs, errs = result
        signals.extend(sigs)
        errors.extend(errs)
    return signals, errors


# ============================================================================
# 4. GitHub Trending — HTML scrape
# ============================================================================


async def fetch_github_trending_lang(
    client: httpx.AsyncClient, language: str, since: str = "daily"
) -> tuple[list[dict], list[str]]:
    """Scrape github.com/trending/<language>?since=daily for trending repos."""
    errors: list[str] = []
    url = f"https://github.com/trending/{language}?since={since}"
    try:
        r = await client.get(url, headers={"User-Agent": BROWSER_USER_AGENT})
        r.raise_for_status()
        html = r.text
    except Exception as e:  # noqa: BLE001
        errors.append(f"github:{language}: fetch failed: {type(e).__name__}: {e}")
        return [], errors

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:  # noqa: BLE001
        errors.append(f"github:{language}: parse failed: {type(e).__name__}: {e}")
        return [], errors

    rows = soup.find_all("article", class_="Box-row")[:GITHUB_REPO_LIMIT]
    now_utc = datetime.now(timezone.utc)
    signals: list[dict] = []

    for row in rows:
        # Repo name (e.g., "openai/whisper") — in h2 > a
        h2 = row.find("h2")
        if not h2:
            continue
        anchor = h2.find("a")
        if not anchor:
            continue
        href = (anchor.get("href") or "").strip().lstrip("/")
        full_name = href  # e.g. "openai/whisper"
        if not full_name:
            continue

        # Description
        desc_p = row.find("p")
        description = desc_p.get_text(strip=True) if desc_p else ""

        # Stars total + today
        stars_total = ""
        stars_today = ""
        for span in row.find_all("span", class_="d-inline-block"):
            txt = span.get_text(strip=True)
            if "star" in txt.lower() and "today" in txt.lower():
                stars_today = txt

        a_tags = row.find_all("a", class_="Link Link--muted")
        if a_tags:
            stars_total = a_tags[0].get_text(strip=True)

        title = full_name
        content_parts = [full_name]
        if description:
            content_parts.append(description)
        meta_bits = []
        if stars_total:
            meta_bits.append(f"★ {stars_total}")
        if stars_today:
            meta_bits.append(stars_today)
        if meta_bits:
            content_parts.append(f"[{' · '.join(meta_bits)}]")
        content = "\n".join(content_parts)

        signals.append({
            "source_type": "github_trending",
            "source_id": f"github/{language}",
            "external_id": full_name,
            "title": title[:500],
            "content": content[:5000],
            "url": f"https://github.com/{full_name}",
            # GitHub trending doesn't expose a per-repo "trending since" timestamp,
            # so we use collection time and mark as fresh.
            "published_at": now_utc.isoformat(),
            "collected_at": now_utc.isoformat(),
            "freshness_score": 75,
            "is_breaking": False,
            "raw_metadata": json.dumps({
                "language": language,
                "stars_total": stars_total,
                "stars_today": stars_today,
                "description": description[:300],
            }),
        })
    return signals, errors


async def fetch_github_trending_all(client: httpx.AsyncClient) -> tuple[list[dict], list[str]]:
    """Fetch trending repos for all configured languages in parallel."""
    results = await asyncio.gather(
        *[fetch_github_trending_lang(client, lang) for lang in GITHUB_LANGUAGES],
        return_exceptions=True,
    )
    signals: list[dict] = []
    errors: list[str] = []
    for lang, result in zip(GITHUB_LANGUAGES, results):
        if isinstance(result, Exception):
            errors.append(f"github:{lang}: unhandled {type(result).__name__}: {result}")
            continue
        sigs, errs = result
        signals.extend(sigs)
        errors.extend(errs)
    return signals, errors


# ============================================================================
# 5 & 6. TechCrunch + Crunchbase + tech RSS — reuses scout.fetch_feed
# ============================================================================


async def fetch_trend_rss_feeds(client: httpx.AsyncClient) -> tuple[list[dict], list[str]]:
    """Fetch all trend RSS feeds in parallel via scout.fetch_feed."""
    results = await asyncio.gather(
        *[fetch_feed(client, src_id, url) for src_id, url in TREND_RSS_FEEDS.items()],
        return_exceptions=True,
    )
    signals: list[dict] = []
    errors: list[str] = []
    for src_id, result in zip(TREND_RSS_FEEDS.keys(), results):
        if isinstance(result, Exception):
            errors.append(f"rss:{src_id}: unhandled {type(result).__name__}: {result}")
            continue
        sigs, errs = result
        signals.extend(sigs)
        errors.extend(errs)
    return signals, errors


# ============================================================================
# Orchestration
# ============================================================================


async def collect_trend_signals() -> tuple[list[dict], int, list[str], dict[str, int]]:
    """Run all 6 source categories in parallel.

    Returns:
        (all_signals, new_in_db, errors, source_counts)
    """
    settings = get_settings()
    headers = {"User-Agent": BROWSER_USER_AGENT}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
        tasks = {
            "hn":           fetch_hackernews(client),
            "producthunt":  fetch_producthunt(client, settings.producthunt_token),
            "reddit":       fetch_reddit_all(client),
            "github":       fetch_github_trending_all(client),
            "rss":          fetch_trend_rss_feeds(client),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    all_signals: list[dict] = []
    errors: list[str] = []
    source_counts: dict[str, int] = {}

    for category, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            errors.append(f"{category}: unhandled {type(result).__name__}: {result}")
            source_counts[category] = 0
            continue
        sigs, errs = result
        all_signals.extend(sigs)
        errors.extend(errs)
        source_counts[category] = len(sigs)

    # Sort by freshness desc — newest first
    all_signals.sort(key=lambda s: s["published_at"], reverse=True)

    new_in_db = await persist_signals(all_signals)
    return all_signals, new_in_db, errors, source_counts


# ============================================================================
# LangGraph node
# ============================================================================


async def trend_node(state: OracleState) -> dict:
    """LangGraph node — replaces the Step 1 placeholder."""
    log.info("trend: fetching HN + ProductHunt + Reddit + GitHub + VC-deals RSS")
    all_signals, new_in_db, errors, counts = await collect_trend_signals()

    breaking = sum(1 for s in all_signals if s["is_breaking"])
    log.info(
        "trend: %d entries · %d new in DB · %d breaking · "
        "hn=%d ph=%d reddit=%d github=%d rss=%d",
        len(all_signals), new_in_db, breaking,
        counts.get("hn", 0),
        counts.get("producthunt", 0),
        counts.get("reddit", 0),
        counts.get("github", 0),
        counts.get("rss", 0),
    )
    if errors:
        log.warning("trend: %d source error(s); first 3: %s", len(errors), errors[:3])

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
        "trend_signals": state_signals,
        "errors": errors,
    }


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.trend`
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
        all_signals, new_in_db, errors, counts = await collect_trend_signals()
        print()
        print("=" * 70)
        print("Trend collection summary")
        print("=" * 70)
        print(f"Sources tried:  hn, producthunt, reddit, github, rss")
        print(f"Errors:         {len(errors)}")
        print(f"Entries:        {len(all_signals)}")
        print(f"New in DB:      {new_in_db}")
        print(f"Breaking (<2h): {sum(1 for s in all_signals if s['is_breaking'])}")
        print()
        print("--- per-source breakdown ---")
        for cat, n in counts.items():
            print(f"  {cat:15s} {n}")
        print()
        print("--- top 8 freshest entries ---")
        for s in all_signals[:8]:
            print(f"  [{s['source_id']:25s}] {s['title'][:75]}")
        if errors:
            print()
            print("--- errors ---")
            for e in errors[:10]:
                print(f"  ! {e}")

    asyncio.run(_main())
