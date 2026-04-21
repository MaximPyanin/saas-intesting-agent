"""Validator agent — Step 12.

Outside-the-LLM sanity check on the critic's verdicts. The critic operates
on the LLM's training-data knowledge of competitors and market timing —
which can be wrong, stale, or hallucinated. The validator does the
real-world reality check:

  1. Web search (DuckDuckGo HTML) for each surviving idea's competitors.
     Verifies that critic-claimed "no competitors exist" actually holds up,
     and surfaces dominant competitors the critic missed.

  2. Brief market sizing from search snippets — even rough numbers help
     the user decide.

  3. For investment scenarios from Step 13: re-check current prices via
     state["market_data"] (collected ~30s earlier in this same run).
     A 4-hour-old "$50k BTC" signal is dangerous if BTC is now $48k.

Output flows two ways:
  - state["validated"]: surviving_ideas augmented with validation_notes,
    market_size_note, validator_verdict, and an updated competitors list
  - state["investment_scenarios"]: in-place augmented with price_drift_pct
    and is_stale flag

Search backend: httpx → html.duckduckgo.com (free, no auth, no new deps).
DDG sometimes rate-limits aggressive scraping but for our 5-10 queries
per ORACLE run it's fine. Per-query failures are isolated.

LLM: gpt-4o-mini (settings.openai_model_light) — reasoning over given
text is cheap and doesn't need full gpt-4o. ~$0.001-0.003 per validator
run. Without OPENAI_API_KEY: web search still runs, validator returns
unannotated ideas (graceful degradation).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from ..config import get_settings
from ..state import OracleState

log = logging.getLogger(__name__)


# ============================================================================
# Output schema
# ============================================================================

ValidatorVerdict = Literal["VALIDATED", "CONCERNS", "REJECTED"]


class IdeaValidation(BaseModel):
    """One validation result per surviving idea, indexed 1-based."""

    idea_index: int = Field(ge=1)
    competitors_verified: list[str] = Field(
        default_factory=list,
        description="Competitor names from the critic's list that were CONFIRMED via search.",
    )
    competitors_added: list[str] = Field(
        default_factory=list,
        description="NEW competitors found in search results that the critic missed.",
    )
    market_size_note: str = Field(
        default="",
        max_length=250,
        description="Brief market sizing inferred from search snippets, e.g. "
                    "'~$2B TAM, growing 30% YoY according to search results'.",
    )
    overall_verdict: ValidatorVerdict = Field(
        description="VALIDATED (critic was right) | CONCERNS (surprise competitor / "
                    "red flag found) | REJECTED (fatal flaw critic missed)",
    )
    validation_notes: str = Field(
        max_length=400,
        description="Concise reasoning for the verdict — what the search found.",
    )


class ValidatorOutput(BaseModel):
    idea_validations: list[IdeaValidation] = Field(default_factory=list)


# ============================================================================
# DuckDuckGo HTML search
# ============================================================================

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
DDG_TIMEOUT = 12.0
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


async def ddg_search(
    client: httpx.AsyncClient, query: str, max_results: int = 5
) -> list[dict]:
    """One DuckDuckGo HTML search. Returns list of {title, url, snippet}."""
    try:
        r = await client.post(
            DDG_HTML_URL,
            data={"q": query, "kl": "wt-wt"},  # wt-wt = no region bias
            headers={"User-Agent": BROWSER_UA},
            timeout=DDG_TIMEOUT,
        )
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        log.warning("ddg[%s]: fetch failed: %s", query[:40], type(e).__name__)
        return []

    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:  # noqa: BLE001
        log.warning("ddg[%s]: parse failed: %s", query[:40], e)
        return []

    results: list[dict] = []
    for div in soup.find_all("div", class_="result")[:max_results]:
        title_a = div.find("a", class_="result__a")
        if not title_a:
            continue
        snippet_el = div.find("a", class_="result__snippet")
        if not snippet_el:
            snippet_el = div.find("div", class_="result__snippet")

        url = title_a.get("href", "")
        # DDG sometimes wraps URLs in a redirector — strip the prefix
        if url.startswith("//duckduckgo.com/l/?uddg="):
            from urllib.parse import unquote
            try:
                url = unquote(url.split("uddg=", 1)[1].split("&", 1)[0])
            except Exception:
                pass

        results.append({
            "title": title_a.get_text(strip=True)[:160],
            "url": url[:200],
            "snippet": (snippet_el.get_text(strip=True) if snippet_el else "")[:300],
        })
    return results


async def search_competitors_for_ideas(
    surviving_ideas: list[dict],
) -> list[list[dict]]:
    """Run one DDG search per idea in parallel. Returns list of result-lists."""
    headers = {"User-Agent": BROWSER_UA}
    async with httpx.AsyncClient(timeout=DDG_TIMEOUT, headers=headers, follow_redirects=True) as client:
        tasks = []
        for idea in surviving_ideas:
            title = idea.get("title", "")
            # Query: title + "alternatives" tends to surface direct competitors
            query = f"{title} alternatives competitor"
            tasks.append(ddg_search(client, query, max_results=5))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    cleaned: list[list[dict]] = []
    for r in results:
        if isinstance(r, Exception):
            cleaned.append([])
        else:
            cleaned.append(r)
    return cleaned


# ============================================================================
# Validator system prompt
# ============================================================================


VALIDATOR_SYSTEM = """\
You are ORACLE's validator — the outside-the-LLM reality check.

You receive 3-5 business ideas that survived the critic's Reflexion loop.
For EACH idea, you also receive:
  - The competitors the critic listed (from the LLM's training-data memory)
  - The critic's notes about the idea
  - 5 LIVE web search results from DuckDuckGo (today, real, fetched seconds ago)

Your job: VERIFY the critic's competitor analysis against the live web results.

For each idea, set:

1. competitors_verified — names from the critic's list that ARE confirmed in
   search results (the search snippets mention them by name).

2. competitors_added — NEW competitors found in search results that the critic
   missed. These are the most valuable signals: critic blind spots.

3. market_size_note — a brief sentence inferred from search snippets. If a
   result mentions "$2B market" or "growing 30% YoY", capture it. If no
   sizing info exists in the snippets, say "no public sizing in search results".

4. overall_verdict — pick ONE:
   VALIDATED  → critic's competitor analysis holds up; no surprise dominant
                player found in search; the idea remains viable
   CONCERNS   → search surfaced a dominant competitor the critic missed,
                OR the listed competitors are massively dominant in the space,
                OR there's a red flag the critic didn't catch
   REJECTED   → search revealed a fatal flaw — clear single dominant player
                with massive scale, or the idea is already commoditized

5. validation_notes — concise reasoning (≤400 chars). Be specific. Cite which
   search result you're drawing from.

CRITICAL: stay grounded in the search results given. Do NOT invent
competitors not present in the snippets. If snippets don't have enough info,
say so in validation_notes and pick VALIDATED (don't punish the idea for
sparse search).

Output one IdeaValidation per input idea, indexed 1-based in the input order.
Respond ONLY with valid JSON.
"""


def format_validation_input(
    surviving_ideas: list[dict],
    search_results: list[list[dict]],
) -> str:
    """Build the user message: each idea + its critic context + search hits."""
    lines: list[str] = []
    for i, (idea, results) in enumerate(zip(surviving_ideas, search_results), start=1):
        title = idea.get("title", "(untitled)")
        listed = ", ".join((idea.get("competitors") or [])[:8]) or "(none listed)"
        notes = (idea.get("critic_notes") or "")[:250]

        lines.append(f"=== IDEA #{i}: {title} ===")
        lines.append(f"Critic-listed competitors: {listed}")
        if notes:
            lines.append(f"Critic notes: {notes}")
        lines.append(f"Live web search results ({len(results)} found):")
        if not results:
            lines.append("  (no search results — DDG returned empty)")
        for j, r in enumerate(results[:5], start=1):
            t = r.get("title", "")
            u = r.get("url", "")
            s = r.get("snippet", "")
            lines.append(f"  [{j}] {t}")
            if u:
                lines.append(f"      {u}")
            if s:
                lines.append(f"      {s}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# LLM call
# ============================================================================


async def call_validator_llm(
    surviving_ideas: list[dict],
    search_results: list[list[dict]],
) -> list[IdeaValidation]:
    """Single OpenAI call. Returns empty list on error."""
    settings = get_settings()
    from ..observability import has_llm_credentials  # noqa: PLC0415
    if not has_llm_credentials():
        log.info("validator: no LLM credentials — skipping LLM validation")
        return []

    try:
        from ..observability import get_openai_client, log_llm_usage  # noqa: PLC0415
    except ImportError:
        log.warning("validator: observability module unavailable")
        return []

    try:
        client = get_openai_client(agent="validator")
    except RuntimeError as e:
        log.error("validator: %s", e)
        return []

    user_msg = format_validation_input(surviving_ideas, search_results)

    log.info(
        "validator: calling %s with %d ideas (~%d KB user msg)",
        settings.openai_model_light,
        len(surviving_ideas),
        len(user_msg) // 1024,
    )

    try:
        response = await client.beta.chat.completions.parse(
            model=settings.openai_model_light,  # gpt-4o-mini — cheap reasoning
            messages=[
                {"role": "system", "content": VALIDATOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=ValidatorOutput,
            temperature=0.1,  # extremely grounded — no creativity
        )
    except Exception as e:  # noqa: BLE001
        log.error("validator: LLM call failed: %s", e)
        return []

    await log_llm_usage("validator", response)

    parsed = response.choices[0].message.parsed
    if not parsed:
        return []

    usage = response.usage
    if usage:
        log.info(
            "validator: %d validations · in=%d out=%d tokens (~$0.001)",
            len(parsed.idea_validations),
            usage.prompt_tokens,
            usage.completion_tokens,
        )
    return parsed.idea_validations


# ============================================================================
# Investment price re-verification
# ============================================================================

PRICE_DRIFT_STALE_THRESHOLD_PCT = 5.0


def lookup_current_price(asset_name: str, market_data: dict) -> float | None:
    """Find an asset's current price across all market_data buckets.

    Step 4 collects market data ~30s before the validator runs, so prices
    here are essentially live for the validation purpose.
    """
    name = asset_name.upper().strip()
    # Direct key match in any bucket
    for bucket_key in ("equities", "stocks", "crypto", "commodities", "forex", "indices"):
        bucket = market_data.get(bucket_key, {}) or {}
        if name in bucket:
            return float(bucket[name].get("price", 0)) or None
    # Fuzzy match: handles "Gold (XAU/USD)" → "GOLD"
    for bucket_key in ("equities", "stocks", "crypto", "commodities", "forex", "indices"):
        bucket = market_data.get(bucket_key, {}) or {}
        for key in bucket:
            if key.upper() in name or name.startswith(key.upper()):
                return float(bucket[key].get("price", 0)) or None
    return None


def validate_investment_prices(
    scenarios: list[dict],
    market_data: dict,
) -> list[dict]:
    """Augment each scenario with price_drift_pct + is_stale_at_validation."""
    if not scenarios or not market_data:
        return scenarios

    augmented: list[dict] = []
    stale_count = 0
    for sig in scenarios:
        sig = dict(sig)  # don't mutate the input
        asset = sig.get("asset", "")
        original_price = float(sig.get("price", 0) or 0)
        current_price = lookup_current_price(asset, market_data)

        if current_price and original_price:
            drift = (current_price - original_price) / original_price * 100
            sig["price_at_validation"] = current_price
            sig["price_drift_pct"] = round(drift, 2)
            sig["is_stale_at_validation"] = abs(drift) > PRICE_DRIFT_STALE_THRESHOLD_PCT
            if sig["is_stale_at_validation"]:
                stale_count += 1
        else:
            sig["price_at_validation"] = None
            sig["price_drift_pct"] = None
            sig["is_stale_at_validation"] = False
        augmented.append(sig)

    if stale_count:
        log.warning("validator: %d/%d investment scenarios are STALE (>%.0f%% drift)",
                    stale_count, len(scenarios), PRICE_DRIFT_STALE_THRESHOLD_PCT)
    return augmented


# ============================================================================
# LangGraph node
# ============================================================================


async def validator_node(state: OracleState) -> dict:
    """Step 12 — replaces the Step 1 placeholder.

    Validates surviving ideas via web search + LLM and re-checks investment
    prices via current market_data. Writes:
      - state["validated"]:           ideas with validation fields merged in
      - state["investment_scenarios"]: in-place augmented with price drift

    Without OpenAI key, web search still runs but the LLM step is skipped;
    ideas are returned as-is. The graph still completes successfully.
    """
    surviving = state.get("surviving_ideas", []) or []
    scenarios = state.get("investment_scenarios", []) or []
    market = state.get("market_data", {}) or {}

    log.info(
        "validator: %d surviving ideas · %d investment scenarios",
        len(surviving), len(scenarios),
    )

    if not surviving and not scenarios:
        return {"validated": [], "investment_scenarios": []}

    # 1. Idea validation: web search + LLM reasoning
    validated_ideas: list[dict] = []
    if surviving:
        log.info("validator: searching DuckDuckGo for %d ideas in parallel", len(surviving))
        search_results = await search_competitors_for_ideas(surviving)
        results_per_idea = sum(len(r) for r in search_results)
        log.info("validator: got %d total search results", results_per_idea)

        validations = await call_validator_llm(surviving, search_results)
        val_by_idx: dict[int, IdeaValidation] = {v.idea_index: v for v in validations}

        for i, idea in enumerate(surviving, start=1):
            idea = dict(idea)  # don't mutate state input
            v = val_by_idx.get(i)
            if v:
                merged_competitors = list(dict.fromkeys(  # dedup, preserve order
                    v.competitors_verified + v.competitors_added
                ))
                # Keep critic-listed competitors that we couldn't verify but
                # don't promote unverified — leave them off until search confirms
                if merged_competitors:
                    idea["competitors"] = merged_competitors
                idea["validation_notes"] = v.validation_notes
                idea["market_size_note"] = v.market_size_note
                idea["validator_verdict"] = v.overall_verdict
                log.info(
                    "  idea #%d %s: %s · %d verified · %d added",
                    i,
                    idea.get("title", "")[:50],
                    v.overall_verdict,
                    len(v.competitors_verified),
                    len(v.competitors_added),
                )
            else:
                idea["validator_verdict"] = "VALIDATED"  # default if LLM skipped
                idea["validation_notes"] = "Validator LLM not invoked (no API key or empty response)"
            validated_ideas.append(idea)

    # 2. Investment price re-check (in-place augment)
    validated_scenarios = validate_investment_prices(scenarios, market) if scenarios else []

    return {
        "validated": validated_ideas,
        "investment_scenarios": validated_scenarios,
    }


# ============================================================================
# Standalone CLI: `uv run python -m oracle.agents.validator`
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
        # Hardcoded sample for isolated testing
        sample_ideas: list[dict] = [
            {
                "title": "LLM Eval Suite for Production Agents",
                "competitors": ["Langfuse", "Helicone", "Arize Phoenix"],
                "critic_notes": "Self-hosted-first vs Langfuse cloud is a real wedge",
            },
            {
                "title": "Vertical Voice Agent Builder for Dental Clinics",
                "competitors": ["Vapi", "Bland.ai"],
                "critic_notes": "Dental vertical specifically — Vapi is horizontal",
            },
        ]
        sample_scenarios = [
            {"asset": "BTC", "price": 70000.0, "signal_type": "Macro"},
            {"asset": "Gold", "price": 4500.0, "signal_type": "Defensive"},
        ]
        sample_market = {
            "crypto": {"BTC": {"price": 69000, "change_24h": -1.4}},
            "commodities": {"GOLD": {"price": 4716}},
        }

        fake_state: OracleState = {
            "scout_signals": [], "market_data": sample_market, "trend_signals": [],
            "custom_signals": [], "synthesized": [], "raw_ideas": [],
            "investment_scenarios": sample_scenarios, "reflexion_round": 3,
            "surviving_ideas": sample_ideas, "validated": [], "final_digest": {},
            "errors": [],
        }
        result = await validator_node(fake_state)

        print()
        print("=" * 70)
        print(f"Validator output: {len(result.get('validated', []))} ideas validated")
        print("=" * 70)
        for idea in result.get("validated", []):
            print()
            print(f"  {idea.get('validator_verdict', '?')}: {idea.get('title')}")
            print(f"    Competitors (post-search): {', '.join(idea.get('competitors', []))}")
            print(f"    Market size: {idea.get('market_size_note', '(none)')}")
            print(f"    Notes: {idea.get('validation_notes', '(none)')[:180]}")

        print()
        print("=" * 70)
        print(f"Investment scenarios re-checked: {len(result.get('investment_scenarios', []))}")
        print("=" * 70)
        for sig in result.get("investment_scenarios", []):
            stale = " ⚠️ STALE" if sig.get("is_stale_at_validation") else ""
            print(f"  {sig['asset']}: {sig.get('price')} → {sig.get('price_at_validation')} "
                  f"({sig.get('price_drift_pct')}%){stale}")

    asyncio.run(_main())
