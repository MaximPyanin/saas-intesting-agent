"""ORACLE Telegram bot — entry point.

Usage:
    uv run python -m oracle.bot.main              # live mode (requires TELEGRAM_BOT_TOKEN in .env)
    uv run python -m oracle.bot.main --dry-run    # render every template to stdout, exercise feedback DB write

Live mode runs polling (long polling). Step 19 will switch to webhook mode
on Azure Container Apps. The bot keeps in-memory storage of the current
digest's items in `application.bot_data` so callbacks can look up which
idea/signal a button refers to.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

# Windows console (cp1251) can't print emoji/Unicode by default — force UTF-8.
# Telegram itself is UTF-8 native; this is only for dry-run stdout printing.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

from ..agents.custom import load_active_sources
from ..config import get_settings
from ..db import init_db
from ..scheduler import setup_scheduler, shutdown_scheduler
from .handlers import (
    ASK_ADJUST_AMOUNT,
    ASK_CATEGORY,
    ASK_CONFIRM,
    ASK_HOLDING_PRICE,
    ASK_HOLDING_QTY,
    ASK_NAME,
    ASK_NEW_AMOUNT,
    ASK_TYPE,
    ASK_URL,
    PICK_NEW_TICKER,
    add_holding_callback_entry,
    add_holding_cancel_cmd,
    add_holding_price,
    add_holding_qty,
    pf_addnew_amount,
    pf_addnew_cancel_cmd,
    pf_addnew_entry,
    pf_addnew_pick,
    pf_adjust_amount,
    pf_adjust_callback_entry,
    pf_adjust_cancel_cmd,
    add_source_callback_entry,
    add_source_cancel_cb,
    add_source_cancel_cmd,
    add_source_category,
    add_source_confirm,
    add_source_name,
    add_source_type,
    add_source_url,
    callback_handler,
    cmd_add_holding,
    cmd_add_source,
    cmd_calibrate,
    cmd_clearfeedback,
    cmd_cost,
    cmd_digest,
    cmd_history,
    cmd_lastdigest,
    cmd_more,
    cmd_morning,
    cmd_pause,
    cmd_preferences,
    cmd_portfolio,
    cmd_remove_holding,
    cmd_saved,
    cmd_settings,
    cmd_sources,
    cmd_start,
    cmd_stats,
)
from .mock import (
    mock_business_ideas,
    mock_investment_signals,
    mock_morning_data,
)
from .views import (
    render_idea_card,
    render_investment_card,
    render_morning_brief,
    render_real_sources_card,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("oracle.bot")


def build_application():
    """Construct the python-telegram-bot Application with all handlers wired up."""
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Copy .env.example to .env and fill in the token from @BotFather."
        )

    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    # /add_source ConversationHandler — MUST be registered before the generic
    # CallbackQueryHandler so it captures button presses for its own states.
    add_source_conv = ConversationHandler(
        entry_points=[
            CommandHandler("add_source", cmd_add_source),
            CallbackQueryHandler(add_source_callback_entry, pattern="^add_source$"),
        ],
        states={
            ASK_TYPE: [
                CallbackQueryHandler(add_source_type, pattern="^addtype_"),
                CallbackQueryHandler(add_source_cancel_cb, pattern="^addsrc_cancel$"),
            ],
            ASK_CATEGORY: [
                CallbackQueryHandler(add_source_category, pattern="^addcat_"),
                CallbackQueryHandler(add_source_cancel_cb, pattern="^addsrc_cancel$"),
            ],
            ASK_URL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_source_url),
            ],
            ASK_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_source_name),
            ],
            ASK_CONFIRM: [
                CallbackQueryHandler(add_source_confirm, pattern="^addconf_"),
                CallbackQueryHandler(add_source_cancel_cb, pattern="^addsrc_cancel$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", add_source_cancel_cmd)],
        per_chat=True,
        per_user=True,
    )
    app.add_handler(add_source_conv)

    # Add-to-portfolio ConversationHandler (Step 20.5).
    # Triggered by "➕ В мой портфель" button under each investment card.
    # Registered BEFORE the generic CallbackQueryHandler so it captures the
    # `inv_add_<sig_id>` pattern first.
    add_holding_conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(add_holding_callback_entry, pattern="^inv_add_"),
        ],
        states={
            ASK_HOLDING_QTY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_holding_qty),
            ],
            ASK_HOLDING_PRICE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_holding_price),
            ],
        },
        fallbacks=[CommandHandler("cancel", add_holding_cancel_cmd)],
        per_chat=True,
        per_user=True,
    )
    app.add_handler(add_holding_conv)

    # Portfolio adjust (+$/-$) ConversationHandler.
    # Captures pf_addusd_<LABEL> and pf_subusd_<LABEL> callbacks before the
    # generic dispatcher, then asks for amount via text.
    pf_adjust_conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(pf_adjust_callback_entry, pattern="^pf_(addusd|subusd)_"),
        ],
        states={
            ASK_ADJUST_AMOUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, pf_adjust_amount),
            ],
        },
        fallbacks=[CommandHandler("cancel", pf_adjust_cancel_cmd)],
        per_chat=True,
        per_user=True,
    )
    app.add_handler(pf_adjust_conv)

    # Portfolio "add new position from scratch" ConversationHandler.
    # Triggered by `pf_addnew` button under /portfolio footer.
    pf_addnew_conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(pf_addnew_entry, pattern="^pf_addnew$"),
        ],
        states={
            PICK_NEW_TICKER: [
                CallbackQueryHandler(pf_addnew_pick, pattern="^pf_pick_"),
            ],
            ASK_NEW_AMOUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, pf_addnew_amount),
            ],
        },
        fallbacks=[CommandHandler("cancel", pf_addnew_cancel_cmd)],
        per_chat=True,
        per_user=True,
    )
    app.add_handler(pf_addnew_conv)

    # Other slash commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("more", cmd_more))
    app.add_handler(CommandHandler("lastdigest", cmd_lastdigest))
    app.add_handler(CommandHandler("preferences", cmd_preferences))
    app.add_handler(CommandHandler("morning", cmd_morning))
    app.add_handler(CommandHandler("sources", cmd_sources))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("saved", cmd_saved))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("calibrate", cmd_calibrate))  # Step 14
    app.add_handler(CommandHandler("clearfeedback", cmd_clearfeedback))
    app.add_handler(CommandHandler("cost", cmd_cost))            # Step 17
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))           # Step 20
    app.add_handler(CommandHandler("add_holding", cmd_add_holding))       # Step 20
    app.add_handler(CommandHandler("remove_holding", cmd_remove_holding)) # Step 20

    # Generic dispatcher for all OTHER inline buttons (likes/dislikes/save/stats etc.)
    # The conversation handler above is registered first so its patterns match first.
    app.add_handler(CallbackQueryHandler(callback_handler))

    return app


async def _post_init(application) -> None:
    """Run after the Application is built but before polling starts.

    1. Ensures the domain DB schema exists so feedback writes succeed.
    2. Starts the APScheduler for 07:00 morning brief + 19:00 evening digest
       (Step 15). The scheduler lives in `application.bot_data["scheduler"]`.
    """
    await init_db()
    setup_scheduler(application)
    log.info("ORACLE bot ready — polling will start now")


async def _post_shutdown(application) -> None:
    """Cleanly stop the scheduler when the bot shuts down."""
    shutdown_scheduler(application)


# ============================================================================
# Dry-run mode — verify templates and DB without needing a real bot token
# ============================================================================


async def dry_run() -> None:
    """Render every template to stdout and exercise the feedback DB write path."""
    print("=" * 70)
    print("ORACLE bot — DRY RUN")
    print("=" * 70)

    # 1. Init DB so feedback writes work
    await init_db()
    print("\n✓ oracle_data.db initialized\n")

    # 2. Render morning brief
    print("\n--- /morning ---\n")
    print(render_morning_brief(mock_morning_data()))

    # 3. Render every idea card
    print("\n--- /digest → IDEA CARDS (3) ---\n")
    for i, idea in enumerate(mock_business_ideas(), start=1):
        print(render_idea_card(idea, i))
        print()

    # 4. Render every investment card
    print("\n--- /digest → INVESTMENT CARDS (3) ---\n")
    for i, sig in enumerate(mock_investment_signals(), start=1):
        print(render_investment_card(sig, i))
        print()

    # 5. Render sources card from REAL user_sources table (Step 8)
    print("\n--- /sources (real data) ---\n")
    sources = await load_active_sources()
    print(render_real_sources_card(sources))

    # 6. Exercise feedback DB write path (without going through Telegram)
    print("\n--- feedback DB write smoke test ---\n")
    from datetime import datetime, timezone
    import json
    from ..db import get_db

    sample_idea = mock_business_ideas()[0]
    async with get_db() as conn:
        await conn.execute(
            """INSERT INTO feedback (item_type, item_id, digest_id, feedback_type,
                                       reason, item_snapshot, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "idea",
                "dry-run-1",
                "dry-run-digest",
                "like",
                None,
                json.dumps(sample_idea.model_dump(), default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await conn.commit()
        async with conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE digest_id = 'dry-run-digest'"
        ) as cur:
            count = (await cur.fetchone())[0]
    print(f"✓ feedback rows for dry-run-digest: {count}")
    print("✓ Step 3 dry-run complete — every template rendered, DB write verified")


def main() -> None:
    parser = argparse.ArgumentParser(description="ORACLE Telegram bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render all templates to stdout, no Telegram API calls",
    )
    args = parser.parse_args()

    if args.dry_run:
        asyncio.run(dry_run())
        return

    try:
        app = build_application()
    except ValueError as e:
        log.error("%s", e)
        log.info("Hint: copy .env.example to .env and set TELEGRAM_BOT_TOKEN")
        return

    log.info("ORACLE bot starting (polling mode)...")
    app.run_polling()


if __name__ == "__main__":
    main()
