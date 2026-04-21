#!/usr/bin/env bash
# ============================================================================
# ORACLE — sync secrets from local .env to Azure Container App (Step 19)
# ============================================================================
# Usage:
#   ./infra/azure/update-secrets.sh [path-to-.env]
#
# Default path: ./.env
#
# What it does:
#   1. Sources every key from the .env file
#   2. Pushes them as Container App secrets (encrypted at rest in Azure)
#   3. Binds secrets to container env vars via secretref:<name>
#   4. Triggers a new revision automatically
#
# The Container App will restart with the new secrets within ~60 seconds.
# Volume data (oracle_data.db, feedback history, etc.) is preserved.
# ============================================================================

set -euo pipefail

ENV_FILE="${1:-.env}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-oracle-rg}"
APP_NAME="${AZURE_CONTAINER_APP:-oracle-bot}"

[[ -f "$ENV_FILE" ]] || { echo "ERROR: $ENV_FILE not found"; exit 1; }
command -v az >/dev/null || { echo "ERROR: az CLI not installed"; exit 1; }
az account show >/dev/null 2>&1 || { echo "ERROR: run 'az login' first"; exit 1; }

# Source the env file safely
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

echo "Syncing secrets from $ENV_FILE to Container App '$APP_NAME'..."
echo

# ---------- 1. Push secrets (encrypted at rest) ----------
# Only include keys that are actually set (non-empty) to avoid blanking
# values already in Azure.
SECRETS=()

add_secret() {
    local name="$1"
    local value="$2"
    if [[ -n "$value" ]]; then
        SECRETS+=("$name=$value")
    fi
}

add_secret "telegram-bot-token"     "${TELEGRAM_BOT_TOKEN:-}"
add_secret "telegram-chat-id"       "${TELEGRAM_CHAT_ID:-}"
add_secret "telegram-api-id"        "${TELEGRAM_API_ID:-}"
add_secret "telegram-api-hash"      "${TELEGRAM_API_HASH:-}"
add_secret "openai-api-key"         "${OPENAI_API_KEY:-}"
add_secret "azure-openai-api-key"   "${AZURE_OPENAI_API_KEY:-}"
add_secret "coingecko-api-key"      "${COINGECKO_API_KEY:-}"
add_secret "fred-api-key"           "${FRED_API_KEY:-}"
add_secret "producthunt-token"      "${PRODUCTHUNT_TOKEN:-}"
add_secret "reddit-client-id"       "${REDDIT_CLIENT_ID:-}"
add_secret "reddit-client-secret"   "${REDDIT_CLIENT_SECRET:-}"
add_secret "youtube-api-key"        "${YOUTUBE_API_KEY:-}"
add_secret "langfuse-public-key"    "${LANGFUSE_PUBLIC_KEY:-}"
add_secret "langfuse-secret-key"    "${LANGFUSE_SECRET_KEY:-}"

if [[ ${#SECRETS[@]} -eq 0 ]]; then
    echo "ERROR: no non-empty values found in $ENV_FILE"
    exit 1
fi

echo "[1/2] Writing ${#SECRETS[@]} encrypted secrets..."
az containerapp secret set \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --secrets "${SECRETS[@]}" \
    -o none

# ---------- 2. Bind secrets to env vars + also set non-sensitive config ----------
echo "[2/2] Binding secrets to container env vars + updating config..."

ENV_VARS=()

# Secrets — reference by secretref
add_ref() {
    local var="$1"
    local secret="$2"
    local source_value="$3"
    if [[ -n "$source_value" ]]; then
        ENV_VARS+=("$var=secretref:$secret")
    fi
}

add_ref "TELEGRAM_BOT_TOKEN"     "telegram-bot-token"     "${TELEGRAM_BOT_TOKEN:-}"
add_ref "TELEGRAM_CHAT_ID"       "telegram-chat-id"       "${TELEGRAM_CHAT_ID:-}"
add_ref "TELEGRAM_API_ID"        "telegram-api-id"        "${TELEGRAM_API_ID:-}"
add_ref "TELEGRAM_API_HASH"      "telegram-api-hash"      "${TELEGRAM_API_HASH:-}"
add_ref "OPENAI_API_KEY"         "openai-api-key"         "${OPENAI_API_KEY:-}"
add_ref "AZURE_OPENAI_API_KEY"   "azure-openai-api-key"   "${AZURE_OPENAI_API_KEY:-}"
add_ref "COINGECKO_API_KEY"      "coingecko-api-key"      "${COINGECKO_API_KEY:-}"
add_ref "FRED_API_KEY"           "fred-api-key"           "${FRED_API_KEY:-}"
add_ref "PRODUCTHUNT_TOKEN"      "producthunt-token"      "${PRODUCTHUNT_TOKEN:-}"
add_ref "REDDIT_CLIENT_ID"       "reddit-client-id"       "${REDDIT_CLIENT_ID:-}"
add_ref "REDDIT_CLIENT_SECRET"   "reddit-client-secret"   "${REDDIT_CLIENT_SECRET:-}"
add_ref "YOUTUBE_API_KEY"        "youtube-api-key"        "${YOUTUBE_API_KEY:-}"
add_ref "LANGFUSE_PUBLIC_KEY"    "langfuse-public-key"    "${LANGFUSE_PUBLIC_KEY:-}"
add_ref "LANGFUSE_SECRET_KEY"    "langfuse-secret-key"    "${LANGFUSE_SECRET_KEY:-}"

# Non-sensitive config — pass through directly
ENV_VARS+=(
    "TIMEZONE=${TIMEZONE:-Europe/Warsaw}"
    "OPENAI_MODEL_HEAVY=${OPENAI_MODEL_HEAVY:-gpt-4o}"
    "OPENAI_MODEL_LIGHT=${OPENAI_MODEL_LIGHT:-gpt-4o-mini}"
    "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-}"
    "AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-10-21}"
    "IDEA_FOCUS_WEIGHT=${IDEA_FOCUS_WEIGHT:-0.70}"
    "REFLEXION_MAX_ROUNDS=${REFLEXION_MAX_ROUNDS:-3}"
    "MAX_IDEAS_PER_DIGEST=${MAX_IDEAS_PER_DIGEST:-3}"
    "MAX_INVESTMENT_SIGNALS=${MAX_INVESTMENT_SIGNALS:-3}"
    "MIN_IDEA_SCORE=${MIN_IDEA_SCORE:-70}"
    "MIN_INVESTMENT_SIGNAL=${MIN_INVESTMENT_SIGNAL:-6}"
    "TELEGRAM_SESSION_NAME=${TELEGRAM_SESSION_NAME:-oracle_reader}"
)

az containerapp update \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --set-env-vars "${ENV_VARS[@]}" \
    -o none

echo
echo "✓ Secrets and env vars updated. New revision rolling out..."
echo
echo "Tail logs to verify the bot restarted cleanly:"
echo
echo "  az containerapp logs show \\"
echo "    --name $APP_NAME \\"
echo "    --resource-group $RESOURCE_GROUP \\"
echo "    --follow"
echo
