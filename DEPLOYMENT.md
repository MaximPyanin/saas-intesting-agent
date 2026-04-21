# ORACLE Deployment Guide — Azure Container Apps

End-to-end deployment guide for Step 19. Takes a fresh Azure subscription
to a running, auto-deploying ORACLE bot in ~15 minutes.

## Architecture

```
           ┌──────────────────────┐
           │  GitHub repository   │
           │    (source of truth) │
           └──────────┬───────────┘
                      │ git push main
                      ▼
           ┌──────────────────────┐
           │  GitHub Actions      │
           │  deploy.yml          │
           │                      │
           │  1. Azure OIDC login │
           │  2. az acr build     │──┐
           │  3. az containerapp  │  │
           │     update           │  │
           └──────────────────────┘  │
                                     │ push image
                                     ▼
           ┌──────────────────────┐
           │ Azure Container      │
           │ Registry (ACR)       │
           │ oracle:sha + latest  │
           └──────────┬───────────┘
                      │ pull
                      ▼
┌─────────────────────────────────────────────┐
│ Azure Container Apps Environment            │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │ oracle-bot container                │    │
│  │ - always on (min=1, max=1)          │    │
│  │ - 0.5 vCPU / 1 Gi                   │    │
│  │ - bot + scheduler + alerts poll     │    │
│  │ - WORKDIR=/app/data                 │    │
│  │                                     │    │
│  │ volume: oracle-data (Azure Files)   │────┼──► Storage Account
│  │   /app/data/oracle.db               │    │   (persistent)
│  │   /app/data/oracle_data.db          │    │
│  │   /app/data/oracle_reader.session   │    │
│  └────────────────────────────────────┘    │
│                                             │
│  Log Analytics Workspace ◄──── logs         │
└─────────────────────────────────────────────┘
                      │
                      ▼ polling
           ┌──────────────────────┐
           │  Telegram Bot API    │
           │  (send/receive)      │
           └──────────────────────┘
```

**Target cost:** ~$13-20/month
- Container App (always-on, 0.5 vCPU, 1 Gi): ~$8-12
- Azure Container Registry Basic: ~$5
- Azure Files (< 1 GB): ~$0.50
- Log Analytics (< 1 GB/month): ~$0-2
- **Plus** LLM costs: ~$11-13/month at 2x/day digests = **$24-33 total/month**

---

## Prerequisites

1. **Azure subscription** (free tier works for first 12 months, then pay-as-you-go)
2. **Azure CLI installed** → https://aka.ms/install-azure-cli
3. **GitHub repository** for this code
4. **Bash shell** (WSL on Windows, native on macOS/Linux)
5. **Bot credentials ready** in `.env`:
   - `TELEGRAM_BOT_TOKEN` from [@BotFather](https://t.me/BotFather)
   - `TELEGRAM_CHAT_ID` from [@userinfobot](https://t.me/userinfobot)
   - `OPENAI_API_KEY` from https://platform.openai.com/api-keys
   - Optional: `TELEGRAM_API_ID` + `TELEGRAM_API_HASH` from https://my.telegram.org
   - Optional: `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` from https://cloud.langfuse.com
   - Optional: `YOUTUBE_API_KEY`, `PRODUCTHUNT_TOKEN`, `FRED_API_KEY`, `REDDIT_*`

---

## Step-by-step deployment

### 1. Azure login

```bash
az login
az account set --subscription "YOUR_SUBSCRIPTION_NAME_OR_ID"
```

### 2. Provision infrastructure (one time)

Run the setup script from the project root:

```bash
chmod +x infra/azure/setup.sh
./infra/azure/setup.sh
```

Optional overrides via env vars:

```bash
AZURE_LOCATION=westeurope \
AZURE_RESOURCE_GROUP=oracle-prod \
AZURE_CONTAINER_APP=oracle-bot-prod \
./infra/azure/setup.sh
```

The script creates:
1. Resource Group (`oracle-rg`)
2. Log Analytics workspace (`oracle-bot-logs`)
3. Storage Account + Azure Files share (`oracle-data`, 5 GB quota)
4. Azure Container Registry (Basic SKU)
5. Container Apps Environment (with Azure Files mount)
6. **Initial image build via ACR Tasks** (runs in Azure, no local Docker needed)
7. Container App bound to image + volume + min-1/max-1 replica

Takes ~5-8 minutes on first run.

### 3. Set secrets from `.env`

```bash
# Make sure .env is filled in first
cp .env.example .env
$EDITOR .env

# Push secrets to Azure
chmod +x infra/azure/update-secrets.sh
./infra/azure/update-secrets.sh .env
```

The bot restarts automatically (~60 seconds for new revision to become healthy).

### 4. Verify the bot is running

Tail live logs:

```bash
az containerapp logs show \
  --name oracle-bot \
  --resource-group oracle-rg \
  --follow
```

You should see:
```
oracle.db: init_db: ensuring schema at oracle_data.db (version 3)
oracle.scheduler: initializing with timezone=Europe/Warsaw
oracle.scheduler: job 'morning_brief' next run = ...
oracle.scheduler: job 'evening_digest' next run = ...
oracle.scheduler: job 'alerts_poll' next run = ...
oracle.bot: ORACLE bot ready — polling will start now
oracle.bot: ORACLE bot starting (polling mode)...
```

Open your bot in Telegram and send `/start` — you should get the welcome message.

### 5. Set up GitHub Actions for auto-deploy

#### 5.1 Create a service principal with federated credential (OIDC)

```bash
# Get your subscription and tenant IDs
SUB_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)

# Create the service principal scoped to the resource group
APP_ID=$(az ad app create --display-name oracle-github-deploy --query appId -o tsv)
az ad sp create --id "$APP_ID" -o none

# Grant it Contributor role on the resource group only
az role assignment create \
  --role Contributor \
  --assignee "$APP_ID" \
  --scope /subscriptions/$SUB_ID/resourceGroups/oracle-rg

# Add federated credential so GitHub Actions can log in without a client secret
# Replace OWNER/REPO below with your GitHub repo path
cat > /tmp/oracle-fic.json <<EOF
{
  "name": "oracle-main-branch",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:OWNER/REPO:ref:refs/heads/main",
  "description": "ORACLE main branch deploys",
  "audiences": ["api://AzureADTokenExchange"]
}
EOF

az ad app federated-credential create \
  --id "$APP_ID" \
  --parameters /tmp/oracle-fic.json

echo "AZURE_CLIENT_ID: $APP_ID"
echo "AZURE_TENANT_ID: $TENANT_ID"
echo "AZURE_SUBSCRIPTION_ID: $SUB_ID"
```

#### 5.2 Add repo secrets on GitHub

Go to: **Settings → Secrets and variables → Actions → New repository secret**

Add these (values from the output of step 5.1 and your setup.sh run):

| Secret | Value |
|---|---|
| `AZURE_CLIENT_ID` | Service principal appId |
| `AZURE_TENANT_ID` | Tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Subscription ID |
| `AZURE_RESOURCE_GROUP` | `oracle-rg` |
| `AZURE_CONTAINER_REGISTRY` | ACR name (NOT the `.azurecr.io` FQDN) |
| `AZURE_CONTAINER_APP` | `oracle-bot` |

#### 5.3 Trigger first auto-deploy

```bash
git add .
git commit -m "Step 19: Azure deployment"
git push origin main
```

Watch the deploy at: **Actions → Deploy ORACLE to Azure Container Apps**

Subsequent pushes to `main` that touch `src/`, `pyproject.toml`, `uv.lock`, `Dockerfile`, or the workflow itself will auto-deploy.

---

## Day-to-day operations

### Tail live logs

```bash
az containerapp logs show --name oracle-bot --resource-group oracle-rg --follow
```

### Get a shell inside the running container

```bash
az containerapp exec --name oracle-bot --resource-group oracle-rg --command bash
```

From inside the container you can inspect the DB:
```bash
cd /app/data
python -c "import sqlite3; c=sqlite3.connect('oracle_data.db'); print(c.execute('SELECT COUNT(*) FROM signals').fetchone())"
```

### One-time Telegram auth (needed for TG custom sources)

```bash
az containerapp exec --name oracle-bot --resource-group oracle-rg --command bash
# inside container:
cd /app/data && python -m oracle.agents.custom auth-tg
# follow the phone number + code prompts
# oracle_reader.session file is created in /app/data (persistent volume)
exit
```

### Manually trigger a digest (via bot)

Open the bot in Telegram and send `/digest`. You don't need to wait for 19:00.

### Pause the bot

```
/pause 3     # in Telegram — pauses scheduled morning/evening + alerts for 3 days
/pause 0     # resume
```

Manual `/digest` and `/morning` commands still work while paused.

### Update secrets without redeploying code

```bash
# Edit .env locally, then:
./infra/azure/update-secrets.sh .env
```

The Container App rolls to a new revision with updated env vars in ~60 seconds. Volume data (feedback history, DBs) is preserved.

### Scale down to save money (pause all costs)

```bash
# Temporarily stop — volume keeps everything
az containerapp update \
  --name oracle-bot --resource-group oracle-rg \
  --min-replicas 0 --max-replicas 0
```

Resume:

```bash
az containerapp update \
  --name oracle-bot --resource-group oracle-rg \
  --min-replicas 1 --max-replicas 1
```

### Tear down EVERYTHING (destructive)

```bash
az group delete --name oracle-rg --yes --no-wait
```

Deletes all resources including the persistent volume. **Feedback history, learning weights, all DBs gone.** Only do this if you're done with ORACLE.

---

## Troubleshooting

### "Container is unhealthy"

Check the healthcheck by entering the container:
```bash
az containerapp exec --name oracle-bot --resource-group oracle-rg --command bash
cd /app/data && python -c "import sqlite3; sqlite3.connect('oracle_data.db').execute('SELECT 1').fetchone()"
```

If this fails, the DB file is missing or corrupt — check Azure Files mount:
```bash
ls -la /app/data
```

### Bot doesn't respond to `/start`

1. Verify `TELEGRAM_BOT_TOKEN` is set:
   ```bash
   az containerapp show --name oracle-bot --resource-group oracle-rg \
     --query properties.template.containers[0].env[?name=='TELEGRAM_BOT_TOKEN']
   ```
2. Check logs for polling errors:
   ```bash
   az containerapp logs show --name oracle-bot --resource-group oracle-rg --tail 100
   ```
3. Make sure the bot is unblocked in Telegram (send it a DM first via @BotFather's link).

### Scheduled jobs not firing at 07:00 / 19:00

1. Verify `TZ=Europe/Warsaw` is set:
   ```bash
   az containerapp exec --name oracle-bot --resource-group oracle-rg --command date
   # should show Warsaw time
   ```
2. Check `/pause` state in the bot — maybe you paused it.
3. Check scheduler logs:
   ```bash
   az containerapp logs show --name oracle-bot --resource-group oracle-rg --tail 200 \
     | grep scheduler
   ```

### GitHub Actions deploy fails with "OIDC login failed"

The federated credential subject must match `repo:OWNER/REPO:ref:refs/heads/main` exactly. Re-check step 5.1.

### Image build fails with "deps not resolvable"

`uv.lock` may be stale. Locally run `uv lock` and commit the updated file.

---

## Monitoring

### Cost

```bash
# Quick cost check for last 30 days on this resource group
az consumption usage list \
  --start-date $(date -d '30 days ago' +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --query "[?contains(instanceName, 'oracle')].{Name:instanceName, Cost:pretaxCost}" \
  -o table
```

### LLM spend (from bot itself)

Send `/cost` to the bot — shows breakdown from `llm_cost_log` table (Step 17).

### Langfuse traces

If `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set, every LLM call is auto-traced. View at https://cloud.langfuse.com/ (free tier).

### Container App metrics

Azure Portal → Container Apps → oracle-bot → Metrics.
Key metrics: CPU, memory, restart count, replica count.

---

## Upgrading ORACLE

To upgrade LLM models (e.g. gpt-4o → gpt-5):

```bash
# Update .env
# OPENAI_MODEL_HEAVY=gpt-5

./infra/azure/update-secrets.sh .env
```

Also add pricing for the new model to `src/oracle/observability.py` `MODEL_PRICING` dict and commit — next push auto-deploys with accurate cost tracking.

To add new collectors or agents: edit `src/oracle/agents/`, push to main, auto-deploy.

To change the scheduler cron times: edit `src/oracle/scheduler.py`, push, auto-deploy.
