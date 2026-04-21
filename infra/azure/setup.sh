#!/usr/bin/env bash
# ============================================================================
# ORACLE — Azure Container Apps one-time setup (Step 19)
# ============================================================================
# Creates all Azure resources ORACLE needs:
#   1. Resource Group
#   2. Log Analytics Workspace (container logs)
#   3. Storage Account + File Share (persistent volume for oracle.db/
#      oracle_data.db/oracle_reader.session)
#   4. Azure Container Registry (image hosting)
#   5. Container Apps Environment (with Log Analytics + Azure Files mount)
#   6. Initial image build via ACR Tasks (no local docker needed)
#   7. Container App bound to image + volume + min 1 replica
#
# Run ONCE after creating the project. Subsequent code pushes deploy via
# GitHub Actions (.github/workflows/deploy.yml).
#
# Prereqs:
#   - Azure CLI installed (`az`)
#   - Logged in: `az login`
#   - Subscription selected: `az account set --subscription <id>`
#   - Bash (WSL/Git Bash on Windows, native on Linux/macOS)
#
# Cost target (per month):
#   - Container App (0.5 vCPU, 1 Gi, min 1 replica, always on): ~$8-12
#   - Container Registry (Basic SKU):                            ~$5
#   - Log Analytics (<1 GB/month):                                ~$0-2
#   - Storage Account (Azure Files, <1 GB):                       ~$0.50
#   - Total:                                                     ~$13-20
# ============================================================================

set -euo pipefail

# ---------- CONFIGURATION (override via env vars) ----------
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-oracle-rg}"
LOCATION="${AZURE_LOCATION:-northeurope}"   # closest to Warsaw (~15ms latency)
ACR_NAME="${AZURE_CONTAINER_REGISTRY:-oraclereg$(date +%s | tail -c 8)}"  # globally unique
ENVIRONMENT="${AZURE_CONTAINER_ENV:-oracle-env}"
APP_NAME="${AZURE_CONTAINER_APP:-oracle-bot}"
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-oraclestor$(date +%s | tail -c 8)}"  # globally unique
FILE_SHARE="${AZURE_FILE_SHARE:-oracle-data}"
STORAGE_NAME_IN_ENV="${AZURE_ENV_STORAGE_NAME:-oracle-data-mount}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "============================================================"
echo "ORACLE Azure Container Apps setup"
echo "============================================================"
echo "Resource Group:     $RESOURCE_GROUP"
echo "Location:           $LOCATION"
echo "Container Registry: $ACR_NAME"
echo "Storage Account:    $STORAGE_ACCOUNT"
echo "File Share:         $FILE_SHARE"
echo "Environment:        $ENVIRONMENT"
echo "App Name:           $APP_NAME"
echo "============================================================"
echo

# ---------- PREREQ CHECKS ----------
command -v az >/dev/null || { echo "ERROR: az CLI not installed. See https://aka.ms/install-azure-cli"; exit 1; }

az account show >/dev/null 2>&1 || { echo "ERROR: not logged in. Run 'az login' first"; exit 1; }

SUBSCRIPTION=$(az account show --query name -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription: $SUBSCRIPTION ($SUBSCRIPTION_ID)"
echo

# Register required providers (idempotent)
echo "[0/7] Registering Azure providers (one-time, may take 1-2 min first run)..."
az provider register --namespace Microsoft.App -o none
az provider register --namespace Microsoft.OperationalInsights -o none
az provider register --namespace Microsoft.ContainerRegistry -o none
az provider register --namespace Microsoft.Storage -o none

# ---------- 1. Resource Group ----------
echo "[1/7] Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none

# ---------- 2. Log Analytics Workspace ----------
echo "[2/7] Creating Log Analytics workspace..."
az monitor log-analytics workspace create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "${APP_NAME}-logs" \
    --location "$LOCATION" \
    -o none

LOG_WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "${APP_NAME}-logs" \
    --query customerId -o tsv)
LOG_WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "${APP_NAME}-logs" \
    --query primarySharedKey -o tsv)

# ---------- 3. Storage Account + File Share (persistent volume) ----------
echo "[3/7] Creating Storage Account for persistent volume..."
az storage account create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$STORAGE_ACCOUNT" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --enable-large-file-share false \
    -o none

STORAGE_KEY=$(az storage account keys list \
    --resource-group "$RESOURCE_GROUP" \
    --account-name "$STORAGE_ACCOUNT" \
    --query '[0].value' -o tsv)

az storage share-rm create \
    --resource-group "$RESOURCE_GROUP" \
    --storage-account "$STORAGE_ACCOUNT" \
    --name "$FILE_SHARE" \
    --quota 5 \
    -o none

# ---------- 4. Container Registry ----------
echo "[4/7] Creating Azure Container Registry..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    -o none

ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query 'passwords[0].value' -o tsv)

# ---------- 5. Container Apps Environment + Azure Files mount ----------
echo "[5/7] Creating Container Apps environment..."
az containerapp env create \
    --name "$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --logs-workspace-id "$LOG_WORKSPACE_ID" \
    --logs-workspace-key "$LOG_WORKSPACE_KEY" \
    -o none

echo "      Mounting Azure Files share into environment..."
az containerapp env storage set \
    --name "$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" \
    --storage-name "$STORAGE_NAME_IN_ENV" \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$FILE_SHARE" \
    --access-mode ReadWrite \
    -o none

# ---------- 6. Build initial image via ACR Tasks (no local Docker required) ----------
echo "[6/7] Building initial image via ACR Tasks..."
# cd to project root (assumes this script is at infra/azure/setup.sh)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

az acr build \
    --registry "$ACR_NAME" \
    --image "oracle:$IMAGE_TAG" \
    --file Dockerfile \
    .

# ---------- 7. Create Container App with volume mount ----------
echo "[7/7] Creating Container App..."

# We need to use a YAML spec to attach the volume mount — az containerapp create
# doesn't accept volume bindings via CLI flags directly.
APP_YAML=$(mktemp /tmp/oracle-app.XXXXXX.yaml)
trap 'rm -f "$APP_YAML"' EXIT

cat > "$APP_YAML" <<YAML
properties:
  managedEnvironmentId: /subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.App/managedEnvironments/${ENVIRONMENT}
  configuration:
    activeRevisionsMode: Single
    registries:
      - server: ${ACR_SERVER}
        username: ${ACR_USERNAME}
        passwordSecretRef: registry-password
    secrets:
      - name: registry-password
        value: ${ACR_PASSWORD}
  template:
    containers:
      - name: oracle-bot
        image: ${ACR_SERVER}/oracle:${IMAGE_TAG}
        resources:
          cpu: 0.5
          memory: 1Gi
        env:
          - name: TZ
            value: Europe/Warsaw
          - name: TIMEZONE
            value: Europe/Warsaw
          - name: PYTHONUNBUFFERED
            value: "1"
          - name: PYTHONUTF8
            value: "1"
          - name: PYTHONIOENCODING
            value: utf-8
        volumeMounts:
          - volumeName: oracle-data
            mountPath: /app/data
    scale:
      minReplicas: 1
      maxReplicas: 1
    volumes:
      - name: oracle-data
        storageType: AzureFile
        storageName: ${STORAGE_NAME_IN_ENV}
YAML

az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --yaml "$APP_YAML" \
    -o none

# ---------- SUCCESS OUTPUT ----------
echo
echo "============================================================"
echo "✓ Azure setup complete"
echo "============================================================"
echo
echo "Resources created:"
echo "  Resource group:     $RESOURCE_GROUP"
echo "  ACR:                $ACR_SERVER"
echo "  Storage account:    $STORAGE_ACCOUNT"
echo "  File share:         $FILE_SHARE"
echo "  Environment:        $ENVIRONMENT"
echo "  Container app:      $APP_NAME"
echo
echo "Next steps:"
echo
echo "1. Set secrets (required to start the bot):"
echo
echo "   ./infra/azure/update-secrets.sh .env"
echo
echo "2. Configure GitHub Actions for auto-deploy on push:"
echo
echo "   Set these repo secrets (Settings → Secrets → Actions):"
echo "     AZURE_CLIENT_ID          (from service principal, step 3 below)"
echo "     AZURE_TENANT_ID          $TENANT_ID"
echo "     AZURE_SUBSCRIPTION_ID    $SUBSCRIPTION_ID"
echo "     AZURE_RESOURCE_GROUP     $RESOURCE_GROUP"
echo "     AZURE_CONTAINER_REGISTRY $ACR_NAME"
echo "     AZURE_CONTAINER_APP      $APP_NAME"
echo
echo "3. Create a service principal for GitHub Actions OIDC:"
echo
echo "   az ad sp create-for-rbac \\"
echo "     --name oracle-github-deploy \\"
echo "     --role contributor \\"
echo "     --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP \\"
echo "     --json-auth"
echo
echo "   Then add federated credential:"
echo "   https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure"
echo
echo "4. Check the bot is running:"
echo
echo "   az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo
echo "5. To tear down everything (DESTRUCTIVE):"
echo
echo "   az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo
