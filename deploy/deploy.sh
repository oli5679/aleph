#!/bin/bash
set -e

# Aleph Lichess Bot - Cloud Run Deployment Script
# Usage: ./deploy.sh <PROJECT_ID> <LICHESS_TOKEN> [REGION] [TIER]

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./deploy.sh <PROJECT_ID> <LICHESS_TOKEN> [REGION] [TIER]"
    echo ""
    echo "  PROJECT_ID    - Your GCP project ID"
    echo "  LICHESS_TOKEN - Your Lichess API token"
    echo "  REGION        - GCP region (default: us-central1)"
    echo "  TIER          - Hardware tier: minimal|standard|competitive (default: standard)"
    echo ""
    echo "Hardware tiers:"
    echo "  minimal     - 1 vCPU, 512MB  (~\$15/mo) - Testing only"
    echo "  standard    - 2 vCPU, 2GB    (~\$40/mo) - Casual play"
    echo "  competitive - 4 vCPU, 8GB    (~\$100/mo) - Stronger play, deeper search"
    exit 1
fi

PROJECT_ID="$1"
LICHESS_TOKEN="$2"
REGION="${3:-us-central1}"
TIER="${4:-standard}"

# Set resources based on tier
case "$TIER" in
    minimal)
        CPU="1"
        MEMORY="512Mi"
        ;;
    standard)
        CPU="2"
        MEMORY="2Gi"
        ;;
    competitive)
        CPU="4"
        MEMORY="8Gi"
        ;;
    *)
        echo "Unknown tier: $TIER (use minimal, standard, or competitive)"
        exit 1
        ;;
esac
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/aleph-repo/aleph-lichess-bot:latest"

echo "=== Aleph Lichess Bot Deployment ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Tier: $TIER ($CPU vCPU, $MEMORY RAM)"
echo ""

# Check if logged in
if ! gcloud auth print-identity-token &>/dev/null; then
    echo "Not logged in to gcloud. Running 'gcloud auth login'..."
    gcloud auth login
fi

# Set project
echo "Setting project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet

# Create artifact registry if it doesn't exist
echo "Ensuring artifact registry exists..."
gcloud artifacts repositories describe aleph-repo \
    --location="$REGION" &>/dev/null || \
gcloud artifacts repositories create aleph-repo \
    --repository-format=docker \
    --location="$REGION" \
    --description="Aleph chess bot images"

# Configure Docker auth
echo "Configuring Docker authentication..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# Build Docker image
echo "Building Docker image..."
cd "$(dirname "$0")/.."
docker build -f deploy/Dockerfile -t "$IMAGE" .

# Push to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push "$IMAGE"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy aleph-lichess-bot \
    --image="$IMAGE" \
    --platform=managed \
    --region="$REGION" \
    --memory="$MEMORY" \
    --cpu="$CPU" \
    --timeout=3600 \
    --min-instances=1 \
    --max-instances=1 \
    --no-cpu-throttling \
    --set-env-vars="LICHESS_TOKEN=$LICHESS_TOKEN" \
    --allow-unauthenticated

echo ""
echo "=== Deployment Complete! ==="
echo ""
echo "Check your bot at: https://lichess.org/@/YOUR_BOT_NAME"
echo ""
echo "View logs with:"
echo "  gcloud run logs read aleph-lichess-bot --region=$REGION --limit=50"
