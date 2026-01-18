# Deploying Aleph to Lichess via Google Cloud Run

This guide walks you through deploying Aleph as a Lichess bot on Google Cloud Run.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Docker** installed locally
4. **Lichess Bot Account** (see below)

## Step 1: Create Lichess Bot Account

1. Create a **new** Lichess account at https://lichess.org/signup
   - Choose a bot name like `AlephBot` or `aleph-engine`
   - **IMPORTANT**: Do NOT play any games on this account

2. Generate an API token at https://lichess.org/account/oauth/token
   - Description: "Aleph Bot Token"
   - Select these permissions:
     - `bot:play` - Play as a bot
     - `challenge:read` - Read incoming challenges
     - `challenge:write` - Create challenges
   - Click "Submit" and copy the token (you won't see it again!)

3. Upgrade to bot account (irreversible!):
   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade \
     -H "Authorization: Bearer YOUR_TOKEN_HERE"
   ```

   You should see: `{"ok":true}`

## Step 2: Setup Google Cloud

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Authenticate (if not already)
gcloud auth login

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com

# Create artifact registry for Docker images
gcloud artifacts repositories create aleph-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Aleph chess bot images"
```

## Step 3: Build and Push Docker Image

From the project root directory:

```bash
# Configure Docker to use gcloud for authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and tag for GCP
docker build -f deploy/Dockerfile \
    -t us-central1-docker.pkg.dev/$PROJECT_ID/aleph-repo/aleph-lichess-bot:latest \
    .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/$PROJECT_ID/aleph-repo/aleph-lichess-bot:latest
```

## Step 4: Deploy to Cloud Run

```bash
# Deploy with your Lichess token
gcloud run deploy aleph-lichess-bot \
    --image=us-central1-docker.pkg.dev/$PROJECT_ID/aleph-repo/aleph-lichess-bot:latest \
    --platform=managed \
    --region=us-central1 \
    --memory=512Mi \
    --cpu=1 \
    --timeout=3600 \
    --min-instances=1 \
    --max-instances=1 \
    --no-cpu-throttling \
    --set-env-vars="LICHESS_TOKEN=YOUR_LICHESS_TOKEN_HERE" \
    --allow-unauthenticated
```

### Important Flags Explained:
- `--min-instances=1`: Keeps the bot always running (required for Lichess connection)
- `--no-cpu-throttling`: Ensures full CPU for chess calculations
- `--timeout=3600`: 1 hour timeout for long games
- `--allow-unauthenticated`: Allows health checks without auth

## Step 5: Verify Deployment

1. Check Cloud Run logs:
   ```bash
   gcloud run logs read aleph-lichess-bot --region=us-central1 --limit=50
   ```

2. View your bot on Lichess:
   - Go to `https://lichess.org/@/YOUR_BOT_NAME`
   - You should see the BOT badge

3. Challenge your bot from another account to test!

## Cost Estimate

With `--min-instances=1` (always-on):
- ~$15-25/month for a small instance running 24/7
- Costs vary by region

To reduce costs (but bot won't be always available):
- Remove `--min-instances=1`
- Use `--cpu-throttling` (default)

## Updating the Bot

After making code changes:

```bash
# Rebuild
docker build -f deploy/Dockerfile \
    -t us-central1-docker.pkg.dev/$PROJECT_ID/aleph-repo/aleph-lichess-bot:latest \
    .

# Push new image
docker push us-central1-docker.pkg.dev/$PROJECT_ID/aleph-repo/aleph-lichess-bot:latest

# Deploy update
gcloud run deploy aleph-lichess-bot \
    --image=us-central1-docker.pkg.dev/$PROJECT_ID/aleph-repo/aleph-lichess-bot:latest \
    --region=us-central1
```

## Troubleshooting

### Bot not responding to challenges
- Check logs: `gcloud run logs read aleph-lichess-bot --region=us-central1`
- Verify token is correct
- Check config.yml challenge filters

### Games timing out
- Increase `--memory` if running out of memory
- Check time management is working (search should stop in time)

### Container keeps restarting
- Usually a token/auth issue
- Check logs for specific error messages

### High latency
- Deploy to region closer to Lichess servers (Europe: `europe-west1`)
- Increase CPU: `--cpu=2`

## Cleanup

To stop and delete the bot:

```bash
gcloud run services delete aleph-lichess-bot --region=us-central1
```
