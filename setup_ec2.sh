#!/bin/bash
# ============================================================
# DocTalk EC2 One-Time Setup Script
# 
# Run this ONCE on your EC2 instance. After this, every
# git push will automatically deploy within ~5 minutes.
#
# Usage: bash setup_ec2.sh YOUR_GITHUB_PAT
# ============================================================

set -e

PAT="$1"

if [ -z "$PAT" ]; then
  echo ""
  echo "❌ Missing GitHub Personal Access Token!"
  echo ""
  echo "Steps to create one:"
  echo "  1. Go to: https://github.com/settings/tokens/new"
  echo "  2. Note: 'EC2 Docker Pull'"
  echo "  3. Expiration: 90 days (or No expiration)"
  echo "  4. Scopes: check ONLY 'read:packages'"
  echo "  5. Click 'Generate token' (green button at bottom)"
  echo "  6. Copy the token (starts with ghp_) and run:"
  echo ""
  echo "     bash setup_ec2.sh ghp_YOUR_TOKEN_HERE"
  echo ""
  exit 1
fi

echo "=========================================="
echo "🚀 DocTalk EC2 One-Time Setup"
echo "=========================================="

# 1. Login to GitHub Container Registry
echo "📦 Logging in to GitHub Container Registry..."
echo "$PAT" | docker login ghcr.io -u YashMittal1503 --password-stdin
echo "✅ GHCR login successful!"

# 2. Find the .env file
ENV_FILE=""
for dir in "$HOME/RAG-PROJECT/backend" "$HOME/rag-project/backend"; do
  if [ -f "$dir/.env" ]; then
    ENV_FILE="$dir/.env"
    break
  fi
done

if [ -z "$ENV_FILE" ]; then
  echo "❌ Could not find backend/.env file!"
  echo "Please make sure your .env file is at ~/RAG-PROJECT/backend/.env"
  exit 1
fi

DATA_DIR="$(dirname "$ENV_FILE")/data"
mkdir -p "$DATA_DIR"

echo "📂 Using env file: $ENV_FILE"
echo "📂 Using data dir: $DATA_DIR"

# 3. Stop and remove old containers (free up ports 8000 and 10000)
echo "🔄 Stopping old containers..."
docker compose -f "$HOME/RAG-PROJECT/docker-compose.yml" down 2>/dev/null || true
docker-compose -f "$HOME/RAG-PROJECT/docker-compose.yml" down 2>/dev/null || true
docker stop doctalk-backend 2>/dev/null || true
docker rm doctalk-backend 2>/dev/null || true
docker stop watchtower 2>/dev/null || true
docker rm watchtower 2>/dev/null || true

# Stop any remaining container that has port 8000 or 10000 mapped
for c in $(docker ps -q); do
  if docker port "$c" 2>/dev/null | grep -q -E "8000|10000"; then
    echo "Stopping container $c occupying port 8000/10000..."
    docker stop "$c" 2>/dev/null || true
    docker rm "$c" 2>/dev/null || true
  fi
done

# Kill any lingering host process on port 8000/10000 if not docker
sudo fuser -k 8000/tcp 2>/dev/null || true
sudo fuser -k 10000/tcp 2>/dev/null || true

# 4. Pull the latest image
echo "📥 Pulling latest DocTalk image..."
docker pull ghcr.io/yashmittal1503/doctalk-backend:latest

# 5. Start the backend container
echo "🚀 Starting DocTalk backend..."
docker run -d \
  --name doctalk-backend \
  --restart unless-stopped \
  -p 8000:10000 \
  -p 10000:10000 \
  --env-file "$ENV_FILE" \
  -v "$DATA_DIR:/app/data" \
  ghcr.io/yashmittal1503/doctalk-backend:latest

# 6. Start Watchtower (auto-updates container when new image is pushed)
echo "👀 Starting Watchtower (auto-update agent)..."
docker run -d \
  --name watchtower \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOME/.docker/config.json:/config.json:ro" \
  containrrr/watchtower \
  --interval 180 \
  --cleanup \
  doctalk-backend

# 7. Verify everything is running
sleep 5
echo ""
echo "=========================================="
echo "📊 Container Status:"
echo "=========================================="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

if docker ps | grep -q doctalk-backend; then
  echo "✅ DocTalk backend is running!"
else
  echo "❌ Backend failed to start. Logs:"
  docker logs doctalk-backend
  exit 1
fi

if docker ps | grep -q watchtower; then
  echo "✅ Watchtower is running (checks for updates every 3 minutes)!"
else
  echo "⚠️  Watchtower failed to start. Auto-updates won't work."
fi

echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "How it works now:"
echo "  1. You push code to GitHub"
echo "  2. GitHub Actions builds a new Docker image (~2-3 min)"
echo "  3. Watchtower detects the new image (~3 min)"
echo "  4. Watchtower auto-pulls and restarts the container"
echo "  5. Your backend is updated! (~5-6 min total)"
echo ""
echo "Useful commands:"
echo "  docker logs -f doctalk-backend    # View live logs"
echo "  docker logs watchtower            # View update logs"
echo "  docker restart doctalk-backend    # Manual restart"
echo ""
