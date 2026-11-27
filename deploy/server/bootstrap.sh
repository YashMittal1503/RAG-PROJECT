#!/usr/bin/env bash
# One-time (and safe to re-run) setup of the DocTalk EC2 host. Replaces the old setup_ec2.sh + Watchtower.
#
#   cd ~/RAG-PROJECT && git pull && sudo bash deploy/server/bootstrap.sh [--nginx DOMAIN]
#
# What it does:
#   1. 2 GB swap (a 1 GB instance runs out of memory on large PDFs without it)
#   2. Docker, if missing
#   3. Installs the deploy agent + backup script into /opt/doctalk and their systemd timers
#   4. Writes /etc/doctalk/deploy.env the first time (finds backend/.env and backend/data)
#   5. Removes Watchtower: the deploy agent replaces it
#   6. Optional --nginx DOMAIN: installs deploy/nginx/doctalk.conf (keeps a backup, reverts if nginx -t fails)
#   7. Runs a first deploy and prints the status
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root: sudo bash $0 $*" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_DIR="$REPO_DIR/deploy/server"
NGINX_DOMAIN=""
if [ "${1:-}" = "--nginx" ]; then
  NGINX_DOMAIN="${2:?usage: bootstrap.sh --nginx your.domain}"
fi

say() { echo "==> $*"; }

# 1. Swap
if [ "$(swapon --show --noheadings | wc -l)" -eq 0 ]; then
  say "Creating 2 GB swap"
  fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
else
  say "Swap already active"
fi

# 2. Docker
if ! command -v docker >/dev/null 2>&1; then
  say "Installing Docker"
  apt-get update -qq
  apt-get install -y -qq docker.io
fi
systemctl enable --now docker >/dev/null

# 3. Agent, backup script and timers
say "Installing deploy agent and backup timer"
install -d -m 755 /opt/doctalk /etc/doctalk /var/lib/doctalk
install -m 755 "$SERVER_DIR/doctalk-deploy.sh" /opt/doctalk/doctalk-deploy.sh
install -m 755 "$SERVER_DIR/backup.sh" /opt/doctalk/backup.sh
for unit in doctalk-deploy.service doctalk-deploy.timer doctalk-backup.service doctalk-backup.timer; do
  install -m 644 "$SERVER_DIR/$unit" "/etc/systemd/system/$unit"
done

# 4. Settings
if [ ! -f /etc/doctalk/deploy.env ]; then
  say "Writing /etc/doctalk/deploy.env"
  ENV_FILE=""
  for candidate in "$REPO_DIR/backend/.env" /home/*/RAG-PROJECT/backend/.env /home/*/rag-project/backend/.env; do
    if [ -f "$candidate" ]; then ENV_FILE="$candidate"; break; fi
  done
  if [ -z "$ENV_FILE" ]; then
    echo "Could not find the backend .env file. Put it at $REPO_DIR/backend/.env and re-run." >&2
    exit 1
  fi
  DATA_DIR="$(dirname "$ENV_FILE")/data"
  sed -e "s|^ENV_FILE=.*|ENV_FILE=$ENV_FILE|" -e "s|^DATA_DIR=.*|DATA_DIR=$DATA_DIR|" \
    "$SERVER_DIR/deploy.env.example" > /etc/doctalk/deploy.env
  chmod 600 /etc/doctalk/deploy.env
else
  say "Keeping existing /etc/doctalk/deploy.env"
fi
# shellcheck source=/dev/null
chmod 600 "$(. /etc/doctalk/deploy.env && echo "$ENV_FILE")" 2>/dev/null || true

# 5. Watchtower is replaced by the agent
if docker ps -a --format '{{.Names}}' | grep -qx watchtower; then
  say "Removing Watchtower"
  docker rm -f watchtower >/dev/null
fi

# 6. nginx (optional)
if [ -n "$NGINX_DOMAIN" ]; then
  say "Installing nginx site for $NGINX_DOMAIN"
  SITE=/etc/nginx/sites-available/doctalk
  BACKUP=""
  if [ -f "$SITE" ]; then BACKUP="$SITE.bak.$(date +%Y%m%d%H%M%S)"; cp "$SITE" "$BACKUP"; fi
  sed "s/__DOMAIN__/$NGINX_DOMAIN/g" "$REPO_DIR/deploy/nginx/doctalk.conf" > "$SITE"
  ln -sf "$SITE" /etc/nginx/sites-enabled/doctalk
  if nginx -t; then
    systemctl reload nginx
  else
    echo "nginx config test failed; restoring the previous one" >&2
    if [ -n "$BACKUP" ]; then cp "$BACKUP" "$SITE"; else rm -f "$SITE" /etc/nginx/sites-enabled/doctalk; fi
    exit 1
  fi
fi

systemctl daemon-reload
systemctl enable --now doctalk-deploy.timer doctalk-backup.timer >/dev/null

# 7. First deploy now (instead of waiting for the timer)
say "Running a deploy now (pulls the production image if it exists yet)"
if systemctl start doctalk-deploy.service; then
  say "Deploy agent finished"
else
  echo "The deploy agent reported a problem; see: journalctl -u doctalk-deploy -n 50" >&2
fi

echo
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
echo
cat <<'EOF'
Done. From now on:
  - merge to main -> CI tests -> image tagged `production` -> this server deploys it within ~2 minutes
  - deploy log:     journalctl -u doctalk-deploy -f
  - app logs:       docker logs -f doctalk-backend
  - deploy now:     sudo systemctl start doctalk-deploy
  - backups:        ls /var/backups/doctalk   (sudo systemctl start doctalk-backup to run one now)
EOF
