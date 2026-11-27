#!/usr/bin/env bash
# DocTalk deploy agent. Runs on the EC2 host every 2 minutes (doctalk-deploy.timer), as root.
#
# 1. Pull the image tagged `production` (CI moves that tag only after the tests pass).
# 2. Same image already running and healthy? Nothing to do.
# 3. Run database migrations with the NEW image. If they fail, stop: the old container keeps serving.
# 4. Replace the container and wait for /api/health.
# 5. Not healthy in time? Put the previous image back, and remember the bad one so it isn't retried
#    every 2 minutes. A newer push (a new image) is tried as usual.
#
# Usage: doctalk-deploy.sh [image-ref]     (default: $IMAGE:$TRACK_TAG)
#        FORCE=1 doctalk-deploy.sh         (retry an image marked bad)
# Settings: /etc/doctalk/deploy.env (see deploy.env.example), or DOCTALK_DEPLOY_CONFIG=<file>.
set -euo pipefail

CONFIG="${DOCTALK_DEPLOY_CONFIG:-/etc/doctalk/deploy.env}"
# shellcheck source=/dev/null
[ -f "$CONFIG" ] && . "$CONFIG"

IMAGE="${IMAGE:-ghcr.io/yashmittal1503/doctalk-backend}"
TRACK_TAG="${TRACK_TAG:-production}"
CONTAINER="${CONTAINER:-doctalk-backend}"
ENV_FILE="${ENV_FILE:-/home/ubuntu/RAG-PROJECT/backend/.env}"
DATA_DIR="${DATA_DIR:-/home/ubuntu/RAG-PROJECT/backend/data}"
HOST_PORTS="${HOST_PORTS:-8000 10000}"     # published on 127.0.0.1 only: nginx is the public entry point
CONTAINER_PORT="${CONTAINER_PORT:-10000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-240}"    # seconds to become healthy (model loading on a small instance)
HEALTH_INTERVAL="${HEALTH_INTERVAL:-5}"
STATE_DIR="${STATE_DIR:-/var/lib/doctalk}"
APP_UID="${APP_UID:-1000}"                 # the image's unprivileged user

TARGET="${1:-$IMAGE:$TRACK_TAG}"
HEALTH_PORT="${HOST_PORTS%% *}"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [doctalk-deploy] $*"; }

mkdir -p "$STATE_DIR"
BAD_LIST="$STATE_DIR/bad-images"
touch "$BAD_LIST"

# One deploy at a time (a slow health wait must not overlap the next timer run).
if command -v flock >/dev/null 2>&1; then
  exec 9>"$STATE_DIR/deploy.lock"
  flock -n 9 || { log "another deploy is running; skipping"; exit 0; }
fi

if [ ! -f "$ENV_FILE" ]; then
  log "ERROR: env file $ENV_FILE not found (set ENV_FILE in $CONFIG)"
  exit 1
fi

if ! docker pull -q "$TARGET" >/dev/null; then
  log "ERROR: could not pull $TARGET"
  exit 1
fi
NEW_ID="$(docker image inspect -f '{{.Id}}' "$TARGET")"
CUR_ID="$(docker inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || true)"
RUNNING="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)"

if [ "$NEW_ID" = "$CUR_ID" ] && [ "$RUNNING" = "true" ]; then
  exit 0   # up to date
fi
if grep -qxF "$NEW_ID" "$BAD_LIST" && [ -z "${FORCE:-}" ]; then
  exit 0   # failed before; waiting for a newer image (or FORCE=1)
fi

commit_of() {
  docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$1" 2>/dev/null | sed -n 's/^GIT_SHA=//p' | head -n1
}
NEW_COMMIT="$(commit_of "$NEW_ID")"
log "deploying $TARGET (commit ${NEW_COMMIT:-unknown}, image ${NEW_ID:7:12})"

mark_bad() { grep -qxF "$NEW_ID" "$BAD_LIST" || echo "$NEW_ID" >> "$BAD_LIST"; }

# The container runs as uid $APP_UID; data written by older root containers must stay writable.
mkdir -p "$DATA_DIR"
chown -R "$APP_UID:$APP_UID" "$DATA_DIR" 2>/dev/null || log "warning: could not chown $DATA_DIR"

# Migrations first, with the new code, while the old container still serves traffic.
if ! MIGRATE_OUT="$(docker run --rm --env-file "$ENV_FILE" "$NEW_ID" alembic upgrade head 2>&1)"; then
  printf '%s\n' "$MIGRATE_OUT"
  if printf '%s' "$MIGRATE_OUT" | grep -q "Can't locate revision"; then
    # Rolling back: the database is at a newer revision than this older code knows. Migrations are
    # not undone; the older code runs against the newer schema.
    log "warning: database is newer than this image's migrations (a rollback); continuing without migrating"
  else
    log "ERROR: database migration failed; keeping the current version"
    mark_bad
    exit 1
  fi
fi

start_container() {
  local image_id="$1" args=()
  for p in $HOST_PORTS; do args+=(-p "127.0.0.1:$p:$CONTAINER_PORT"); done
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" --restart unless-stopped \
    "${args[@]}" \
    --env-file "$ENV_FILE" \
    -e "PORT=$CONTAINER_PORT" \
    -v "$DATA_DIR:/app/data" \
    --log-opt max-size=20m --log-opt max-file=5 \
    "$image_id" >/dev/null
}

wait_healthy() {
  local waited=0 state body
  while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
    state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)"
    if [ "$state" = "exited" ] || [ "$state" = "dead" ] || [ "$state" = "missing" ]; then
      return 1
    fi
    if body="$(curl -fsS --max-time 5 "http://127.0.0.1:$HEALTH_PORT/api/health" 2>/dev/null)" \
        && printf '%s' "$body" | grep -q '"status":"ok"'; then
      return 0
    fi
    sleep "$HEALTH_INTERVAL"
    waited=$((waited + HEALTH_INTERVAL))
  done
  return 1
}

if [ -n "$CUR_ID" ]; then
  docker tag "$CUR_ID" "$IMAGE:previous" || true   # keeps the rollback image from being pruned
fi

start_container "$NEW_ID"
if wait_healthy; then
  echo "$NEW_ID" > "$STATE_DIR/current-image"
  log "OK: commit ${NEW_COMMIT:-unknown} is live"
  docker image prune -f >/dev/null 2>&1 || true   # dangling layers from older pulls
  exit 0
fi

FAIL_LOG="$STATE_DIR/failed-$(date -u +%Y%m%dT%H%M%SZ).log"
docker logs --tail 300 "$CONTAINER" > "$FAIL_LOG" 2>&1 || true
log "ERROR: commit ${NEW_COMMIT:-unknown} did not become healthy in ${HEALTH_TIMEOUT}s (logs: $FAIL_LOG)"
mark_bad

if [ -n "$CUR_ID" ]; then
  log "rolling back to image ${CUR_ID:7:12} (commit $(commit_of "$CUR_ID"))"
  start_container "$CUR_ID"
  if wait_healthy; then
    log "rollback OK"
  else
    log "ERROR: the previous version is not healthy either; check 'docker logs $CONTAINER'"
  fi
fi
exit 1
