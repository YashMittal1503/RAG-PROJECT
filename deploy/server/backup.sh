#!/usr/bin/env bash
# Daily backup of the backend's on-disk data (per-user DuckDB spreadsheet databases).
# Everything else (Postgres, vectors, uploaded files) lives in Supabase and Qdrant Cloud.
# Runs as root from doctalk-backup.timer. Keeps BACKUP_KEEP_DAYS days locally; optionally copies to S3.
set -euo pipefail

CONFIG="${DOCTALK_DEPLOY_CONFIG:-/etc/doctalk/deploy.env}"
# shellcheck source=/dev/null
[ -f "$CONFIG" ] && . "$CONFIG"

DATA_DIR="${DATA_DIR:-/home/ubuntu/RAG-PROJECT/backend/data}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/doctalk}"
BACKUP_KEEP_DAYS="${BACKUP_KEEP_DAYS:-14}"
BACKUP_S3_URI="${BACKUP_S3_URI:-}"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [doctalk-backup] $*"; }

if [ ! -d "$DATA_DIR" ]; then
  log "nothing to back up: $DATA_DIR does not exist"
  exit 0
fi

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
ARCHIVE="$BACKUP_DIR/doctalk-data-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
# DuckDB files are opened per query and closed again, so a copy taken between queries is consistent.
tar -czf "$ARCHIVE" -C "$DATA_DIR" .
log "wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

find "$BACKUP_DIR" -name 'doctalk-data-*.tar.gz' -mtime +"$BACKUP_KEEP_DAYS" -delete

if [ -n "$BACKUP_S3_URI" ]; then
  if command -v aws >/dev/null 2>&1; then
    aws s3 cp "$ARCHIVE" "$BACKUP_S3_URI/" --only-show-errors && log "copied to $BACKUP_S3_URI"
  else
    log "warning: BACKUP_S3_URI is set but the AWS CLI is not installed"
  fi
fi
