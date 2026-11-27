# Backend deployment

How the DocTalk backend gets from a commit to the live API at `https://yash--mittal.duckdns.org`, and
what to do when something goes wrong.

## The big picture

```
 pull request ──► CI: backend tests · deploy-script tests · frontend lint+build · image builds
                      (all must pass before you merge)

 merge to main ──► Release backend workflow
                      1. CI again, on the merged code
                      2. build the Docker image, push it to GHCR as  doctalk-backend:<commit>
                      3. move the  doctalk-backend:production  tag to that image
                      4. wait (up to 15 min) until the live API reports <commit>  ──► green ✔ / red ✖

 EC2 server ──► doctalk-deploy.timer, every 2 minutes
                      new `production` image?  ─ no ─► nothing to do
                            │ yes
                      run database migrations with the NEW image   ─ fail ─► keep the old version, stop
                            │ ok
                      replace the container, wait for /api/health  ─ fail ─► restore the old version,
                            │ healthy                                          never retry this image
                      done: the new commit is live
```

Pieces, and where they live:

| Piece | What it is | Where |
|---|---|---|
| Docker image | The backend packaged with Python, system libraries and the embedding/reranker models | `backend/Dockerfile`, published to `ghcr.io/yashmittal1503/doctalk-backend` |
| CI | Tests and checks on every pull request | `.github/workflows/ci.yml` |
| Release | Builds the image and moves `production` after CI passes on `main` | `.github/workflows/deploy.yml` |
| Rollback | Puts an older commit back in production | `.github/workflows/rollback.yml` |
| Deploy agent | Runs on the server; pulls, migrates, health-checks, rolls back | `deploy/server/doctalk-deploy.sh` + `doctalk-deploy.timer` |
| Backups | Daily archive of the server's on-disk data (DuckDB spreadsheets) | `deploy/server/backup.sh` + `doctalk-backup.timer` |
| nginx | Public HTTPS entry point (Let's Encrypt certificate), proxies to the container | `deploy/nginx/doctalk.conf` (template) |
| Server setup | One command to (re)install all of the above on the EC2 host | `deploy/server/bootstrap.sh` |

Everything else the backend needs is a managed service: Supabase (auth, Postgres, file storage), Qdrant
Cloud (vectors) and the LLM APIs. The server holds only the container, `backend/.env` and `backend/data`.

Why the server pulls instead of GitHub pushing: the server's SSH port is closed to the internet, which is
safer. The agent reaches out to GHCR; nothing needs to reach in.

## First-time setup (once)

On the EC2 server (SSH in the way you normally do):

```bash
cd ~/RAG-PROJECT
git pull                      # get this branch's files (after it is merged: main)
sudo bash deploy/server/bootstrap.sh
```

It adds swap if missing, installs the deploy agent and the backup timer, writes `/etc/doctalk/deploy.env`
(finding your `backend/.env` and `backend/data`), removes Watchtower, and runs a first deploy. It is safe to
run again after pulling changes to `deploy/`.

nginx is left alone unless you ask: `sudo bash deploy/server/bootstrap.sh --nginx yash--mittal.duckdns.org`
installs `deploy/nginx/doctalk.conf`, keeping a backup of the current site and restoring it if `nginx -t`
fails. Compare first (`sudo cat /etc/nginx/sites-enabled/*`): the template proxies to `127.0.0.1:8000` with
streaming (SSE) support and a 100 MB upload limit.

Order matters once: **merge this change first, then run bootstrap.** Until the first release run, there is
no `production` image to deploy; bootstrap then leaves the current container running.

Optional, in GitHub: Settings → Environments → `production` (created by the first release run) can require
your approval before each release, and Settings → Variables → `PRODUCTION_URL` overrides the API address.

## Day to day

- **Ship:** merge a pull request into `main`. Watch Actions → "Release backend". Green means the new commit
  is serving traffic.
- **What is live?** `curl https://yash--mittal.duckdns.org/api/health`: `commit` is the deployed commit.
- **Roll back:** Actions → "Roll back backend" → Run workflow → paste a full commit SHA of an earlier
  successful release. Database migrations are not undone, so pick a commit that works with today's schema.
- **Logs on the server:**

  | | |
  |---|---|
  | deploy agent | `journalctl -u doctalk-deploy -f` |
  | backend | `docker logs -f doctalk-backend` |
  | why a version failed | `ls /var/lib/doctalk/failed-*.log` |
  | deploy right now | `sudo systemctl start doctalk-deploy` |
  | retry an image marked bad | `sudo FORCE=1 /opt/doctalk/doctalk-deploy.sh` |
  | backups | `ls /var/backups/doctalk` (`sudo systemctl start doctalk-backup` runs one now) |

## When the release run is red

| Failed step | Meaning | What to do |
|---|---|---|
| CI | Tests, lint or build failed | Nothing shipped. Fix on a branch, open a PR. |
| Build & push image | Docker build failed | Nothing shipped. The same build runs on every PR, so this is rare. |
| Wait for the server | The server did not start the new commit within 15 minutes | Check `journalctl -u doctalk-deploy -n 100`. Usually a failed migration or health check: the old version is still (or again) live. If the server itself is down, the API is unreachable in the log. |

## Running the image locally

```bash
docker compose up --build     # migrations, then the API on http://127.0.0.1:8000
```

## Notes

- The image runs as an unprivileged user (uid 1000); the agent makes `backend/data` writable for it.
- Ports 8000 and 10000 are bound to `127.0.0.1` only; nginx on 443 is the only public entry point.
- Container logs are capped (5 × 20 MB) so they cannot fill the disk.
- Images older than this pipeline report `"commit": "dev"`, so a rollback to one of them deploys but the
  rollback run cannot confirm it.
- Backups stay on the same server unless `BACKUP_S3_URI` is set in `/etc/doctalk/deploy.env` (needs the AWS CLI
  and an instance role allowed to write to that bucket). A copy on the same disk does not survive losing
  the server.
