"""
Runs deploy/server/doctalk-deploy.sh against fake `docker` and `curl` commands, so the agent's decisions
(skip / deploy / refuse on failed migration / roll back on failed health check / don't retry a bad image)
are tested without a server. Needs bash (CI: ubuntu; locally on Windows: Git Bash).
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

AGENT = Path(__file__).resolve().parents[1] / "server" / "doctalk-deploy.sh"
BASH = os.environ.get("BASH_BIN") or shutil.which("bash")

FAKE_DOCKER = r"""#!/usr/bin/env bash
# Minimal stand-in for the docker CLI calls the agent makes. State lives in $FAKE_STATE.
S="$FAKE_STATE"
echo "$*" >> "$S/calls"
case "$1" in
  pull) exit "${FAKE_PULL_RC:-0}" ;;
  image)
    if [ "$2" = "inspect" ]; then
      ref="${@: -1}"
      case "$4" in
        *Config.Env*) id="$ref"; [ "${ref#sha256:}" = "$ref" ] && id="$FAKE_NEW_ID"; echo "GIT_SHA=commit-of-${id#sha256:}" ;;
        *) echo "$FAKE_NEW_ID" ;;
      esac
      exit 0
    fi
    exit 0 ;;   # image prune
  inspect)
    [ -f "$S/container" ] || exit 1
    case "$3" in
      *Image*) cat "$S/container" ;;
      *Running*) echo true ;;
      *Status*) echo running ;;
    esac ;;
  run)
    if [ "$2" = "--rm" ]; then   # the migration container
      [ -n "${FAKE_MIGRATE_OUT:-}" ] && echo "$FAKE_MIGRATE_OUT"
      exit "${FAKE_MIGRATE_RC:-0}"
    fi
    echo "${@: -1}" > "$S/container"; echo started ;;
  rm) rm -f "$S/container" ;;
  tag|logs) exit 0 ;;
esac
"""

FAKE_CURL = r"""#!/usr/bin/env bash
img="$(cat "$FAKE_STATE/container" 2>/dev/null)"
for ok in $FAKE_HEALTHY; do
  if [ "$img" = "$ok" ]; then echo '{"status":"ok","commit":"x"}'; exit 0; fi
done
exit 7
"""


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def agent(tmp_path):
    if not BASH:
        pytest.skip("bash not available")
    bin_dir, state, work = tmp_path / "bin", tmp_path / "fake", tmp_path / "state"
    for d in (bin_dir, state, work):
        d.mkdir()
    _exe(bin_dir / "docker", FAKE_DOCKER)
    _exe(bin_dir / "curl", FAKE_CURL)
    _exe(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    env_file = tmp_path / "backend.env"
    env_file.write_text("X=1\n")
    config = tmp_path / "deploy.env"
    config.write_text(
        f"ENV_FILE={env_file.as_posix()}\nDATA_DIR={(tmp_path / 'data').as_posix()}\n"
        f"STATE_DIR={work.as_posix()}\nHEALTH_TIMEOUT=3\nHEALTH_INTERVAL=1\nAPP_UID=$(id -u)\n",
        encoding="utf-8",
    )

    def run(new_id, running=None, healthy=(), **fake):
        if running:
            (state / "container").write_text(running + "\n")
        env = {
            **os.environ,
            "PATH": f"{bin_dir.as_posix()}{os.pathsep}{os.environ['PATH']}",
            "FAKE_STATE": state.as_posix(),
            "FAKE_NEW_ID": new_id,
            "FAKE_HEALTHY": " ".join(healthy),
            "DOCTALK_DEPLOY_CONFIG": config.as_posix(),
            **{k: str(v) for k, v in fake.items()},
        }
        p = subprocess.run([BASH, AGENT.as_posix()], env=env, capture_output=True, text=True)
        container = (state / "container").read_text().strip() if (state / "container").exists() else None
        calls = (state / "calls").read_text() if (state / "calls").exists() else ""
        bad = (work / "bad-images").read_text().split() if (work / "bad-images").exists() else []
        return p, container, calls, bad

    return run


def test_up_to_date_does_nothing(agent):
    p, container, calls, _ = agent("sha256:new", running="sha256:new", healthy=["sha256:new"])
    assert p.returncode == 0 and container == "sha256:new"
    assert "alembic" not in calls and "run -d" not in calls


def test_new_image_is_migrated_then_started(agent):
    p, container, calls, bad = agent("sha256:new", running="sha256:old", healthy=["sha256:new"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert container == "sha256:new" and not bad
    assert calls.index("alembic upgrade head") < calls.index("run -d")      # migrate before switching
    assert "-p 127.0.0.1:8000:10000" in calls and "-p 127.0.0.1:10000:10000" in calls   # never 0.0.0.0
    assert "OK: commit commit-of-new is live" in p.stdout


def test_failed_migration_keeps_the_old_container(agent):
    p, container, calls, bad = agent("sha256:new", running="sha256:old", healthy=["sha256:new", "sha256:old"],
                                     FAKE_MIGRATE_RC=1, FAKE_MIGRATE_OUT="sqlalchemy error")
    assert p.returncode == 1
    assert container == "sha256:old" and "run -d" not in calls
    assert bad == ["sha256:new"]


def test_rollback_migration_mismatch_still_deploys(agent):
    p, container, _, bad = agent("sha256:older", running="sha256:new", healthy=["sha256:older"],
                                 FAKE_MIGRATE_RC=1, FAKE_MIGRATE_OUT="Can't locate revision identified by 'abc'")
    assert p.returncode == 0 and container == "sha256:older" and not bad


def test_unhealthy_new_version_is_rolled_back_and_not_retried(agent):
    p, container, _, bad = agent("sha256:new", running="sha256:old", healthy=["sha256:old"])
    assert p.returncode == 1
    assert container == "sha256:old"                       # previous version restored
    assert bad == ["sha256:new"] and "rollback OK" in p.stdout
    p2, container2, calls2, _ = agent("sha256:new", healthy=["sha256:old"])   # next timer run
    assert p2.returncode == 0 and container2 == "sha256:old"
    assert calls2.count("run -d") == 2                     # nothing new started on the second run


def test_force_retries_a_bad_image(agent):
    agent("sha256:new", running="sha256:old", healthy=["sha256:old"])
    p, container, _, _ = agent("sha256:new", healthy=["sha256:new"], FORCE=1)
    assert p.returncode == 0 and container == "sha256:new"


def test_pull_failure_changes_nothing(agent):
    p, container, calls, _ = agent("sha256:new", running="sha256:old", healthy=["sha256:old"], FAKE_PULL_RC=1)
    assert p.returncode == 1 and container == "sha256:old" and "run" not in calls.replace("pull", "")
