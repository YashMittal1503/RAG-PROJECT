"""The health endpoint reports the running commit, which the release pipeline waits for."""

import pytest

from app.routers.health import health_check


@pytest.mark.anyio
async def test_health_reports_commit(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "abc123")
    body = await health_check()
    assert body.status == "ok" and body.commit == "abc123"


@pytest.mark.anyio
async def test_health_commit_defaults_to_dev(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    assert (await health_check()).commit == "dev"
