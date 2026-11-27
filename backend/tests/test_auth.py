"""
Supabase JWT verification (app/auth.py): the dependency that decides which user every request acts as.
No network: the JWKS client and the Supabase admin client are mocked.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app import auth

SECRET = "legacy-hs256-secret-for-tests-only-0123456789"
USER = "11111111-2222-4333-8444-555555555555"


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _claims(**over):
    c = {"sub": USER, "aud": "authenticated", "exp": int(time.time()) + 3600}
    c.update(over)
    return {k: v for k, v in c.items() if v is not None}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    auth._verified_token_cache.clear()
    monkeypatch.setattr(auth.settings, "supabase_jwt_secret", SECRET)
    # Default: the JWKS path doesn't recognise the token (as for HS256 tokens) and the Supabase API rejects it.
    jwks = MagicMock()
    jwks.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError("no matching key")
    monkeypatch.setattr(auth, "jwks_client", jwks)
    admin = MagicMock()
    admin.auth.get_user.side_effect = RuntimeError("invalid JWT")
    monkeypatch.setattr(auth, "get_supabase_admin", lambda: admin)
    yield
    auth._verified_token_cache.clear()


async def _user(token: str) -> str:
    return await auth.get_current_user(_creds(token))


@pytest.mark.anyio
async def test_es256_token_verified_with_jwks_key(monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())
    token = jwt.encode(_claims(), key, algorithm="ES256")
    jwks = MagicMock()
    jwks.get_signing_key_from_jwt.return_value = SimpleNamespace(key=key.public_key())
    monkeypatch.setattr(auth, "jwks_client", jwks)
    assert await _user(token) == USER


@pytest.mark.anyio
async def test_hs256_token_verified_with_legacy_secret():
    assert await _user(jwt.encode(_claims(), SECRET, algorithm="HS256")) == USER


@pytest.mark.anyio
@pytest.mark.parametrize("token_factory, detail", [
    (lambda: jwt.encode(_claims(exp=int(time.time()) - 10), SECRET, algorithm="HS256"), "expired"),
    (lambda: jwt.encode(_claims(), "some-other-secret-0123456789abcdef", algorithm="HS256"), "validate"),
    (lambda: jwt.encode(_claims(aud="anon"), SECRET, algorithm="HS256"), "validate"),
    (lambda: jwt.encode(_claims(sub=None), SECRET, algorithm="HS256"), "identifier"),
    (lambda: jwt.encode(_claims(), None, algorithm="none"), "validate"),   # unsigned token
    (lambda: "not-a-jwt", "validate"),
])
async def test_bad_tokens_are_rejected(token_factory, detail):
    with pytest.raises(HTTPException) as e:
        await _user(token_factory())
    assert e.value.status_code == 401 and detail in e.value.detail.lower()


@pytest.mark.anyio
async def test_hs256_token_cannot_pass_the_asymmetric_path(monkeypatch):
    """Algorithm confusion: an HS256 token signed with a public key must not verify as ES256/RS256."""
    key = ec.generate_private_key(ec.SECP256R1())
    jwks = MagicMock()
    jwks.get_signing_key_from_jwt.return_value = SimpleNamespace(key=key.public_key())
    monkeypatch.setattr(auth, "jwks_client", jwks)
    monkeypatch.setattr(auth.settings, "supabase_jwt_secret", None)
    forged = jwt.encode(_claims(), "attacker-chosen-secret-0123456789abcd", algorithm="HS256")
    with pytest.raises(HTTPException):
        await _user(forged)


@pytest.mark.anyio
async def test_supabase_api_fallback_when_local_checks_fail(monkeypatch):
    monkeypatch.setattr(auth.settings, "supabase_jwt_secret", None)
    admin = MagicMock()
    admin.auth.get_user.return_value = SimpleNamespace(user=SimpleNamespace(id=USER))
    monkeypatch.setattr(auth, "get_supabase_admin", lambda: admin)
    assert await _user("opaque-token-supabase-accepts") == USER


@pytest.mark.anyio
async def test_verified_tokens_are_cached_briefly_and_never_past_expiry():
    token = jwt.encode(_claims(), SECRET, algorithm="HS256")
    assert await _user(token) == USER
    with patch.object(auth.jwt, "decode", side_effect=AssertionError("should be served from cache")):
        assert await _user(token) == USER
    _, until = auth._verified_token_cache[token]
    assert until - time.time() <= 60                        # at most 60 s, even for a 1-hour token

    soon = jwt.encode(_claims(exp=int(time.time()) + 5), SECRET, algorithm="HS256")
    await _user(soon)
    assert auth._verified_token_cache[soon][1] <= time.time() + 5   # not past the token's own expiry


def test_token_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(auth, "_MAX_TOKEN_CACHE_SIZE", 3)
    for i in range(5):
        auth._cache_user(f"t{i}", USER)
    assert list(auth._verified_token_cache) == ["t2", "t3", "t4"]   # least recently used evicted first
