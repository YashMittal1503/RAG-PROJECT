"""
JWT-based authentication dependency for FastAPI.

Verifies Supabase Auth JWTs locally using the project's JWKS endpoint (for asymmetric keys like ES256),
with fallbacks to legacy HS256 secret or Supabase API verification.
"""

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client

from app.config import settings

# HTTPBearer extracts the token from the "Authorization: Bearer <token>" header.
security = HTTPBearer()

jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
jwks_client = PyJWKClient(jwks_url)

_supabase_admin = None


def get_supabase_admin():
    global _supabase_admin
    if _supabase_admin is None:
        _supabase_admin = create_client(settings.supabase_url, settings.supabase_key)
    return _supabase_admin


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency that extracts and verifies the Supabase JWT.

    Returns the user_id (UUID string from the 'sub' claim).
    Raises HTTP 401 if the token is invalid, expired, or malformed.
    """
    token = credentials.credentials
    payload = None

    # 1. Try JWKS verification (modern Supabase ECC / RSA tokens)
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256", "HS256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please sign in again.",
        )
    except Exception:
        # 2. Try HS256 with legacy secret if available
        if settings.supabase_jwt_secret:
            try:
                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated",
                )
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired. Please sign in again.",
                )
            except Exception:
                pass

    # 3. Fallback to Supabase Auth API verification
    if not payload:
        try:
            client = get_supabase_admin()
            user_resp = client.auth.get_user(token)
            if user_resp and user_resp.user:
                return str(user_resp.user.id)
        except Exception as api_err:
            print(f"[AUTH ERROR] Supabase API verification failed: {api_err}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials.",
            )

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
        )

    # The 'sub' claim contains the Supabase user UUID
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing user identifier.",
        )

    return str(user_id)
