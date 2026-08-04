import logging
from typing import Optional
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth
import config

logger = logging.getLogger(__name__)

# auto_error=False allows optional header inspection
security = HTTPBearer(auto_error=False)


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> str:
    """
    Validates the Firebase ID token or WAYFINDER_API_KEY in the Authorization header.
    - If a valid Firebase ID token is provided, returns the verified Firebase UID.
    - If WAYFINDER_API_KEY matches, returns 'api_key_user'.
    - If a token is provided but invalid or expired, raises 401 so the client can refresh.
    - If an email is unverified, raises 403.
    - If no credentials and WAYFINDER_API_KEY is unset, falls back to 'guest_user' for local dev.
    """
    if credentials and credentials.credentials:
        token = credentials.credentials.strip()

        # 1. Check if token matches WAYFINDER_API_KEY
        if config.WAYFINDER_API_KEY and token == config.WAYFINDER_API_KEY:
            return "api_key_user"

        # 2. Check Firebase token
        if firebase_admin._apps:
            try:
                decoded_token = auth.verify_id_token(token)
                uid = decoded_token.get("uid")
                if not uid:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid token payload: missing UID.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                email = decoded_token.get("email")
                email_verified = decoded_token.get("email_verified", False)

                if email and not email_verified:
                    logger.warning(f"Rejected unverified email access: {email} (UID: {uid})")
                    raise HTTPException(
                        status_code=403,
                        detail="Email address not verified. Please check your inbox and verify your email to access Wayfinder.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                return uid

            except HTTPException:
                raise
            except auth.ExpiredIdTokenError:
                logger.info("Firebase ID token has expired.")
                raise HTTPException(
                    status_code=401,
                    detail="Firebase ID token has expired. Please refresh your session.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            except auth.InvalidIdTokenError as e:
                logger.warning(f"Invalid Firebase ID token: {e}")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid Firebase ID token.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            except Exception as e:
                logger.warning(f"Firebase token verification failed: {e}")
                raise HTTPException(
                    status_code=401,
                    detail=f"Token verification error: {str(e)}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        else:
            # Firebase Admin not initialized, but token was sent and did not match API key
            logger.warning("Firebase Admin SDK is not initialized; cannot verify Bearer token.")
            raise HTTPException(
                status_code=401,
                detail="Authentication failed: Unable to verify token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 3. If WAYFINDER_API_KEY is configured and no credentials provided, reject with 401
    if config.WAYFINDER_API_KEY and config.WAYFINDER_API_KEY.strip():
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please provide a valid Bearer token or API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Unauthenticated fallback for local dev when WAYFINDER_API_KEY is unset
    return "guest_user"


# Alias for explicit naming
require_firebase_user = get_current_user
