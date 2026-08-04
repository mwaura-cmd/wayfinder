import logging
from typing import Optional
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth
import config

logger = logging.getLogger(__name__)

# Use auto_error=False so missing credentials can fall back to guest access when WAYFINDER_API_KEY is empty
security = HTTPBearer(auto_error=False)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> str:
    """
    Validates the Firebase ID token or WAYFINDER_API_KEY in the Authorization header.
    If no credentials are provided and WAYFINDER_API_KEY is not set, allows guest access.
    """
    if credentials and credentials.credentials:
        token = credentials.credentials.strip()

        # 1. Check if token matches WAYFINDER_API_KEY
        if config.WAYFINDER_API_KEY and token == config.WAYFINDER_API_KEY:
            return "api_key_user"

        # 2. Check if Firebase is initialized and verify token
        if firebase_admin._apps:
            try:
                decoded_token = auth.verify_id_token(token)
                uid = decoded_token['uid']

                email = decoded_token.get('email')
                email_verified = decoded_token.get('email_verified', False)

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
            except Exception as e:
                logger.warning(f"Firebase token verification failed: {e}")

    # 3. If WAYFINDER_API_KEY is set but credentials failed or were missing, reject with 401
    if config.WAYFINDER_API_KEY and config.WAYFINDER_API_KEY.strip():
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Default: Guest access for local dev / unauthenticated visits when WAYFINDER_API_KEY is not set
    return "guest_user"
