import logging
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth
import config

logger = logging.getLogger(__name__)

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Validates the Firebase ID token in the Authorization header.
    Ensures that the user's email is verified if registered via email.
    Returns the user's UID if valid.
    """
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        
        # Enforce email verification for email-based accounts
        email = decoded_token.get('email')
        email_verified = decoded_token.get('email_verified', False)
        
        # If the account has an email and it is not verified, block access
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
        logger.error(f"Error verifying Firebase ID token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

