import os
import json
import logging
import base64
from pathlib import Path
from typing import Optional
import firebase_admin
from firebase_admin import credentials, firestore, auth
import config

logger = logging.getLogger(__name__)


def init_firebase() -> Optional[firebase_admin.App]:
    """
    Initializes the Firebase Admin SDK using the best available credential source:
    1. FIREBASE_CREDENTIALS_JSON env variable (raw JSON string or base64 encoded)
    2. FIREBASE_CREDENTIALS_PATH or known secret file locations (Render /etc/secrets)
    3. GOOGLE_APPLICATION_CREDENTIALS env variable
    4. Project ID fallback (options={'projectId': config.FIREBASE_PROJECT_ID}) which
       allows auth.verify_id_token to verify tokens against Google's public keys.
    """
    if firebase_admin._apps:
        return firebase_admin.get_app()

    # Priority 1: JSON String in environment variable (Render / Heroku / Cloud Run)
    if config.FIREBASE_CREDENTIALS_JSON and config.FIREBASE_CREDENTIALS_JSON.strip():
        try:
            raw_str = config.FIREBASE_CREDENTIALS_JSON.strip()
            # Strip accidental surrounding quotes from dashboard copy-pastes
            if (raw_str.startswith("'") and raw_str.endswith("'")) or (raw_str.startswith('"') and raw_str.endswith('"')):
                raw_str = raw_str[1:-1].strip()

            # Check if base64 encoded
            if not raw_str.startswith("{"):
                try:
                    raw_str = base64.b64decode(raw_str).decode("utf-8")
                except Exception:
                    pass

            cert_dict = json.loads(raw_str)
            cred = credentials.Certificate(cert_dict)
            app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK successfully initialized via FIREBASE_CREDENTIALS_JSON.")
            return app
        except Exception as e:
            logger.warning(f"Failed to parse FIREBASE_CREDENTIALS_JSON: {e}")

    # Priority 2: File path
    possible_paths = [
        Path(config.FIREBASE_CREDENTIALS_PATH),
        Path("/etc/secrets/firebase-adminsdk.json"),
        Path("/etc/secrets/serviceAccountKey.json"),
        Path(__file__).parent.parent / "firebase-adminsdk.json",
    ]

    for p in possible_paths:
        if p and p.exists():
            try:
                cred = credentials.Certificate(str(p))
                app = firebase_admin.initialize_app(cred)
                logger.info(f"Firebase Admin SDK successfully initialized via credentials file: {p}")
                return app
            except Exception as e:
                logger.warning(f"Failed to load credentials file from {p}: {e}")

    # Priority 3: Fallback with Project ID (allows public key token verification)
    try:
        project_id = config.FIREBASE_PROJECT_ID or "wayfinder-b98c7"
        app = firebase_admin.initialize_app(options={"projectId": project_id})
        logger.info(f"Firebase Admin SDK initialized with projectId='{project_id}' for public ID token verification.")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin with projectId: {e}")

    return None


# Run initialization on module import
init_firebase()


def get_db():
    """Returns the Firestore client if initialized with credentials, otherwise None."""
    try:
        if not firebase_admin._apps:
            init_firebase()
        return firestore.client()
    except Exception as e:
        logger.debug(f"Firestore not available: {e}")
        return None
