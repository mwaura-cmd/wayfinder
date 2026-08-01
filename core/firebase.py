import os
import json
import logging
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
import config

logger = logging.getLogger(__name__)

# Initialize Firebase only once
if not firebase_admin._apps:
    try:
        # Priority 1: Check if credentials were provided as a JSON string (good for PaaS deployments)
        if config.FIREBASE_CREDENTIALS_JSON:
            cert_dict = json.loads(config.FIREBASE_CREDENTIALS_JSON)
            cred = credentials.Certificate(cert_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized via JSON string.")
        
        # Priority 2: Check if file exists
        elif Path(config.FIREBASE_CREDENTIALS_PATH).exists():
            cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized via service account file.")
            
        else:
            logger.warning("No Firebase credentials found. Firebase features will fail if called.")
            
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")

def get_db():
    """Returns the Firestore client, initializing if necessary."""
    try:
        return firestore.client()
    except ValueError:
        # App hasn't been initialized
        logger.error("Firestore requested but Firebase is not initialized.")
        return None
