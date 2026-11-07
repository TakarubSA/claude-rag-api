import logging
import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from pwdlib import PasswordHash


load_dotenv()

logger = logging.getLogger(__name__)

password_hash = PasswordHash.recommended()

JWT_SECRET = os.getenv("JWT_SECRET")

# Only dashboard logins need this: without it the chat API keeps working
# and dashboard login is refused.
if not JWT_SECRET:
    logger.error("JWT_SECRET is not configured; dashboard login is disabled")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_HOURS = 12


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    return password_hash.verify(
        password,
        hashed_password,
    )


def create_access_token(
    user_id: int,
    client_id: int | None,
    role: str,
) -> str:
    payload = {
        "user_id": user_id,
        "client_id": client_id,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_HOURS),
    }

    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not configured")

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict | None:
    """Return the token's payload, or None if it is invalid or expired."""
    if not JWT_SECRET:
        return None

    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
