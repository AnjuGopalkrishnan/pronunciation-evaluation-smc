from __future__ import annotations

from datetime import datetime, timedelta
from jose import jwt


SECRET_KEY = "b94f8e6272fcef848060d16721461f19439147462768dadfaf9e132b5e7d5dca"
ALGORITHM = "HS256"


def create_access_token(payload: dict, expires_delta: timedelta | None = None):
    payload_copy = payload.copy()
    if expires_delta:
        expires = datetime.utcnow() + expires_delta
    else:
        expires = datetime.utcnow() + timedelta(minutes=15)
    payload_copy.update({"exp": expires})
    token = jwt.encode(payload_copy, SECRET_KEY, algorithm=ALGORITHM)
    return token


def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, ALGORITHM)