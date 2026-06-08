"""Security primitives -- dependency-free (stdlib only).

- Password hashing: PBKDF2-HMAC-SHA256 with a per-hash salt (encoded as a single
  string). Prod can upgrade to argon2 by swapping these two functions.
- JWT: compact HS256 tokens via hmac (no third-party JWT lib).
- Secrets: numeric email PINs and opaque reset/refresh tokens, plus a one-way
  hash for storing them at rest.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

# --- passwords ---------------------------------------------------------------

_PBKDF2_ROUNDS = 240_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, rounds, salt_b64, dk_b64 = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), _unb64(salt_b64), int(rounds))
        return hmac.compare_digest(dk, _unb64(dk_b64))
    except (ValueError, TypeError):
        return False


# --- opaque secrets / pins ---------------------------------------------------

def generate_pin(digits: int = 6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(digits))


def generate_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def hash_secret(secret: str) -> str:
    """One-way hash for PINs/tokens stored at rest."""
    return hashlib.sha256(secret.encode()).hexdigest()


def verify_secret(secret: str, secret_hash: str) -> bool:
    return hmac.compare_digest(hash_secret(secret), secret_hash)


# --- JWT (HS256) -------------------------------------------------------------

class JWTError(Exception):
    pass


def encode_jwt(claims: dict, secret: str, ttl_seconds: int, algorithm: str = "HS256") -> str:
    if algorithm != "HS256":
        raise JWTError(f"unsupported algorithm: {algorithm}")
    now = int(time.time())
    payload = {**claims, "iat": now, "exp": now + ttl_seconds}
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [_b64url(json.dumps(header).encode()), _b64url(json.dumps(payload).encode())]
    signature = _sign(".".join(segments), secret)
    segments.append(_b64url(signature))
    return ".".join(segments)


def decode_jwt(token: str, secret: str, algorithm: str = "HS256") -> dict:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise JWTError("malformed token") from exc
    expected = _sign(f"{header_b64}.{payload_b64}", secret)
    if not hmac.compare_digest(expected, _unb64url(sig_b64)):
        raise JWTError("bad signature")
    payload = json.loads(_unb64url(payload_b64))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise JWTError("token expired")
    return payload


def _sign(message: str, secret: str) -> bytes:
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()


# --- base64 helpers ----------------------------------------------------------

def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64url(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
