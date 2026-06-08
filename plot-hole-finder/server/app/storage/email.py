"""Email sending -- dev-friendly by default.

Backends:

- ``console`` (default, dev): prints the message (incl. verification PIN) to
  stdout, so signup/reset flows are testable with zero infrastructure.
- ``file`` (dev): writes each message as an ``.eml`` file you can open.
- ``smtp`` (prod, or local MailHog/Mailpit on :1025): real SMTP.

Sent mail is represented by :class:`EmailMessage`; templating helpers build the
verification-PIN and password-reset messages the auth service will send.
"""

from __future__ import annotations

import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage as MimeMessage

from ..config import Settings, get_settings


@dataclass
class EmailMessage:
    to: str
    subject: str
    body: str
    from_addr: str


class EmailSender(ABC):
    @abstractmethod
    def send(self, message: EmailMessage) -> None:
        ...


class ConsoleEmailSender(EmailSender):
    """Dev: print the email so PINs/links are visible in the server log."""

    def send(self, message: EmailMessage) -> None:
        print(
            "\n===== EMAIL (console backend) =====\n"
            f"From:    {message.from_addr}\n"
            f"To:      {message.to}\n"
            f"Subject: {message.subject}\n"
            "-----------------------------------\n"
            f"{message.body}\n"
            "===================================\n",
            flush=True,
        )


class FileEmailSender(EmailSender):
    """Dev: write each message as a timestamped .eml file."""

    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

    def send(self, message: EmailMessage) -> None:
        mime = MimeMessage()
        mime["From"] = message.from_addr
        mime["To"] = message.to
        mime["Subject"] = message.subject
        mime.set_content(message.body)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = os.path.join(self.directory, f"{stamp}-{message.to}.eml")
        with open(path, "wb") as fh:
            fh.write(bytes(mime))


class ResendEmailSender(EmailSender):
    """Prod: send via the Resend HTTP API (https://resend.com).

    Uses the stdlib HTTP client so there's no extra dependency. ``from`` must be
    an address on a domain you've verified in Resend (for quick tests, Resend
    provides ``onboarding@resend.dev``).
    """

    ENDPOINT = "https://api.resend.com/emails"

    def __init__(self, api_key: str, endpoint: str | None = None):
        if not api_key:
            raise ValueError("RESEND_API_KEY is required for the resend email backend")
        self.api_key = api_key
        self.endpoint = endpoint or self.ENDPOINT

    @staticmethod
    def _payload(message: EmailMessage) -> dict:
        return {
            "from": message.from_addr,
            "to": [message.to],
            "subject": message.subject,
            "text": message.body,
        }

    def send(self, message: EmailMessage) -> None:
        import json
        import urllib.error
        import urllib.request

        data = json.dumps(self._payload(message)).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:  # surface Resend's error body
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Resend send failed ({exc.code}): {detail}") from exc


class SmtpEmailSender(EmailSender):
    """Prod (or local MailHog/Mailpit): send via SMTP."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def send(self, message: EmailMessage) -> None:
        mime = MimeMessage()
        mime["From"] = message.from_addr
        mime["To"] = message.to
        mime["Subject"] = message.subject
        mime.set_content(message.body)

        s = self.settings
        with smtplib.SMTP(s.smtp_host, s.smtp_port) as client:
            if s.smtp_use_tls:
                client.starttls()
            if s.smtp_user:
                client.login(s.smtp_user, s.smtp_password)
            client.send_message(mime)


def get_email_sender(settings: Settings | None = None) -> EmailSender:
    settings = settings or get_settings()
    if settings.email_backend == "resend":
        return ResendEmailSender(settings.resend_api_key)
    if settings.email_backend == "smtp":
        return SmtpEmailSender(settings)
    if settings.email_backend == "file":
        return FileEmailSender(settings.email_file_dir)
    return ConsoleEmailSender()


# --- message templates -------------------------------------------------------

def verification_pin_message(settings: Settings, to: str, pin: str) -> EmailMessage:
    return EmailMessage(
        to=to,
        from_addr=settings.email_from,
        subject="Your plot-hole-finder verification code",
        body=(
            f"Welcome! Your verification code is: {pin}\n\n"
            f"It expires in {settings.email_pin_ttl_seconds // 60} minutes.\n"
            "If you didn't request this, you can ignore this email."
        ),
    )


def password_reset_message(settings: Settings, to: str, reset_url: str) -> EmailMessage:
    return EmailMessage(
        to=to,
        from_addr=settings.email_from,
        subject="Reset your plot-hole-finder password",
        body=(
            "We received a request to reset your password.\n\n"
            f"Reset it here: {reset_url}\n\n"
            f"This link expires in {settings.password_reset_ttl_seconds // 60} minutes.\n"
            "If you didn't request this, you can ignore this email."
        ),
    )
