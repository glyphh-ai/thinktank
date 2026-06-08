"""Swappable infrastructure: blob storage and email sending."""

from .blob import BlobStore, LocalDiskBlobStore, R2BlobStore, get_blob_store
from .email import (
    ConsoleEmailSender,
    EmailMessage,
    EmailSender,
    FileEmailSender,
    ResendEmailSender,
    SmtpEmailSender,
    get_email_sender,
    password_reset_message,
    verification_pin_message,
)

__all__ = [
    "BlobStore",
    "LocalDiskBlobStore",
    "R2BlobStore",
    "get_blob_store",
    "EmailSender",
    "EmailMessage",
    "ConsoleEmailSender",
    "FileEmailSender",
    "ResendEmailSender",
    "SmtpEmailSender",
    "get_email_sender",
    "verification_pin_message",
    "password_reset_message",
]
