"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# --- auth ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    pin: str


class ResendPinRequest(BaseModel):
    email: EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    email_verified: bool
    plan: str


# --- projects ---
class CreateProjectRequest(BaseModel):
    name: str
    text: str  # paste flow; upload uses multipart endpoint


class ProjectResponse(BaseModel):
    id: str
    name: str
    status: str
    word_count: int
    created_at: datetime


class DocumentResponse(BaseModel):
    id: str
    filename: str
    word_count: int


# --- analysis / issues ---
class EvidenceResponse(BaseModel):
    role: str
    char_start: int
    char_end: int
    quote: str


class IssueResponse(BaseModel):
    id: str
    category: str
    severity: str
    status: str
    title: str
    description: str
    confidence: float
    suggested_fix: str | None
    evidence: list[EvidenceResponse]


class RunResponse(BaseModel):
    id: str
    kind: str
    status: str
    stats: dict
    error: str | None


# --- billing ---
class CheckoutRequest(BaseModel):
    tier: str  # "basic" | "pro"


class CheckoutResponse(BaseModel):
    url: str


# --- generic ---
class MessageResponse(BaseModel):
    message: str
