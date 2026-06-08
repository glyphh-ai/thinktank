"""Domain exceptions, mapped to HTTP status codes in the API layer."""

from __future__ import annotations


class AppError(Exception):
    status_code = 400
    code = "error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class PlanLimitExceeded(Forbidden):
    code = "plan_limit_exceeded"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"
