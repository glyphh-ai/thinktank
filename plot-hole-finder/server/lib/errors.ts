export class AppError extends Error {
  statusCode = 400;
  code = "error";
  constructor(message: string) {
    super(message);
  }
}
export class ValidationError extends AppError { statusCode = 422; code = "validation_error"; }
export class Unauthorized extends AppError { statusCode = 401; code = "unauthorized"; }
export class Forbidden extends AppError { statusCode = 403; code = "forbidden"; }
export class NotFound extends AppError { statusCode = 404; code = "not_found"; }
export class Conflict extends AppError { statusCode = 409; code = "conflict"; }
export class PlanLimitExceeded extends Forbidden { code = "plan_limit_exceeded"; }
