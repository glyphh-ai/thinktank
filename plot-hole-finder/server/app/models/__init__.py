"""All ORM models and the shared declarative ``Base``.

Import this module (or ``Base``) before ``create_all`` so every table is
registered on the metadata.
"""

from .base import Base, Bipolar, Embedding, new_id
from .billing import StripeEvent, Subscription, SubscriptionStatus
from .narrative import (
    AnalysisRun,
    Entity,
    EntityType,
    EvidenceRole,
    Fact,
    FactKind,
    Issue,
    IssueCategory,
    IssueEvidence,
    IssueStatus,
    RunKind,
    RunStatus,
    Severity,
)
from .project import Document, Project, ProjectStatus, SourceType
from .usage import UsageEvent, UsageMetric, sum_usage
from .user import RefreshToken, TokenKind, User, UserStatus, VerificationToken
from .vectors import (
    EMBEDDING_DIM,
    CortexLevel,
    HypervectorRecord,
    PassageEmbedding,
)

__all__ = [
    "Base",
    "Bipolar",
    "Embedding",
    "new_id",
    # user
    "User",
    "UserStatus",
    "VerificationToken",
    "TokenKind",
    "RefreshToken",
    # billing
    "Subscription",
    "SubscriptionStatus",
    "StripeEvent",
    # project
    "Project",
    "ProjectStatus",
    "SourceType",
    "Document",
    # narrative
    "Entity",
    "EntityType",
    "Fact",
    "FactKind",
    "AnalysisRun",
    "RunKind",
    "RunStatus",
    "Issue",
    "IssueCategory",
    "Severity",
    "IssueStatus",
    "IssueEvidence",
    "EvidenceRole",
    # usage
    "UsageEvent",
    "UsageMetric",
    "sum_usage",
    # vectors
    "PassageEmbedding",
    "HypervectorRecord",
    "CortexLevel",
    "EMBEDDING_DIM",
]
