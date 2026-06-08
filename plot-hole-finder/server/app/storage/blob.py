"""Blob storage for uploads -- R2 in prod, local disk in dev.

Both behind one interface so the rest of the app never knows which is active.
Key scheme (caller's responsibility, but recommended):

    users/{user_id}/projects/{project_id}/documents/{document_id}
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from ..config import Settings, get_settings


class BlobStore(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store bytes under ``key``; return the key."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...


class LocalDiskBlobStore(BlobStore):
    """Dev backend: writes blobs under a local root directory."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        # Prevent path traversal escaping the root.
        full = os.path.abspath(os.path.join(self.root, key))
        if not full.startswith(self.root + os.sep):
            raise ValueError(f"unsafe blob key: {key!r}")
        return full

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return key

    def get(self, key: str) -> bytes:
        with open(self._path(key), "rb") as fh:
            return fh.read()

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


class R2BlobStore(BlobStore):
    """Prod backend: Cloudflare R2 via the S3-compatible API (boto3, lazy)."""

    def __init__(self, settings: Settings):
        import boto3  # lazy: only needed when R2 is selected

        self.bucket = settings.r2_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
        )

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def get(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


def get_blob_store(settings: Settings | None = None) -> BlobStore:
    settings = settings or get_settings()
    if settings.blob_backend == "r2":
        return R2BlobStore(settings)
    return LocalDiskBlobStore(settings.blob_local_root)
