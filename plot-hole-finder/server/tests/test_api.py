"""End-to-end API tests over the full stack (offline adapters).

Exercises the real routers/services/pipeline against a temp SQLite DB with the
heuristic extractor, hashing embedder, console-recording email, and disabled
billing -- no network, no API key.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.storage.email import EmailMessage, EmailSender

# A short manuscript with one planted continuity error (Elena's eyes) and one
# *licensed* change that must NOT be flagged (Marcus dyes his hair).
STORY = """\
Elena's eyes were green as she stood at the village gate that morning.

The road to the city was long. Marcus had black hair and walked beside her.

Halfway through the journey, Marcus dyed his hair blonde at a roadside inn.

By the time they reached the castle, Marcus had blonde hair once more.

In the throne room, Elena's eyes were blue, though no one remarked upon it.
"""


class RecordingEmail(EmailSender):
    def __init__(self):
        self.messages: list[EmailMessage] = []

    def send(self, message: EmailMessage) -> None:
        self.messages.append(message)

    def last_pin(self) -> str:
        m = re.search(r"\b(\d{6})\b", self.messages[-1].body)
        assert m, "no PIN in email"
        return m.group(1)

    def last_reset_token(self) -> str:
        m = re.search(r"token=([\w\-]+)", self.messages[-1].body)
        assert m, "no reset token in email"
        return m.group(1)


@pytest.fixture
def client(tmp_path):
    from fakes import StubEmbedder, StubExtractor, StubVerifier

    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path}/api.db",
        blob_local_root=str(tmp_path / "blobs"),
        jwt_secret="test-secret",
        email_backend="console",
        billing_backend="disabled",
    )
    mailbox = RecordingEmail()
    # Inject deterministic test doubles for the pipeline plumbing (no Claude,
    # no model download); quality is measured against real Claude in eval/.
    app = create_app(settings, email_sender=mailbox, extractor=StubExtractor(),
                     verifier=StubVerifier(), embedder=StubEmbedder())
    c = TestClient(app)
    c.mailbox = mailbox
    return c


def _register_and_login(client, email="writer@example.com", password="supersecret"):
    r = client.post("/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    pin = client.mailbox.last_pin()
    r = client.post("/auth/verify-email", json={"email": email, "pin": pin})
    assert r.status_code == 200, r.text
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_register_requires_verification_before_login(client):
    client.post("/auth/register", json={"email": "x@y.com", "password": "supersecret"})
    r = client.post("/auth/login", json={"email": "x@y.com", "password": "supersecret"})
    assert r.status_code == 403  # email not verified


def test_wrong_pin_rejected(client):
    client.post("/auth/register", json={"email": "x@y.com", "password": "supersecret"})
    r = client.post("/auth/verify-email", json={"email": "x@y.com", "pin": "000000"})
    assert r.status_code == 401


def test_full_flow_finds_planted_plot_hole(client):
    token = _register_and_login(client)

    me = client.get("/auth/me", headers=_auth(token)).json()
    assert me["plan"] == "free"

    r = client.post("/projects", headers=_auth(token), json={"name": "Novel", "text": STORY})
    assert r.status_code == 201, r.text
    project_id = r.json()["id"]

    r = client.post(f"/projects/{project_id}/analyze", headers=_auth(token))
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "succeeded"  # inline runner ran it synchronously
    assert job["result"]["issues"] >= 1
    # job is observable via the jobs API
    polled = client.get(f"/jobs/{job['id']}", headers=_auth(token)).json()
    assert polled["progress"] == 1.0

    issues = client.get(f"/projects/{project_id}/issues", headers=_auth(token)).json()
    # Elena's eye colour is flagged; Marcus's licensed hair change is not.
    titles = " ".join(i["title"].lower() for i in issues)
    assert "elena" in titles
    assert "marcus" not in titles
    assert all(len(i["evidence"]) == 2 for i in issues)


def test_free_plan_project_limit(client):
    token = _register_and_login(client)
    r1 = client.post("/projects", headers=_auth(token), json={"name": "One", "text": STORY})
    assert r1.status_code == 201
    r2 = client.post("/projects", headers=_auth(token), json={"name": "Two", "text": STORY})
    assert r2.status_code == 403
    assert r2.json()["error"] == "plan_limit_exceeded"


def test_upgrade_unlocks_limits_and_fix(client):
    token = _register_and_login(client)
    # Upgrade to Pro (disabled billing flips the plan directly).
    r = client.post("/billing/checkout", headers=_auth(token), json={"tier": "pro"})
    assert r.status_code == 200, r.text
    assert client.get("/auth/me", headers=_auth(token)).json()["plan"] == "pro"

    # Now a second project is allowed.
    client.post("/projects", headers=_auth(token), json={"name": "One", "text": STORY})
    r = client.post("/projects", headers=_auth(token), json={"name": "Two", "text": STORY})
    assert r.status_code == 201
    project_id = r.json()["id"]

    client.post(f"/projects/{project_id}/analyze", headers=_auth(token))
    r = client.post(f"/projects/{project_id}/fix", headers=_auth(token))
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "succeeded"
    issues = client.get(f"/projects/{project_id}/issues", headers=_auth(token)).json()
    assert any(i["suggested_fix"] for i in issues)


def test_docx_upload_parses_and_analyzes(client):
    from pathlib import Path

    token = _register_and_login(client)
    docx = (Path(__file__).parent / "fixtures" / "sample.docx").read_bytes()
    r = client.post(
        "/projects/upload",
        headers=_auth(token),
        data={"name": "Legal Doc"},
        files={"file": ("sample.docx", docx,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert r.status_code == 201, r.text
    project_id = r.json()["id"]
    assert r.json()["word_count"] > 0  # text was extracted from the .docx

    r = client.post(f"/projects/{project_id}/analyze", headers=_auth(token))
    assert r.status_code == 202, r.text
    assert r.json()["result"]["issues"] >= 1  # Elena's eye-colour contradiction


def test_unsupported_upload_rejected(client):
    token = _register_and_login(client)
    r = client.post(
        "/projects/upload",
        headers=_auth(token),
        data={"name": "Bad"},
        files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "unsupported_document"


def test_password_reset_flow(client):
    email, pw = "reset@me.com", "supersecret"
    _register_and_login(client, email, pw)
    client.post("/auth/forgot-password", json={"email": email})
    reset_token = client.mailbox.last_reset_token()

    r = client.post("/auth/reset-password", json={"token": reset_token, "password": "newpassword1"})
    assert r.status_code == 200
    assert client.post("/auth/login", json={"email": email, "password": pw}).status_code == 401
    assert client.post("/auth/login", json={"email": email, "password": "newpassword1"}).status_code == 200


def test_refresh_rotation(client):
    token = _register_and_login(client)
    login = client.post("/auth/login", json={"email": "writer@example.com", "password": "supersecret"})
    refresh = login.json()["refresh_token"]
    r = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    # old refresh token is now revoked (rotation)
    assert client.post("/auth/refresh", json={"refresh_token": refresh}).status_code == 401
