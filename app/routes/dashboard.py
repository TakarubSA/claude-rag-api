from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.auth.security import (
    JWT_SECRET,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.dashboard import repository
from app.knowledge.ingestion import ingest_file


# Routes are plain `def`: DB writes and embedding block, so FastAPI runs
# them in its thread pool.
router = APIRouter(prefix="/api")

MAX_UPLOAD_CHARS = 5_000_000


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateProviderRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=8, max_length=200)
    anthropic_api_key: str | None = None
    credit_limit_usd: float | None = Field(default=None, ge=0)


class UpdateProviderRequest(BaseModel):
    anthropic_api_key: str | None = None
    credit_limit_usd: float | None = Field(default=None, ge=0)


class UploadRequest(BaseModel):
    filename: str = Field(max_length=300)
    content: str = Field(max_length=MAX_UPLOAD_CHARS)


# ----------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------

def current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not logged in")

    payload = decode_access_token(authorization[len("Bearer "):].strip())

    if payload is None:
        raise HTTPException(status_code=401, detail="Session expired, please log in again")

    return payload


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return user


def require_provider(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "provider" or user.get("client_id") is None:
        raise HTTPException(status_code=403, detail="Providers only")
    return user


@router.post("/auth/login")
def login(request: LoginRequest):
    if not JWT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Dashboard login is not configured (JWT_SECRET is missing)",
        )

    user = repository.get_user_by_email(request.email.strip())

    if user is None or user.role not in ("admin", "provider") or not verify_password(
        request.password, user.password_hash
    ):
        raise HTTPException(status_code=401, detail="Wrong email or password")

    return {
        "token": create_access_token(user.id, user.client_id, user.role),
        "role": user.role,
        "name": user.name,
    }


# ----------------------------------------------------------------------
# Admin
# ----------------------------------------------------------------------

def _provider_or_404(client_id: int) -> dict:
    provider = repository.get_provider(client_id)

    if provider is None:
        raise HTTPException(status_code=404, detail="Provider not found")

    return provider


def _upload(client_id: int, request: UploadRequest) -> dict:
    try:
        count = ingest_file(request.filename, request.content, client_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"count": count}


@router.get("/admin/providers")
def admin_list_providers(_: dict = Depends(require_admin)):
    return repository.list_providers()


@router.post("/admin/providers")
def admin_create_provider(
    request: CreateProviderRequest,
    _: dict = Depends(require_admin),
):
    email = request.email.strip()

    if repository.email_exists(email):
        raise HTTPException(status_code=409, detail="This email is already used")

    client_id = repository.create_provider(
        name=request.name.strip(),
        email=email,
        password_hash=hash_password(request.password),
        anthropic_api_key=(request.anthropic_api_key or "").strip() or None,
        credit_limit_usd=request.credit_limit_usd,
    )

    return repository.get_provider(client_id)


@router.patch("/admin/providers/{client_id}")
def admin_update_provider(
    client_id: int,
    request: UpdateProviderRequest,
    _: dict = Depends(require_admin),
):
    _provider_or_404(client_id)

    # Only fields actually sent are changed; an empty key clears it
    # (the provider then falls back to the server's default key).
    changes = request.model_dump(include=request.model_fields_set)

    if "anthropic_api_key" in changes:
        changes["anthropic_api_key"] = (changes["anthropic_api_key"] or "").strip() or None

    repository.update_provider(client_id, changes)

    return repository.get_provider(client_id)


@router.post("/admin/providers/{client_id}/regenerate-token")
def admin_regenerate_token(client_id: int, _: dict = Depends(require_admin)):
    _provider_or_404(client_id)
    repository.regenerate_token(client_id)
    return repository.get_provider(client_id)


@router.post("/admin/providers/{client_id}/knowledge")
def admin_upload_knowledge(
    client_id: int,
    request: UploadRequest,
    _: dict = Depends(require_admin),
):
    _provider_or_404(client_id)
    return _upload(client_id, request)


# ----------------------------------------------------------------------
# Provider
# ----------------------------------------------------------------------

@router.get("/provider/me")
def provider_me(user: dict = Depends(require_provider)):
    return _provider_or_404(user["client_id"])


@router.post("/provider/knowledge")
def provider_upload_knowledge(
    request: UploadRequest,
    user: dict = Depends(require_provider),
):
    return _upload(user["client_id"], request)
