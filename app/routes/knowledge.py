from fastapi import APIRouter, Header, HTTPException, Body

from app.clients.repository import get_client_by_token
from app.knowledge.ingestion import ingest_data


router = APIRouter()


# Plain `def` (not async): embedding and DB writes block, so FastAPI must run
# this in its thread pool instead of freezing the event loop for every user.
@router.post("/knowledge/upload")
def upload_knowledge(
    document: dict = Body(...),
    authorization: str | None = Header(default=None),
):
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is required",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format",
        )

    token = authorization[len("Bearer "):].strip()

    client = get_client_by_token(token)

    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    count = ingest_data(
        document,
        client.id,
    )

    return {
        "message": "Knowledge uploaded successfully",
        "count": count,
    }