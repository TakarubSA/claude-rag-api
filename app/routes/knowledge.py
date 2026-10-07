from fastapi import APIRouter, Header, HTTPException, Body

from app.clients.repository import get_client_by_token
from app.knowledge.ingestion import ingest_data


router = APIRouter()


@router.post("/knowledge/upload")
async def upload_knowledge(
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

    token = authorization.replace("Bearer ", "", 1).strip()

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