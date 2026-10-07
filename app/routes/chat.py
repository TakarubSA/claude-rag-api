from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.clients.repository import get_client_by_token
from app.knowledge.gaps_repository import insert_gap
from app.services.intent import classify_intent
from app.services.rag import answer_question


router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    # 1. Validate Authorization header
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

    # 2. Extract token
    token = authorization.replace(
        "Bearer ",
        "",
        1,
    ).strip()

    # 3. Resolve token → client
    client = get_client_by_token(token)

    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    # 4. Classify intent
    intent = classify_intent(
        request.message
    )

    print("MESSAGE:", request.message)
    print("CLIENT ID:", client.id)
    print("INTENT:", intent)

    # 5. Search and answer using this client's knowledge
    result = answer_question(
        question=request.message,
        client_id=client.id,
    )

    # 6. Save knowledge gaps
    if result["status"] in {
        "RELATED_BUT_UNKNOWN",
    }:
        insert_gap(
            client_id=client.id,
            message=request.message,
            gap_type=result["status"],
        )

    # 7. Return response
    return {
        "action": intent,
        "answer": result["answer"],
        "best_distance": result["best_distance"],
        "status": result["status"],
    }
