from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.clients.repository import get_client_by_token
from app.services.rag import answer_question
from app.services.intent import classify_intent


router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
def chat(
    request: ChatRequest,
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

    # Classify the user's message
    intent = classify_intent(request.message)

    print("MESSAGE:", request.message)
    print("INTENT:", intent)

    # Current behavior: continue using RAG
    answer = answer_question(
        request.message,
        client_id=client.id,
    )

    return {
        "answer": answer
    }