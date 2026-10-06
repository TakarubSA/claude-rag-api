from fastapi import FastAPI

from app.routes.chat import router as chat_router
from app.routes.knowledge import router as knowledge_router


app = FastAPI()

app.include_router(chat_router)
app.include_router(knowledge_router)