import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database.migrations import run_migrations
from app.embeddings.voyage import create_embeddings
from app.routes.chat import router as chat_router
from app.routes.dashboard import router as dashboard_router
from app.routes.knowledge import router as knowledge_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_migrations()

    # Load and run the embedding model once at boot (~4-5 s) so the first
    # user request after a restart does not pay for it.
    started = time.perf_counter()
    create_embeddings(["warm up"], input_type="query")
    logger.info(
        "Embedding model ready in %.1f s",
        time.perf_counter() - started,
    )
    yield


app = FastAPI(lifespan=lifespan)

# The dashboard is a separate web app, so its origin must be allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "DASHBOARD_ORIGINS",
        "http://localhost:5173",
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = [
        {
            "loc": error.get("loc"),
            "type": error.get("type"),
            "msg": error.get("msg"),
        }
        for error in exc.errors()
    ]

    logger.warning(
        "REQUEST VALIDATION FAILED: path=%s errors=%s",
        request.url.path,
        errors,
    )

    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )


app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(dashboard_router)