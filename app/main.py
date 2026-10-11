import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routes.chat import router as chat_router
from app.routes.knowledge import router as knowledge_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


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