import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")

# pool_pre_ping replaces connections the database has closed (restarts,
# idle timeouts) instead of failing the next request with them.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
