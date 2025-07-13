from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from ..core.config import settings

# The engine is the entry point to the database.
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

# A session factory, to create new session objects when needed.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency for FastAPI endpoints to get a DB session.
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()