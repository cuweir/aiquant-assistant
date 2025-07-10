from fastapi import FastAPI

from .core.lifespan import lifespan
from .core.config import settings
from .api.endpoints import analysis

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

# Include the API router
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.PROJECT_VERSION}