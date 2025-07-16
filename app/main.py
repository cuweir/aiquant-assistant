from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.lifespan import lifespan
from .core.config import settings
from .api.endpoints import analysis

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

origins = [
    "http://localhost",
    "http://localhost:5173",  # Your Vue dev server
]

if settings.SERVER_PUBLIC_IP:
    # Assuming frontend runs on port 5173
    frontend_origin_on_server = f"http://{settings.SERVER_PUBLIC_IP}:5173"
    print(f"INFO: Dynamically adding '{frontend_origin_on_server}' to allowed CORS origins.")
    origins.append(frontend_origin_on_server)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True, # Allow cookies (not needed now, but good practice)
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Include the API router
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.PROJECT_VERSION}