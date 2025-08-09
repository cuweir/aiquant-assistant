import sys
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .core.lifespan import lifespan
from .core.config import settings
from .api.endpoints import analysis, backtest, system

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 1. Console Handler (for Docker logs)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# 2. Rotating File Handler (for persistent logs)
# This will create log files in a 'logs' directory.
# maxBytes=20*1024*1024 means 20MB per file.
# backupCount=3 means it will keep the main log file + 3 old ones.
log_file_handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=20 * 1024 * 1024,
    backupCount=3
)
log_file_handler.setFormatter(log_formatter)

# Get the root logger and add the handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(log_file_handler)

# Redirect print statements to the logger
def redirect_print_to_log():
    class LogWriter:
        def write(self, message):
            if message.strip(): # Avoid logging empty lines
                logging.info(message.strip())
        def flush(self):
            pass
    sys.stdout = LogWriter()
    sys.stderr = LogWriter()

redirect_print_to_log()
logging.info("Logging configured. Print statements will be redirected to logs.")

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
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])
app.include_router(system.router, prefix="/api/v1/system", tags=["System Diagnostics"])


@app.get("/", tags=["Root"])
async def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.PROJECT_VERSION}