from fastapi import FastAPI
from .api import router as tasks_router
from .config import settings

app = FastAPI(
    title=settings.app_name,
    description="A simple Task Manager REST API",
    version="1.0.0"
)

app.include_router(tasks_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}
