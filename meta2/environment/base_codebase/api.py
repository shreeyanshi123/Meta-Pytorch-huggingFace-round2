from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import httpx
import uuid

from .utils import sanitize_string, paginate_results, parse_date, setup_logger
from .config import settings

router = APIRouter()
logger = setup_logger("task_manager_api")

tasks_db = {}

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: str

class TaskResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    due_date: str
    timezone: str
    status: str

async def get_timezone_for_date(date_str: str) -> str:
    """Mock external API call using httpx."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://worldtimeapi.org/api/timezone/Etc/UTC")
            if response.status_code == 200:
                data = response.json()
                return data.get("timezone", "UTC")
            return "UTC"
    except Exception as e:
        logger.error(f"Failed to fetch timezone: {e}")
        return "UTC"

@router.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(task: TaskCreate):
    title = sanitize_string(task.title)
    if not title:
        raise HTTPException(status_code=400, detail="Title is required and must contain alphanumeric characters")
    
    try:
        parsed_date = parse_date(task.due_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    task_id = str(uuid.uuid4())
    timezone = await get_timezone_for_date(task.due_date)
    
    new_task = {
        "id": task_id,
        "title": title,
        "description": task.description,
        "due_date": task.due_date,
        "timezone": timezone,
        "status": "pending"
    }
    
    tasks_db[task_id] = new_task
    logger.info(f"Created task {task_id}")
    return new_task

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/tasks", response_model=dict)
async def list_tasks(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    tasks_list = list(tasks_db.values())
    paginated_tasks, meta = paginate_results(tasks_list, page, page_size)
    return {
        "data": paginated_tasks,
        "meta": meta
    }

@router.put("/tasks/{task_id}", response_model=TaskResponse)
async def update_task(task_id: str, task: TaskCreate):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
        
    title = sanitize_string(task.title)
    if not title:
        raise HTTPException(status_code=400, detail="Invalid title")
        
    try:
        parse_date(task.due_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    timezone = await get_timezone_for_date(task.due_date)
    
    tasks_db[task_id].update({
        "title": title,
        "description": task.description,
        "due_date": task.due_date,
        "timezone": timezone
    })
    
    return tasks_db[task_id]

@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    del tasks_db[task_id]
    logger.info(f"Deleted task {task_id}")
    return None
