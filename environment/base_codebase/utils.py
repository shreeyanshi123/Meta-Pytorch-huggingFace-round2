import logging
from datetime import datetime
from typing import List, Any, Tuple
import re

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger("task_manager_utils")

def format_date(date: datetime) -> str:
    """Formats a datetime object to an ISO 8601 string."""
    return date.isoformat()

def parse_date(date_str: str) -> datetime:
    """Parses an ISO 8601 string to a datetime object."""
    try:
        return datetime.fromisoformat(date_str)
    except ValueError as e:
        logger.error(f"Error parsing date {date_str}: {e}")
        raise ValueError("Invalid date format. Expected ISO 8601.")

def sanitize_string(input_str: str) -> str:
    """Removes non-alphanumeric characters from a string (allows spaces)."""
    if not isinstance(input_str, str):
        return ""
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', input_str)
    return sanitized.strip()

def paginate_results(items: List[Any], page: int = 1, page_size: int = 10) -> Tuple[List[Any], dict]:
    """Paginates a list of items."""
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10
        
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_items = items[start_idx:end_idx]
    
    total_items = len(items)
    total_pages = (total_items + page_size - 1) // page_size
    
    meta = {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }
    
    return paginated_items, meta

def calculate_priority(due_date: datetime, current_date: datetime = None) -> str:
    """Calculates task priority based on due date proximity."""
    if current_date is None:
        current_date = datetime.now()
        
    delta = due_date - current_date
    if delta.days < 0:
        return "OVERDUE"
    elif delta.days <= 1:
        return "HIGH"
    elif delta.days <= 3:
        return "MEDIUM"
    else:
        return "LOW"

def validate_task_data(data: dict) -> bool:
    """Validates basic task constraints."""
    if not data.get("title"):
        return False
    if len(data.get("title", "")) > 100:
        return False
    return True
