from fastapi import APIRouter, HTTPException

from schemas.user_schema import UserListResponse, UserResponse
from services.user_service import get_all_users, get_user_detail

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=UserListResponse)
def list_users():
    """Return all user IDs currently in the system."""
    users = get_all_users()
    return {"users": users, "count": len(users)}


@router.get("/{user_id}", response_model=UserResponse)
def user_detail(user_id: str):
    """Return profile (likes, dislikes, history size) for a specific user."""
    data = get_user_detail(user_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    return data
