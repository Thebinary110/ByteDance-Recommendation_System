from pydantic import BaseModel
from typing import List, Optional


class UserResponse(BaseModel):
    user_id:          str
    likes:            List[str]
    dislikes:         List[str]
    history_size:     int
    last_interaction: Optional[int] = None


class UserListResponse(BaseModel):
    users: List[str]
    count: int
