from fastapi import APIRouter
from app.schemas.user import UserCreate
from app.services.user_service import create_user, get_all_users

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/")
def add_user(user: UserCreate):
    user_id = create_user(user.dict())
    return {"message": "User added", "id": user_id}

@router.get("/")
def read_users():
    return get_all_users()
