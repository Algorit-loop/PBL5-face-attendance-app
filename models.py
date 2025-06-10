from typing import Optional
from pydantic import BaseModel

class Employee(BaseModel):
    """
    Model for employee data
    """
    id: Optional[int] = None
    full_name: str
    birth_date: str
    email: str
    phone: str
    address: str
    gender: str
    position: str
    department: str
    username: str
    password: Optional[str] = None

class User(BaseModel):
    """
    Model for user authentication
    """
    username: str
    password: str

class UserSession(BaseModel):
    """
    Model for user session data
    """
    id: int
    username: str
    full_name: str
    position: str
    department: str
    authenticated: bool = True

class APIResponse(BaseModel):
    """
    Model for standard API response
    """
    success: bool
    message: str
    data: Optional[dict] = None 