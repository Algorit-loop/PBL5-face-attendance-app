from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime, time

class Department(BaseModel):
    """
    Model for department data
    """
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

class Position(BaseModel):
    """
    Model for position data
    """
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

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
    username: Optional[str] = None
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

class Shift(BaseModel):
    id: Optional[int] = None
    name: str
    start_time: str  # Format "HH:MM"
    end_time: str    # Format "HH:MM"
    
class ShiftCreate(BaseModel):
    name: str
    start_time: str  # Format "HH:MM"
    end_time: str    # Format "HH:MM"
    
class ShiftUpdate(BaseModel):
    name: Optional[str] = None
    start_time: Optional[str] = None  # Format "HH:MM"
    end_time: Optional[str] = None    # Format "HH:MM"

class Attendance(BaseModel):
    id: Optional[int] = None
    employee_id: int
    shift_id: int
    check_in_time: str  # Format "YYYY-MM-DD HH:MM:SS"
    status: str  # "on_time", "late", "absent"
    
class AttendanceCreate(BaseModel):
    employee_id: int
    shift_id: int
    check_in_time: Optional[str] = None  # Nếu không cung cấp, sẽ dùng thời gian hiện tại
    
class AttendanceFilter(BaseModel):
    employee_id: Optional[int] = None
    shift_id: Optional[int] = None
    start_date: Optional[str] = None  # Format "YYYY-MM-DD"
    end_date: Optional[str] = None    # Format "YYYY-MM-DD"
    status: Optional[str] = None      # "on_time", "late", "absent" 