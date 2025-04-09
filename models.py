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