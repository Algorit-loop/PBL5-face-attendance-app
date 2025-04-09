import json
import os
from typing import List, Dict, Any
from models import Employee

# Database file path
EMPLOYEES_FILE = 'employees.json'

def load_employees() -> List[Dict[str, Any]]:
    """
    Load employees from the JSON file
    
    Returns:
        List of employee data
    """
    if os.path.exists(EMPLOYEES_FILE):
        with open(EMPLOYEES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_employees(employees_data: List[Dict[str, Any]]) -> None:
    """
    Save employees to the JSON file
    
    Args:
        employees_data: List of employee data to save
    """
    with open(EMPLOYEES_FILE, 'w', encoding='utf-8') as f:
        json.dump(employees_data, f, ensure_ascii=False, indent=2)

# Initialize employees from file
employees = load_employees()
if not employees:
    # Default employees if file is empty
    employees = [
        {
            "id": 1,
            "full_name": "Nguyễn Văn A",
            "birth_date": "1990-01-01",
            "email": "nguyenvana@example.com",
            "phone": "0123456789",
            "address": "Hà Nội",
            "gender": "Nam",
            "position": "Nhân viên"
        },
        {
            "id": 2,
            "full_name": "Trần Thị B",
            "birth_date": "1992-05-15",
            "email": "tranthib@example.com",
            "phone": "0987654321",
            "address": "TP.HCM",
            "gender": "Nữ",
            "position": "Quản lý"
        }
    ]
    save_employees(employees) 