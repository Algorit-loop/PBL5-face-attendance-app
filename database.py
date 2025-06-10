import json
import os
from typing import List, Dict, Any
from models import Employee, User

# Database file path
DATA_FILE = 'data.json'

def load_data() -> List[Dict[str, Any]]:
    """
    Load data from the JSON file
    
    Returns:
        List of employee data
    """
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_data(data: List[Dict[str, Any]]) -> None:
    """
    Save data to the JSON file
    
    Args:
        data: List of employee data to save
    """
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_all_employees() -> List[Dict[str, Any]]:
    """
    Get all employees from database
    
    Returns:
        List of all employees
    """
    return load_data()

def get_employee_by_id(employee_id: int) -> Dict[str, Any]:
    """
    Get employee by ID
    
    Args:
        employee_id: Employee ID
        
    Returns:
        Employee data or None if not found
    """
    employees = load_data()
    for employee in employees:
        if employee["id"] == employee_id:
            return employee
    return None

def add_employee(employee_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new employee to the database
    
    Args:
        employee_data: Employee data
        
    Returns:
        Added employee data with ID
    """
    employees = load_data()
    
    # Generate new ID
    max_id = 0
    for employee in employees:
        if employee["id"] > max_id:
            max_id = employee["id"]
    
    employee_data["id"] = max_id + 1
    employees.append(employee_data)
    save_data(employees)
    
    return employee_data

def update_employee(employee_id: int, employee_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update employee data
    
    Args:
        employee_id: Employee ID
        employee_data: New employee data
        
    Returns:
        Updated employee data or None if not found
    """
    employees = load_data()
    
    for i, employee in enumerate(employees):
        if employee["id"] == employee_id:
            # Preserve the ID
            employee_data["id"] = employee_id
            
            # If password is empty, keep the old one
            if "password" in employee_data and not employee_data["password"]:
                employee_data["password"] = employee["password"]
                
            employees[i] = employee_data
            save_data(employees)
            return employee_data
            
    return None

def delete_employee(employee_id: int) -> bool:
    """
    Delete employee by ID
    
    Args:
        employee_id: Employee ID
        
    Returns:
        True if deleted, False if not found
    """
    employees = load_data()
    
    for i, employee in enumerate(employees):
        if employee["id"] == employee_id:
            del employees[i]
            save_data(employees)
            return True
            
    return False

def get_user_by_username(username: str) -> Dict[str, Any]:
    """
    Get user by username
    
    Args:
        username: Username
        
    Returns:
        User data or None if not found
    """
    employees = load_data()
    
    for employee in employees:
        if employee.get("username") == username:
            return employee
            
    return None

def verify_credentials(username: str, password: str) -> Dict[str, Any]:
    """
    Verify user credentials
    
    Args:
        username: Username
        password: Password
        
    Returns:
        User data if credentials are valid, None otherwise
    """
    user = get_user_by_username(username)
    
    if user and user.get("password") == password:
        return user
        
    return None

# Initialize with default data if file doesn't exist
if not os.path.exists(DATA_FILE):
    default_data = [
        {
            "id": 1,
            "full_name": "Admin",
            "birth_date": "2000-01-01",
            "email": "admin@gmail.com",
            "phone": "0123456789",
            "address": "Admin Address",
            "gender": "Nam",
            "position": "admin",
            "department": "IT",
            "username": "admin",
            "password": "admin123"
        },
        {
            "id": 2,
            "full_name": "User Test",
            "birth_date": "2000-01-02",
            "email": "user@gmail.com",
            "phone": "0987654321",
            "address": "User Address",
            "gender": "Nam",
            "position": "employee",
            "department": "HR",
            "username": "user",
            "password": "user123"
        }
    ]
    save_data(default_data) 