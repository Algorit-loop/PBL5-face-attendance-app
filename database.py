import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from models import Employee, User, Shift, Attendance

# Database file paths
DATA_FILE = 'data.json'
SHIFTS_FILE = 'shifts.json'
ATTENDANCE_FILE = 'attendance.json'

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file
    
    Returns:
        List of data
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_data(file_path: str, data: List[Dict[str, Any]]) -> None:
    """
    Save data to a JSON file
    
    Args:
        file_path: Path to save the data
        data: List of data to save
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Employee Functions
def get_all_employees() -> List[Dict[str, Any]]:
    """
    Get all employees from database
    
    Returns:
        List of all employees
    """
    return load_data(DATA_FILE)

def get_employee_by_id(employee_id: int) -> Dict[str, Any]:
    """
    Get employee by ID
    
    Args:
        employee_id: Employee ID
        
    Returns:
        Employee data or None if not found
    """
    employees = load_data(DATA_FILE)
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
    employees = load_data(DATA_FILE)
    
    # Generate new ID
    max_id = 0
    for employee in employees:
        if employee["id"] > max_id:
            max_id = employee["id"]
    
    employee_data["id"] = max_id + 1
    employees.append(employee_data)
    save_data(DATA_FILE, employees)
    
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
    employees = load_data(DATA_FILE)
    
    for i, employee in enumerate(employees):
        if employee["id"] == employee_id:
            # Preserve the ID
            employee_data["id"] = employee_id
            
            # If password is empty, keep the old one
            if "password" in employee_data and not employee_data["password"]:
                employee_data["password"] = employee["password"]
                
            employees[i] = employee_data
            save_data(DATA_FILE, employees)
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
    employees = load_data(DATA_FILE)
    
    for i, employee in enumerate(employees):
        if employee["id"] == employee_id:
            del employees[i]
            save_data(DATA_FILE, employees)
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
    employees = load_data(DATA_FILE)
    
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

# Shift Functions
def get_all_shifts() -> List[Dict[str, Any]]:
    """
    Get all shifts from database
    
    Returns:
        List of all shifts
    """
    return load_data(SHIFTS_FILE)

def get_shift_by_id(shift_id: int) -> Dict[str, Any]:
    """
    Get shift by ID
    
    Args:
        shift_id: Shift ID
        
    Returns:
        Shift data or None if not found
    """
    shifts = load_data(SHIFTS_FILE)
    for shift in shifts:
        if shift["id"] == shift_id:
            return shift
    return None

def add_shift(shift_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new shift to the database
    
    Args:
        shift_data: Shift data
        
    Returns:
        Added shift data with ID
    """
    shifts = load_data(SHIFTS_FILE)
    
    # Generate new ID
    max_id = 0
    for shift in shifts:
        if shift["id"] > max_id:
            max_id = shift["id"]
    
    shift_data["id"] = max_id + 1
    shifts.append(shift_data)
    save_data(SHIFTS_FILE, shifts)
    
    return shift_data

def update_shift(shift_id: int, shift_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update shift data
    
    Args:
        shift_id: Shift ID
        shift_data: New shift data
        
    Returns:
        Updated shift data or None if not found
    """
    shifts = load_data(SHIFTS_FILE)
    
    for i, shift in enumerate(shifts):
        if shift["id"] == shift_id:
            # Preserve the ID
            shift_data["id"] = shift_id
            shifts[i] = shift_data
            save_data(SHIFTS_FILE, shifts)
            return shift_data
            
    return None

def delete_shift(shift_id: int) -> bool:
    """
    Delete shift by ID
    
    Args:
        shift_id: Shift ID
        
    Returns:
        True if deleted, False if not found
    """
    shifts = load_data(SHIFTS_FILE)
    
    for i, shift in enumerate(shifts):
        if shift["id"] == shift_id:
            del shifts[i]
            save_data(SHIFTS_FILE, shifts)
            return True
            
    return False

# Attendance Functions
def get_all_attendance() -> List[Dict[str, Any]]:
    """
    Get all attendance records from database
    
    Returns:
        List of all attendance records
    """
    return load_data(ATTENDANCE_FILE)

def get_attendance_by_id(attendance_id: int) -> Dict[str, Any]:
    """
    Get attendance record by ID
    
    Args:
        attendance_id: Attendance ID
        
    Returns:
        Attendance data or None if not found
    """
    attendance_records = load_data(ATTENDANCE_FILE)
    for record in attendance_records:
        if record["id"] == attendance_id:
            return record
    return None

def add_attendance(attendance_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new attendance record to the database
    
    Args:
        attendance_data: Attendance data
        
    Returns:
        Added attendance data with ID
    """
    attendance_records = load_data(ATTENDANCE_FILE)
    
    # Generate new ID
    max_id = 0
    for record in attendance_records:
        if record["id"] > max_id:
            max_id = record["id"]
    
    attendance_data["id"] = max_id + 1
    
    # If check_in_time is not provided, use current time
    if "check_in_time" not in attendance_data or not attendance_data["check_in_time"]:
        attendance_data["check_in_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine status based on shift start time
    shift = get_shift_by_id(attendance_data["shift_id"])
    if shift:
        # Parse times
        check_in_datetime = datetime.strptime(attendance_data["check_in_time"], "%Y-%m-%d %H:%M:%S")
        check_in_date = check_in_datetime.date()
        check_in_time = check_in_datetime.time()
        shift_start = datetime.strptime(shift["start_time"], "%H:%M").time()
        
        # Create datetime objects for comparison
        shift_start_datetime = datetime.combine(check_in_date, shift_start)
        
        # Calculate late threshold (15 minutes after shift start)
        late_threshold = shift_start_datetime + timedelta(minutes=15)
        
        # Determine status
        if check_in_datetime <= shift_start_datetime:
            attendance_data["status"] = "on_time"
        elif check_in_datetime <= late_threshold:
            attendance_data["status"] = "late"
        else:
            attendance_data["status"] = "very_late"
    else:
        # Default to on_time if shift not found
        attendance_data["status"] = "on_time"
    
    attendance_records.append(attendance_data)
    save_data(ATTENDANCE_FILE, attendance_records)
    
    return attendance_data

def update_attendance(attendance_id: int, attendance_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update attendance record
    
    Args:
        attendance_id: Attendance ID
        attendance_data: New attendance data
        
    Returns:
        Updated attendance data or None if not found
    """
    attendance_records = load_data(ATTENDANCE_FILE)
    
    for i, record in enumerate(attendance_records):
        if record["id"] == attendance_id:
            # Preserve the ID
            attendance_data["id"] = attendance_id
            
            # Ensure essential fields are preserved
            if "employee_id" not in attendance_data and "employee_id" in record:
                attendance_data["employee_id"] = record["employee_id"]
                
            if "shift_id" not in attendance_data and "shift_id" in record:
                attendance_data["shift_id"] = record["shift_id"]
            
            attendance_records[i] = attendance_data
            save_data(ATTENDANCE_FILE, attendance_records)
            return attendance_data
            
    return None

def delete_attendance(attendance_id: int) -> bool:
    """
    Delete attendance record by ID
    
    Args:
        attendance_id: Attendance ID
        
    Returns:
        True if deleted, False if not found
    """
    attendance_records = load_data(ATTENDANCE_FILE)
    
    for i, record in enumerate(attendance_records):
        if record["id"] == attendance_id:
            del attendance_records[i]
            save_data(ATTENDANCE_FILE, attendance_records)
            return True
            
    return False

def filter_attendance(
    employee_id: Optional[int] = None,
    shift_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter attendance records
    
    Args:
        employee_id: Filter by employee ID
        shift_id: Filter by shift ID
        start_date: Filter by start date (YYYY-MM-DD)
        end_date: Filter by end date (YYYY-MM-DD)
        status: Filter by status
        
    Returns:
        Filtered attendance records
    """
    attendance_records = load_data(ATTENDANCE_FILE)
    filtered_records = []
    
    for record in attendance_records:
        # Apply filters
        if employee_id is not None and record["employee_id"] != employee_id:
            continue
            
        if shift_id is not None and record["shift_id"] != shift_id:
            continue
            
        if start_date is not None:
            record_date = record["check_in_time"].split(" ")[0]
            if record_date < start_date:
                continue
                
        if end_date is not None:
            record_date = record["check_in_time"].split(" ")[0]
            if record_date > end_date:
                continue
                
        if status is not None and record["status"] != status:
            continue
            
        filtered_records.append(record)
    
    return filtered_records

# Initialize with default data if files don't exist
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
    save_data(DATA_FILE, default_data)

if not os.path.exists(SHIFTS_FILE):
    default_shifts = [
        {
            "id": 1,
            "name": "Ca sáng",
            "start_time": "08:00",
            "end_time": "12:00"
        },
        {
            "id": 2,
            "name": "Ca chiều",
            "start_time": "13:00",
            "end_time": "17:00"
        },
        {
            "id": 3,
            "name": "Ca tối",
            "start_time": "18:00",
            "end_time": "22:00"
        }
    ]
    save_data(SHIFTS_FILE, default_shifts)

if not os.path.exists(ATTENDANCE_FILE):
    # Empty attendance records initially
    save_data(ATTENDANCE_FILE, []) 