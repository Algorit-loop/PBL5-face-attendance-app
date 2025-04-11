from fastapi import HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from typing import List

from models import Employee
import database
import camera

# Root endpoint
async def read_root():
    """
    Serve the main HTML page
    
    Returns:
        HTML file response
    """
    return FileResponse('static/index.html')

# Camera endpoints
async def video_feed():
    """
    Stream video from the camera
    
    Returns:
        Streaming response with camera frames
    """
    return StreamingResponse(camera.generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

async def scan_video_feed():
    """
    Stream video specifically for face scanning during registration
    
    Returns:
        Streaming response with camera frames optimized for face scanning
    """
    return StreamingResponse(camera.scan_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

async def get_camera_info_endpoint():
    """
    Get camera information
    
    Returns:
        Dictionary with camera properties
    """
    return camera.get_camera_info()

async def toggle_camera():
    """
    Toggle camera on/off
    
    Returns:
        Dictionary with camera status
    """
    if camera.camera_running:
        camera.release_camera()
    else:
        if not camera.init_camera():
            return {"status": "error", "message": "Không thể khởi tạo camera"}
    return {"status": "on" if camera.camera_running else "off"}

async def get_camera_status():
    """
    Get current camera status
    
    Returns:
        Dictionary with camera status
    """
    return {"status": "on" if camera.camera_running else "off"}

# Face scanning endpoints
async def start_face_scan(employee_id: str):
    """
    Start face scanning process for a new employee
    """
    camera.face_scanner.start_scan(employee_id)
    return {"status": "success", "message": "Face scanning started"}

async def get_face_scan_status():
    """
    Get current face scanning status
    """
    return camera.face_scanner.get_status()

async def stop_face_scan():
    """
    Stop face scanning process
    """
    camera.face_scanner.scanning = False
    return {"status": "success", "message": "Face scanning stopped"}

# Employee endpoints
async def get_employees():
    """
    Get all employees
    
    Returns:
        List of all employees
    """
    return database.employees

async def get_employee(employee_id: int):
    """
    Get employee by ID
    
    Args:
        employee_id: ID of the employee to get
        
    Returns:
        Employee data
        
    Raises:
        HTTPException: If employee not found
    """
    employee = next((emp for emp in database.employees if emp["id"] == employee_id), None)
    if not employee:
        raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
    return employee

async def create_employee(employee: Employee):
    """
    Create a new employee
    
    Args:
        employee: Employee data to create
        
    Returns:
        Created employee data
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Check for duplicate email
        if any(emp["email"] == employee.email for emp in database.employees):
            raise HTTPException(
                status_code=400,
                detail="Email đã tồn tại trong hệ thống"
            )
            
        # Check for duplicate phone
        if any(emp["phone"] == employee.phone for emp in database.employees):
            raise HTTPException(
                status_code=400,
                detail="Số điện thoại đã tồn tại trong hệ thống"
            )
            
        new_id = max(emp["id"] for emp in database.employees) + 1 if database.employees else 1
        employee_dict = employee.dict()
        employee_dict["id"] = new_id
        database.employees.append(employee_dict)
        database.save_employees(database.employees)
        return employee_dict
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def update_employee(employee_id: int, employee: Employee):
    """
    Update an existing employee
    
    Args:
        employee_id: ID of the employee to update
        employee: Updated employee data
        
    Returns:
        Updated employee data
        
    Raises:
        HTTPException: If employee not found or validation fails
    """
    try:
        existing_employee = next((i for i, emp in enumerate(database.employees) if emp["id"] == employee_id), None)
        if existing_employee is None:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy nhân viên"
            )
        
        # Check for duplicate email, excluding current employee
        if any(emp["email"] == employee.email and emp["id"] != employee_id for emp in database.employees):
            raise HTTPException(
                status_code=400,
                detail="Email đã tồn tại trong hệ thống"
            )
            
        # Check for duplicate phone, excluding current employee
        if any(emp["phone"] == employee.phone and emp["id"] != employee_id for emp in database.employees):
            raise HTTPException(
                status_code=400,
                detail="Số điện thoại đã tồn tại trong hệ thống"
            )
            
        employee_dict = employee.dict()
        employee_dict["id"] = employee_id
        database.employees[existing_employee] = employee_dict
        database.save_employees(database.employees)
        return employee_dict
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def delete_employee(employee_id: int):
    """
    Delete an employee
    
    Args:
        employee_id: ID of the employee to delete
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If employee not found
    """
    employee_index = next((i for i, emp in enumerate(database.employees) if emp["id"] == employee_id), None)
    if employee_index is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
    
    database.employees.pop(employee_index)
    database.save_employees(database.employees)
    return {"message": "Đã xóa nhân viên thành công"}