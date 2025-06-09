from fastapi import HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from typing import List, Optional
from datetime import datetime

from models import Employee
import database
import camera
from employeecontroller import EmployeeController
from attendance_controller import AttendanceController

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
    return await EmployeeController.get_all()

async def get_employee(employee_id: int):
    return await EmployeeController.get_by_id(employee_id)

async def create_employee(employee: Employee):
    return await EmployeeController.create(employee)

async def update_employee(employee_id: int, employee: Employee):
    return await EmployeeController.update(employee_id, employee)

async def delete_employee(employee_id: int):
    return await EmployeeController.delete(employee_id)

# Attendance endpoints
async def get_attendance(employee_id: Optional[str] = None, date: Optional[str] = None):
    """
    Get attendance records with optional filtering by employee and date
    
    Args:
        employee_id: Optional employee ID to filter records
        date: Optional date to filter records
        
    Returns:
        Dictionary containing attendance records
    """
    try:
        # Get attendance records
        result = await AttendanceController.get_employee_attendance(employee_id, date)
        
        if result["success"]:
            # If employee_id is provided, get employee name
            if employee_id:
                employee = await EmployeeController.get_by_id(int(employee_id))
                if employee:
                    for record in result["data"]:
                        record["employee_name"] = employee["full_name"]
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def record_attendance(employee_id: str, check_type: str):
    """
    Record attendance (check-in or check-out) for an employee
    
    Args:
        employee_id: Employee ID
        check_type: Type of check ('in' for check-in, 'out' for check-out)
        
    Returns:
        Dictionary containing attendance record status
    """
    try:
        if not employee_id or not check_type:
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        result = await AttendanceController.record_attendance(employee_id, check_type)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))