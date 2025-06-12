from fastapi import FastAPI, Request, Response, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from typing import List, Dict, Any, Optional
import threading
import subprocess
import os
import json
import time
import cv2
from datetime import datetime
import uvicorn
import base64
import sys

from models import Employee, APIResponse, Shift, ShiftCreate, ShiftUpdate, Attendance, AttendanceCreate, AttendanceFilter
from controllers.employee_controller import EmployeeController
from controllers.shift_controller import ShiftController
from controllers.attendance_controller import AttendanceController
import database
import camera

# Create FastAPI app
app = FastAPI(title="Hệ thống điểm danh", description="API cho hệ thống điểm danh nhân viên")

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your_secret_key_here")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="static/pages")

# Load users from data.json
def load_users():
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
        return []

# Authentication middleware
def login_required(request: Request):
    if not request.session.get("logged_in"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return request.session

def admin_required(request: Request):
    session = login_required(request)
    if not session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return session

# Root endpoint - redirect to login if not logged in, otherwise to dashboard
@app.get("/")
async def root(request: Request):
    if not request.session.get("logged_in"):
        return RedirectResponse(url="/login")
    
    if request.session.get("is_admin"):
        return RedirectResponse(url="/dashboard")
    else:
        return RedirectResponse(url="/user_dashboard")

# Login page
@app.get("/login")
async def login_page(request: Request):
    if request.session.get("logged_in"):
        if request.session.get("is_admin"):
            return RedirectResponse(url="/dashboard")
        else:
            return RedirectResponse(url="/user_dashboard")
    return templates.TemplateResponse("login.html", {"request": request})

# Login endpoint
@app.post("/api/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    users = load_users()
    user = next((u for u in users if u["username"] == username and u["password"] == password), None)
    
    if user:
        # Set session data
        request.session["logged_in"] = True
        request.session["user_id"] = user["id"]
        request.session["username"] = user["username"]
        request.session["full_name"] = user["full_name"]
        request.session["is_admin"] = user["position"] == "admin"
        
        # Remove password from user object
        user_copy = dict(user)
        user_copy.pop("password", None)
        
        return {"success": True, "user": user_copy}
    else:
        return {"success": False, "message": "Tên đăng nhập hoặc mật khẩu không đúng"}

# Logout endpoint
@app.post("/api/logout")
async def logout(request: Request):
    request.session.clear()
    return {"success": True}

# Dashboard page
@app.get("/dashboard")
async def dashboard_page(request: Request):
    try:
        session = admin_required(request)
        return RedirectResponse(url="/static/pages/dashboard.html")
    except HTTPException:
        return RedirectResponse(url="/login")

# User dashboard page
@app.get("/user_dashboard")
async def user_dashboard_page(request: Request):
    try:
        session = login_required(request)
        return RedirectResponse(url="/static/pages/user_dashboard.html")
    except HTTPException:
        return RedirectResponse(url="/login")

# Get current user info
@app.get("/api/user")
async def get_user(request: Request):
    try:
        session = login_required(request)
        users = load_users()
        user = next((u for u in users if u["id"] == session.get("user_id")), None)
        
        if user:
            user_copy = dict(user)
            user_copy.pop("password", None)
            return user_copy
        else:
            return HTTPException(status_code=404, detail="User not found")
    except HTTPException as e:
        return e

# Get all employees (admin only)
@app.get("/api/employees")
async def get_employees(request: Request):
    try:
        session = admin_required(request)
        # Get employees using controller
        employee_controller = EmployeeController()
        employees = await employee_controller.get_all()
        
        # Remove passwords for security
        for employee in employees:
            if "password" in employee:
                employee.pop("password", None)
                
        return employees
    except HTTPException as e:
        return e

# Create new employee
@app.post("/api/employees")
async def create_employee(request: Request, employee: Employee):
    try:
        session = admin_required(request)
        # Create employee using controller
        employee_controller = EmployeeController()
        created_employee = await employee_controller.create(employee)
        
        # Start training the model with the new face data
        threading.Thread(target=run_training).start()
        
        return JSONResponse(content={
            "success": True,
            "message": "Employee created successfully",
            "employee": created_employee
        })
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Get employee by ID
@app.get("/api/employees/{employee_id}")
async def get_employee(request: Request, employee_id: int):
    try:
        session = login_required(request)
        
        # Get employee using controller
        employee_controller = EmployeeController()
        employee = await employee_controller.get_by_id(employee_id)
        
        # Check if requesting own data or admin
        if session.get("user_id") != employee_id and not session.get("is_admin"):
            return HTTPException(status_code=403, detail="Not authorized to access this employee data")
        
        # Remove password for security
        if "password" in employee:
            employee.pop("password", None)
            
        return employee
    except HTTPException as e:
        return e

# Update employee
@app.put("/api/employees/{employee_id}")
async def update_employee(request: Request, employee_id: int, employee: Employee):
    try:
        session = admin_required(request)
        # Update employee using controller
        employee_controller = EmployeeController()
        updated_employee = await employee_controller.update(employee_id, employee)
        
        return JSONResponse(content={
            "success": True,
            "message": "Employee updated successfully",
            "employee": updated_employee
        })
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Delete employee
@app.delete("/api/employees/{employee_id}")
async def delete_employee(request: Request, employee_id: int):
    try:
        session = admin_required(request)
        # Delete employee using controller
        employee_controller = EmployeeController()
        result = await employee_controller.delete(employee_id)
        
        # Start training the model after deleting an employee
        threading.Thread(target=run_training).start()
        
        return JSONResponse(content={
            "success": True,
            "message": "Employee deleted successfully"
        })
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Get dashboard data
@app.get("/api/dashboard")
async def get_dashboard_data(request: Request):
    try:
        session = admin_required(request)
        
        # Mock data - in a real app this would come from a database
        return {
            "total_employees": 12,
            "present_today": 8,
            "late_today": 2,
            "absent_today": 2,
            "recent_activities": [
                {
                    "type": "Check-in",
                    "description": "Employee ID 101 checked in",
                    "time": "09:00 AM"
                },
                {
                    "type": "Check-out",
                    "description": "Employee ID 103 checked out",
                    "time": "05:30 PM"
                },
                {
                    "type": "New Employee",
                    "description": "Added new employee: John Doe",
                    "time": "02:15 PM"
                }
            ]
        }
    except HTTPException as e:
        return e

# Video feed endpoint
@app.get("/video_feed")
async def video_feed(request: Request):
    try:
        session = login_required(request)
        # Ensure camera is initialized
        if not camera.camera_running:
            camera.init_camera()
        # Return video stream
        return StreamingResponse(
            camera.generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except HTTPException as e:
        return e

# Face scan video feed
@app.get("/scan_feed")
async def scan_feed(request: Request):
    try:
        session = admin_required(request)
        # Initialize camera if not running
        if not camera.camera_running:
            camera.init_camera()
        # Use scan_frames() for face scanning mode
        return StreamingResponse(
            camera.scan_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except HTTPException as e:
        return e

# Start face scanning
@app.post("/api/face-scan/start/{employee_id}")
async def start_face_scan(request: Request, employee_id: int):
    try:
        session = admin_required(request)
        camera.face_scanner.start_scan(employee_id)
        return {"success": True}
    except HTTPException as e:
        return e

# Start face capture (after scanning has begun)
@app.post("/api/face-scan/start-capture")
async def start_face_capture(request: Request):
    try:
        session = admin_required(request)
        if camera.face_scanner.start_capture():
            return {"success": True}
        else:
            return {"success": False, "message": "Cannot start capture - scan not active"}
    except HTTPException as e:
        return e

# Get face scan status
@app.get("/api/face-scan/status")
async def face_scan_status(request: Request):
    try:
        session = login_required(request)
        return camera.face_scanner.get_status()
    except HTTPException as e:
        return e

# Capture a face
@app.post("/api/capture")
async def capture_face(request: Request):
    try:
        session = login_required(request)
        # Ensure camera is initialized
        if not camera.camera_running:
            camera.init_camera()
            
        # Get current frame
        ret, frame = camera.camera.read()
        if not ret or frame is None:
            return {"success": False, "message": "Failed to capture frame"}
            
        # Run face detection using YOLO
        results = camera.model(frame, verbose=False)[0]
        if len(results.boxes) == 0:
            return {"success": False, "message": "No face detected"}
            
        # Get the largest face (closest to camera)
        largest_face = None
        max_area = 0
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_face = (x1, y1, x2, y2)
                
        if largest_face is None:
            return {"success": False, "message": "No valid face detected"}
            
        # Extract and encode the face image
        x1, y1, x2, y2 = largest_face
        face_img = frame[y1:y2, x1:x2]
        _, img_encoded = cv2.imencode('.jpg', face_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        return {
            "success": True,
            "image": img_base64,
            "bbox": largest_face
        }
    except Exception as e:
        print(f"Error capturing face: {str(e)}")
        return {"success": False, "message": str(e)}

# Verify face
@app.post("/api/verify")
async def verify_face(request: Request):
    try:
        session = admin_required(request)
        # Ensure camera is initialized
        if not camera.camera_running:
            print("Camera not running, initializing...")
            if not camera.init_camera():
                return {"success": False, "message": "Failed to initialize camera"}
        
        # Get the current recognition status
        if camera.recognized_employee and camera.recognition_count >= 3:
            # Employee has been recognized in 3 consecutive frames
            employee_id = camera.recognized_employee["id"]
            print(f"Recognized employee ID: {employee_id}, type: {type(employee_id)}")
            
            # Ensure employee_id is an integer for database lookup
            try:
                employee_id = int(employee_id)
            except (ValueError, TypeError):
                print(f"Error converting employee_id to int: {employee_id}")
                return {"success": False, "message": f"Invalid employee ID format: {employee_id}"}
            
            # Get employee details using the controller
            employee_controller = EmployeeController()
            employee = await employee_controller.get_by_id(employee_id)
            
            if not employee:
                print(f"Employee with ID {employee_id} not found in database")
                # Reset recognition because we can't verify this employee
                camera.reset_recognition()
                return {
                    "success": False, 
                    "message": f"Employee with ID {employee_id} not found in database",
                    "recognition_count": 0
                }
            
            print(f"Found employee: {employee['full_name']}")
            return {
                "success": True,
                "employee": {
                    "id": employee["id"],
                    "full_name": employee["full_name"],
                    "department": employee["department"],
                    "position": employee["position"]
                },
                "verified": True
            }
        elif camera.recognized_employee and camera.recognition_count > 0:
            # Employee is being recognized but not yet verified
            return {
                "success": False, 
                "message": "Waiting for consistent recognition",
                "recognition_count": int(camera.recognition_count),
                "employee_id": camera.recognized_employee["id"]
            }
        else:
            # No recognition yet
            return {
                "success": False, 
                "message": "Waiting for face detection",
                "recognition_count": 0
            }
    except Exception as e:
        print(f"Error verifying face: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e)}

# Mark attendance
@app.post("/api/attendance")
async def mark_attendance(request: Request, data: dict):
    try:
        session = login_required(request)
        employee_id = data.get("employee_id")
        shift_id = data.get("shift_id", 1)  # Default to first shift if not specified
        image_data = data.get("image_data")  # Base64 encoded image
        
        if not employee_id:
            return {"success": False, "message": "Employee ID is required"}
        
        # Create attendance record
        attendance_data = AttendanceCreate(
            employee_id=employee_id,
            shift_id=shift_id
        )
        
        attendance_controller = AttendanceController()
        attendance = await attendance_controller.create(attendance_data)
        
        # Save image if available
        if image_data and attendance.get("id"):
            try:
                # Ensure directory exists
                img_dir = "img_historyreport"
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                    print(f"Created directory: {img_dir}")
                
                # Save image
                attendance_id = attendance.get("id")
                image_path = os.path.join(img_dir, f"imgreport_{attendance_id}.jpg")
                
                # Decode base64 image
                try:
                    image_bytes = base64.b64decode(image_data)
                    
                    # Write image to file
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    print(f"Saved attendance image to {image_path}, size: {len(image_bytes)} bytes")
                    attendance["image_path"] = image_path
                except Exception as decode_error:
                    print(f"Error decoding image: {str(decode_error)}")
                    print(f"Image data length: {len(image_data) if image_data else 'None'}")
            except Exception as img_error:
                print(f"Error saving attendance image: {str(img_error)}")
                import traceback
                traceback.print_exc()
        else:
            print("No image data provided or attendance ID not available")
            if not image_data:
                print("Image data is missing")
            if not attendance.get("id"):
                print("Attendance ID is missing")
        
        # Reset recognition state after marking attendance
        camera.reset_recognition()
            
        return {
            "success": True,
            "message": f"Attendance marked for employee {employee_id}",
            "attendance": attendance
        }
    except Exception as e:
        print(f"Error marking attendance: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e)}

# Camera status
@app.get("/api/camera/status")
async def camera_status(request: Request):
    try:
        session = login_required(request)
        return {"is_running": camera.camera_running}
    except HTTPException as e:
        return e

# Close camera
@app.post("/api/camera/close")
async def close_camera(request: Request):
    try:
        session = login_required(request)
        if camera.camera_running:
            camera.release_camera()
            # Also stop face scanning if it's active
            camera.face_scanner.stop()
        return {"success": True, "status": "closed"}
    except HTTPException as e:
        return e

# Toggle camera
@app.post("/api/camera/toggle")
async def toggle_camera(request: Request):
    try:
        session = admin_required(request)
        if camera.camera_running:
            camera.release_camera()
            return {"status": "off"}
        else:
            if camera.init_camera():
                return {"status": "on"}
            else:
                raise HTTPException(status_code=500, detail="Failed to initialize camera")
    except HTTPException as e:
        return e

# Camera info
@app.get("/api/camera/info")
async def camera_info(request: Request):
    try:
        session = login_required(request)
        return camera.get_camera_info()
    except HTTPException as e:
        return e

# Reset face recognition
@app.post("/api/reset-recognition")
async def reset_recognition(request: Request):
    try:
        session = login_required(request)
        # Reset recognition tracking
        camera.reset_recognition()
        return {"success": True, "message": "Recognition reset successfully"}
    except HTTPException as e:
        return e

@app.post("/api/train")
async def train_model(request: Request):
    try:
        session = admin_required(request)
        if training_status["is_training"]:
            return {"status": "training"}
        training_status["is_training"] = True
        thread = threading.Thread(target=run_training)
        thread.start()
        return {"status": "started"}
    except HTTPException as e:
        return e

# Training status
training_status = {
    "is_training": False,
    "progress": 0,
    "message": ""
}

@app.get("/api/train/status")
async def train_status():
    return training_status

def run_training():
        print(">>> Đang chạy training.py ...", flush=True)

    # Dùng chính interpreter của chương trình cha (đang ở trong venv)
        cmd = [sys.executable, "training.py"]

    # KHÔNG dùng shell=True để tránh lỗi parse lệnh Windows
        proc = subprocess.Popen(
        cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # In realtime output
        while True:
            line = proc.stdout.readline()
            if not line:
                break
        print(line.rstrip(), flush=True)

    # Chờ tiến trình kết thúc và kiểm tra lỗi
        proc.wait()
        if proc.returncode != 0:
            print(">>> Lỗi khi chạy training.py:", flush=True)
            for line in proc.stderr:
                print(line.rstrip(), flush=True)
        else:
            print(">>> Training hoàn thành thành công", flush=True)

    # Reload model sau khi training
        from camera import Camera
        Camera.reload_model()
        print(">>> Model đã được reload", flush=True)
    
    # Reset training status
        training_status["is_training"] = False

# Page routes
@app.get("/verify")
async def verify_page(request: Request):
    try:
        session = login_required(request)
        return RedirectResponse(url="/static/pages/verify.html")
    except HTTPException:
        return RedirectResponse(url="/login")

@app.get("/employees")
async def employees_page(request: Request):
    try:
        session = admin_required(request)
        return RedirectResponse(url="/static/pages/employees.html")
    except HTTPException:
        return RedirectResponse(url="/login")

@app.get("/add_employee")
async def add_employee_page(request: Request):
    try:
        session = admin_required(request)
        return RedirectResponse(url="/static/pages/add_employee.html")
    except HTTPException:
        return RedirectResponse(url="/login")

# Shift API endpoints
@app.get("/api/shifts")
async def get_shifts(request: Request):
    try:
        session = login_required(request)
        shift_controller = ShiftController()
        shifts = await shift_controller.get_all()
        return shifts
    except HTTPException as e:
        return e

@app.get("/api/shifts/{shift_id}")
async def get_shift(request: Request, shift_id: int):
    try:
        session = login_required(request)
        shift_controller = ShiftController()
        shift = await shift_controller.get_by_id(shift_id)
        if not shift:
            return HTTPException(status_code=404, detail="Shift not found")
        return shift
    except HTTPException as e:
        return e

@app.post("/api/shifts")
async def create_shift(request: Request, shift: ShiftCreate):
    try:
        session = admin_required(request)
        shift_controller = ShiftController()
        created_shift = await shift_controller.create(shift)
        return JSONResponse(content={
            "success": True,
            "message": "Shift created successfully",
            "shift": created_shift
        })
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@app.put("/api/shifts/{shift_id}")
async def update_shift(request: Request, shift_id: int, shift: ShiftUpdate):
    try:
        session = admin_required(request)
        shift_controller = ShiftController()
        updated_shift = await shift_controller.update(shift_id, shift)
        return JSONResponse(content={
            "success": True,
            "message": "Shift updated successfully",
            "shift": updated_shift
        })
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@app.delete("/api/shifts/{shift_id}")
async def delete_shift(request: Request, shift_id: int):
    try:
        session = admin_required(request)
        shift_controller = ShiftController()
        result = await shift_controller.delete(shift_id)
        return result
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Attendance API endpoints
@app.post("/api/attendance/record")
async def record_attendance(request: Request, attendance: AttendanceCreate):
    try:
        session = login_required(request)
        attendance_controller = AttendanceController()
        recorded_attendance = await attendance_controller.create(attendance)
        return JSONResponse(content={
            "success": True,
            "message": "Attendance recorded successfully",
            "attendance": recorded_attendance
        })
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@app.post("/api/attendance/filter")
async def filter_attendance(request: Request, filter_params: AttendanceFilter):
    try:
        session = login_required(request)
        attendance_controller = AttendanceController()
        attendance_records = await attendance_controller.filter(filter_params)
        return attendance_records
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@app.get("/api/attendance/statistics")
async def get_attendance_statistics(
    request: Request, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    department: Optional[str] = None
):
    try:
        session = login_required(request)
        attendance_controller = AttendanceController()
        statistics = await attendance_controller.get_statistics(start_date, end_date, department)
        return statistics
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Page routes for new features
@app.get("/shifts")
async def shifts_page(request: Request):
    try:
        session = admin_required(request)
        return RedirectResponse(url="/static/pages/shifts.html")
    except HTTPException:
        return RedirectResponse(url="/login")

@app.get("/reports")
async def reports_page(request: Request):
    try:
        session = login_required(request)
        return RedirectResponse(url="/static/pages/reports.html")
    except HTTPException:
        return RedirectResponse(url="/login")

@app.post("/api/check-directory")
async def check_directory(request: Request, data: dict):
    try:
        session = login_required(request)
        directory = data.get("directory")
        
        if not directory:
            return {"success": False, "message": "Directory name is required"}
        
        # Check if directory exists
        if not os.path.exists(directory):
            # Create directory
            os.makedirs(directory)
            return {"success": True, "message": f"Directory {directory} created successfully", "created": True}
        
        return {"success": True, "message": f"Directory {directory} already exists", "created": False}
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "message": e.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

