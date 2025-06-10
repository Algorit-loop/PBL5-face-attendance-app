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
from datetime import datetime
import uvicorn
import base64

from models import Employee, APIResponse
from controllers.employee_controller import EmployeeController
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
        session = login_required(request)
        # Ensure camera is initialized
        if not camera.camera_running:
            camera.init_camera()
        # Return scanning video stream
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
        session = login_required(request)
        # Ensure camera is initialized
        if not camera.camera_running:
            camera.init_camera()
            
        # Get current frame
        ret, frame = camera.camera.read()
        if not ret or frame is None:
            return {"success": False, "message": "Failed to capture frame"}
            
        # Process with YOLO and R50
        # Create YOLO input frame
        yolo_frame = cv2.resize(frame, (640, 640))
        
        # Run inference
        camera.yolo_r50_inference(frame, yolo_frame)
        
        # Check results
        if camera.global_result is None or len(camera.global_result) == 0:
            return {"success": False, "message": "No face detected"}
            
        # Get face with highest confidence
        best_face = None
        for face in camera.global_result:
            if face['id_user'] != "Unknown":
                # If we already have a match, just return it
                best_face = face
                break
                
        if best_face is None:
            return {"success": False, "message": "No recognized face"}
            
        # Get employee details
        employee_controller = EmployeeController()
        employee = await employee_controller.get_by_id(best_face['id_user'])
        
        if not employee:
            return {"success": False, "message": "Employee not found"}
            
        return {
            "success": True,
            "employee": {
                "id": employee["id"],
                "full_name": employee["full_name"],
                "department": employee["department"],
                "position": employee["position"]
            },
            "confidence": best_face['conf']
        }
    except Exception as e:
        print(f"Error verifying face: {str(e)}")
        return {"success": False, "message": str(e)}

# Mark attendance
@app.post("/api/attendance")
async def mark_attendance(request: Request, data: dict):
    try:
        session = login_required(request)
        employee_id = data.get("employee_id")
        status = data.get("status", "present")
        
        if not employee_id:
            return {"success": False, "message": "Employee ID is required"}
            
        # In a real app, this would save to a database
        # For now, just return success
        return {
            "success": True,
            "message": f"Attendance marked for employee {employee_id} as {status}"
        }
    except Exception as e:
        print(f"Error marking attendance: {str(e)}")
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

# def run_training():
#     try:
#         print(">>> Đang chạy training.py ...", flush=True)
#         # Sử dụng "source venv/bin/activate && python training.py" với shell=True
#         # command = ". venv/Scripts/activate && python training.py"
#         # command = "python training.py"
#         command = "venv\\Scripts\\Activate.ps1; python training.py"

#         proc = subprocess.Popen(
#             command,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             shell=True,
#             text=True
#         )
#         # In realtime output
#         while True:
#             print("ccacaacc")
#             line = proc.stdout.readline()
#             if not line:
#                 break
#             print(line, flush=True)
#         proc.wait()
#         if proc.returncode != 0:
#             for line in proc.stderr:
#                 print(line, flush=True)
#         from camera import Camera
#         Camera.reload_model()
#         print(">>> Model đã được reload", flush=True)
#     except Exception as e:
#         print(f"Training error: {e}", flush=True)
#     finally:
#         training_status["is_training"] = False
import sys, subprocess, textwrap

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

    # Đọc realtime
    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    if proc.returncode != 0:
        print(proc.stderr.read(), flush=True)

    from camera import Camera
    Camera.reload_model()
    print(">>> Model đã được reload", flush=True)

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

# Import OpenCV here to avoid circular imports
import cv2

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

