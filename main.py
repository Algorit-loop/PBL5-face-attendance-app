from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import uvicorn
from typing import List, Optional
from pydantic import BaseModel, EmailStr, validator
from datetime import date
import json
import os
import time
import numpy as np
import re

app = FastAPI()

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

# Camera setup
camera = None
camera_running = False
last_frame_time = 0
frame_interval = 1/30  # 30 FPS

def init_camera():
    global camera, camera_running
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise Exception("Không thể mở camera")
            camera_running = True
            return True
    except Exception as e:
        print(f"Lỗi khi khởi tạo camera: {str(e)}")
        camera_running = False
        return False
    return True

def release_camera():
    global camera, camera_running
    try:
        if camera is not None:
            camera.release()
            camera = None
        camera_running = False
    except Exception as e:
        print(f"Lỗi khi giải phóng camera: {str(e)}")
        camera_running = False

# Get camera properties
def get_camera_info():
    if camera is None or not camera.isOpened():
        return {
            "width": 0,
            "height": 0,
            "fps": 0,
            "error": "Camera chưa được khởi tạo"
        }
    try:
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = camera.get(cv2.CAP_PROP_FPS)
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "error": None
        }
    except Exception as e:
        return {
            "width": 0,
            "height": 0,
            "fps": 0,
            "error": str(e)
        }

def generate_frames():
    global last_frame_time
    while True:
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            if camera_running and camera is not None and camera.isOpened():
                try:
                    success, frame = camera.read()
                    if not success:
                        # Create a black frame if camera read fails
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        # Resize frame to a reasonable size
                        frame = cv2.resize(frame, (640, 480))
                        # Flip frame horizontally
                        frame = cv2.flip(frame, 1) # lat anh cho de nhin ne, ok valcut 
                except Exception as e:
                    print(f"Lỗi khi đọc frame: {str(e)}")
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Create a black frame when camera is off
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            last_frame_time = current_time
        else:
            time.sleep(0.001)  # Small delay to prevent CPU overuse

# Employee model
class Employee(BaseModel):
    id: Optional[int] = None
    full_name: str
    birth_date: str
    email: str
    phone: str
    address: str
    gender: str
    position: str

# Database simulation with file storage
EMPLOYEES_FILE = 'employees.json'

def load_employees():
    if os.path.exists(EMPLOYEES_FILE):
        with open(EMPLOYEES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_employees(employees_data):
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

# API endpoints
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/camera/info")
async def get_camera_info_endpoint():
    return get_camera_info()

@app.post("/camera/toggle")
async def toggle_camera():
    global camera_running
    if camera_running:
        release_camera()
    else:
        if not init_camera():
            return {"status": "error", "message": "Không thể khởi tạo camera"}
    return {"status": "on" if camera_running else "off"}

@app.get("/camera/status")
async def get_camera_status():
    return {"status": "on" if camera_running else "off"}

@app.get("/employees", response_model=List[Employee])
async def get_employees():
    return employees

@app.get("/employees/{employee_id}", response_model=Employee)
async def get_employee(employee_id: int):
    employee = next((emp for emp in employees if emp["id"] == employee_id), None)
    if not employee:
        raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
    return employee

@app.post("/employees", response_model=Employee)
async def create_employee(employee: Employee):
    try:
        # Check for duplicate email
        if any(emp["email"] == employee.email for emp in employees):
            raise HTTPException(
                status_code=400,
                detail="Email đã tồn tại trong hệ thống"
            )
            
        # Check for duplicate phone
        if any(emp["phone"] == employee.phone for emp in employees):
            raise HTTPException(
                status_code=400,
                detail="Số điện thoại đã tồn tại trong hệ thống"
            )
            
        new_id = max(emp["id"] for emp in employees) + 1 if employees else 1
        employee_dict = employee.dict()
        employee_dict["id"] = new_id
        employees.append(employee_dict)
        save_employees(employees)
        return employee_dict
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/employees/{employee_id}", response_model=Employee)
async def update_employee(employee_id: int, employee: Employee):
    try:
        existing_employee = next((i for i, emp in enumerate(employees) if emp["id"] == employee_id), None)
        if existing_employee is None:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy nhân viên"
            )
        
        # Check for duplicate email, excluding current employee
        if any(emp["email"] == employee.email and emp["id"] != employee_id for emp in employees):
            raise HTTPException(
                status_code=400,
                detail="Email đã tồn tại trong hệ thống"
            )
            
        # Check for duplicate phone, excluding current employee
        if any(emp["phone"] == employee.phone and emp["id"] != employee_id for emp in employees):
            raise HTTPException(
                status_code=400,
                detail="Số điện thoại đã tồn tại trong hệ thống"
            )
            
        employee_dict = employee.dict()
        employee_dict["id"] = employee_id
        employees[existing_employee] = employee_dict
        save_employees(employees)
        return employee_dict
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/employees/{employee_id}")
async def delete_employee(employee_id: int):
    employee_index = next((i for i, emp in enumerate(employees) if emp["id"] == employee_id), None)
    if employee_index is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
    
    employees.pop(employee_index)
    save_employees(employees)
    return {"message": "Đã xóa nhân viên thành công"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
