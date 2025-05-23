from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import threading
import subprocess
import os

from models import Employee
import routes

# Create FastAPI app
app = FastAPI(title="Hệ thống điểm danh", description="API cho hệ thống điểm danh nhân viên")

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

@app.on_event("shutdown")
async def shutdown_event():
    print("Đang giải phóng camera và dọn dẹp...")
    release_camera()  # Đảm bảo rằng hàm release_camera() được định nghĩa ở đâu đó

# Root endpoint
@app.get("/")
async def read_root():
    return await routes.read_root()

# Camera endpoints
@app.get("/video_feed")
async def video_feed():
    return await routes.video_feed()

@app.get("/camera/info")
async def get_camera_info():
    return await routes.get_camera_info_endpoint()

@app.post("/camera/toggle")
async def toggle_camera():
    return await routes.toggle_camera()

@app.get("/camera/status")
async def get_camera_status():
    return await routes.get_camera_status()

@app.get("/scan_video_feed")
async def scan_video_feed():
    return await routes.scan_video_feed()

# Employee endpoints
@app.get("/employees", response_model=List[Employee])
async def get_employees():
    return await routes.get_employees()

@app.get("/employees/{employee_id}", response_model=Employee)
async def get_employee(employee_id: int):
    return await routes.get_employee(employee_id)

@app.post("/employees", response_model=Employee)
async def create_employee(employee: Employee):
    return await routes.create_employee(employee)

@app.put("/employees/{employee_id}", response_model=Employee)
async def update_employee(employee_id: int, employee: Employee):
    return await routes.update_employee(employee_id, employee)

@app.delete("/employees/{employee_id}")
async def delete_employee(employee_id: int):
    return await routes.delete_employee(employee_id)

# Face scanning endpoints
@app.post("/face-scan/start/{employee_id}")
async def start_face_scan(employee_id: str):
    return await routes.start_face_scan(employee_id)

@app.get("/face-scan/status")
async def get_face_scan_status():
    return await routes.get_face_scan_status()

@app.post("/face-scan/stop")
async def stop_face_scan():
    return await routes.stop_face_scan()

# Training endpoints
training_status = {"is_training": False}

@app.post("/train")
async def train_endpoint():
    if training_status["is_training"]:
        return {"status": "training"}
    training_status["is_training"] = True
    thread = threading.Thread(target=run_training)
    thread.start()
    return {"status": "started"}

@app.get("/train/status")
async def train_status():
    return training_status

def run_training():
    try:
        print(">>> Đang chạy training.py ...", flush=True)
        # Sử dụng "source venv/bin/activate && python training.py" với shell=True
        command = ". venv/bin/activate && python training.py"
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True
        )
        # In realtime output
        while True:
            print("ccacaacc")
            line = proc.stdout.readline()
            if not line:
                break
            print(line, flush=True)
        proc.wait()
        if proc.returncode != 0:
            for line in proc.stderr:
                print(line, flush=True)
        from camera import Camera
        Camera.reload_model()
        print(">>> Model đã được reload", flush=True)
    except Exception as e:
        print(f"Training error: {e}", flush=True)
    finally:
        training_status["is_training"] = False

