from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List

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