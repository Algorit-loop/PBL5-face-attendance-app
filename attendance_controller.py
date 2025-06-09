import asyncio
from datetime import datetime
import json
import os
from employeecontroller import EmployeeController

class AttendanceController:
    ATTENDANCE_FILE = "attendance_data.json"
    
    @classmethod
    async def record_attendance(cls, employee_id: str, check_type: str) -> dict:
        """
        Record attendance (check-in or check-out) for an employee
        check_type: 'in' for check-in, 'out' for check-out
        """
        try:
            # Load existing attendance data
            attendance_data = cls._load_attendance_data()
            
            # Get current date and time
            current_time = datetime.now()
            date_str = current_time.strftime("%Y-%m-%d")
            time_str = current_time.strftime("%H:%M:%S")
            
            # Initialize employee's attendance record if not exists
            if employee_id not in attendance_data:
                attendance_data[employee_id] = {}
            
            # Initialize date record if not exists
            if date_str not in attendance_data[employee_id]:
                attendance_data[employee_id][date_str] = {
                    "check_in": None,
                    "check_out": None
                }
            
            # Record check-in/check-out
            if check_type == "in":
                attendance_data[employee_id][date_str]["check_in"] = time_str
            elif check_type == "out":
                attendance_data[employee_id][date_str]["check_out"] = time_str
            
            # Save updated data
            cls._save_attendance_data(attendance_data)
            
            return {
                "success": True,
                "message": f"Check-{check_type} recorded successfully",
                "time": time_str
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error recording attendance: {str(e)}"
            }
    
    @classmethod
    async def get_employee_attendance(cls, employee_id: str = None, date: str = None) -> dict:
        """
        Get attendance records for an employee or all employees
        If date is provided, returns attendance for that specific date
        Otherwise returns all attendance records
        Luôn trả về đủ danh sách nhân viên cho mỗi ngày (kể cả người không có bản ghi)
        """
        try:
            attendance_data = cls._load_attendance_data()
            employees = await EmployeeController.get_all()
            emp_map = {str(emp["id"]): emp["full_name"] for emp in employees}
            emp_ids = [str(emp["id"]) for emp in employees]

            records = []
            if date:
                # Trả về đủ danh sách nhân viên cho ngày đó
                for emp_id in emp_ids:
                    rec = attendance_data.get(emp_id, {}).get(date, None)
                    records.append({
                        "date": date,
                        "employee_id": emp_id,
                        "employee_name": emp_map.get(emp_id, "Unknown"),
                        "check_in": rec["check_in"] if rec else None,
                        "check_out": rec["check_out"] if rec else None
                    })
            elif employee_id:
                # Chỉ lấy cho 1 nhân viên
                if employee_id not in attendance_data:
                    return {"success": True, "data": []}
                for d, rec in attendance_data[employee_id].items():
                    records.append({
                        "date": d,
                        "employee_id": employee_id,
                        "employee_name": emp_map.get(str(employee_id), "Unknown"),
                        "check_in": rec["check_in"],
                        "check_out": rec["check_out"]
                    })
            else:
                # Lấy cho tất cả nhân viên, tất cả ngày (tổng hợp)
                all_dates = set()
                for days in attendance_data.values():
                    all_dates.update(days.keys())
                for d in all_dates:
                    for emp_id in emp_ids:
                        rec = attendance_data.get(emp_id, {}).get(d, None)
                        records.append({
                            "date": d,
                            "employee_id": emp_id,
                            "employee_name": emp_map.get(emp_id, "Unknown"),
                            "check_in": rec["check_in"] if rec else None,
                            "check_out": rec["check_out"] if rec else None
                        })
            # Sắp xếp theo ngày mới nhất, sau đó theo tên nhân viên
            records = sorted(records, key=lambda x: (x["date"], x["employee_name"]), reverse=True)
            return {"success": True, "data": records}
        except Exception as e:
            return {"success": False, "message": f"Error getting attendance: {str(e)}"}
    
    @classmethod
    def _load_attendance_data(cls) -> dict:
        """Load attendance data from JSON file"""
        if os.path.exists(cls.ATTENDANCE_FILE):
            with open(cls.ATTENDANCE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def _save_attendance_data(cls, data: dict) -> None:
        """Save attendance data to JSON file"""
        with open(cls.ATTENDANCE_FILE, 'w') as f:
            json.dump(data, f, indent=4) 