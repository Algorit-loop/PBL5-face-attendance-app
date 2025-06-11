import asyncio
from datetime import datetime, date, timedelta
import json
import os
from employeecontroller import EmployeeController
import calendar
from typing import Optional

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
    async def get_employee_attendance(cls, employee_id: Optional[str] = None, date: Optional[str] = None) -> dict:
        """
        Get attendance records for an employee or all employees, optionally filtered by date or month.
        If date is provided, returns attendance for that specific date or month.
        """
        try:
            print(f"Backend: Received employee_id={employee_id}, date={date}") # Debug print
            attendance_data = cls._load_attendance_data()
            employees = await EmployeeController.get_all()
            emp_map = {str(emp["id"]): emp["full_name"] for emp in employees}
            emp_ids = [str(emp["id"]) for emp in employees]

            # Validate employee_id if provided
            if employee_id and str(employee_id) not in emp_map:
                return {
                    "success": False,
                    "message": f"Không tìm thấy nhân viên với ID {employee_id}"
                }

            # Validate date format if provided
            if date:
                if len(date) not in [7, 10]:  # YYYY-MM or YYYY-MM-DD
                    return {
                        "success": False,
                        "message": "Định dạng ngày không hợp lệ. Sử dụng YYYY-MM hoặc YYYY-MM-DD"
                    }
                try:
                    if len(date) == 7:  # YYYY-MM
                        year, month = map(int, date.split('-'))
                        if not (1 <= month <= 12):
                            raise ValueError("Tháng không hợp lệ")
                    else:  # YYYY-MM-DD
                        datetime.strptime(date, "%Y-%m-%d")
                except ValueError as e:
                    return {
                        "success": False,
                        "message": f"Ngày không hợp lệ: {str(e)}"
                    }

            records = []

            if employee_id and date and len(date) == 7: # YYYY-MM format for a specific employee
                year, month_num = map(int, date.split('-'))
                num_days = calendar.monthrange(year, month_num)[1]
                start_date = datetime(year, month_num, 1).date()

                for i in range(num_days):
                    current_date = start_date + timedelta(days=i)
                    d_str = current_date.strftime("%Y-%m-%d")
                    rec = attendance_data.get(str(employee_id), {}).get(d_str, None)
                    print(f"Backend: Processing date {d_str}, record: {rec}") # New debug log
                    records.append({
                        "date": d_str,
                        "employee_id": employee_id,
                        "employee_name": emp_map.get(str(employee_id), "Unknown"),
                        "check_in": rec.get("check_in") if rec else None,
                        "check_out": rec.get("check_out") if rec else None
                    })
            elif employee_id: # Get all attendance for a specific employee
                if str(employee_id) not in attendance_data:
                    return {"success": True, "data": [], "message": "Không có dữ liệu điểm danh cho nhân viên này"}
                
                # Filter by date or month if provided and valid
                for d, rec in attendance_data[str(employee_id)].items():
                    if date:
                        if len(date) == 7:  # YYYY-MM format, filter by month
                            if not d.startswith(date):
                                continue
                        elif len(date) == 10:  # YYYY-MM-DD format, filter by specific date
                            if d != date:
                                continue
                    records.append({
                        "date": d,
                        "employee_id": employee_id,
                        "employee_name": emp_map.get(str(employee_id), "Unknown"),
                        "check_in": rec.get("check_in"),
                        "check_out": rec.get("check_out")
                    })
            elif date and len(date) == 10: # YYYY-MM-DD format for all employees on a specific date
                for emp_id in emp_ids:
                    rec = attendance_data.get(emp_id, {}).get(date, None)
                    records.append({
                        "date": date,
                        "employee_id": emp_id,
                        "employee_name": emp_map.get(emp_id, "Unknown"),
                        "check_in": rec.get("check_in") if rec else None,
                        "check_out": rec.get("check_out") if rec else None
                    })
            else: # Get all attendance records
                if date and len(date) != 7 and len(date) != 10:
                    return {
                        "success": False,
                        "message": "Định dạng ngày không hợp lệ cho truy vấn tất cả nhân viên. Sử dụng YYYY-MM hoặc YYYY-MM-DD"
                    }

                all_dates = set()
                for days in attendance_data.values():
                    all_dates.update(days.keys())
                for d in sorted(list(all_dates), reverse=True):
                    if date and not d.startswith(date):
                        continue
                    for emp_id in emp_ids:
                        rec = attendance_data.get(emp_id, {}).get(d, None)
                        records.append({
                            "date": d,
                            "employee_id": emp_id,
                            "employee_name": emp_map.get(emp_id, "Unknown"),
                            "check_in": rec.get("check_in") if rec else None,
                            "check_out": rec.get("check_out") if rec else None
                        })

            # Sort records by date (newest first), then by employee name
            records = sorted(records, key=lambda x: (x["date"], x["employee_name"]), reverse=True)
            
            print(f"Backend: Returning records: {records}") # New debug log

            if not records:
                return {
                    "success": True,
                    "data": [],
                    "message": "Không có dữ liệu điểm danh cho khoảng thời gian này"
                }
            
            return {"success": True, "data": records}
        except Exception as e:
            print(f"Backend Error (attendance_controller): Type={type(e)}, Value={e}") # Debug print
            return {
                "success": False,
                "message": f"Lỗi khi lấy dữ liệu điểm danh: {str(e)}"
            }
    
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