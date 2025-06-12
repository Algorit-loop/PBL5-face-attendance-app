from fastapi import HTTPException
from typing import List, Dict, Any, Optional
from models import Attendance, AttendanceCreate, AttendanceFilter
import database
from datetime import datetime

class AttendanceController:
    @staticmethod
    async def get_all():
        """
        Get all attendance records
        
        Returns:
            List of all attendance records
        """
        return database.get_all_attendance()
        
    @staticmethod
    async def get_by_id(attendance_id: int):
        """
        Get attendance by ID
        
        Args:
            attendance_id: Attendance ID
            
        Returns:
            Attendance data or None if not found
        """
        try:
            attendance = database.get_attendance_by_id(attendance_id)
            if not attendance:
                print(f"Attendance with ID {attendance_id} not found in database")
                return None
            return attendance
        except Exception as e:
            print(f"Error in get_by_id: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    async def create(attendance: AttendanceCreate):
        """
        Create a new attendance record
        
        Args:
            attendance: Attendance data
            
        Returns:
            Created attendance data
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            # Check if employee exists
            employee = database.get_employee_by_id(attendance.employee_id)
            if not employee:
                raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
                
            # Check if shift exists
            shift = database.get_shift_by_id(attendance.shift_id)
            if not shift:
                raise HTTPException(status_code=404, detail="Không tìm thấy ca làm")
            
            # Convert to dict for storage
            attendance_dict = attendance.dict()
            
            # Add attendance record to database
            return database.add_attendance(attendance_dict)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def filter(filter_params: AttendanceFilter):
        """
        Filter attendance records
        
        Args:
            filter_params: Filter parameters
            
        Returns:
            Filtered attendance records
        """
        try:
            # Convert filter params to dict
            filter_dict = filter_params.dict(exclude_unset=True)
            
            # Filter attendance records
            records = database.filter_attendance(**filter_dict)
            
            # Enhance records with employee and shift info
            enhanced_records = []
            for record in records:
                enhanced_record = dict(record)
                
                # Add employee info
                employee = database.get_employee_by_id(record["employee_id"])
                if employee:
                    enhanced_record["employee_name"] = employee["full_name"]
                    enhanced_record["department"] = employee["department"]
                    
                # Add shift info
                shift = database.get_shift_by_id(record["shift_id"])
                if shift:
                    enhanced_record["shift_name"] = shift["name"]
                    enhanced_record["start_time"] = shift["start_time"]
                    enhanced_record["end_time"] = shift["end_time"]
                    
                enhanced_records.append(enhanced_record)
            
            return enhanced_records
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    @staticmethod
    async def get_statistics(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        department: Optional[str] = None
    ):
        """
        Get attendance statistics
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            department: Department
            
        Returns:
            Attendance statistics
        """
        try:
            # Get all employees
            employees = database.get_all_employees()
            
            # Filter employees by department if specified
            if department:
                employees = [emp for emp in employees if emp["department"] == department]
                
            # Get all attendance records
            filter_params = {}
            if start_date:
                filter_params["start_date"] = start_date
            if end_date:
                filter_params["end_date"] = end_date
                
            attendance_records = database.filter_attendance(**filter_params)
            
            # Calculate statistics
            total_employees = len(employees)
            employee_stats = {}
            
            for employee in employees:
                employee_id = employee["id"]
                employee_stats[employee_id] = {
                    "id": employee_id,
                    "name": employee["full_name"],
                    "department": employee["department"],
                    "on_time": 0,
                    "late": 0,
                    "very_late": 0,
                    "absent": 0,
                    "total": 0
                }
            
            # Count attendance by status
            for record in attendance_records:
                employee_id = record["employee_id"]
                status = record["status"]
                
                if employee_id in employee_stats:
                    employee_stats[employee_id]["total"] += 1
                    if status == "on_time":
                        employee_stats[employee_id]["on_time"] += 1
                    elif status == "late":
                        employee_stats[employee_id]["late"] += 1
                    elif status == "very_late":
                        employee_stats[employee_id]["very_late"] += 1
                    elif status == "absent":
                        employee_stats[employee_id]["absent"] += 1
            
            # Convert to list
            employee_stats_list = list(employee_stats.values())
            
            # Calculate overall statistics
            total_records = len(attendance_records)
            on_time_count = sum(emp["on_time"] for emp in employee_stats_list)
            late_count = sum(emp["late"] for emp in employee_stats_list)
            very_late_count = sum(emp["very_late"] for emp in employee_stats_list)
            absent_count = sum(emp["absent"] for emp in employee_stats_list)
            
            # Calculate percentages
            on_time_percentage = (on_time_count / total_records * 100) if total_records > 0 else 0
            late_percentage = (late_count / total_records * 100) if total_records > 0 else 0
            very_late_percentage = (very_late_count / total_records * 100) if total_records > 0 else 0
            absent_percentage = (absent_count / total_records * 100) if total_records > 0 else 0
            
            return {
                "total_employees": total_employees,
                "total_records": total_records,
                "on_time_count": on_time_count,
                "late_count": late_count,
                "very_late_count": very_late_count,
                "absent_count": absent_count,
                "on_time_percentage": round(on_time_percentage, 2),
                "late_percentage": round(late_percentage, 2),
                "very_late_percentage": round(very_late_percentage, 2),
                "absent_percentage": round(absent_percentage, 2),
                "employee_stats": employee_stats_list
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) 