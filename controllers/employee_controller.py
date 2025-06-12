from fastapi import HTTPException
from typing import List, Dict, Any, Optional
from models import Employee
import database

class EmployeeController:
    @staticmethod
    async def get_all():
        """
        Get all employees
        
        Returns:
            List of all employees
        """
        return database.get_all_employees()
        
    @staticmethod
    async def get_by_id(employee_id: int):
        """
        Get employee by ID
        
        Args:
            employee_id: Employee ID
            
        Returns:
            Employee data or None if not found
        """
        try:
            employee = database.get_employee_by_id(employee_id)
            if not employee:
                print(f"Employee with ID {employee_id} not found in database")
                return None
            return employee
        except Exception as e:
            print(f"Error in get_by_id: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    async def create(employee: Employee):
        """
        Create a new employee
        
        Args:
            employee: Employee data
            
        Returns:
            Created employee data
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            # Get all employees
            employees = database.get_all_employees()
            
            # Check duplicates
            if any(emp["email"] == employee.email for emp in employees):
                raise HTTPException(status_code=400, detail="Email đã tồn tại trong hệ thống")
                
            if any(emp["phone"] == employee.phone for emp in employees):
                raise HTTPException(status_code=400, detail="Số điện thoại đã tồn tại trong hệ thống")
                
            if any(emp.get("username") == employee.username for emp in employees):
                raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại trong hệ thống")
            
            # Convert to dict for storage
            employee_dict = employee.dict()
            
            # Add employee to database
            return database.add_employee(employee_dict)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def update(employee_id: int, employee: Employee):
        """
        Update an existing employee
        
        Args:
            employee_id: Employee ID
            employee: New employee data
            
        Returns:
            Updated employee data
            
        Raises:
            HTTPException: If employee not found or validation fails
        """
        try:
            # Check if employee exists
            existing_employee = database.get_employee_by_id(employee_id)
            if not existing_employee:
                raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")

            # Get all employees
            employees = database.get_all_employees()
            
            # Check duplicates excluding current employee
            if any(emp["email"] == employee.email and emp["id"] != employee_id for emp in employees):
                raise HTTPException(status_code=400, detail="Email đã tồn tại trong hệ thống")
                
            if any(emp["phone"] == employee.phone and emp["id"] != employee_id for emp in employees):
                raise HTTPException(status_code=400, detail="Số điện thoại đã tồn tại trong hệ thống")
                
            if any(emp.get("username") == employee.username and emp["id"] != employee_id for emp in employees):
                raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại trong hệ thống")
            
            # Convert to dict for storage
            employee_dict = employee.dict()
            
            # Update employee in database
            updated_employee = database.update_employee(employee_id, employee_dict)
            if not updated_employee:
                raise HTTPException(status_code=404, detail="Cập nhật nhân viên thất bại")
                
            return updated_employee
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def delete(employee_id: int):
        """
        Delete an employee
        
        Args:
            employee_id: Employee ID
            
        Returns:
            Success message
            
        Raises:
            HTTPException: If employee not found
        """
        # Check if employee exists
        existing_employee = database.get_employee_by_id(employee_id)
        if not existing_employee:
            raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
        
        # Delete employee
        success = database.delete_employee(employee_id)
        if not success:
            raise HTTPException(status_code=500, detail="Xóa nhân viên thất bại")
            
        return {"success": True, "message": "Đã xóa nhân viên thành công"} 