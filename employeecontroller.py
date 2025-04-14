from fastapi import HTTPException
from models import Employee
import database

class EmployeeController:
    @staticmethod
    async def get_all():
        """Get all employees"""
        return database.employees
        
    @staticmethod
    async def get_by_id(employee_id: int):
        """Get employee by ID"""
        employee = next((emp for emp in database.employees if emp["id"] == employee_id), None)
        if not employee:
            raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
        return employee

    @staticmethod
    async def create(employee: Employee):
        """Create a new employee"""
        try:
            # Check duplicates
            if any(emp["email"] == employee.email for emp in database.employees):
                raise HTTPException(status_code=400, detail="Email đã tồn tại trong hệ thống")
                
            if any(emp["phone"] == employee.phone for emp in database.employees):
                raise HTTPException(status_code=400, detail="Số điện thoại đã tồn tại trong hệ thống")
                
            new_id = max(emp["id"] for emp in database.employees) + 1 if database.employees else 1
            employee_dict = employee.dict()
            employee_dict["id"] = new_id
            database.employees.append(employee_dict)
            database.save_employees(database.employees)
            return employee_dict
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def update(employee_id: int, employee: Employee):
        """Update an existing employee"""
        try:
            existing_employee = next((i for i, emp in enumerate(database.employees) if emp["id"] == employee_id), None)
            if existing_employee is None:
                raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")

            # Check duplicates excluding current employee
            if any(emp["email"] == employee.email and emp["id"] != employee_id for emp in database.employees):
                raise HTTPException(status_code=400, detail="Email đã tồn tại trong hệ thống")
                
            if any(emp["phone"] == employee.phone and emp["id"] != employee_id for emp in database.employees):
                raise HTTPException(status_code=400, detail="Số điện thoại đã tồn tại trong hệ thống")
                
            employee_dict = employee.dict()
            employee_dict["id"] = employee_id
            database.employees[existing_employee] = employee_dict
            database.save_employees(database.employees)
            return employee_dict
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def delete(employee_id: int):
        """Delete an employee"""
        employee_index = next((i for i, emp in enumerate(database.employees) if emp["id"] == employee_id), None)
        if employee_index is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy nhân viên")
        
        database.employees.pop(employee_index)
        database.save_employees(database.employees)
        return {"message": "Đã xóa nhân viên thành công"}
