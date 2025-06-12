from fastapi import HTTPException
from typing import List, Dict, Any, Optional
from models import Shift, ShiftCreate, ShiftUpdate
import database

class ShiftController:
    @staticmethod
    async def get_all():
        """
        Get all shifts
        
        Returns:
            List of all shifts
        """
        return database.get_all_shifts()
        
    @staticmethod
    async def get_by_id(shift_id: int):
        """
        Get shift by ID
        
        Args:
            shift_id: Shift ID
            
        Returns:
            Shift data or None if not found
        """
        try:
            shift = database.get_shift_by_id(shift_id)
            if not shift:
                print(f"Shift with ID {shift_id} not found in database")
                return None
            return shift
        except Exception as e:
            print(f"Error in get_by_id: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    async def create(shift: ShiftCreate):
        """
        Create a new shift
        
        Args:
            shift: Shift data
            
        Returns:
            Created shift data
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            # Convert to dict for storage
            shift_dict = shift.dict()
            
            # Add shift to database
            return database.add_shift(shift_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def update(shift_id: int, shift: ShiftUpdate):
        """
        Update an existing shift
        
        Args:
            shift_id: Shift ID
            shift: New shift data
            
        Returns:
            Updated shift data
            
        Raises:
            HTTPException: If shift not found or validation fails
        """
        try:
            # Check if shift exists
            existing_shift = database.get_shift_by_id(shift_id)
            if not existing_shift:
                raise HTTPException(status_code=404, detail="Không tìm thấy ca làm")

            # Convert to dict for storage
            shift_dict = shift.dict(exclude_unset=True)
            
            # Merge with existing data
            updated_data = {**existing_shift, **shift_dict}
            
            # Update shift in database
            updated_shift = database.update_shift(shift_id, updated_data)
            if not updated_shift:
                raise HTTPException(status_code=404, detail="Cập nhật ca làm thất bại")
                
            return updated_shift
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def delete(shift_id: int):
        """
        Delete a shift
        
        Args:
            shift_id: Shift ID
            
        Returns:
            Success message
            
        Raises:
            HTTPException: If shift not found
        """
        # Check if shift exists
        existing_shift = database.get_shift_by_id(shift_id)
        if not existing_shift:
            raise HTTPException(status_code=404, detail="Không tìm thấy ca làm")
        
        # Delete shift
        success = database.delete_shift(shift_id)
        if not success:
            raise HTTPException(status_code=500, detail="Xóa ca làm thất bại")
            
        return {"success": True, "message": "Đã xóa ca làm thành công"} 