// DOM Elements
const dateFilter = document.getElementById('dateFilter');
const employeeFilter = document.getElementById('employeeFilter');
const refreshBtn = document.getElementById('refreshBtn');
const attendanceTableBody = document.getElementById('attendanceTableBody');

// Set default date to today
dateFilter.valueAsDate = new Date();

// Load employees for filter
async function loadEmployees() {
    try {
        const response = await fetch('/employees');
        const data = await response.json();
        
        // Clear existing options except the first one
        while (employeeFilter.options.length > 1) {
            employeeFilter.remove(1);
        }
        
        // Add employee options
        data.forEach(employee => {
            const option = document.createElement('option');
            option.value = employee.id;
            option.textContent = employee.full_name;
            employeeFilter.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading employees:', error);
    }
}

// Load attendance records
async function loadAttendance() {
    try {
        const date = dateFilter.value;
        const employeeId = employeeFilter.value;
        
        let url = '/api/attendance';
        const params = new URLSearchParams();
        if (date) params.append('date', date);
        if (employeeId) params.append('employee_id', employeeId);
        if (params.toString()) url += '?' + params.toString();
        
        const response = await fetch(url);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to load attendance data');
        }
        
        const data = await response.json();
        
        // Clear existing records
        attendanceTableBody.innerHTML = '';
        
        if (data.success && data.data.length > 0) {
            data.data.forEach(record => {
                const row = document.createElement('tr');
                
                // Format date
                const date = new Date(record.date);
                const formattedDate = date.toLocaleDateString('vi-VN', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                });
                
                // Determine status
                let status = 'absent';
                let statusText = 'Vắng mặt';
                let statusClass = 'status-absent';
                
                if (record.check_in && record.check_out) {
                    status = 'complete';
                    statusText = 'Đầy đủ';
                    statusClass = 'status-complete';
                } else if (record.check_in) {
                    status = 'incomplete';
                    statusText = 'Chưa hoàn thành';
                    statusClass = 'status-incomplete';
                }
                
                row.innerHTML = `
                    <td>${formattedDate}</td>
                    <td>${record.employee_name}</td>
                    <td>${record.check_in || '-'}</td>
                    <td>${record.check_out || '-'}</td>
                    <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                `;
                
                attendanceTableBody.appendChild(row);
            });
        } else {
            // Show no records message
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="5" class="text-center py-4">
                    <i class="fas fa-info-circle me-2"></i>
                    ${data.message || 'Không có dữ liệu điểm danh'}
                </td>
            `;
            attendanceTableBody.appendChild(row);
        }
    } catch (error) {
        console.error('Error loading attendance:', error);
        // Show error message in the table
        attendanceTableBody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center py-4 text-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    ${error.message || 'Không thể tải dữ liệu điểm danh'}
                </td>
            </tr>
        `;
    }
}

// Event Listeners
dateFilter.addEventListener('change', loadAttendance);
employeeFilter.addEventListener('change', loadAttendance);
refreshBtn.addEventListener('click', loadAttendance);

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    loadEmployees();
    loadAttendance();
}); 