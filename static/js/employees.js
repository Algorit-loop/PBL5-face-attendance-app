/**
 * Employees page functionality
 */

// Employee data array
let employees = [];
let currentEmployeeId = null;

// Initialize employees page
document.addEventListener('DOMContentLoaded', function() {
    // Load employee data
    fetchEmployees();
    
    // Setup event listeners
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Add employee button
    const addEmployeeBtn = document.getElementById('addEmployeeBtn');
    if (addEmployeeBtn) {
        addEmployeeBtn.addEventListener('click', () => showEmployeeModal());
    }
    
    // Save employee button
    const saveEmployeeBtn = document.getElementById('saveEmployeeBtn');
    if (saveEmployeeBtn) {
        saveEmployeeBtn.addEventListener('click', saveEmployee);
    }
}

// Fetch employees from server
function fetchEmployees() {
    fetch('/api/employees')
        .then(response => response.json())
        .then(data => {
            employees = data;
            renderEmployeeList();
        })
        .catch(error => {
            showToast('Lỗi tải dữ liệu nhân viên: ' + error.message, 'error');
        });
}

// Render employee list
function renderEmployeeList() {
    const employeeList = document.getElementById('employee-list');
    if (!employeeList) return;
    
    employeeList.innerHTML = '';
    
    if (employees.length === 0) {
        employeeList.innerHTML = `
            <tr>
                <td colspan="9" class="text-center">Không có dữ liệu nhân viên</td>
            </tr>
        `;
        return;
    }
    
    employees.forEach(employee => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${employee.id}</td>
            <td>${employee.fullName}</td>
            <td>${formatDate(employee.birthDate)}</td>
            <td>${employee.email}</td>
            <td>${employee.phone}</td>
            <td>${employee.address}</td>
            <td>${employee.gender}</td>
            <td>${employee.position}</td>
            <td>
                <div class="action-buttons">
                    <button class="action-btn edit-btn" onclick="editEmployee(${employee.id})">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="action-btn delete-btn" onclick="deleteEmployee(${employee.id})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        `;
        employeeList.appendChild(row);
    });
}

// Show employee modal (add or edit)
function showEmployeeModal(employeeId = null) {
    currentEmployeeId = employeeId;
    
    // Get modal elements
    const modalTitle = document.getElementById('modalTitleText');
    const employeeForm = document.getElementById('employeeForm');
    
    // Reset form
    employeeForm.reset();
    
    if (employeeId) {
        // Edit mode
        modalTitle.textContent = 'Sửa thông tin nhân viên';
        
        // Find employee
        const employee = employees.find(emp => emp.id === employeeId);
        if (employee) {
            // Fill form with employee data
            document.getElementById('employeeId').value = employee.id;
            document.getElementById('fullName').value = employee.fullName;
            document.getElementById('birthDate').value = formatDateForInput(employee.birthDate);
            document.getElementById('email').value = employee.email;
            document.getElementById('phone').value = employee.phone;
            document.getElementById('address').value = employee.address;
            document.getElementById('gender').value = employee.gender;
            document.getElementById('position').value = employee.position;
        }
    } else {
        // Add mode
        modalTitle.textContent = 'Thêm nhân viên';
        document.getElementById('employeeId').value = '';
    }
    
    // Show modal
    const employeeModal = new bootstrap.Modal(document.getElementById('employeeModal'));
    employeeModal.show();
}

// Format date for input field (YYYY-MM-DD)
function formatDateForInput(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    return date.toISOString().split('T')[0];
}

// Save employee (add or update)
function saveEmployee() {
    // Get form data
    const employeeId = document.getElementById('employeeId').value;
    const fullName = document.getElementById('fullName').value;
    const birthDate = document.getElementById('birthDate').value;
    const email = document.getElementById('email').value;
    const phone = document.getElementById('phone').value;
    const address = document.getElementById('address').value;
    const gender = document.getElementById('gender').value;
    const position = document.getElementById('position').value;
    
    // Validate form
    if (!fullName || !birthDate || !email || !phone || !address || !gender || !position) {
        showToast('Vui lòng điền đầy đủ thông tin', 'error');
        return;
    }
    
    // Prepare employee data
    const employeeData = {
        fullName,
        birthDate,
        email,
        phone,
        address,
        gender,
        position
    };
    
    // Add or update employee
    if (currentEmployeeId) {
        // Update existing employee
        employeeData.id = currentEmployeeId;
        
        fetch(`/api/employees/${currentEmployeeId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(employeeData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Cập nhật nhân viên thành công', 'success');
                fetchEmployees(); // Refresh list
                hideEmployeeModal();
            } else {
                showToast('Lỗi: ' + data.error, 'error');
            }
        })
        .catch(error => {
            showToast('Lỗi kết nối: ' + error.message, 'error');
        });
    } else {
        // Add new employee
        fetch('/api/employees', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(employeeData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Thêm nhân viên thành công', 'success');
                fetchEmployees(); // Refresh list
                hideEmployeeModal();
            } else {
                showToast('Lỗi: ' + data.error, 'error');
            }
        })
        .catch(error => {
            showToast('Lỗi kết nối: ' + error.message, 'error');
        });
    }
}

// Hide employee modal
function hideEmployeeModal() {
    const employeeModal = bootstrap.Modal.getInstance(document.getElementById('employeeModal'));
    if (employeeModal) {
        employeeModal.hide();
    }
    currentEmployeeId = null;
}

// Edit employee
function editEmployee(employeeId) {
    showEmployeeModal(employeeId);
}

// Delete employee
function deleteEmployee(employeeId) {
    if (confirm('Bạn có chắc chắn muốn xóa nhân viên này?')) {
        fetch(`/api/employees/${employeeId}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Xóa nhân viên thành công', 'success');
                fetchEmployees(); // Refresh list
            } else {
                showToast('Lỗi: ' + data.error, 'error');
            }
        })
        .catch(error => {
            showToast('Lỗi kết nối: ' + error.message, 'error');
        });
    }
} 