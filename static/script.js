// Global variables
let employees = [];
const modal = new bootstrap.Modal(document.getElementById('employeeModal'));
let cameraStatus = false;
let employeeModal = null;
let employeeForm = null;

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap modal
    employeeModal = new bootstrap.Modal(document.getElementById('employeeModal'));
    employeeForm = document.getElementById('employeeForm');
    
    // Load employees
    loadEmployees();
    
    // Add event listeners
    document.getElementById('toggleCamera').addEventListener('click', toggleCamera);
    
    // Add form validation
    employeeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        saveEmployee();
    });
});

// Load camera information
async function loadCameraInfo() {
    try {
        const response = await fetch('/camera/info');
        const info = await response.json();
        if (info.error) {
            document.getElementById('resolution').textContent = 'Lỗi: ' + info.error;
            document.getElementById('fps').textContent = 'Lỗi: ' + info.error;
            return;
        }
        if (info.width === 0 && info.height === 0) {
            document.getElementById('resolution').textContent = 'Chưa khởi động';
            document.getElementById('fps').textContent = 'Chưa khởi động';
        } else {
            document.getElementById('resolution').textContent = `${info.width}x${info.height}`;
            document.getElementById('fps').textContent = info.fps.toFixed(2);
        }
    } catch (error) {
        console.error('Error loading camera info:', error);
        document.getElementById('resolution').textContent = 'Lỗi kết nối';
        document.getElementById('fps').textContent = 'Lỗi kết nối';
    }
}

// Update camera status
async function updateCameraStatus() {
    try {
        const response = await fetch('/camera/status');
        const status = await response.json();
        cameraStatus = status.status === 'on';
        document.getElementById('cameraStatus').textContent = cameraStatus ? 'Đang bật' : 'Đang tắt';
        document.getElementById('toggleCamera').textContent = cameraStatus ? 'Tắt Camera' : 'Bật Camera';
        if (cameraStatus) {
            loadCameraInfo();
        } else {
            document.getElementById('resolution').textContent = 'Chưa khởi động';
            document.getElementById('fps').textContent = 'Chưa khởi động';
        }
    } catch (error) {
        console.error('Error updating camera status:', error);
        document.getElementById('cameraStatus').textContent = 'Lỗi kết nối';
    }
}

// Toggle camera
async function toggleCamera() {
    const button = document.getElementById('toggleCamera');
    const spinner = document.getElementById('cameraSpinner');
    const cameraFeed = document.getElementById('camera-feed');
    
    try {
        if (!cameraStatus) {
            // Show loading spinner
            spinner.style.display = 'block';
            cameraFeed.style.display = 'none';
            
            const response = await fetch('/camera/toggle', {
                method: 'POST'
            });
            
            const data = await response.json();
            if (data.status === 'on') {
                cameraStatus = true;
                button.innerHTML = '<i class="fas fa-power-off me-2"></i>Tắt Camera';
                cameraFeed.style.display = 'block';
                spinner.style.display = 'none';
                
                // Update camera info
                updateCameraInfo();
            }
        } else {
            const response = await fetch('/camera/toggle', {
                method: 'POST'
            });
            
            const data = await response.json();
            if (data.status === 'off') {
                cameraStatus = false;
                button.innerHTML = '<i class="fas fa-power-off me-2"></i>Bật Camera';
                cameraFeed.style.display = 'none';
                spinner.style.display = 'none';
            }
        }
    } catch (error) {
        console.error('Error toggling camera:', error);
        showToast('Lỗi khi thao tác với camera', 'danger');
    }
}

async function updateCameraInfo() {
    try {
        const response = await fetch('/camera/info');
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('resolution').textContent = 'Lỗi';
            document.getElementById('fps').textContent = 'Lỗi';
            document.getElementById('cameraStatus').textContent = 'Lỗi';
            return;
        }
        
        document.getElementById('resolution').textContent = `${data.width}x${data.height}`;
        document.getElementById('fps').textContent = data.fps.toFixed(2);
        document.getElementById('cameraStatus').textContent = 'Đang hoạt động';
    } catch (error) {
        console.error('Error updating camera info:', error);
    }
}

// Load all employees
async function loadEmployees() {
    const employeeList = document.getElementById('employee-list');
    employeeList.innerHTML = '<tr><td colspan="9" class="text-center"><div class="spinner"></div></td></tr>';
    
    try {
        const response = await fetch('/employees');
        employees = await response.json();
        
        if (employees.length === 0) {
            employeeList.innerHTML = '<tr><td colspan="9" class="text-center">Không có nhân viên nào</td></tr>';
            return;
        }
        
        employeeList.innerHTML = employees.map(employee => `
            <tr class="fade-in">
                <td>${employee.id}</td>
                <td>${employee.full_name}</td>
                <td>${formatDate(employee.birth_date)}</td>
                <td>${employee.email}</td>
                <td>${employee.phone}</td>
                <td>${employee.address}</td>
                <td>${employee.gender}</td>
                <td>${employee.position}</td>
                <td>
                    <button class="btn btn-sm btn-primary me-1" onclick="editEmployee(${employee.id})">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="deleteEmployee(${employee.id})">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading employees:', error);
        employeeList.innerHTML = '<tr><td colspan="9" class="text-center text-danger">Lỗi khi tải danh sách nhân viên</td></tr>';
        showToast('Lỗi khi tải danh sách nhân viên', 'danger');
    }
}

// Show add employee modal
function showAddEmployeeModal() {
    document.getElementById('modalTitleText').textContent = 'Thêm nhân viên';
    document.getElementById('employeeId').value = '';
    employeeForm.reset();
    employeeModal.show();
}

// Show edit employee modal
async function editEmployee(id) {
    try {
        const response = await fetch(`/employees/${id}`);
        const employee = await response.json();
        
        document.getElementById('modalTitleText').textContent = 'Chỉnh sửa nhân viên';
        document.getElementById('employeeId').value = employee.id;
        document.getElementById('fullName').value = employee.full_name;
        document.getElementById('birthDate').value = employee.birth_date;
        document.getElementById('email').value = employee.email;
        document.getElementById('phone').value = employee.phone;
        document.getElementById('address').value = employee.address;
        document.getElementById('gender').value = employee.gender;
        document.getElementById('position').value = employee.position;
        
        employeeModal.show();
    } catch (error) {
        console.error('Error loading employee:', error);
        showToast('Lỗi khi tải thông tin nhân viên', 'danger');
    }
}

// Save employee (create or update)
async function saveEmployee() {
    const employeeId = document.getElementById('employeeId').value;
    const employee = {
        full_name: document.getElementById('fullName').value,
        birth_date: document.getElementById('birthDate').value,
        email: document.getElementById('email').value,
        phone: document.getElementById('phone').value,
        address: document.getElementById('address').value,
        gender: document.getElementById('gender').value,
        position: document.getElementById('position').value
    };
    
    try {
        const url = employeeId ? `/employees/${employeeId}` : '/employees';
        const method = employeeId ? 'PUT' : 'POST';
        
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(employee)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Lỗi khi lưu nhân viên');
        }
        
        employeeModal.hide();
        showToast(employeeId ? 'Cập nhật nhân viên thành công' : 'Thêm nhân viên thành công', 'success');
        loadEmployees();
    } catch (error) {
        console.error('Error saving employee:', error);
        showToast(error.message, 'danger');
    }
}

// Delete employee
async function deleteEmployee(id) {
    if (!confirm('Bạn có chắc chắn muốn xóa nhân viên này?')) {
        return;
    }
    
    try {
        const response = await fetch(`/employees/${id}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Lỗi khi xóa nhân viên');
        }
        
        showToast('Xóa nhân viên thành công', 'success');
        loadEmployees();
    } catch (error) {
        console.error('Error deleting employee:', error);
        showToast(error.message, 'danger');
    }
}

// Utility functions
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN');
}

function showToast(message, type = 'success') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    // Add toast to container
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    toastContainer.appendChild(toast);
    
    // Initialize and show toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: 3000
    });
    bsToast.show();
    
    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
} 