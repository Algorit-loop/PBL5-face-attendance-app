// Global variables
let employees = [];
const modal = new bootstrap.Modal(document.getElementById('employeeModal'));
let cameraStatus = false;

// Load employees and camera info when page loads
document.addEventListener('DOMContentLoaded', () => {
    loadEmployees();
    updateCameraStatus();
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
    try {
        const response = await fetch('/camera/toggle', {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.status === 'error') {
            alert(result.message);
            return;
        }
        
        cameraStatus = result.status === 'on';
        document.getElementById('cameraStatus').textContent = cameraStatus ? 'Đang bật' : 'Đang tắt';
        document.getElementById('toggleCamera').textContent = cameraStatus ? 'Tắt Camera' : 'Bật Camera';
        
        if (cameraStatus) {
            loadCameraInfo();
        } else {
            document.getElementById('resolution').textContent = 'Chưa khởi động';
            document.getElementById('fps').textContent = 'Chưa khởi động';
        }
    } catch (error) {
        console.error('Error toggling camera:', error);
        alert('Không thể kết nối với camera. Vui lòng thử lại sau.');
    }
}

// Add event listener for camera toggle button
document.getElementById('toggleCamera').addEventListener('click', toggleCamera);

// Load all employees
async function loadEmployees() {
    try {
        const response = await fetch('/employees');
        employees = await response.json();
        displayEmployees();
    } catch (error) {
        console.error('Error loading employees:', error);
    }
}

// Display employees in the table
function displayEmployees() {
    const tbody = document.getElementById('employee-list');
    tbody.innerHTML = '';
    
    employees.forEach(employee => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${employee.id}</td>
            <td>${employee.full_name}</td>
            <td>${employee.birth_date}</td>
            <td>${employee.email}</td>
            <td>${employee.phone}</td>
            <td>${employee.address}</td>
            <td>${employee.gender}</td>
            <td>${employee.position}</td>
            <td>
                <button class="btn btn-sm btn-warning" onclick="editEmployee(${employee.id})">Sửa</button>
                <button class="btn btn-sm btn-danger" onclick="deleteEmployee(${employee.id})">Xóa</button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Show add employee modal
function showAddEmployeeModal() {
    document.getElementById('modalTitle').textContent = 'Thêm nhân viên';
    document.getElementById('employeeForm').reset();
    document.getElementById('employeeId').value = '';
    modal.show();
}

// Show edit employee modal
async function editEmployee(id) {
    try {
        const response = await fetch(`/employees/${id}`);
        const employee = await response.json();
        
        document.getElementById('modalTitle').textContent = 'Sửa thông tin nhân viên';
        document.getElementById('employeeId').value = employee.id;
        document.getElementById('fullName').value = employee.full_name;
        document.getElementById('birthDate').value = employee.birth_date;
        document.getElementById('email').value = employee.email;
        document.getElementById('phone').value = employee.phone;
        document.getElementById('address').value = employee.address;
        document.getElementById('gender').value = employee.gender;
        document.getElementById('position').value = employee.position;
        
        modal.show();
    } catch (error) {
        console.error('Error loading employee:', error);
    }
}

// Save employee (create or update)
async function saveEmployee() {
    const form = document.getElementById('employeeForm');
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const employee = {
        full_name: document.getElementById('fullName').value,
        birth_date: document.getElementById('birthDate').value,
        email: document.getElementById('email').value,
        phone: document.getElementById('phone').value,
        address: document.getElementById('address').value,
        gender: document.getElementById('gender').value,
        position: document.getElementById('position').value
    };

    const id = document.getElementById('employeeId').value;
    const url = id ? `/employees/${id}` : '/employees';
    const method = id ? 'PUT' : 'POST';

    try {
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(employee)
        });

        if (response.ok) {
            modal.hide();
            loadEmployees();
        } else {
            const errorData = await response.json();
            if (errorData.detail) {
                alert("Lỗi: " + errorData.detail);
            } else {
                alert("Lỗi: Không thể lưu thông tin nhân viên");
            }
        }
    } catch (error) {
        console.error('Error saving employee:', error);
        alert('Không thể kết nối với máy chủ. Vui lòng thử lại sau.');
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

        if (response.ok) {
            loadEmployees();
        } else {
            console.error('Error deleting employee:', await response.text());
        }
    } catch (error) {
        console.error('Error deleting employee:', error);
    }
} 