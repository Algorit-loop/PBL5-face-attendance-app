// Utility function to format date
const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN');
};

// Update datetime in navbar
function updateDateTime() {
    const now = new Date();
    const dateTimeString = now.toLocaleString('vi-VN');
    document.getElementById('datetime').textContent = dateTimeString;
}

// Employee management class
class EmployeeManager {
    constructor() {
        this.employees = [];
        this.initializeEventListeners();
        this.loadEmployees();
    }

    initializeEventListeners() {
        // Train model button
        const trainModelBtn = document.getElementById('trainModelBtn');
        if (trainModelBtn) {
            trainModelBtn.addEventListener('click', async () => await this.handleTrainModel());
        }

        // Save employee button
        const saveEmployeeBtn = document.getElementById('saveEmployeeBtn');
        if (saveEmployeeBtn) {
            saveEmployeeBtn.addEventListener('click', () => this.handleSaveEmployee());
        }
    }

    async loadEmployees() {
        try {
            const response = await fetch('/employees');
            if (!response.ok) throw new Error('Failed to fetch employees');
            this.employees = await response.json();
            this.renderEmployeeTable();
        } catch (error) {
            console.error('Error loading employees:', error);
            alert('Không thể tải danh sách nhân viên');
        }
    }

    renderEmployeeTable() {
        const tableBody = document.getElementById('employee-list');
        if (!tableBody) return;

        tableBody.innerHTML = this.employees.map(employee => `
            <tr>
                <td>${employee.id}</td>
                <td>${employee.full_name}</td>
                <td>${formatDate(employee.birth_date)}</td>
                <td>${employee.email}</td>
                <td>${employee.phone}</td>
                <td>${employee.address}</td>
                <td>${employee.gender}</td>
                <td>${employee.position}</td>
                <td>
                    <div class="btn-group">
                        <button class="btn btn-sm btn-primary" onclick="employeeManager.editEmployee(${employee.id})">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="employeeManager.deleteEmployee(${employee.id})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    async handleTrainModel() {
        const trainModelBtn = document.getElementById('trainModelBtn');
        const trainingStatus = document.getElementById('trainingStatus');
        
        try {
            trainModelBtn.disabled = true;
            trainingStatus.style.display = 'block';
            trainingStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang training model...';

            const response = await fetch('/train', { method: 'POST' });
            if (!response.ok) throw new Error('Training failed');

            trainingStatus.innerHTML = '<i class="fas fa-check-circle text-success"></i> Training thành công!';
            setTimeout(() => {
                trainingStatus.style.display = 'none';
                trainModelBtn.disabled = false;
            }, 3000);
        } catch (error) {
            console.error('Training error:', error);
            trainingStatus.innerHTML = '<i class="fas fa-exclamation-circle text-danger"></i> Training thất bại';
            trainModelBtn.disabled = false;
        }
    }

    async handleSaveEmployee() {
        const employeeData = {
            fullName: document.getElementById('fullName').value,
            birthDate: document.getElementById('birthDate').value,
            email: document.getElementById('email').value,
            phone: document.getElementById('phone').value,
            address: document.getElementById('address').value,
            gender: document.getElementById('gender').value,
            position: document.getElementById('position').value
        };

        const employeeId = document.getElementById('employeeId').value;
        const isNewEmployee = !employeeId;

        try {
            const url = isNewEmployee ? '/employees' : `/employees/${employeeId}`;
            const method = isNewEmployee ? 'POST' : 'PUT';
            
            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(employeeData)
            });

            if (!response.ok) throw new Error('Failed to save employee');

            // Close modal and refresh list
            const modal = bootstrap.Modal.getInstance(document.getElementById('employeeModal'));
            modal.hide();
            await this.loadEmployees();

            alert(isNewEmployee ? 'Thêm nhân viên thành công!' : 'Cập nhật thông tin thành công!');
        } catch (error) {
            console.error('Error saving employee:', error);
            alert('Không thể lưu thông tin nhân viên');
        }
    }

    editEmployee(employeeId) {
        const employee = this.employees.find(emp => emp.id === employeeId);
        if (!employee) return;

        // Fill form with employee data
        document.getElementById('employeeId').value = employee.id;
        document.getElementById('fullName').value = employee.fullName;
        document.getElementById('birthDate').value = employee.birthDate.split('T')[0];
        document.getElementById('email').value = employee.email;
        document.getElementById('phone').value = employee.phone;
        document.getElementById('address').value = employee.address;
        document.getElementById('gender').value = employee.gender;
        document.getElementById('position').value = employee.position;

        // Update modal title
        document.getElementById('modalTitleText').textContent = 'Cập nhật thông tin nhân viên';

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('employeeModal'));
        modal.show();
    }

    async deleteEmployee(employeeId) {
        if (!confirm('Bạn có chắc chắn muốn xóa nhân viên này?')) return;

        try {
            const response = await fetch(`/employees/${employeeId}`, {
                method: 'DELETE'
            });

            if (!response.ok) throw new Error('Failed to delete employee');

            await this.loadEmployees();
            alert('Xóa nhân viên thành công!');
        } catch (error) {
            console.error('Error deleting employee:', error);
            alert('Không thể xóa nhân viên');
        }
    }
}

// Initialize
const employeeManager = new EmployeeManager();

// Update datetime every second
setInterval(updateDateTime, 1000);
updateDateTime();