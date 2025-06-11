// Utility function to format date
const formatDate = (dateString) => {
    if (!dateString) return '-';
    const [year, month, day] = dateString.split('-');
    return `${parseInt(day)}/${parseInt(month)}/${year}`;
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
                        <button class="btn btn-sm btn-info" onclick="employeeManager.viewEmployeeDetails(${employee.id})">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-sm btn-info" onclick="employeeManager.viewAttendanceHistory(${employee.id}, '${employee.full_name}')">
                            <i class="fas fa-history"></i>
                        </button>
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

    // Add new methods for attendance history
    async viewAttendanceHistory(employeeId, employeeName) {
        // Set employee name in modal
        document.getElementById('employeeName').textContent = employeeName;
        
        // Populate month select
        const monthSelect = document.getElementById('monthSelect');
        const currentDate = new Date();
        const months = [];
        
        for (let i = 0; i < 12; i++) {
            const date = new Date(currentDate.getFullYear(), currentDate.getMonth() - i, 1);
            const year = date.getFullYear();
            const month = (date.getMonth() + 1).toString().padStart(2, '0'); // Ensure 2 digits, 0-indexed month
            const monthValue = `${year}-${month}`;

            const monthStr = date.toLocaleString('vi-VN', { month: 'long', year: 'numeric' });
            months.push({ value: monthValue, label: monthStr });
        }
        
        monthSelect.innerHTML = months.map(month => 
            `<option value="${month.value}">${month.label}</option>`
        ).join('');

        // Log the value that is about to be used for the initial load
        console.log('DEBUG: Initial monthSelect.value for loadAttendanceData:', monthSelect.value);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('attendanceHistoryModal'));
        modal.show();

        // Load initial data
        await this.loadAttendanceData(employeeId, monthSelect.value);

        // Add event listener for month change
        monthSelect.addEventListener('change', () => {
            this.loadAttendanceData(employeeId, monthSelect.value);
        });
    }

    async loadAttendanceData(employeeId, month) {
        try {
            console.log(`Frontend: Fetching attendance for employeeId=${employeeId}, month=${month}`); // More detailed debug
            const response = await fetch(`/api/attendance?employee_id=${employeeId}&date=${month}`);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Frontend: Network response was not ok:', response.status, errorText); // Log status and error body
                throw new Error('Failed to fetch attendance data');
            }
            
            const data = await response.json();
            console.log('Frontend: Received attendance data:', data); // Log the full data object
            
            if (data.success) {
                this.renderAttendanceChart(data.data);
                this.renderAttendanceDetails(data.data);
            } else {
                console.error('Frontend: Backend reported success: false', data.message);
                throw new Error(data.message || 'Failed to fetch attendance data');
            }
        } catch (error) {
            console.error('Error loading attendance data:', error);
            alert('Không thể tải dữ liệu điểm danh');
        }
    }

    renderAttendanceChart(data) {
        const ctx = document.getElementById('attendanceChart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.attendanceChart) {
            this.attendanceChart.destroy();
        }

        // Prepare data for chart
        const labels = data.map(record => {
            if (record.date) {
                return parseInt(record.date.split('-')[2]); // Extract day from YYYY-MM-DD
            }
            return ''; // Fallback for missing date
        });
        const checkInData = data.map(record => record.check_in ? new Date(`2000-01-01T${record.check_in}`).getHours() + new Date(`2000-01-01T${record.check_in}`).getMinutes() / 60 : null);
        const checkOutData = data.map(record => record.check_out ? new Date(`2000-01-01T${record.check_out}`).getHours() + new Date(`2000-01-01T${record.check_out}`).getMinutes() / 60 : null);

        // Create new chart
        this.attendanceChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Giờ vào',
                        data: checkInData,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        pointRadius: 5,
                        pointHoverRadius: 7
                    },
                    {
                        label: 'Giờ ra',
                        data: checkOutData,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1,
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Giờ'
                        },
                        min: 0,
                        max: 24 // Assuming hours 0-24
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Ngày'
                        },
                        type: 'category',
                        labels: labels // Use category type with labels for discrete days
                    }
                },
                tooltips: {
                    callbacks: {
                        label: function(tooltipItem, data) {
                            const datasetLabel = data.datasets[tooltipItem.datasetIndex].label || '';
                            const timeInHours = tooltipItem.yLabel;
                            const hours = Math.floor(timeInHours);
                            const minutes = Math.round((timeInHours - hours) * 60);
                            return `${datasetLabel}: ${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
                        }
                    }
                }
            }
        });
    }

    renderAttendanceDetails(data) {
        const tbody = document.getElementById('attendanceDetails');
        tbody.innerHTML = ''; // Clear existing records

        if (data && data.length > 0) {
            tbody.innerHTML = data.map(record => `
                <tr>
                    <td>${formatDate(record.date)}</td>
                    <td>${record.check_in ? new Date(`2000-01-01T${record.check_in}`).toLocaleTimeString('vi-VN') : '-'}</td>
                    <td>${record.check_out ? new Date(`2000-01-01T${record.check_out}`).toLocaleTimeString('vi-VN') : '-'}</td>
                    <td>
                        <span class="badge ${this.getStatusBadgeClass(record)}">
                            ${this.getStatusText(record)}
                        </span>
                    </td>
                </tr>
            `).join('');
        } else {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="4" class="text-center py-4">
                    <i class="fas fa-info-circle me-2"></i>
                    Không có dữ liệu điểm danh
                </td>
            `;
            tbody.appendChild(row);
        }
    }

    getStatusBadgeClass(record) {
        if (!record.check_in) return 'bg-danger';
        if (!record.check_out) return 'bg-warning';
        return 'bg-success';
    }

    getStatusText(record) {
        if (!record.check_in) return 'Vắng mặt';
        if (!record.check_out) return 'Đang làm việc';
        return 'Hoàn thành';
    }

    // Add new method for viewing employee details
    async viewEmployeeDetails(employeeId) {
        try {
            const response = await fetch(`/employees/${employeeId}`);
            if (!response.ok) throw new Error('Failed to fetch employee details');
            
            const employee = await response.json();
            
            // Update modal content
            document.getElementById('employeeFullName').textContent = employee.full_name;
            document.getElementById('employeeId').textContent = employee.id;
            document.getElementById('employeeBirthDate').textContent = formatDate(employee.birth_date);
            document.getElementById('employeeEmail').textContent = employee.email;
            document.getElementById('employeePhone').textContent = employee.phone;
            document.getElementById('employeeAddress').textContent = employee.address;
            document.getElementById('employeeGender').textContent = employee.gender;
            document.getElementById('employeePosition').textContent = employee.position;
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('employeeDetailsModal'));
            modal.show();
        } catch (error) {
            console.error('Error loading employee details:', error);
            alert('Không thể tải thông tin chi tiết nhân viên');
        }
    }
}

// Initialize
const employeeManager = new EmployeeManager();

// Update datetime every second
setInterval(updateDateTime, 1000);
updateDateTime();