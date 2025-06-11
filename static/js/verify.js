/**
 * Verification page functionality
 */

// Camera variables
let cameraActive = false;
let cameraFeed = null;
let fpsCounter = 0;
let lastFpsUpdate = Date.now();

// Camera resolution settings (target 960x720)
const targetWidth = 960;
const targetHeight = 720;

// Attendance functionality
let employeeSelect = document.getElementById('employeeSelect');
let checkInBtn = document.getElementById('checkInBtn');
let checkOutBtn = document.getElementById('checkOutBtn');
let attendanceTableBody = document.getElementById('attendanceTableBody');

// Initialize verification page
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    cameraFeed = document.getElementById('videoFeed');
    const toggleCameraBtn = document.getElementById('toggleCamera');
    
    // Add event listeners
    toggleCameraBtn.addEventListener('click', toggleCamera);
    
    // Update FPS counter every second
    setInterval(updateFps, 1000);
    
    // Check camera status on load
    checkCameraStatus();

    // Add resize handler to maintain aspect ratio on mobile
    window.addEventListener('resize', adjustCameraHeight);
    adjustCameraHeight();
    
    // Initialize attendance functionality
    loadEmployees();
    loadTodayAttendance();
    
    // Refresh attendance table every minute
    setInterval(loadTodayAttendance, 60000);
});

// Load employees into select dropdown
async function loadEmployees() {
    try {
        const response = await fetch('/employees');
        const employees = await response.json();
        
        employeeSelect.innerHTML = '<option value="">-- Chọn nhân viên --</option>';
        employees.forEach(employee => {
            const option = document.createElement('option');
            option.value = employee.id;
            option.textContent = employee.full_name;
            employeeSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading employees:', error);
    }
}

// Load today's attendance
async function loadTodayAttendance() {
    try {
        const today = new Date().toISOString().split('T')[0];
        const response = await fetch(`/api/attendance?date=${today}`);
        const result = await response.json();
        
        if (result.success) {
            displayAttendance(result.data);
        } else {
            console.error('Error loading attendance:', result.message);
        }
    } catch (error) {
        console.error('Error loading attendance:', error);
    }
}

// Display attendance records in table
function displayAttendance(records) {
    const attendanceTableBody = document.getElementById('attendanceTableBody');
    attendanceTableBody.innerHTML = '';
    
    if (records.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="3" class="text-center">Chưa có dữ liệu điểm danh hôm nay</td>
        `;
        attendanceTableBody.appendChild(row);
        return;
    }
    
    // The backend's /api/attendance?date=today endpoint already returns records
    // with check_in and check_out directly available for each employee.
    // So, we can directly iterate and display them.
    records.forEach(record => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${record.employee_name}</td>
            <td>${record.check_in || '-'}</td>
            <td>${record.check_out || '-'}</td>
        `;
        attendanceTableBody.appendChild(row);
    });
}

// Record attendance
async function recordAttendance(employeeId, checkType) {
    try {
        const response = await fetch(`/api/attendance/record?employee_id=${employeeId}&check_type=${checkType}`, {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
            // Reload attendance table
            await loadTodayAttendance();
            // Show success message
            alert(`Điểm danh ${checkType === 'in' ? 'check-in' : 'check-out'} thành công!`);
        } else {
            alert(result.message || 'Có lỗi xảy ra khi điểm danh');
        }
    } catch (error) {
        console.error('Error recording attendance:', error);
        alert('Có lỗi xảy ra khi điểm danh');
    }
}

// Add event listeners for manual attendance
checkInBtn.addEventListener('click', () => {
    const employeeId = employeeSelect.value;
    if (!employeeId) {
        alert('Vui lòng chọn nhân viên');
        return;
    }
    recordAttendance(employeeId, 'in');
});

checkOutBtn.addEventListener('click', () => {
    const employeeId = employeeSelect.value;
    if (!employeeId) {
        alert('Vui lòng chọn nhân viên');
        return;
    }
    recordAttendance(employeeId, 'out');
});

// Check camera status
function checkCameraStatus() {
    fetch('/camera/status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'on') {
                cameraActive = true;
                updateCameraUI(true);
                startCapturing();
            }
        })
        .catch(error => {
            console.error('Error checking camera status:', error);
        });
}

// Toggle camera on/off
function toggleCamera() {
    const toggleCameraBtn = document.getElementById('toggleCamera');
    
    // Toggle camera via API
    fetch('/camera/toggle', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'on') {
                cameraActive = true;
                updateCameraUI(true);
                startCapturing();
            } else {
                cameraActive = false;
                updateCameraUI(false);
            }
        })
        .catch(error => {
            console.error('Error toggling camera:', error);
            cameraActive = false;
            updateCameraUI(false);
        });
}

// Update camera UI based on status
function updateCameraUI(isActive) {
    const toggleCameraBtn = document.getElementById('toggleCamera');
    
    if (isActive) {
        toggleCameraBtn.innerHTML = '<i class="fas fa-camera"></i> Tắt Camera';
    } else {
        toggleCameraBtn.innerHTML = '<i class="fas fa-camera"></i> Bật Camera';
        if (cameraFeed.src) {
            cameraFeed.src = '';
        }
    }
}

// Start capturing frames
function startCapturing() {
    const timestamp = new Date().getTime();
    cameraFeed.src = `/video_feed?timestamp=${timestamp}`;
    
    cameraFeed.onload = function() {
        fpsCounter++;
        adjustCameraHeight();
    };
}

// Update FPS counter
function updateFps() {
    if (!cameraActive) return;
    
    const now = Date.now();
    const elapsed = (now - lastFpsUpdate) / 1000;
    const fps = Math.round(fpsCounter / elapsed);
    
    // Reset counter
    fpsCounter = 0;
    lastFpsUpdate = now;
}

// Adjust camera height to maintain aspect ratio
function adjustCameraHeight() {
    const cameraFrame = document.querySelector('.camera-frame');
    if (window.innerWidth <= 992) {
        const width = cameraFrame.offsetWidth;
        const height = width * (720 / 960); // Maintain 960x720 aspect ratio
        cameraFrame.style.height = height + 'px';
    } else {
        cameraFrame.style.height = '720px';
    }
} 