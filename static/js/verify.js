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

// Initialize verification page
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    cameraFeed = document.getElementById('camera-feed');
    const toggleCameraBtn = document.getElementById('toggleCamera');
    const cameraSpinner = document.getElementById('cameraSpinner');
    
    // Add event listeners
    toggleCameraBtn.addEventListener('click', toggleCamera);
    
    // Update FPS counter every second
    setInterval(updateFps, 1000);
    
    // Check for verification status updates
    setInterval(checkVerificationStatus, 2000);
});

// Toggle camera on/off
function toggleCamera() {
    const toggleCameraBtn = document.getElementById('toggleCamera');
    const toggleCameraText = document.getElementById('toggleCameraText');
    const cameraSpinner = document.getElementById('cameraSpinner');
    const cameraStatus = document.getElementById('cameraStatus');
    
    if (!cameraActive) {
        // Turn camera on
        toggleCameraText.textContent = 'Tắt';
        cameraStatus.textContent = 'Đang khởi động';
        cameraStatus.style.color = '#ffc107'; // warning color
        cameraSpinner.style.display = 'block';
        
        // Start video stream
        fetch('/start_camera', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    cameraActive = true;
                    cameraStatus.textContent = 'Đang hoạt động';
                    cameraStatus.style.color = '#28a745'; // success color
                    
                    // Start capturing frames
                    startCapturing();
                } else {
                    showToast('Không thể khởi động camera: ' + data.error, 'error');
                    resetCameraUI();
                }
            })
            .catch(error => {
                showToast('Lỗi kết nối: ' + error.message, 'error');
                resetCameraUI();
            })
            .finally(() => {
                cameraSpinner.style.display = 'none';
            });
    } else {
        // Turn camera off
        cameraActive = false;
        resetCameraUI();
        
        fetch('/stop_camera', { method: 'POST' })
            .catch(error => {
                console.error('Error stopping camera:', error);
            });
    }
}

// Reset camera UI to default state
function resetCameraUI() {
    const toggleCameraText = document.getElementById('toggleCameraText');
    const cameraStatus = document.getElementById('cameraStatus');
    const resolutionDisplay = document.getElementById('resolution');
    const fpsDisplay = document.getElementById('fps');
    
    toggleCameraText.textContent = 'Bật';
    cameraStatus.textContent = 'Đang tắt';
    cameraStatus.style.color = '#dc3545'; // danger color
    resolutionDisplay.textContent = 'Chưa khởi động';
    fpsDisplay.textContent = 'Chưa khởi động';
    
    // Clear the camera feed
    if (cameraFeed.src && cameraFeed.src !== '') {
        cameraFeed.src = '';
    }
}

// Start capturing frames
function startCapturing() {
    // Update camera feed with timestamp to prevent caching
    cameraFeed.src = '/video_feed?timestamp=' + new Date().getTime();
    
    // Set resolution info
    const resolutionDisplay = document.getElementById('resolution');
    resolutionDisplay.textContent = `${targetWidth}x${targetHeight}`;
    
    // Listen for frame updates to count FPS
    cameraFeed.onload = function() {
        fpsCounter++;
    };
}

// Update FPS counter
function updateFps() {
    if (!cameraActive) return;
    
    const now = Date.now();
    const elapsed = (now - lastFpsUpdate) / 1000;
    const fps = Math.round(fpsCounter / elapsed);
    
    const fpsDisplay = document.getElementById('fps');
    fpsDisplay.textContent = `${fps}`;
    
    // Reset counter
    fpsCounter = 0;
    lastFpsUpdate = now;
}

// Check for verification status updates
function checkVerificationStatus() {
    if (!cameraActive) return;
    
    fetch('/verification_status')
        .then(response => response.json())
        .then(data => {
            updateVerificationUI(data);
        })
        .catch(error => {
            console.error('Error checking verification status:', error);
        });
}

// Update verification UI based on status data
function updateVerificationUI(data) {
    const statusMessage = document.getElementById('statusMessage');
    const employeeInfo = document.getElementById('employeeInfo');
    
    if (data.verified) {
        statusMessage.textContent = 'Nhân viên đã được xác nhận';
        statusMessage.style.color = '#28a745'; // success color
        
        // Display employee info
        if (data.employee) {
            employeeInfo.innerHTML = `
                <p><strong>ID:</strong> ${data.employee.id}</p>
                <p><strong>Họ tên:</strong> ${data.employee.name}</p>
                <p><strong>Thời gian:</strong> ${new Date().toLocaleTimeString('vi-VN')}</p>
            `;
        }
    } else if (data.processing) {
        statusMessage.textContent = 'Đang xử lý...';
        statusMessage.style.color = '#ffc107'; // warning color
        employeeInfo.innerHTML = '';
    } else {
        statusMessage.textContent = 'Chờ nhận diện nhân viên';
        statusMessage.style.color = '#6c757d'; // secondary color
        employeeInfo.innerHTML = '';
    }
} 