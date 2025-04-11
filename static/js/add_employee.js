let isScanning = false;
let cameraStarted = false;
const maxFrames = 10;

document.addEventListener('DOMContentLoaded', function() {
    const startScanBtn = document.getElementById('startScanBtn');
    const cameraFeed = document.getElementById('cameraFeed');
    const cameraPlaceholder = document.getElementById('cameraPlaceholder');
    const scanProgress = document.getElementById('scanProgress');
    const employeeForm = document.getElementById('employeeForm');
    const scanSection = document.getElementById('scanSection');
    
    // Start scanning button click handler
    startScanBtn.addEventListener('click', async function() {
        try {
            // Turn on camera
            const response = await fetch('/camera/toggle', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();
            
            if (data.status === 'on') {
                cameraStarted = true;
                
                // Generate temporary ID for the scanning process
                const tempId = Date.now().toString();
                
                // Start face scanning
                const scanResponse = await fetch(`/face-scan/start/${tempId}`, {
                    method: 'POST'
                });
                
                if (scanResponse.ok) {
                    isScanning = true;
                    startScanBtn.style.display = 'none';
                    cameraPlaceholder.style.display = 'none';
                    cameraFeed.style.display = 'block';
                    scanProgress.style.display = 'block';
                    
                    // Start video feed
                    cameraFeed.src = '/scan_video_feed';
                    
                    // Start progress monitoring
                    updateScanningStatus();
                }
            } else {
                throw new Error('Failed to start camera');
            }
        } catch (error) {
            showToast('Không thể khởi động camera. Vui lòng thử lại.', 'error');
        }
    });
    
    // Function to show toast messages
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type} show`;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
    
    // Function to update scanning status
    async function updateScanningStatus() {
        if (!isScanning) return;
        
        try {
            const response = await fetch('/face-scan/status');
            const status = await response.json();
            
            if (status.scanning) {
                // Update direction and frame count
                document.getElementById('scanDirection').textContent = `Hướng: ${status.current_direction}`;
                document.getElementById('frameCount').textContent = 
                    `Frames: ${status.frames_captured}/${status.max_frames}`;
                
                // Update progress bar
                const progress = (status.frames_captured / status.max_frames) * 100;
                document.querySelector('.progress-bar').style.width = `${progress}%`;
                
                // Update per-direction progress
                Object.entries(status.direction_progress).forEach(([direction, count]) => {
                    const element = document.getElementById(`${direction}Progress`);
                    if (element) {
                        element.textContent = `${count}/${status.max_frames}`;
                        const statusBox = element.closest('.direction-status');
                        if (count === status.max_frames) {
                            statusBox.classList.add('completed');
                        }
                    }
                });
            }
            
            // Check for scan completion regardless of scanning status
            if (status.scan_complete) {
                isScanning = false;
                
                // Show success message
                showToast('Quét khuôn mặt thành công!', 'success');
                
                // Turn off camera
                await fetch('/camera/toggle', { method: 'POST' });
                cameraStarted = false;
                
                // Smooth transition to employee form
                scanSection.style.opacity = '0';
                setTimeout(() => {
                    scanSection.style.display = 'none';
                    employeeForm.style.display = 'block';
                    setTimeout(() => {
                        employeeForm.style.opacity = '1';
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                    }, 50);
                }, 500);
            } else if (isScanning) {
                // Continue updating status
                setTimeout(updateScanningStatus, 500);
            }
        } catch (error) {
            console.error('Error updating scan status:', error);
            showToast('Lỗi khi cập nhật trạng thái quét', 'error');
        }
    }
    
    // Handle employee form submission
    const addEmployeeForm = document.getElementById('addEmployeeForm');
    if (addEmployeeForm) {
        addEmployeeForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const employeeData = {
                full_name: document.getElementById('fullName').value,
                birth_date: document.getElementById('birthDate').value,
                email: document.getElementById('email').value,
                phone: document.getElementById('phone').value,
                address: document.getElementById('address').value,
                gender: document.getElementById('gender').value,
                position: document.getElementById('position').value
            };
            
            try {
                const response = await fetch('/employees', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(employeeData)
                });
                
                if (response.ok) {
                    showToast('Thêm nhân viên thành công!', 'success');
                    setTimeout(() => {
                        window.location.href = '/static/pages/employees.html';
                    }, 1500);
                } else {
                    const result = await response.json();
                    showToast(result.detail || 'Không thể thêm nhân viên', 'error');
                }
            } catch (error) {
                console.error('Error adding employee:', error);
                showToast('Có lỗi xảy ra khi thêm nhân viên', 'error');
            }
        });
    }
});