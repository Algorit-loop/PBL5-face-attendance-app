/**
 * Common functionality for all pages
 */

// Toast notification
function showToast(message, type = 'info', duration = 3000) {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => toast.remove());
    
    // Create new toast
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = message;
    
    // Add to document
    document.body.appendChild(toast);
    
    // Show toast
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // Hide toast after duration
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, duration);
}

// Format date for display
function formatDate(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN', { 
        year: 'numeric', 
        month: 'numeric', 
        day: 'numeric' 
    });
}

// Update date and time display
function updateDateTime() {
    const dateTimeElements = document.querySelectorAll('.date-time');
    if (dateTimeElements.length === 0) return;
    
    const now = new Date();
    const dateOptions = { weekday: 'short', year: 'numeric', month: 'numeric', day: 'numeric' };
    const timeOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    
    const dateStr = now.toLocaleDateString('vi-VN', dateOptions);
    const timeStr = now.toLocaleTimeString('vi-VN', timeOptions);
    
    dateTimeElements.forEach(element => {
        element.innerHTML = `${dateStr} | <span class="fw-bold">${timeStr}</span>`;
    });
}

// Initialize date time and update every second
document.addEventListener('DOMContentLoaded', function() {
    updateDateTime();
    setInterval(updateDateTime, 1000);
    
    // Set active link in navbar
    const currentPage = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (currentPage.includes(href) && href !== '#') {
            link.classList.add('active');
        }
    });
}); 