// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    // Toggle sidebar on hamburger click
    const hamburger = document.getElementById('hamburger');
    const sidebar = document.getElementById('sidebar');
    
    if (hamburger) {
        hamburger.addEventListener('click', function() {
            sidebar.classList.toggle('active');
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        if (window.innerWidth <= 1023 && 
            !sidebar.contains(event.target) && 
            !hamburger.contains(event.target)) {
            sidebar.classList.remove('active');
        }
    });

    // Update layout on window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 1023) {
            sidebar.classList.remove('active');
        }
    });

    // Tab functionality
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Update active tab content
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === tabName) {
                    content.classList.add('active');
                }
            });
        });
    });

    // Initialize first tab as active if exists
    const firstTab = document.querySelector('.tab');
    if (firstTab) {
        firstTab.click();
    }
});

// File upload functionality
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const uploadText = document.getElementById('uploadText');

    if (!uploadArea || !fileInput) return;

    // Click on upload area triggers file input
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFileSelection(this.files[0]);
        }
    });

    function handleFileSelection(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Please select a valid image file (JPEG, PNG, or GIF).', 'error');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showAlert('File size must be less than 10MB.', 'error');
            return;
        }

        // Show loading state
        uploadButton.disabled = true;
        uploadText.textContent = 'Processing...';
        loadingSpinner.style.display = 'inline-block';

        // Create FormData and send to server
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Store results and redirect to results page
            localStorage.setItem('lastScanResult', JSON.stringify(data));
            window.location.href = '/results';
        })
        .catch(error => {
            showAlert('Error uploading file: ' + error.message, 'error');
        })
        .finally(() => {
            // Reset loading state
            uploadButton.disabled = false;
            uploadText.textContent = 'Drag image here or click to upload';
            loadingSpinner.style.display = 'none';
            fileInput.value = '';
        });
    }
}

// Alert functionality
function showAlert(message, type = 'error') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    // Insert at the top of main content
    const mainContent = document.querySelector('.main-content');
    mainContent.insertBefore(alertDiv, mainContent.firstChild);
    
    // Remove alert after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    setupFileUpload();
});