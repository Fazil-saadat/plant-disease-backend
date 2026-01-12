// ==============================
// Navigation & UI functionality
// ==============================
document.addEventListener('DOMContentLoaded', function () {
    const hamburger = document.getElementById('hamburger');
    const sidebar = document.getElementById('sidebar');

    // Toggle sidebar
    if (hamburger && sidebar) {
        hamburger.addEventListener('click', function (e) {
            e.stopPropagation();
            sidebar.classList.toggle('active');
        });
    }

    // Close sidebar on outside click (mobile)
    document.addEventListener('click', function (event) {
        if (
            window.innerWidth <= 1023 &&
            sidebar &&
            hamburger &&
            !sidebar.contains(event.target) &&
            !hamburger.contains(event.target)
        ) {
            sidebar.classList.remove('active');
        }
    });

    // Reset sidebar on resize
    window.addEventListener('resize', function () {
        if (window.innerWidth > 1023 && sidebar) {
            sidebar.classList.remove('active');
        }
    });

    // Tabs
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', function () {
            const tabName = this.getAttribute('data-tab');

            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');

            tabContents.forEach(content => {
                content.classList.toggle('active', content.id === tabName);
            });
        });
    });

    if (tabs.length > 0) {
        tabs[0].click();
    }

    setupFileUpload();
});

// ==============================
// File Upload functionality
// ==============================
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const uploadText = document.getElementById('uploadText');

    if (!uploadArea || !fileInput || !uploadButton) return;

    let isUploading = false;

    // ONLY button opens file picker
    uploadButton.addEventListener('click', function (e) {
        e.preventDefault();
        if (!isUploading) fileInput.click();
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function () {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length && !isUploading) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File selected
    fileInput.addEventListener('change', function () {
        if (this.files.length && !isUploading) {
            handleFile(this.files[0]);
        }
    });

    async function handleFile(file) {
        isUploading = true;

        const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Invalid image format (JPEG, PNG, GIF only)');
            isUploading = false;
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showAlert('Image must be under 10MB');
            isUploading = false;
            return;
        }

        uploadButton.disabled = true;
        if (uploadText) uploadText.textContent = 'Processing...';
        if (loadingSpinner) loadingSpinner.style.display = 'inline-block';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            if (data.uploaded_image && !data.uploaded_image.startsWith('/')) {
                data.uploaded_image = '/uploads/' + data.uploaded_image;
            }

            localStorage.setItem('lastScanResult', JSON.stringify(data));
            window.location.href = '/results';

        } catch (error) {
            showAlert(error.message);
        } finally {
            uploadButton.disabled = false;
            if (uploadText) uploadText.textContent = 'Drag image here or click to upload';
            if (loadingSpinner) loadingSpinner.style.display = 'none';
            fileInput.value = '';
            isUploading = false;
        }
    }
}

// ==============================
// Alert helper
// ==============================
function showAlert(message, type = 'error') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    const mainContent = document.querySelector('.main-content') || document.body;
    mainContent.insertBefore(alertDiv, mainContent.firstChild);

    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}
