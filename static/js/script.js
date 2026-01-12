// ==============================
// Navigation & UI functionality
// ==============================
document.addEventListener('DOMContentLoaded', function () {
    const hamburger = document.getElementById('hamburger');
    const sidebar = document.getElementById('sidebar');

    if (hamburger && sidebar) {
        hamburger.addEventListener('click', function (e) {
            e.stopPropagation();
            sidebar.classList.toggle('active');
        });
    }

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

    window.addEventListener('resize', function () {
        if (window.innerWidth > 1023 && sidebar) {
            sidebar.classList.remove('active');
        }
    });

    setupFileUpload();
});

// ==============================
// File Upload (SINGLE SOURCE)
// ==============================
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const uploadText = document.getElementById('uploadText');
    const languageSelect = document.getElementById('globalLanguage');

    if (!uploadArea || !fileInput || !uploadButton) return;

    let isUploading = false;
    let pickerLocked = false;

    // Button opens picker
    uploadButton.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();

        if (isUploading || pickerLocked) return;

        pickerLocked = true;
        fileInput.click();
    });

    // Picker closed / file selected
    fileInput.addEventListener('change', function () {
        pickerLocked = false;

        if (this.files.length && !isUploading) {
            handleFile(this.files[0]);
        }
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', e => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files.length && !isUploading) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    async function handleFile(file) {
        isUploading = true;

        const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Invalid image format');
            reset();
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showAlert('Image must be under 10MB');
            reset();
            return;
        }

        uploadButton.disabled = true;
        if (uploadText) uploadText.textContent = 'Processing...';
        if (loadingSpinner) loadingSpinner.style.display = 'inline-block';

        const formData = new FormData();
        formData.append('file', file);
        formData.append(
            'language',
            languageSelect ? languageSelect.value : 'en'
        );

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            localStorage.setItem('lastScanResult', JSON.stringify(data));
            window.location.href = '/results';

        } catch (err) {
            showAlert(err.message);
        } finally {
            reset();
        }
    }

    function reset() {
        isUploading = false;
        pickerLocked = false;
        uploadButton.disabled = false;
        if (uploadText) uploadText.textContent = 'Drag image here or click to upload';
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        fileInput.value = '';
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

    setTimeout(() => alertDiv.remove(), 5000);
}
