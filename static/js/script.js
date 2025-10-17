document.getElementById('predictBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) return alert("Please upload an image.");

    const formData = new FormData();
    formData.append('file', file);

    // Show preview with subtle animation
    const preview = document.getElementById('preview');
    const reader = new FileReader();
    reader.onload = () => {
        preview.src = reader.result;
        preview.classList.remove('show');
        setTimeout(() => preview.classList.add('show'), 50);
    };
    reader.readAsDataURL(file);

    // Send file to backend
    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        // Update prediction and confidence
        document.getElementById('prediction').innerText = data.prediction;
        document.getElementById('confidence').innerText = data.confidence ;

        // Update table dynamically with animation
        const tableBody = document.querySelector('#treatmentTable tbody');
        tableBody.innerHTML = `
            <tr class="highlight">
                <td>${data.prediction}</td>
                <td>${data.treatment}</td>
            </tr>
        `;
        setTimeout(() => {
            document.querySelector('#treatmentTable tbody tr').classList.remove('highlight');
        }, 1200);

    } catch (err) {
        alert("Prediction failed: " + err);
    }
});
