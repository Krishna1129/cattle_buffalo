document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const previewSection = document.getElementById('preview-section');
    const previewImage = document.getElementById('preview-image');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const breedSection = document.getElementById('breed-section');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Handle click to upload
    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file input change
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                uploadFile(file);
                displayPreview(file);
            } else {
                alert('Please upload an image file');
            }
        }
    }

    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewSection.classList.remove('d-none');
            loading.classList.remove('d-none');
            results.classList.add('d-none');
        }
        reader.readAsDataURL(file);
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        loading.classList.remove('d-none');
        results.classList.add('d-none');

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            alert('Error: ' + error.message);
        })
        .finally(() => {
            loading.classList.add('d-none');
        });
    }

    function displayResults(data) {
        // Update cattle type and confidence
        document.getElementById('cattle-type').textContent = data.cattle_type;
        const cattleConfidence = document.getElementById('cattle-confidence');
        cattleConfidence.style.width = data.cattle_confidence;
        cattleConfidence.textContent = data.cattle_confidence;

        // Update breed information if available
        if (data.breed_result) {
            document.getElementById('breed-name').textContent = data.breed_result.name;
            const breedConfidence = document.getElementById('breed-confidence');
            breedConfidence.style.width = data.breed_result.confidence;
            breedConfidence.textContent = data.breed_result.confidence;
            breedSection.classList.remove('d-none');
        } else {
            breedSection.classList.add('d-none');
        }

        // Show results section
        results.classList.remove('d-none');
        
        // Animate results appearance
        results.classList.add('fade-in');
    }
});