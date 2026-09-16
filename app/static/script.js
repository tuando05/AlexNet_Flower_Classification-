document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const dropZoneContent = document.getElementById('dropZoneContent');
    const previewWrapper = document.getElementById('previewWrapper');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const predictBtn = document.getElementById('predictBtn');
    const spinner = document.getElementById('spinner');
    const btnText = predictBtn.querySelector('.btn-text');

    const emptyState = document.getElementById('emptyState');
    const resultsContent = document.getElementById('resultsContent');
    const winnerNameVi = document.getElementById('winnerNameVi');
    const winnerNameEn = document.getElementById('winnerNameEn');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const flowerIcon = document.getElementById('flowerIcon');
    const probList = document.getElementById('probList');

    const sampleCards = document.querySelectorAll('.sample-card');

    let selectedFile = null;

    // Flower Icon Map
    const flowerIcons = {
        'rose': '🌹',
        'daisy': '🌼',
        'lily': '🪷'
    };

    const fillClasses = {
        'rose': 'rose-fill',
        'daisy': 'daisy-fill',
        'lily': 'lily-fill'
    };

    // Drag & Drop Event Listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFileInput();
    });

    // Handle Image Selection
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn tệp định dạng hình ảnh!');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZoneContent.classList.add('hidden');
            previewWrapper.classList.remove('hidden');
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetFileInput() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewWrapper.classList.add('hidden');
        dropZoneContent.classList.remove('hidden');
        predictBtn.disabled = true;
        emptyState.classList.remove('hidden');
        resultsContent.classList.add('hidden');
    }

    // Sample Image Presets
    sampleCards.forEach(card => {
        card.addEventListener('click', async () => {
            const sampleType = card.getAttribute('data-sample');
            createSampleCanvas(sampleType);
        });
    });

    function createSampleCanvas(type) {
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 300;
        const ctx = canvas.getContext('2d');

        // Draw colorful floral sample background
        let gradient;
        if (type === 'rose') {
            gradient = ctx.createRadialGradient(150, 150, 20, 150, 150, 150);
            gradient.addColorStop(0, '#F43F5E');
            gradient.addColorStop(1, '#881337');
        } else if (type === 'daisy') {
            gradient = ctx.createRadialGradient(150, 150, 20, 150, 150, 150);
            gradient.addColorStop(0, '#FDE047');
            gradient.addColorStop(1, '#78350F');
        } else {
            gradient = ctx.createRadialGradient(150, 150, 20, 150, 150, 150);
            gradient.addColorStop(0, '#34D399');
            gradient.addColorStop(1, '#064E3B');
        }

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 300, 300);

        // Draw emoji icon in center
        ctx.font = '100px serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(flowerIcons[type] || '🌸', 150, 150);

        canvas.toBlob((blob) => {
            const file = new File([blob], `sample_${type}.png`, { type: 'image/png' });
            handleFileSelect(file);
        });
    }

    // Handle Prediction Request
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Loading state
        predictBtn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Lỗi máy chủ khi xử lý phân loại ảnh.');
            }

            const result = await response.json();

            if (result.success) {
                renderResults(result.data);
            } else {
                alert('Không thể nhận kết quả phân loại từ máy chủ.');
            }
        } catch (error) {
            console.error(error);
            alert(`Lỗi: ${error.message}`);
        } finally {
            predictBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    });

    // Render Prediction Results to UI
    function renderResults(data) {
        emptyState.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        // Render Winner Info
        const winnerKey = data.prediction;
        winnerNameVi.textContent = data.prediction_vi;
        winnerNameEn.textContent = `${data.prediction.toUpperCase()} (Class: ${winnerKey})`;
        confidenceBadge.textContent = `${data.confidence_percentage}% Confidence`;
        flowerIcon.textContent = flowerIcons[winnerKey] || '🌸';

        // Render Probability List Bars
        probList.innerHTML = '';

        const sortedProbs = Object.values(data.probabilities).sort((a, b) => b.percentage - a.percentage);

        sortedProbs.forEach(item => {
            const probItem = document.createElement('div');
            probItem.className = 'prob-item';

            const fillClass = fillClasses[item.name_en] || 'rose-fill';

            probItem.innerHTML = `
                <div class="prob-header">
                    <span>${flowerIcons[item.name_en] || '🌸'} ${item.name_vi} (${item.name_en})</span>
                    <span>${item.percentage}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill ${fillClass}" style="width: 0%;"></div>
                </div>
            `;

            probList.appendChild(probItem);

            // Animate bar width
            setTimeout(() => {
                const barFill = probItem.querySelector('.prob-bar-fill');
                barFill.style.width = `${item.percentage}%`;
            }, 100);
        });
    }
});
