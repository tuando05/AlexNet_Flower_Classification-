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
    const modelWarning = document.getElementById('modelWarning');

    const sampleCards = document.querySelectorAll('.sample-card');

    let selectedFile = null;

    // Giới hạn dung lượng upload, sinh ra từ template (config.MAX_UPLOAD_SIZE_MB)
    const MAX_UPLOAD_MB = parseFloat(dropZone.dataset.maxUploadMb || '10');
    const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

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

    // Hiển thị cảnh báo nếu server báo model chưa được huấn luyện
    function setModelWarning(isTrained) {
        if (!modelWarning) return;
        modelWarning.classList.toggle('hidden', isTrained);
    }

    function showError(message) {
        alert(`Lỗi: ${message}`);
    }

    // Cảnh báo ngay khi tải trang nếu server báo chưa có model đã huấn luyện
    setModelWarning(document.body.dataset.modelTrained !== 'false');

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
            showError('Vui lòng chọn tệp định dạng hình ảnh!');
            return;
        }

        if (file.size > MAX_UPLOAD_BYTES) {
            showError(`Ảnh ${(file.size / 1024 / 1024).toFixed(1)}MB vượt quá giới hạn ${MAX_UPLOAD_MB}MB. Vui lòng chọn ảnh nhỏ hơn.`);
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

    // Ảnh mẫu thật (server phục vụ tại /samples/*.jpg)
    // Trước đây UI tự vẽ gradient + emoji bằng canvas nên kết quả dự đoán vô nghĩa.
    sampleCards.forEach(card => {
        card.addEventListener('click', async () => {
            const url = card.getAttribute('data-url');
            if (!url) return;

            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`Không tải được ảnh mẫu (HTTP ${response.status}).`);
                }
                const blob = await response.blob();
                const filename = url.split('/').pop();
                handleFileSelect(new File([blob], filename, { type: blob.type || 'image/jpeg' }));
            } catch (error) {
                console.error(error);
                showError(error.message);
            }
        });
    });

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

            const payload = await response.json().catch(() => null);

            if (!response.ok) {
                // FastAPI trả chi tiết lỗi trong trường 'detail'
                const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
                throw new Error(detail);
            }

            if (payload && payload.success) {
                renderResults(payload.data);
            } else {
                throw new Error('Máy chủ trả về kết quả không hợp lệ.');
            }
        } catch (error) {
            console.error(error);
            showError(error.message);
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
        winnerNameEn.textContent = `Class ID: ${winnerKey}`;
        confidenceBadge.textContent = `${data.confidence_percentage}% Confidence`;
        flowerIcon.textContent = flowerIcons[winnerKey] || '🌸';

        // Cảnh báo nếu server báo model chưa được huấn luyện
        setModelWarning(data.model_trained !== false);

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
