// const MODEL_PATH = './eff_b0_kfold_add_focal.onnx';
const MODEL_PATH = './eff_b0_kfold_add_focal_r2.onnx';
// const MODEL_PATH = './eff_b0_kfold_add_focal_r3_edit.onnx';
const IMAGE_SIZE = 640;
const CLASS_NAMES = ['Good', 'Out of Focus', 'Tracking Error', 'Over Saturated', 'No Star', 'Satellite'];
const LOW_CONFIDENCE_THRESHOLD = 0.6;
const MODEL_CONFIDENCE_WEIGHT = 0.75;
const RULE_CONFIDENCE_WEIGHT = 0.25;
const RULE_NEUTRAL_SCORE = 0.5;
const RULE_IMAGE_SIZE = 256;
const LINE_SCAN_ANGLES = [-70, -55, -40, -25, -10, 0, 10, 25, 40, 55, 70];

const upload = document.getElementById('upload');
const dropZone = document.getElementById('dropZone');
const runButton = document.getElementById('runButton');
const exportButton = document.getElementById('exportButton');
const clearButton = document.getElementById('clearButton');
const resultsGrid = document.getElementById('resultsGrid');
const modelStatus = document.getElementById('modelStatus');
const queueStatus = document.getElementById('queueStatus');
const processedCount = document.getElementById('processedCount');
const totalImages = document.getElementById('totalImages');
const avgConfidence = document.getElementById('avgConfidence');
const avgLatency = document.getElementById('avgLatency');
const dominantClass = document.getElementById('dominantClass');
const classDistribution = document.getElementById('classDistribution');
const cardTemplate = document.getElementById('resultCardTemplate');
const filterChips = document.getElementById('filterChips');
const filteredCount = document.getElementById('filteredCount');
const sortSelect = document.getElementById('sortSelect');
const imageModal = document.getElementById('imageModal');
const modalImage = document.getElementById('modalImage');
const modalTitle = document.getElementById('modalTitle');
const modalCloseButton = document.getElementById('modalCloseButton');
const modalPrevButton = document.getElementById('modalPrevButton');
const modalNextButton = document.getElementById('modalNextButton');
const modalTopClass = document.getElementById('modalTopClass');
const modalTopConfidence = document.getElementById('modalTopConfidence');
const modalMeta = document.getElementById('modalMeta');
const modalRankList = document.getElementById('modalRankList');
const modalRuleVerdictBox = document.getElementById('modalRuleVerdictBox');
const modalRuleClassLabel = document.getElementById('modalRuleClassLabel');
const modalRuleBadge = document.getElementById('modalRuleBadge');
const modalRuleDesc = document.getElementById('modalRuleDesc');
const modalPhysicsSection = document.getElementById('modalPhysicsSection');
const modalPhysicsGrid = document.getElementById('modalPhysicsGrid');

let sessionPromise;
let queuedItems = [];
let inferenceResults = [];
let activeFilter = { type: 'all', value: 'all' };
let isRunningInference = false;
let modalZoom = 1;
let modalPan = { x: 0, y: 0 };
let modalDrag = null;
let currentModalItem = null;

init();

function init() {
    preloadModel();
    updateFilterChips();
    applyResultFilter();
    upload.addEventListener('change', (event) => addFiles(event.target.files));
    runButton.addEventListener('click', runBatchInference);
    exportButton.addEventListener('click', exportClassifiedImages);
    clearButton.addEventListener('click', clearAll);
    sortSelect.addEventListener('change', applyResultFilter);
    filterChips.addEventListener('click', (event) => {
        const chip = event.target.closest('.filter-chip');
        if (!chip) return;
        activeFilter = {
            type: chip.dataset.filterType,
            value: chip.dataset.filterValue,
        };
        updateFilterChips();
        applyResultFilter();
    });
    modalCloseButton.addEventListener('click', closeImageModal);
    modalPrevButton.addEventListener('click', () => navigateModalImage(-1));
    modalNextButton.addEventListener('click', () => navigateModalImage(1));
    imageModal.addEventListener('click', (event) => {
        if (event.target.matches('[data-close-modal]')) {
            closeImageModal();
        }
    });
    modalImage.addEventListener('wheel', handleModalImageWheel, { passive: false });
    modalImage.addEventListener('pointerdown', handleModalImagePointerDown);
    modalImage.addEventListener('pointermove', handleModalImagePointerMove);
    modalImage.addEventListener('pointerup', endModalImageDrag);
    modalImage.addEventListener('pointercancel', endModalImageDrag);
    modalImage.addEventListener('dblclick', resetModalImageTransform);
    document.addEventListener('keydown', (event) => {
        if (!imageModal.classList.contains('is-open')) return;

        if (event.key === 'Escape') {
            closeImageModal();
            return;
        }

        if (event.key === 'ArrowLeft') {
            event.preventDefault();
            navigateModalImage(-1);
        }

        if (event.key === 'ArrowRight') {
            event.preventDefault();
            navigateModalImage(1);
        }
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add('is-dragging');
        });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove('is-dragging');
        });
    });

    dropZone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));
}

async function preloadModel() {
    try {
        modelStatus.textContent = 'Loading model';

        // 1. Check backend API status first
        try {
            const healthRes = await fetch('/api/health');
            if (healthRes.ok) {
                const healthData = await healthRes.json();
                if (healthData.onnx_model_loaded) {
                    modelStatus.textContent = 'Model ready';
                    modelStatus.classList.add('ready');
                    return;
                }
            }
        } catch {
            // Backend offline, fallback to in-browser ONNX
        }

        // 2. In-browser ONNX fallback
        const options = { 
            executionProviders: ['webgl', 'wasm'] 
        };
        sessionPromise = ort.InferenceSession.create(MODEL_PATH, options);
        await sessionPromise;

        modelStatus.textContent = 'Model ready';
        modelStatus.classList.add('ready');
    } catch (error) {
        modelStatus.textContent = 'Model load failed';
        modelStatus.classList.add('error');
        showGridMessage(`Could not load model: ${error.message}`);
    }
}

function addFiles(fileList) {
    const imageFiles = Array.from(fileList).filter((file) => file.type.startsWith('image/'));
    if (imageFiles.length === 0) return;

    if (resultsGrid.classList.contains('empty-state')) {
        resultsGrid.classList.remove('empty-state');
        resultsGrid.innerHTML = '';
    }

    const newItems = imageFiles.map((file) => {
        const item = {
            id: createItemId(),
            file,
            objectUrl: URL.createObjectURL(file),
            card: null,
            dimensions: null,
            prediction: null,
        };
        item.card = createResultCard(item);
        resultsGrid.appendChild(item.card);
        return item;
    });

    queuedItems = [...queuedItems, ...newItems];
    inferenceResults = [];
    activeFilter = { type: 'all', value: 'all' };
    updateControls();
    updateMetrics();
    updateFilterChips();
    applyResultFilter();
}

function createItemId() {
    if (window.crypto?.randomUUID) {
        return window.crypto.randomUUID();
    }

    return `image-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createResultCard(item) {
    const card = cardTemplate.content.firstElementChild.cloneNode(true);
    card.dataset.state = 'queued';
    const thumb = card.querySelector('.thumb');
    thumb.src = item.objectUrl;
    thumb.alt = item.file.name;
    thumb.addEventListener('load', () => {
        item.dimensions = {
            width: thumb.naturalWidth,
            height: thumb.naturalHeight,
        };
        card.querySelector('.dimensions').textContent = formatDimensions(item.dimensions);
    }, { once: true });
    card.querySelector('.thumb-wrap').addEventListener('click', () => openImageModal(item));
    card.querySelector('.remove-card-button').addEventListener('click', () => removeItem(item));
    card.querySelector('.file-name').textContent = item.file.name;
    card.querySelector('.size').textContent = formatBytes(item.file.size);
    return card;
}

function openImageModal(item) {
    currentModalItem = item;
    renderModalItem(item);
    imageModal.classList.add('is-open');
    imageModal.setAttribute('aria-hidden', 'false');
    modalCloseButton.focus();
}

function renderModalItem(item) {
    resetModalImageTransform();
    modalImage.src = item.objectUrl;
    modalImage.alt = item.file.name;
    modalTitle.textContent = item.file.name;
    renderModalPrediction(item);
    updateModalNavigationButtons();
}

function closeImageModal() {
    imageModal.classList.remove('is-open');
    imageModal.setAttribute('aria-hidden', 'true');
    currentModalItem = null;
    modalImage.removeAttribute('src');
    modalImage.alt = '';
    modalTitle.textContent = 'Image preview';
    modalTopClass.textContent = '-';
    modalTopConfidence.textContent = '-';
    modalMeta.textContent = '-';
    if (modalRuleVerdictBox) modalRuleVerdictBox.style.display = 'none';
    if (modalPhysicsSection) modalPhysicsSection.style.display = 'none';
    modalRankList.innerHTML = '<p>Run prediction to see all 6 quality groups.</p>';
    resetModalImageTransform();
}

function navigateModalImage(direction) {
    const visibleItems = getVisibleSortedItems();
    if (!currentModalItem || visibleItems.length <= 1) return;

    const currentIndex = visibleItems.findIndex((item) => item.id === currentModalItem.id);
    if (currentIndex === -1) return;

    const nextIndex = (currentIndex + direction + visibleItems.length) % visibleItems.length;
    currentModalItem = visibleItems[nextIndex];
    renderModalItem(currentModalItem);
}

function updateModalNavigationButtons() {
    const visibleItems = getVisibleSortedItems();
    const canNavigate = visibleItems.length > 1;
    modalPrevButton.disabled = !canNavigate;
    modalNextButton.disabled = !canNavigate;
}

async function runBatchInference() {
    if (queuedItems.length === 0) return;

    isRunningInference = true;
    runButton.disabled = true;
    runButton.textContent = 'Running...';
    inferenceResults = [];
    queuedItems.forEach((item) => {
        item.card.dataset.state = 'queued';
        delete item.card.dataset.predictedClass;
        delete item.card.dataset.lowConfidence;
        delete item.card.dataset.disagree;
        delete item.card.dataset.confidence;
        delete item.card.dataset.latency;
        item.card.classList.remove('low-confidence');
        item.card.querySelector('.latency').classList.remove('slow-latency');
        item.prediction = null;
    });
    activeFilter = { type: 'all', value: 'all' };
    updateFilterChips();
    applyResultFilter();

    try {
        const CLASS_MAP = {
            '01_Good': 'Good',
            '02_Out_of_Focus': 'Out of Focus',
            '03_Tracking_Error': 'Tracking Error',
            '04_Over_Saturated': 'Over Saturated',
            '05_No_Star': 'No Star',
            '06_Satellite': 'Satellite',
        };

        for (const item of [...queuedItems]) {
            if (!queuedItems.includes(item)) continue;
            setCardStatus(item.card, 'Processing', 'running');
            const start = performance.now();

            let result = null;

            // Try backend API first (sub-30ms CNN + Physics V2)
            try {
                const formData = new FormData();
                formData.append('file', item.file);
                const res = await fetch('/api/analyze', { method: 'POST', body: formData });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                const latency = performance.now() - start;

                ranked = Object.entries(data.cnn_model.probabilities_pct).map(([cname, probPct]) => {
                    const prob = probPct / 100.0;
                    const shortName = CLASS_MAP[cname] || cname;
                    return {
                        className: shortName,
                        probability: prob,
                        modelProbability: prob,
                        adjustedProbability: prob,
                    };
                }).sort((a, b) => b.modelProbability - a.modelProbability);

                const top = ranked[0];
                const ruleTop = CLASS_MAP[data.rule_based.top_class] || data.rule_based.top_class;
                const ruleTopScore = data.rule_based.top_score_pct / 100.0;
                const isMatch = (data.consensus === 'MATCH');

                // Commented out to eliminate redundant in-browser computation latency (results already computed by Python backend)
                // const imageElement = await loadImage(item.objectUrl);
                // const localRule = await analyzePredictionWithRules(imageElement, ranked[0]);

                const ruleAnalysis = {
                    available: true,
                    ruleTopClass: ruleTop,
                    ruleTopScore: ruleTopScore,
                    isMatch: isMatch,
                    consensus: data.consensus,
                    features: {
                        starCount: data.physics_metrics.star_count,
                        fwhm: data.physics_metrics.median_fwhm,
                        eccentricity: data.physics_metrics.median_eccentricity,
                        circularity: data.physics_metrics.mean_circularity,
                        hollowness: data.physics_metrics.mean_hollowness,
                        saturatedRatio: data.physics_metrics.saturated_ratio,
                        sharpness: data.physics_metrics.sharpness,
                        aspectRatio: data.physics_metrics.mean_aspect_ratio,
                        streakRatio: data.physics_metrics.max_streak_length_ratio,
                    },
                    ruleScore: ruleTopScore,
                    finalConfidence: top.modelProbability,
                    label: isMatch
                        ? 'Rule evidence supports this class'
                        : `Rule flag: ${ruleTop} (${(ruleTopScore * 100).toFixed(0)}%)`,
                };
                top.ruleScore = ruleTopScore;
                top.ruleLabel = ruleAnalysis.label;
                top.ruleTopClass = ruleTop;
                console.log(`[API SUCCESS] ${item.file.name}: Server processed in ${data.backend_latency_ms || latency.toFixed(0)} ms (Network total: ${latency.toFixed(0)} ms)`);

                const displayLatency = data.backend_latency_ms ? data.backend_latency_ms : latency;
                result = { item, ranked, latency: displayLatency, ruleAnalysis };
            } catch (err) {
                console.warn(`[API FAILED] Fallback to in-browser ONNX for ${item.file.name}:`, err);
                // In-browser ONNX fallback if backend is unavailable
                const session = await sessionPromise;
                const inputName = session.inputNames[0];
                const outputName = session.outputNames[0];

                const imageElement = await loadImage(item.objectUrl);
                const tensor = preprocessImage(imageElement);
                const outputs = await session.run({ [inputName]: tensor });
                const latency = performance.now() - start;
                const scores = Array.from(outputs[outputName].data);
                ranked = rankPredictions(scores);
                const ruleAnalysis = await analyzePredictionWithRules(imageElement, ranked[0]);
                applyRuleAdjustment(ranked, ruleAnalysis);
                const top = ranked[0];
                top.ruleTopClass = top.className;
                top.ruleTopScore = ruleAnalysis.ruleScore;
                top.isMatch = ruleAnalysis.ruleScore >= 0.72;
                ruleAnalysis.ruleTopClass = top.className;
                ruleAnalysis.ruleTopScore = ruleAnalysis.ruleScore;
                ruleAnalysis.isMatch = top.isMatch;
                result = { item, ranked, latency, ruleAnalysis };
            }

            inferenceResults.push(result);
            renderPrediction(result);
            updateLatencyHighlights();
            updateMetrics();
            updateFilterChips();
            applyResultFilter();
        }
    } catch (error) {
        showGridMessage(`Inference failed: ${error.message}`);
    } finally {
        isRunningInference = false;
        runButton.textContent = 'Run prediction';
        exportButton.textContent = 'Download .zip';
        updateControls();
    }
}

async function exportClassifiedImages() {
    const completedItems = getCompletedItems();
    if (completedItems.length === 0) return;

    exportButton.disabled = true;
    exportButton.textContent = 'Preparing .zip...';

    try {
        await downloadClassifiedImagesZip(completedItems);
    } catch (error) {
        alert(`Could not create zip file: ${error.message}`);
    } finally {
        exportButton.textContent = 'Download .zip';
        updateControls();
    }
}

async function downloadClassifiedImagesZip(completedItems) {
    if (!window.JSZip) {
        throw new Error('ZIP library is not available. Please check your internet connection and reload the page.');
    }

    const zip = new JSZip();
    CLASS_NAMES.forEach((className) => zip.folder(safeFolderName(className)));

    completedItems.forEach((item) => {
        const [top] = item.prediction.ranked;
        const folderName = safeFolderName(top.className);
        const fileName = uniqueExportFileName(item, completedItems);
        zip.folder(folderName).file(fileName, item.file);
    });

    zip.file('classification_manifest.csv', toCsv(buildManifestRows(completedItems)));
    const blob = await zip.generateAsync({ type: 'blob' });
    downloadBlob(blob, `astro-classified-${formatDateStamp(new Date())}.zip`);
}

function buildManifestRows(completedItems) {
    const rows = [
        [
            'file_name',
            'cnn_predicted_class',
            'cnn_confidence',
            'rule_predicted_class',
            'rule_confidence',
            'consensus',
            'rule_label',
            ...CLASS_NAMES.map((className) => `prob_${toColumnName(className)}`),
            'star_count',
            'fwhm_px',
            'eccentricity',
            'circularity',
            'hollowness',
            'saturated_ratio',
            'sharpness',
            'latency_ms',
            'width',
            'height',
            'size_bytes',
        ],
    ];

    completedItems.forEach((item) => {
        const { ranked, latency, ruleAnalysis } = item.prediction;
        const [top] = ranked;
        const confidenceByClass = Object.fromEntries(
            ranked.map((prediction) => [prediction.className, prediction.modelProbability]),
        );
        const features = ruleAnalysis?.features;
        rows.push([
            item.file.name,
            top.className,
            (top.modelProbability * 100).toFixed(2) + '%',
            ruleAnalysis?.ruleTopClass ?? '',
            ruleAnalysis?.ruleTopScore !== undefined ? (ruleAnalysis.ruleTopScore * 100).toFixed(1) + '%' : '',
            ruleAnalysis?.isMatch ? 'MATCH' : 'DISAGREE',
            ruleAnalysis?.label ?? '',
            ...CLASS_NAMES.map((className) => ((confidenceByClass[className] ?? 0) * 100).toFixed(2) + '%'),
            features?.starCount ?? '',
            features?.fwhm?.toFixed(2) ?? '',
            features?.eccentricity?.toFixed(3) ?? '',
            features?.circularity?.toFixed(3) ?? '',
            features?.hollowness?.toFixed(4) ?? '',
            features?.saturatedRatio !== undefined ? (features.saturatedRatio * 100).toFixed(3) + '%' : '',
            features?.sharpness?.toFixed(1) ?? '',
            latency.toFixed(0),
            item.dimensions?.width ?? '',
            item.dimensions?.height ?? '',
            item.file.size,
        ]);
    });

    return rows;
}

function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(imageElement, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

    const { data } = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const tensorData = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
    const mean = [0.353340744972229, 0.353340744972229, 0.353340744972229];
    const std = [0.13850915431976318, 0.13850915431976318, 0.13850915431976318];
    const channelSize = IMAGE_SIZE * IMAGE_SIZE;

    for (let i = 0; i < channelSize; i += 1) {
        const r = data[i * 4] / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;

        tensorData[i] = (r - mean[0]) / std[0];
        tensorData[i + channelSize] = (g - mean[1]) / std[1];
        tensorData[i + channelSize * 2] = (b - mean[2]) / std[2];
    }

    return new ort.Tensor('float32', tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function rankPredictions(scores) {
    const probabilities = softmax(scores);
    return probabilities
        .map((probability, index) => ({
            className: CLASS_NAMES[index] ?? `Class ${index}`,
            probability,
            modelProbability: probability,
            adjustedProbability: probability,
        }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 6);
}

async function analyzePredictionWithRules(imageElement, topPrediction) {
    let features = null;
    try {
        features = extractCanvasFeatures(imageElement);
    } catch {
        features = null;
    }
    const ruleScore = features
        ? scorePredictionRule(topPrediction.className, features)
        : RULE_NEUTRAL_SCORE;

    return {
        available: Boolean(features),
        features,
        ruleScore,
        finalConfidence: combineConfidence(topPrediction.modelProbability, ruleScore),
        label: describeRuleSupport(ruleScore, Boolean(features)),
    };
}

function extractCanvasFeatures(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = RULE_IMAGE_SIZE;
    canvas.height = RULE_IMAGE_SIZE;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(imageElement, 0, 0, RULE_IMAGE_SIZE, RULE_IMAGE_SIZE);

    const { data } = context.getImageData(0, 0, RULE_IMAGE_SIZE, RULE_IMAGE_SIZE);
    const gray = new Float32Array(RULE_IMAGE_SIZE * RULE_IMAGE_SIZE);
    let saturatedCount = 0;
    let highlightCount = 0;
    let total = 0;

    for (let i = 0, pixel = 0; i < data.length; i += 4, pixel += 1) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const value = r * 0.299 + g * 0.587 + b * 0.114;
        gray[pixel] = value;
        total += value;

        if (r >= 250 || g >= 250 || b >= 250) {
            saturatedCount += 1;
        }

        if (value >= 235) {
            highlightCount += 1;
        }
    }

    const mean = total / gray.length;
    let variance = 0;
    for (let i = 0; i < gray.length; i += 1) {
        variance += Math.pow(gray[i] - mean, 2);
    }

    const stdDev = Math.sqrt(variance / gray.length);
    const brightThreshold = clamp(mean + stdDev * 2.6, 150, 245);
    const lineThreshold = clamp(mean + stdDev * 1.65, 125, 235);
    const brightAreaThreshold = clamp(mean + stdDev * 1.25, 135, 225);
    let brightAreaCount = 0;
    for (let i = 0; i < gray.length; i += 1) {
        if (gray[i] >= brightAreaThreshold) {
            brightAreaCount += 1;
        }
    }

    const components = findBrightComponents(gray, RULE_IMAGE_SIZE, brightThreshold);
    const starComponents = components.filter((component) => (
        component.area >= 2
        && component.area <= 180
        && component.width <= 36
        && component.height <= 36
    ));
    const elongatedStars = starComponents.filter((star) => star.aspectRatio >= 2.4);

    return {
        sharpness: estimateSharpness(gray, RULE_IMAGE_SIZE),
        saturatedRatio: saturatedCount / gray.length,
        brightAreaRatio: brightAreaCount / gray.length,
        highlightRatio: highlightCount / gray.length,
        meanBrightness: mean / 255,
        starCount: starComponents.length,
        elongatedStarRatio: starComponents.length === 0 ? 0 : elongatedStars.length / starComponents.length,
        longLineCount: Math.max(
            estimateComponentLineCount(components, RULE_IMAGE_SIZE),
            estimateScannedLineCount(gray, RULE_IMAGE_SIZE, lineThreshold),
        ),
        brightThreshold,
        lineThreshold,
    };
}

function estimateSharpness(gray, size) {
    let total = 0;
    let totalSq = 0;
    let count = 0;

    for (let y = 1; y < size - 1; y += 1) {
        for (let x = 1; x < size - 1; x += 1) {
            const index = y * size + x;
            const laplacian = (
                gray[index - size]
                + gray[index - 1]
                - gray[index] * 4
                + gray[index + 1]
                + gray[index + size]
            );
            total += laplacian;
            totalSq += laplacian * laplacian;
            count += 1;
        }
    }

    const mean = total / count;
    return (totalSq / count) - (mean * mean);
}

function findBrightComponents(gray, size, threshold) {
    const visited = new Uint8Array(gray.length);
    const components = [];
    const stack = [];

    for (let start = 0; start < gray.length; start += 1) {
        if (visited[start] || gray[start] < threshold) continue;

        let area = 0;
        let minX = size;
        let maxX = 0;
        let minY = size;
        let maxY = 0;
        stack.push(start);
        visited[start] = 1;

        while (stack.length > 0) {
            const index = stack.pop();
            const x = index % size;
            const y = Math.floor(index / size);
            area += 1;
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);

            for (let dy = -1; dy <= 1; dy += 1) {
                for (let dx = -1; dx <= 1; dx += 1) {
                    if (dx === 0 && dy === 0) continue;
                    const nextX = x + dx;
                    const nextY = y + dy;
                    if (nextX < 0 || nextX >= size || nextY < 0 || nextY >= size) continue;

                    const nextIndex = nextY * size + nextX;
                    if (!visited[nextIndex] && gray[nextIndex] >= threshold) {
                        visited[nextIndex] = 1;
                        stack.push(nextIndex);
                    }
                }
            }
        }

        const width = maxX - minX + 1;
        const height = maxY - minY + 1;
        components.push({
            area,
            width,
            height,
            aspectRatio: Math.max(width, height) / Math.max(1, Math.min(width, height)),
        });
    }

    return components;
}

function estimateComponentLineCount(components, size) {
    const minLength = size * 0.18;
    return components.filter((component) => (
        component.area >= 24
        && Math.max(component.width, component.height) >= minLength
        && component.aspectRatio >= 4
    )).length;
}

function estimateScannedLineCount(gray, size, threshold) {
    const binary = new Uint8Array(gray.length);
    for (let i = 0; i < gray.length; i += 1) {
        binary[i] = gray[i] >= threshold ? 1 : 0;
    }

    const bestRuns = LINE_SCAN_ANGLES.map((angle) => findBestLineRun(binary, size, angle));
    const strongRuns = bestRuns.filter((run) => (
        run.length >= size * 0.42
        && run.hitRatio >= 0.38
        && run.longestGap <= size * 0.12
    ));

    return strongRuns.length > 0 ? 1 : 0;
}

function findBestLineRun(binary, size, angleDegrees) {
    const radians = angleDegrees * Math.PI / 180;
    const dx = Math.cos(radians);
    const dy = Math.sin(radians);
    const normalX = -dy;
    const normalY = dx;
    const center = (size - 1) / 2;
    const offsetLimit = Math.ceil(size * 0.72);
    let best = { length: 0, hitRatio: 0, longestGap: Infinity };

    for (let offset = -offsetLimit; offset <= offsetLimit; offset += 2) {
        const startX = center + normalX * offset - dx * size;
        const startY = center + normalY * offset - dy * size;
        const endX = center + normalX * offset + dx * size;
        const endY = center + normalY * offset + dy * size;
        const run = measureLineRun(binary, size, startX, startY, endX, endY);

        if (
            run.length > best.length
            || (run.length === best.length && run.hitRatio > best.hitRatio)
        ) {
            best = run;
        }
    }

    return best;
}

function measureLineRun(binary, size, startX, startY, endX, endY) {
    const steps = Math.ceil(Math.hypot(endX - startX, endY - startY));
    let currentLength = 0;
    let currentHits = 0;
    let currentGap = 0;
    let currentLongestGap = 0;
    let best = { length: 0, hitRatio: 0, longestGap: Infinity };
    const maxGap = Math.ceil(size * 0.055);

    for (let step = 0; step <= steps; step += 1) {
        const t = step / steps;
        const x = Math.round(startX + (endX - startX) * t);
        const y = Math.round(startY + (endY - startY) * t);
        const inBounds = x >= 0 && x < size && y >= 0 && y < size;
        const hasHit = inBounds && hasLinePixel(binary, size, x, y);

        if (hasHit) {
            currentLength += 1;
            currentHits += 1;
            currentLongestGap = Math.max(currentLongestGap, currentGap);
            currentGap = 0;
            continue;
        }

        if (!inBounds || currentGap > maxGap) {
            best = pickBetterLineRun(best, currentLength, currentHits, currentLongestGap);
            currentLength = 0;
            currentHits = 0;
            currentGap = 0;
            currentLongestGap = 0;
            continue;
        }

        currentLength += 1;
        currentGap += 1;
    }

    return pickBetterLineRun(best, currentLength, currentHits, currentLongestGap);
}

function hasLinePixel(binary, size, x, y) {
    for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
            const nextX = x + dx;
            const nextY = y + dy;
            if (nextX < 0 || nextX >= size || nextY < 0 || nextY >= size) continue;
            if (binary[nextY * size + nextX]) return true;
        }
    }

    return false;
}

function pickBetterLineRun(best, length, hits, longestGap) {
    if (length <= 0 || hits <= 0) return best;

    const candidate = {
        length,
        hitRatio: hits / length,
        longestGap,
    };

    if (
        candidate.length > best.length
        || (candidate.length === best.length && candidate.hitRatio > best.hitRatio)
    ) {
        return candidate;
    }

    return best;
}

function scorePredictionRule(className, features) {
    switch (className) {
        case 'Good':
            return average([
                scoreHigh(features.starCount, 12, 45),
                scoreHigh(features.sharpness, 35, 150),
                scoreLow(features.saturatedRatio, 0.004, 0.025),
                scoreLow(features.brightAreaRatio, 0.10, 0.32),
                scoreLow(features.elongatedStarRatio, 0.08, 0.28),
                scoreLow(features.longLineCount, 0.5, 2),
            ]);
        case 'Out of Focus':
            return average([
                scoreLow(features.sharpness, 20, 85),
                scoreHigh(features.starCount, 5, 18),
                scoreLow(features.longLineCount, 0.5, 2),
            ]);
        case 'Tracking Error':
            return average([
                scoreHigh(features.elongatedStarRatio, 0.12, 0.38),
                scoreHigh(features.starCount, 5, 18),
                scoreLow(features.longLineCount, 1, 4),
            ]);
        case 'Over Saturated':
            return average([
                scoreHigh(features.saturatedRatio, 0.002, 0.018),
                scoreHigh(features.highlightRatio, 0.01, 0.09),
                scoreHigh(features.brightAreaRatio, 0.16, 0.42),
                scoreHigh(features.meanBrightness, 0.18, 0.36),
                scoreHigh(features.starCount, 5, 18),
            ]);
        case 'No Star':
            return average([
                scoreLow(features.starCount, 2, 8),
                scoreLow(features.saturatedRatio, 0.004, 0.025),
                scoreLow(features.brightAreaRatio, 0.12, 0.35),
                scoreLow(features.longLineCount, 0.5, 2),
            ]);
        case 'Satellite':
            return average([
                scoreHigh(features.longLineCount, 0.25, 1),
                scoreLow(features.saturatedRatio, 0.008, 0.04),
            ]);
        default:
            return RULE_NEUTRAL_SCORE;
    }
}

function applyRuleAdjustment(ranked, ruleAnalysis) {
    const [top] = ranked;
    top.ruleScore = ruleAnalysis.ruleScore;
    top.ruleLabel = ruleAnalysis.label;
    top.adjustedProbability = ruleAnalysis.finalConfidence;
    top.probability = ruleAnalysis.finalConfidence;
}

function combineConfidence(modelConfidence, ruleScore) {
    return clamp(
        modelConfidence * MODEL_CONFIDENCE_WEIGHT + ruleScore * RULE_CONFIDENCE_WEIGHT,
        0,
        1,
    );
}

function scoreHigh(value, weakAt, strongAt) {
    return clamp((value - weakAt) / (strongAt - weakAt), 0, 1);
}

function scoreLow(value, strongAt, weakAt) {
    return 1 - scoreHigh(value, strongAt, weakAt);
}

function describeRuleSupport(ruleScore, isAvailable) {
    if (!isAvailable) return 'Rule evidence unavailable, model confidence only';
    if (ruleScore >= 0.72) return 'Rule evidence supports this class';
    if (ruleScore >= 0.45) return 'Rule evidence is inconclusive';
    return 'Rule evidence disagrees, review suggested';
}

function softmax(values) {
    const maxValue = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - maxValue));
    const sum = exps.reduce((total, value) => total + value, 0);
    return exps.map((value) => value / sum);
}

function renderPrediction({ item, ranked, latency, ruleAnalysis }) {
    const [top] = ranked;
    const card = item.card;
    const isLowConfidence = top.modelProbability < LOW_CONFIDENCE_THRESHOLD;
    const isDisagree = Boolean(ruleAnalysis && !ruleAnalysis.isMatch);
    item.prediction = { ranked, latency, ruleAnalysis };
    card.dataset.state = 'done';
    card.dataset.predictedClass = top.className;
    card.dataset.lowConfidence = isLowConfidence ? 'true' : 'false';
    card.dataset.disagree = isDisagree ? 'true' : 'false';
    card.dataset.confidence = top.modelProbability.toString();
    card.dataset.latency = latency.toString();
    card.classList.toggle('low-confidence', isLowConfidence);
    card.querySelector('.top-class').textContent = top.className;
    card.querySelector('.top-confidence').textContent = `${formatPercent(top.modelProbability)} CNN confidence`;

    const ruleSummaryEl = card.querySelector('.rule-summary');
    if (ruleAnalysis?.ruleTopClass) {
        ruleSummaryEl.textContent = `Rule: ${ruleAnalysis.ruleTopClass} (${formatPercent(ruleAnalysis.ruleTopScore)}) • ${ruleAnalysis.isMatch ? 'MATCH' : 'DISAGREE'}`;
        ruleSummaryEl.className = `rule-summary ${ruleAnalysis.isMatch ? 'match' : 'disagree'}`;
        ruleSummaryEl.title = ruleAnalysis.label ?? '';
    } else {
        ruleSummaryEl.textContent = `Model ${formatPercent(top.modelProbability)} | Rule ${formatPercent(top.ruleScore ?? RULE_NEUTRAL_SCORE)}`;
    }

    card.querySelector('.latency').textContent = `${latency.toFixed(0)} ms`;
    setCardStatus(card, 'Done', 'done');
}

function updateLatencyHighlights() {
    queuedItems.forEach((item) => {
        const latencyElement = item.card.querySelector('.latency');
        latencyElement.classList.remove('slow-latency');
        latencyElement.title = '';
    });

    if (inferenceResults.length < 3) return;

    const latencies = inferenceResults.map((result) => result.latency);
    const averageLatency = average(latencies);
    const slowThreshold = averageLatency * 1.5;

    queuedItems.forEach((item) => {
        const latencyElement = item.card.querySelector('.latency');
        const latency = Number(item.card.dataset.latency);
        const isSlow = Number.isFinite(latency) && latency > slowThreshold;
        latencyElement.classList.toggle('slow-latency', isSlow);
        latencyElement.title = isSlow
            ? `Slower than batch average (${averageLatency.toFixed(0)} ms)`
            : '';
    });
}

function renderModalPrediction(item) {
    modalMeta.textContent = `${formatBytes(item.file.size)} | ${formatDimensions(item.dimensions)}`;

    if (!item.prediction) {
        modalTopClass.textContent = '-';
        modalTopConfidence.textContent = 'Prediction not run yet';
        if (modalRuleVerdictBox) modalRuleVerdictBox.style.display = 'none';
        if (modalPhysicsSection) modalPhysicsSection.style.display = 'none';
        modalRankList.innerHTML = '<p>Run prediction to see all 6 quality groups.</p>';
        return;
    }

    const { ranked, latency, ruleAnalysis } = item.prediction;
    const [top] = ranked;
    modalTopClass.textContent = `${top.className} (${formatPercent(top.modelProbability)})`;
    modalTopConfidence.textContent = `CNN Confidence: ${formatPercent(top.modelProbability)} | Latency: ${latency.toFixed(0)} ms`;

    // Rule Prediction & Verdict Card
    if (ruleAnalysis && ruleAnalysis.ruleTopClass && modalRuleVerdictBox) {
        modalRuleVerdictBox.style.display = 'grid';
        modalRuleVerdictBox.className = `rule-verdict-box ${ruleAnalysis.isMatch ? 'match' : 'disagree'}`;
        modalRuleClassLabel.innerHTML = `Rule Prediction: <strong>${ruleAnalysis.ruleTopClass} (${formatPercent(ruleAnalysis.ruleTopScore)})</strong>`;
        modalRuleBadge.textContent = ruleAnalysis.isMatch ? 'MATCH' : 'DISAGREE';
        modalRuleBadge.className = `rule-verdict-badge ${ruleAnalysis.isMatch ? 'match' : 'disagree'}`;
        modalRuleDesc.textContent = ruleAnalysis.label;
    }

    // Physics Metrics Mini-Cards Grid
    if (ruleAnalysis?.features && modalPhysicsSection && modalPhysicsGrid) {
        const m = ruleAnalysis.features;
        modalPhysicsSection.style.display = 'block';
        modalPhysicsGrid.innerHTML = `
            <div class="metric-mini-card"><span>Stars</span><strong>${m.starCount ?? '-'}</strong></div>
            <div class="metric-mini-card"><span>FWHM</span><strong>${m.fwhm ? m.fwhm.toFixed(1) + ' px' : '-'}</strong></div>
            <div class="metric-mini-card"><span>Eccentricity</span><strong>${m.eccentricity ? m.eccentricity.toFixed(2) : '-'}</strong></div>
            <div class="metric-mini-card"><span>Circularity</span><strong>${m.circularity ? m.circularity.toFixed(2) : '-'}</strong></div>
            <div class="metric-mini-card"><span>Hollowness</span><strong>${m.hollowness ? m.hollowness.toFixed(3) : '-'}</strong></div>
            <div class="metric-mini-card"><span>Saturation</span><strong>${formatPercent(m.saturatedRatio)}</strong></div>
            <div class="metric-mini-card"><span>Sharpness</span><strong>${m.sharpness ? m.sharpness.toFixed(0) : '-'}</strong></div>
            <div class="metric-mini-card"><span>Aspect Ratio</span><strong>${m.aspectRatio ? m.aspectRatio.toFixed(2) : '-'}</strong></div>
        `;
    }

    modalRankList.innerHTML = ranked.map((prediction, index) => `
        <div class="rank-row">
            <span>${index + 1}. ${prediction.className}</span>
            <strong>${formatPercent(prediction.modelProbability)}</strong>
            <div class="bar" aria-hidden="true">
                <i style="width: ${Math.max(prediction.modelProbability * 100, 2)}%"></i>
            </div>
        </div>
    `).join('');
}

function updateMetrics() {
    totalImages.textContent = queuedItems.length.toString();
    processedCount.textContent = `${inferenceResults.length} processed`;
    queueStatus.textContent = queuedItems.length === 0
        ? 'No images'
        : `${queuedItems.length} images queued`;

    if (inferenceResults.length === 0) {
        avgConfidence.textContent = '-';
        avgLatency.textContent = '-';
        dominantClass.textContent = '-';
        classDistribution.className = 'distribution empty';
        classDistribution.textContent = 'No predictions yet';
        return;
    }

    const topPredictions = inferenceResults.map((result) => result.ranked[0]);
    const averageConfidence = average(topPredictions.map((prediction) => prediction.probability));
    const averageLatency = average(inferenceResults.map((result) => result.latency));
    const classCounts = countBy(topPredictions, (prediction) => prediction.className);
    const [dominantName] = Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0];

    avgConfidence.textContent = formatPercent(averageConfidence);
    avgLatency.textContent = `${averageLatency.toFixed(0)} ms`;
    dominantClass.textContent = dominantName;
    renderClassDistribution(classCounts);
}

function updateFilterChips() {
    const counts = getFilterCounts();
    const existingClassFilter = activeFilter.type === 'class' ? activeFilter.value : null;
    const classButtons = CLASS_NAMES.map((className) => {
        const count = counts.classes[className] ?? 0;
        const isActive = activeFilter.type === 'class' && activeFilter.value === className;
        return `
            <button class="filter-chip ${isActive ? 'active' : ''}" type="button" data-filter-type="class" data-filter-value="${className}">
                ${className} <span>${count}</span>
            </button>
        `;
    }).join('');

    if (existingClassFilter && !CLASS_NAMES.includes(existingClassFilter)) {
        activeFilter = { type: 'all', value: 'all' };
    }

    filterChips.innerHTML = `
        <button class="filter-chip ${activeFilter.type === 'all' ? 'active' : ''}" type="button" data-filter-type="all" data-filter-value="all">
            All <span>${counts.all}</span>
        </button>
        <button class="filter-chip warning ${activeFilter.type === 'low' ? 'active' : ''}" type="button" data-filter-type="low" data-filter-value="low">
            Low confidence &lt; 60% <span>${counts.low}</span>
        </button>
        <button class="filter-chip disagree-chip ${activeFilter.type === 'disagree' ? 'active' : ''}" type="button" data-filter-type="disagree" data-filter-value="disagree">
            Disagree (CNN ≠ Rule) <span>${counts.disagree}</span>
        </button>
        ${classButtons}
    `;
}

function getFilterCounts() {
    const counts = {
        all: queuedItems.length,
        low: 0,
        disagree: 0,
        classes: Object.fromEntries(CLASS_NAMES.map((className) => [className, 0])),
    };

    inferenceResults.forEach((result) => {
        const top = result.ranked[0];
        counts.classes[top.className] = (counts.classes[top.className] ?? 0) + 1;
        if (top.probability < LOW_CONFIDENCE_THRESHOLD) {
            counts.low += 1;
        }
        if (result.ruleAnalysis && !result.ruleAnalysis.isMatch) {
            counts.disagree += 1;
        }
    });

    return counts;
}

function applyResultFilter() {
    if (queuedItems.length === 0 || resultsGrid.classList.contains('empty-state')) {
        filteredCount.textContent = '0 shown';
        updateFilterChips();
        return;
    }

    let shown = 0;
    queuedItems.forEach((item) => {
        const shouldShow = matchesActiveFilter(item.card);
        item.card.hidden = !shouldShow;
        if (shouldShow) shown += 1;
    });

    sortResultCards();
    filteredCount.textContent = `${shown} shown`;
    resultsGrid.classList.toggle('has-no-filtered-results', shown === 0);
}

function sortResultCards() {
    getSortedItems().forEach((item) => resultsGrid.appendChild(item.card));
    if (imageModal.classList.contains('is-open')) {
        updateModalNavigationButtons();
    }
}

function getVisibleSortedItems() {
    return getSortedItems().filter((item) => !item.card.hidden);
}

function getSortedItems() {
    const sortedItems = [...queuedItems];

    if (sortSelect.value === 'confidence-asc') {
        sortedItems.sort((a, b) => {
            const aConfidence = Number(a.card.dataset.confidence);
            const bConfidence = Number(b.card.dataset.confidence);
            if (!Number.isFinite(aConfidence) && !Number.isFinite(bConfidence)) return 0;
            if (!Number.isFinite(aConfidence)) return 1;
            if (!Number.isFinite(bConfidence)) return -1;
            return aConfidence - bConfidence;
        });
    }

    return sortedItems;
}

function matchesActiveFilter(card) {
    if (activeFilter.type === 'all') return true;
    if (card.dataset.state !== 'done') return false;
    if (activeFilter.type === 'low') return card.dataset.lowConfidence === 'true';
    if (activeFilter.type === 'disagree') return card.dataset.disagree === 'true';
    if (activeFilter.type === 'class') return card.dataset.predictedClass === activeFilter.value;
    return true;
}

function renderClassDistribution(classCounts) {
    const maxCount = Math.max(...Object.values(classCounts));
    classDistribution.className = 'distribution';
    classDistribution.innerHTML = CLASS_NAMES.map((className) => {
        const count = classCounts[className] ?? 0;
        const width = maxCount === 0 ? 0 : (count / maxCount) * 100;
        return `
            <div class="dist-row">
                <span>${className}</span>
                <strong>${count}</strong>
                <div class="bar" aria-hidden="true"><i style="width: ${width}%"></i></div>
            </div>
        `;
    }).join('');
}

function setCardStatus(card, text, state) {
    const pill = card.querySelector('.status-pill');
    pill.textContent = text;
    pill.dataset.state = state;
}

function updateControls() {
    runButton.disabled = isRunningInference || queuedItems.length === 0 || modelStatus.classList.contains('error');
    exportButton.disabled = isRunningInference || getCompletedItems().length === 0 || !window.JSZip;
    clearButton.disabled = isRunningInference || queuedItems.length === 0;
    queuedItems.forEach((item) => {
        item.card.querySelector('.remove-card-button').disabled = isRunningInference;
    });
}

function clearAll() {
    closeImageModal();
    queuedItems.forEach((item) => URL.revokeObjectURL(item.objectUrl));
    queuedItems = [];
    inferenceResults = [];
    activeFilter = { type: 'all', value: 'all' };
    upload.value = '';
    showGridMessage('Upload images to start testing the model');
    updateControls();
    updateMetrics();
    updateFilterChips();
    applyResultFilter();
}

function removeItem(item) {
    if (isRunningInference) return;

    closeImageModal();
    URL.revokeObjectURL(item.objectUrl);
    item.card.remove();
    queuedItems = queuedItems.filter((candidate) => candidate.id !== item.id);
    inferenceResults = inferenceResults.filter((result) => result.item.id !== item.id);

    if (activeFilter.type === 'class' && getFilterCounts().classes[activeFilter.value] === 0) {
        activeFilter = { type: 'all', value: 'all' };
    }

    if (queuedItems.length === 0) {
        upload.value = '';
        showGridMessage('Upload images to classify them into 6 quality groups');
    } else {
        resultsGrid.classList.remove('empty-state');
        updateLatencyHighlights();
    }

    updateControls();
    updateMetrics();
    updateFilterChips();
    applyResultFilter();
}

function showGridMessage(message) {
    resultsGrid.className = 'results-grid empty-state';
    resultsGrid.innerHTML = `<p>${message}</p>`;
}

function handleModalImageWheel(event) {
    if (!imageModal.classList.contains('is-open')) return;
    event.preventDefault();

    const direction = event.deltaY > 0 ? -1 : 1;
    const nextZoom = clamp(modalZoom + direction * 0.25, 1, 8);
    modalZoom = nextZoom;

    if (modalZoom === 1) {
        modalPan = { x: 0, y: 0 };
    }

    updateModalImageTransform();
}

function handleModalImagePointerDown(event) {
    if (modalZoom <= 1) return;
    event.preventDefault();
    modalDrag = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        panX: modalPan.x,
        panY: modalPan.y,
    };
    modalImage.setPointerCapture(event.pointerId);
    modalImage.classList.add('is-dragging');
}

function handleModalImagePointerMove(event) {
    if (!modalDrag || modalDrag.pointerId !== event.pointerId) return;
    modalPan = {
        x: modalDrag.panX + event.clientX - modalDrag.startX,
        y: modalDrag.panY + event.clientY - modalDrag.startY,
    };
    updateModalImageTransform();
}

function endModalImageDrag(event) {
    if (!modalDrag || modalDrag.pointerId !== event.pointerId) return;
    if (modalImage.hasPointerCapture(event.pointerId)) {
        modalImage.releasePointerCapture(event.pointerId);
    }
    modalDrag = null;
    modalImage.classList.remove('is-dragging');
}

function resetModalImageTransform() {
    modalZoom = 1;
    modalPan = { x: 0, y: 0 };
    modalDrag = null;
    updateModalImageTransform();
}

function updateModalImageTransform() {
    modalImage.style.transform = `translate(${modalPan.x}px, ${modalPan.y}px) scale(${modalZoom})`;
    modalImage.classList.toggle('is-zoomed', modalZoom > 1);
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error('Could not read image'));
        image.src = src;
    });
}

function average(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function countBy(values, getKey) {
    return values.reduce((counts, value) => {
        const key = getKey(value);
        counts[key] = (counts[key] ?? 0) + 1;
        return counts;
    }, {});
}

function formatPercent(value) {
    return `${(value * 100).toFixed(2)}%`;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const unitIndex = Math.floor(Math.log(bytes) / Math.log(1024));
    const value = bytes / Math.pow(1024, unitIndex);
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDimensions(dimensions) {
    if (!dimensions) return 'Size pending';
    return `${dimensions.width} x ${dimensions.height}px`;
}

function getCompletedItems() {
    return queuedItems.filter((item) => item.prediction);
}

function safeFolderName(value) {
    return value.replace(/[<>:"/\\|?*]+/g, '-').trim();
}

function toColumnName(value) {
    return value.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
}

function uniqueExportFileName(item, allItems) {
    const sameNameIndex = allItems
        .filter((candidate) => candidate.file.name === item.file.name)
        .findIndex((candidate) => candidate.id === item.id);

    if (sameNameIndex <= 0) return item.file.name;

    const dotIndex = item.file.name.lastIndexOf('.');
    if (dotIndex <= 0) return `${item.file.name}-${sameNameIndex + 1}`;

    return `${item.file.name.slice(0, dotIndex)}-${sameNameIndex + 1}${item.file.name.slice(dotIndex)}`;
}

function toCsv(rows) {
    return rows
        .map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(','))
        .join('\n');
}

function downloadBlob(blob, fileName) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

function formatDateStamp(date) {
    const pad = (value) => String(value).padStart(2, '0');
    return [
        date.getFullYear(),
        pad(date.getMonth() + 1),
        pad(date.getDate()),
        '-',
        pad(date.getHours()),
        pad(date.getMinutes()),
    ].join('');
}
