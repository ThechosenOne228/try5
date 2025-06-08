// Configuration
const MODEL_PATH = 'assets/yolov8m.onnx';
const CONFIDENCE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.45;
const TARGET_FPS = 15;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

// Class colors for bounding boxes
const CLASS_COLORS = {
    person: '#FF0000',
    car: '#0000FF',
    phone: '#00FF00',
    tv: '#FFFF00',
    laptop: '#FF00FF',
    shoe: '#00FFFF',
    shirt: '#FFA500',
    pants: '#800080',
    bag: '#008000'
};

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const exportBtn = document.getElementById('exportBtn');
const statusEl = document.getElementById('status');
const showBrand = document.getElementById('showBrand');
const showPrice = document.getElementById('showPrice');
const showDescription = document.getElementById('showDescription');

// State
let model = null;
let session = null;
let isRunning = false;
let lastFrameTime = 0;
let productDb = null;
let currentDetections = [];

// Initialize
async function init() {
    try {
        // Load product database
        const response = await fetch('assets/product_db.json');
        productDb = await response.json();
        
        // Load ONNX model
        statusEl.textContent = 'Loading model...';
        session = await ort.InferenceSession.create(MODEL_PATH);
        model = session;
        statusEl.textContent = 'Model loaded! Click "Start Camera" to begin.';
        
        // Enable buttons
        startBtn.disabled = false;
    } catch (error) {
        console.error('Initialization error:', error);
        statusEl.textContent = 'Error loading model. Please refresh the page.';
    }
}

// Start webcam
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();
        
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Enable controls
        captureBtn.disabled = false;
        exportBtn.disabled = false;
        startBtn.disabled = true;
        
        // Start detection loop
        isRunning = true;
        detectFrame();
    } catch (error) {
        console.error('Camera error:', error);
        statusEl.textContent = 'Error accessing camera. Please check permissions.';
    }
}

// Process a single frame
async function detectFrame() {
    if (!isRunning) return;
    
    const now = performance.now();
    const elapsed = now - lastFrameTime;
    
    if (elapsed >= FRAME_INTERVAL) {
        // Draw video frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Prepare input tensor
        const inputTensor = preprocessFrame();
        
        // Run inference
        const results = await runInference(inputTensor);
        
        // Process and draw results
        processResults(results);
        
        lastFrameTime = now;
    }
    
    // Schedule next frame
    requestAnimationFrame(detectFrame);
}

// Preprocess frame for model input
function preprocessFrame() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data } = imageData;
    
    // Convert to float32 and normalize
    const input = new Float32Array(3 * canvas.width * canvas.height);
    for (let i = 0; i < data.length / 4; i++) {
        input[i] = data[i * 4] / 255.0;
        input[i + canvas.width * canvas.height] = data[i * 4 + 1] / 255.0;
        input[i + 2 * canvas.width * canvas.height] = data[i * 4 + 2] / 255.0;
    }
    
    return new ort.Tensor('float32', input, [1, 3, canvas.height, canvas.width]);
}

// Run model inference
async function runInference(inputTensor) {
    const feeds = { images: inputTensor };
    const results = await model.run(feeds);
    return results.output0.data;
}

// Process detection results
function processResults(results) {
    // Clear previous detections
    currentDetections = [];
    
    // Process each detection
    for (let i = 0; i < results.length; i += 6) {
        const [x1, y1, x2, y2, confidence, classId] = results.slice(i, i + 6);
        
        if (confidence >= CONFIDENCE_THRESHOLD) {
            const detection = {
                bbox: [x1, y1, x2, y2],
                confidence,
                classId: Math.round(classId),
                class: getClassName(classId)
            };
            
            currentDetections.push(detection);
            drawDetection(detection);
        }
    }
}

// Draw detection on canvas
function drawDetection(detection) {
    const [x1, y1, x2, y2] = detection.bbox;
    const { class: className, confidence } = detection;
    const color = CLASS_COLORS[className] || '#FFFFFF';
    
    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    
    // Draw label
    const label = `${className} ${confidence.toFixed(2)}`;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(x1, y1 - 20, ctx.measureText(label).width + 10, 20);
    ctx.fillStyle = color;
    ctx.fillText(label, x1 + 5, y1 - 5);
    
    // Draw product info if available
    if (productDb[className]) {
        const info = productDb[className];
        let infoText = '';
        
        if (showBrand.checked) infoText += `${info.brand} `;
        if (showPrice.checked) infoText += `${info.price} `;
        if (showDescription.checked) infoText += `— ${info.description}`;
        
        if (infoText) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            const textWidth = ctx.measureText(infoText).width;
            ctx.fillRect(x1, y2, textWidth + 10, 20);
            ctx.fillStyle = '#FFFFFF';
            ctx.fillText(infoText, x1 + 5, y2 + 15);
        }
    }
}

// Get class name from ID
function getClassName(classId) {
    const classes = ['person', 'car', 'phone', 'tv', 'laptop', 'shoe', 'shirt', 'pants', 'bag'];
    return classes[classId] || 'unknown';
}

// Capture current frame
function captureFrame() {
    const link = document.createElement('a');
    link.download = `detection-${Date.now()}.png`;
    link.href = canvas.toDataURL();
    link.click();
}

// Export detection data
function exportData() {
    const data = {
        timestamp: Date.now(),
        detections: currentDetections.map(d => ({
            ...d,
            productInfo: productDb[d.class]
        }))
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.download = `detections-${Date.now()}.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
}

// Event Listeners
startBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', captureFrame);
exportBtn.addEventListener('click', exportData);

// Initialize on load
window.addEventListener('load', init); 