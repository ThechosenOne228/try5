// Global variables
let video;
let canvas;
let ctx;
let model;
let isModelLoading = true;
let currentStream = null;
let isFrontCamera = true;
let productDb = {};
let lastDetectionTime = 0;
let frameCount = 0;
let fps = 0;
let lastFpsUpdate = 0;
let currentDetections = [];

// UI state
const uiState = {
    showBrand: true,
    showDescription: true,
    showPrice: true
};

// Colors for different classes
const classColors = {
    person: '#FF0000',
    car: '#0000FF',
    'cell phone': '#00FF00',
    tv: '#FFFF00',
    laptop: '#FF00FF',
    shoe: '#00FFFF',
    backpack: '#FFA500',
    bottle: '#800080',
    chair: '#008000',
    cup: '#800000'
};

// Show error message
function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    errorElement.textContent = message;
    errorElement.classList.remove('hidden');
    setTimeout(() => {
        errorElement.classList.add('hidden');
    }, 5000);
}

// Show/hide loading indicator
function setLoading(isLoading) {
    const loadingElement = document.getElementById('loadingIndicator');
    loadingElement.style.display = isLoading ? 'flex' : 'none';
}

// Initialize the application
async function init() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Show loading indicator
    setLoading(true);
    
    // Load product database
    try {
        const response = await fetch('assets/product_db.json');
        if (!response.ok) {
            throw new Error('Failed to load product database');
        }
        productDb = await response.json();
    } catch (error) {
        console.error('Error loading product database:', error);
        showError('Failed to load product database');
    }

    // Set up event listeners
    document.getElementById('switchCamera').addEventListener('click', switchCamera);
    document.getElementById('captureFrame').addEventListener('click', captureFrame);
    document.getElementById('downloadResults').addEventListener('click', downloadResults);
    
    // Set up UI toggles
    document.getElementById('showBrand').addEventListener('change', (e) => uiState.showBrand = e.target.checked);
    document.getElementById('showDescription').addEventListener('change', (e) => uiState.showDescription = e.target.checked);
    document.getElementById('showPrice').addEventListener('change', (e) => uiState.showPrice = e.target.checked);

    // Load ONNX model
    try {
        model = await ort.InferenceSession.create('yolov8m.onnx');
        isModelLoading = false;
        updateStatus('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
        showError('Failed to load model. Please check if yolov8m.onnx exists in the root directory.');
        return;
    }

    // Start camera
    try {
        await startCamera();
    } catch (error) {
        console.error('Error starting camera:', error);
        showError('Failed to start camera. Please check camera permissions.');
        return;
    }
    
    // Hide loading indicator
    setLoading(false);
    
    // Start detection loop
    requestAnimationFrame(detectFrame);
}

// Switch between front and back cameras
async function switchCamera() {
    try {
        // Show loading state
        const switchButton = document.getElementById('switchCamera');
        const originalIcon = switchButton.innerHTML;
        switchButton.innerHTML = '<i class="fas fa-spinner fa-spin text-xl"></i>';
        switchButton.disabled = true;

        isFrontCamera = !isFrontCamera;
        await startCamera();

        // Update button icon based on current camera
        switchButton.innerHTML = isFrontCamera 
            ? '<i class="fas fa-camera-rotate text-xl"></i>'
            : '<i class="fas fa-camera-rotate fa-flip-horizontal text-xl"></i>';
        
        // Show camera mode in status
        updateStatus(`Switched to ${isFrontCamera ? 'front' : 'back'} camera`);
    } catch (error) {
        console.error('Error switching camera:', error);
        showError('Failed to switch camera: ' + error.message);
    } finally {
        // Re-enable button
        switchButton.disabled = false;
    }
}

// Start camera with specified constraints
async function startCamera() {
    const constraints = {
        video: {
            facingMode: isFrontCamera ? 'user' : 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    };

    try {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }

        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        await video.play();

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Update camera mode indicator
        const switchButton = document.getElementById('switchCamera');
        switchButton.innerHTML = isFrontCamera 
            ? '<i class="fas fa-camera-rotate text-xl"></i>'
            : '<i class="fas fa-camera-rotate fa-flip-horizontal text-xl"></i>';
        
        updateStatus(`Camera started (${isFrontCamera ? 'front' : 'back'})`);
    } catch (error) {
        console.error('Error starting camera:', error);
        showError('Error starting camera: ' + error.message);
        throw error;
    }
}

// Main detection loop
async function detectFrame() {
    if (!isModelLoading && video.readyState === video.HAVE_ENOUGH_DATA) {
        try {
            // Calculate FPS
            const now = performance.now();
            frameCount++;
            
            if (now - lastFpsUpdate >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsUpdate = now;
            }

            // Prepare input tensor
            const inputTensor = preprocessImage(video);
            
            // Run inference
            const results = await model.run({ images: inputTensor });
            
            // Process results
            currentDetections = processResults(results);
            
            // Draw results
            drawDetections(currentDetections);
            
            // Update status with FPS
            updateStatus(`Running at ${fps} FPS`);

            // Clean up tensors
            inputTensor.delete();
            results.output0.delete();
        } catch (error) {
            console.error('Error in detection loop:', error);
            showError('Detection error: ' + error.message);
        }
    }
    
    requestAnimationFrame(detectFrame);
}

// Preprocess image for model input
function preprocessImage(video) {
    try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 640;  // YOLOv8 input size
        tempCanvas.height = 640;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw and resize video frame
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        
        // Get image data
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        
        // Convert to float32 array and normalize
        const inputTensor = new Float32Array(imageData.data.length / 4);
        for (let i = 0; i < imageData.data.length / 4; i++) {
            inputTensor[i] = imageData.data[i * 4] / 255.0;
        }
        
        return new ort.Tensor('float32', inputTensor, [1, 3, 640, 640]);
    } catch (error) {
        console.error('Error preprocessing image:', error);
        throw new Error('Failed to preprocess image: ' + error.message);
    }
}

// Process model results
function processResults(results) {
    const output = results.output0.data;
    const detections = [];
    
    // Process YOLOv8 output format
    for (let i = 0; i < output.length; i += 85) {  // 80 classes + 4 bbox + 1 confidence
        const confidence = output[i + 4];
        if (confidence > 0.5) {  // Confidence threshold
            const classId = output.slice(i + 5, i + 85).indexOf(Math.max(...output.slice(i + 5, i + 85)));
            const bbox = output.slice(i, i + 4);
            
            detections.push({
                bbox: bbox,
                classId: classId,
                confidence: confidence,
                class: getClassName(classId)
            });
        }
    }
    
    return detections;
}

// Draw detections on canvas
function drawDetections(detections) {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    detections.forEach(detection => {
        const [x, y, w, h] = detection.bbox;
        const color = classColors[detection.class] || '#FFFFFF';
        
        // Scale bbox to canvas size
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 640;
        
        const boxX = x * scaleX;
        const boxY = y * scaleY;
        const boxWidth = w * scaleX;
        const boxHeight = h * scaleY;
        
        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
        
        // Draw label
        const label = `${detection.class} ${Math.round(detection.confidence * 100)}%`;
        ctx.fillStyle = color;
        ctx.font = '16px Arial';
        ctx.fillText(label, boxX, boxY - 5);
        
        // Draw metadata if available
        if (productDb[detection.class]) {
            const metadata = productDb[detection.class];
            let metadataText = '';
            
            if (uiState.showBrand) metadataText += `${metadata.brand} `;
            if (uiState.showDescription) metadataText += `${metadata.description} `;
            if (uiState.showPrice) metadataText += metadata.price;
            
            if (metadataText) {
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(boxX, boxY + boxHeight, ctx.measureText(metadataText).width + 10, 20);
                ctx.fillStyle = '#FFFFFF';
                ctx.fillText(metadataText, boxX + 5, boxY + boxHeight + 15);
            }
        }
    });
}

// Get class name from class ID
function getClassName(classId) {
    const classNames = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ];
    return classNames[classId] || 'unknown';
}

// Capture current frame
function captureFrame() {
    const link = document.createElement('a');
    link.download = `detection-${new Date().toISOString()}.png`;
    link.href = canvas.toDataURL();
    link.click();
}

// Download detection results
function downloadResults() {
    const results = {
        timestamp: new Date().toISOString(),
        detections: currentDetections
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.download = `detections-${new Date().toISOString()}.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
}

// Update status message
function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

// Initialize the app when the page loads
window.addEventListener('load', init); 