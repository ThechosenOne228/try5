import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const App = () => {
    const [detections, setDetections] = useState([]);
    const [isModelLoading, setIsModelLoading] = useState(true);
    const [showToast, setShowToast] = useState(false);
    const modelRef = useRef(null);
    const videoRef = useRef(document.getElementById('video-feed'));

    // Load the COCO-SSD model
    useEffect(() => {
        const loadModel = async () => {
            try {
                const model = await cocoSsd.load();
                modelRef.current = model;
                setIsModelLoading(false);
            } catch (err) {
                console.error('Error loading model:', err);
            }
        };
        loadModel();
    }, []);

    // Run detection
    useEffect(() => {
        let animationFrameId;
        const detectObjects = async () => {
            if (modelRef.current && videoRef.current) {
                const predictions = await modelRef.current.detect(videoRef.current);
                setDetections(predictions);
            }
            animationFrameId = requestAnimationFrame(detectObjects);
        };

        if (!isModelLoading) {
            detectObjects();
        }

        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    }, [isModelLoading]);

    const handleShare = () => {
        const items = detections
            .filter(det => ['person', 'backpack', 'handbag', 'suitcase', 'bottle', 'wine glass', 'cup', 'bowl', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'].includes(det.class))
            .map(det => det.class);
        
        const text = `Today's Fit: ${items.join(', ')}`;
        navigator.clipboard.writeText(text);
        setShowToast(true);
        setTimeout(() => setShowToast(false), 2000);
    };

    return (
        <div className="relative w-full h-screen">
            {/* Detection Labels */}
            {detections.map((detection, index) => (
                <div
                    key={index}
                    className="absolute bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm"
                    style={{
                        left: `${detection.bbox[0]}px`,
                        top: `${detection.bbox[1]}px`,
                        width: `${detection.bbox[2]}px`,
                        height: `${detection.bbox[3]}px`,
                        border: '2px solid #fff'
                    }}
                >
                    <div className="flex items-center gap-2">
                        <span>{detection.class}</span>
                        <a
                            href={`https://www.google.com/search?tbm=shop&q=${detection.class}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-400 hover:text-blue-300"
                        >
                            🔍
                        </a>
                    </div>
                </div>
            ))}

            {/* Share Button */}
            <button
                onClick={handleShare}
                className="fixed bottom-8 left-1/2 transform -translate-x-1/2 bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-full shadow-lg transition-all duration-200"
            >
                Share My Fit
            </button>

            {/* Loading Indicator */}
            {isModelLoading && (
                <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-black bg-opacity-50 text-white px-4 py-2 rounded">
                    Loading model...
                </div>
            )}

            {/* Toast Message */}
            {showToast && (
                <div className="fixed top-8 left-1/2 transform -translate-x-1/2 bg-green-500 text-white px-4 py-2 rounded shadow-lg transition-all duration-200">
                    Link copied!
                </div>
            )}
        </div>
    );
};

export default App; 