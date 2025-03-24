document.addEventListener('DOMContentLoaded', function() {
    const socket = io();
    const form = document.getElementById('upload-form');
    const streamImg = document.getElementById('stream');
    const outputVideo = document.getElementById('output-video');
    const videoSource = document.getElementById('video-source');
    const processedImage = document.getElementById('processed-image');
    const statusDiv = document.getElementById('status');
    const placeholder = document.querySelector('.placeholder');
    const pauseResumeBtn = document.getElementById('pause-resume-btn');
    const toggleHeatmapBtn = document.getElementById('toggle-heatmap-btn');
    const showClassCheckbox = document.getElementById('show-class');
    const showConfidenceCheckbox = document.getElementById('show-confidence');
    const showBboxCheckbox = document.getElementById('show-bbox');
    const confThresholdSlider = document.getElementById('conf-threshold');
    const confValueSpan = document.getElementById('conf-value');
    const statsContent = document.getElementById('stats-content');
    const canvas = document.getElementById('roi-canvas');
    const ctx = canvas.getContext('2d');
    const resetBtn = document.getElementById('reset-btn');
    let currentTaskId = null;
    let zoomLevel = 1.0;
    let zoomStep = 0.1;
    let roi = null;
    let lastRoi = null;
    let videoWidth = 0, videoHeight = 0;
    let isZooming = false;
    let zoomTimeout;

    confThresholdSlider.addEventListener('input', () => {
        confValueSpan.textContent = confThresholdSlider.value;
        if (currentTaskId) {
            socket.emit('update_conf_threshold', { task_id: currentTaskId, threshold: parseFloat(confThresholdSlider.value) });
        }
    });

    socket.on('class_stats', (data) => {
        if (data.task_id === currentTaskId) {
            if (Object.keys(data.stats).length === 0) {
                statsContent.textContent = 'No objects detected';
            } else {
                const statsText = Object.entries(data.stats)
                    .map(([className, count]) => `${className}: ${count}`)
                    .join(', ');
                statsContent.textContent = statsText;
            }
        }
    });

    function resizeCanvas() {
        canvas.width = streamImg.width;
        canvas.height = streamImg.height;
        updateZoomROI();
    }
    streamImg.onload = resizeCanvas;
    window.addEventListener('resize', resizeCanvas);

    function updateZoomROI(mouseX = null, mouseY = null) {
        if (videoWidth === 0 || videoHeight === 0) return;

        const scaleX = videoWidth / canvas.width;
        const scaleY = videoHeight / canvas.height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (zoomLevel <= 1.0) {
            roi = { x: 0, y: 0, width: videoWidth, height: videoHeight };
            lastRoi = null;
        } else {
            let roiX, roiY, roiWidth, roiHeight;

            const scale = 1 / zoomLevel;
            roiWidth = canvas.width * scale;
            roiHeight = canvas.height * scale;

            if (lastRoi && mouseX !== null && mouseY !== null) {
                const prevRoiX = lastRoi.x / scaleX;
                const prevRoiY = lastRoi.y / scaleY;
                const prevRoiWidth = lastRoi.width / scaleX;
                const prevRoiHeight = lastRoi.height / scaleY;

                const relativeX = (mouseX - prevRoiX) / prevRoiWidth;
                const relativeY = (mouseY - prevRoiY) / prevRoiHeight;

                roiX = mouseX - (roiWidth * relativeX);
                roiY = mouseY - (roiHeight * relativeY);

                roiX = Math.max(0, Math.min(roiX, canvas.width - roiWidth));
                roiY = Math.max(0, Math.min(roiY, canvas.height - roiHeight));
            } else if (lastRoi) {
                const prevRoiX = lastRoi.x / scaleX;
                const prevRoiY = lastRoi.y / scaleY;
                const prevRoiWidth = lastRoi.width / scaleX;
                const prevRoiHeight = lastRoi.height / scaleY;

                roiX = prevRoiX + (prevRoiWidth - roiWidth) / 2;
                roiY = prevRoiY + (prevRoiHeight - roiHeight) / 2;

                roiX = Math.max(0, Math.min(roiX, canvas.width - roiWidth));
                roiY = Math.max(0, Math.min(roiY, canvas.height - roiHeight));
            } else {
                roiX = (canvas.width - roiWidth) / 2;
                roiY = (canvas.height - roiHeight) / 2;

                if (mouseX !== null && mouseY !== null) {
                    roiX = mouseX - (roiWidth / 2);
                    roiY = mouseY - (roiHeight / 2);
                    roiX = Math.max(0, Math.min(roiX, canvas.width - roiWidth));
                    roiY = Math.max(0, Math.min(roiY, canvas.height - roiHeight));
                }
            }

            roi = {
                x: roiX * scaleX,
                y: roiY * scaleY,
                width: roiWidth * scaleX,
                height: roiHeight * scaleY
            };

            lastRoi = { x: roi.x, y: roi.y, width: roi.width, height: roi.height };

            if (isZooming && mouseX !== null && mouseY !== null) {
                const drawRoiX = mouseX - roiWidth / 2;
                const drawRoiY = mouseY - roiHeight / 2;
                const boundedDrawRoiX = Math.max(0, Math.min(drawRoiX, canvas.width - roiWidth));
                const boundedDrawRoiY = Math.max(0, Math.min(drawRoiY, canvas.height - roiHeight));

                ctx.strokeStyle = 'yellow';
                ctx.lineWidth = 5;
                ctx.strokeRect(boundedDrawRoiX, boundedDrawRoiY, roiWidth, roiHeight);
            }
        }

        if (currentTaskId && statusDiv.textContent.includes('Streaming')) {
            socket.emit('update_roi', { task_id: currentTaskId, roi });
        }
    }

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (!statusDiv.textContent.includes('Streaming')) return;

        isZooming = true;

        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const prevZoomLevel = zoomLevel;
        zoomLevel += e.deltaY < 0 ? zoomStep : -zoomStep;
        zoomLevel = Math.max(1.0, Math.min(zoomLevel, 10.0));

        const zoomTooltip = document.getElementById('zoom-tooltip');
        if (zoomLevel > prevZoomLevel) {
            zoomTooltip.textContent = 'Zoom In';
        } else if (zoomLevel < prevZoomLevel) {
            zoomTooltip.textContent = 'Zoom Out';
        }
        zoomTooltip.classList.add('active');

        updateZoomROI(mouseX, mouseY);

        clearTimeout(zoomTimeout);
        zoomTimeout = setTimeout(() => {
            isZooming = false;
            updateZoomROI();
            zoomTooltip.classList.remove('active');
        }, 1000);
    });

    socket.on('video_frame', (data) => {
        if (data.task_id === currentTaskId) {
            streamImg.src = `data:image/jpeg;base64,${data.frame}`;
            streamImg.classList.add('active');
            placeholder.style.display = 'none';
            resizeCanvas();
            if (data.frame_width && data.frame_height) {
                videoWidth = data.frame_width;
                videoHeight = data.frame_height;
            }
            updateZoomROI();
        }
    });

    socket.on('status_update', (data) => {
        statusDiv.textContent = `Status: ${data.status.charAt(0).toUpperCase() + data.status.slice(1)}`;
        if (data.status === 'streaming') {
            statusDiv.classList.add('streaming');
            statusDiv.classList.remove('completed', 'error', 'stopped');
            pauseResumeBtn.disabled = false;
            toggleHeatmapBtn.disabled = false;
            showClassCheckbox.disabled = false;
            showConfidenceCheckbox.disabled = false;
            showBboxCheckbox.disabled = false;
            confThresholdSlider.disabled = false;
            pauseResumeBtn.textContent = 'Pause';
            pauseResumeBtn.classList.remove('paused');
            pauseResumeBtn.setAttribute('aria-label', 'Pause video processing');
            updateZoomROI();
        } else if (data.status === 'paused') {
            statusDiv.classList.add('streaming');
            statusDiv.classList.remove('completed', 'error', 'stopped');
            pauseResumeBtn.disabled = false;
            toggleHeatmapBtn.disabled = false;
            showClassCheckbox.disabled = false;
            showConfidenceCheckbox.disabled = false;
            showBboxCheckbox.disabled = false;
            confThresholdSlider.disabled = false;
            pauseResumeBtn.textContent = 'Resume';
            pauseResumeBtn.classList.add('paused');
            pauseResumeBtn.setAttribute('aria-label', 'Resume video processing');
        } else if (data.status === 'completed' && data.url) {
            streamImg.classList.remove('active');
            videoSource.src = data.url;
            outputVideo.load();
            outputVideo.classList.add('active');
            statusDiv.classList.remove('streaming', 'error', 'stopped');
            statusDiv.classList.add('completed');
            statusDiv.setAttribute('aria-busy', 'false');
            pauseResumeBtn.disabled = true;
            toggleHeatmapBtn.disabled = true;
            showClassCheckbox.disabled = true;
            showConfidenceCheckbox.disabled = true;
            showBboxCheckbox.disabled = true;
            confThresholdSlider.disabled = true;
            statsContent.textContent = 'No objects detected yet';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            zoomLevel = 1.0;
            lastRoi = null;
        } else if (data.status === 'error') {
            streamImg.classList.remove('active');
            outputVideo.classList.remove('active');
            processedImage.classList.remove('active');
            statusDiv.classList.remove('streaming', 'completed', 'stopped');
            statusDiv.classList.add('error');
            statusDiv.setAttribute('aria-busy', 'false');
            pauseResumeBtn.disabled = true;
            toggleHeatmapBtn.disabled = true;
            showClassCheckbox.disabled = true;
            showConfidenceCheckbox.disabled = true;
            showBboxCheckbox.disabled = true;
            confThresholdSlider.disabled = true;
            statsContent.textContent = 'No objects detected yet';
            placeholder.style.display = 'block';
        } else if (data.status === 'stopped') {
            streamImg.classList.remove('active');
            outputVideo.classList.remove('active');
            processedImage.classList.remove('active');
            statusDiv.classList.remove('streaming', 'completed', 'error');
            statusDiv.classList.add('stopped');
            statusDiv.setAttribute('aria-busy', 'false');
            pauseResumeBtn.disabled = true;
            toggleHeatmapBtn.disabled = true;
            showClassCheckbox.disabled = true;
            showConfidenceCheckbox.disabled = true;
            showBboxCheckbox.disabled = true;
            confThresholdSlider.disabled = true;
            statsContent.textContent = 'No objects detected yet';
            placeholder.style.display = 'block';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            zoomLevel = 1.0;
            lastRoi = null;
            currentTaskId = null;
        }
    });

    socket.on('heatmap_status', (data) => {
        if (data.task_id === currentTaskId) {
            if (data.state === 'on') {
                toggleHeatmapBtn.textContent = 'Turn Off Heatmap';
                toggleHeatmapBtn.classList.add('active');
            } else {
                toggleHeatmapBtn.textContent = 'Turn On Heatmap';
                toggleHeatmapBtn.classList.remove('active');
            }
        }
    });

    function sendDisplayOptions() {
        if (currentTaskId) {
            const options = {
                class_name: showClassCheckbox.checked,
                confidence: showConfidenceCheckbox.checked,
                bbox: showBboxCheckbox.checked
            };
            socket.emit('update_display_options', { task_id: currentTaskId, options: options });
        }
    }

    showClassCheckbox.addEventListener('change', sendDisplayOptions);
    showConfidenceCheckbox.addEventListener('change', sendDisplayOptions);
    showBboxCheckbox.addEventListener('change', sendDisplayOptions);

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const fileInput = form.querySelector('input[type="file"]');
        const streamUrlInput = form.querySelector('input[name="stream-url"]');
        const file = fileInput.files[0];
        const streamUrl = streamUrlInput.value.trim();
        if (!file && !streamUrl) {
            statusDiv.textContent = 'Status: Please select a file or enter a stream URL';
            statusDiv.classList.add('error');
            return;
        }

        placeholder.style.display = 'none';
        streamImg.classList.remove('active');
        outputVideo.classList.remove('active');
        processedImage.classList.remove('active');
        pauseResumeBtn.disabled = true;
        toggleHeatmapBtn.disabled = true;
        showClassCheckbox.disabled = true;
        showConfidenceCheckbox.disabled = true;
        showBboxCheckbox.disabled = true;
        showClassCheckbox.checked = true;
        showConfidenceCheckbox.checked = false;
        showBboxCheckbox.checked = true;
        confThresholdSlider.disabled = true;
        confThresholdSlider.value = 0.25;
        confValueSpan.textContent = '0.25';
        statsContent.textContent = 'No objects detected yet';
        toggleHeatmapBtn.textContent = 'Turn On Heatmap';
        toggleHeatmapBtn.classList.remove('active');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        zoomLevel = 1.0;
        lastRoi = null;
        roi = null;
        statusDiv.textContent = 'Status: Initializing...';
        statusDiv.classList.remove('error', 'completed', 'streaming', 'stopped');
        statusDiv.setAttribute('aria-busy', 'true');

        const formData = new FormData();
        if (file) {
            formData.append('file', file);
        } else if (streamUrl) {
            formData.append('stream_url', streamUrl);
        }

        try {
            const response = await fetch('/', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`Server error: ${response.status}`);
            const data = await response.json();

            if (data.task_id) {
                currentTaskId = data.task_id;
                socket.emit('start_stream', { 
                    task_id: data.task_id,
                    stream_url: data.stream_url || ''
                });
                sendDisplayOptions();
                socket.emit('update_conf_threshold', { task_id: currentTaskId, threshold: parseFloat(confThresholdSlider.value) });
            } else if (data.img_data) {
                processedImage.src = `data:image/png;base64,${data.img_data}`;
                processedImage.classList.add('active');
                statusDiv.textContent = 'Status: Completed';
                statusDiv.classList.remove('streaming', 'error', 'stopped');
                statusDiv.classList.add('completed');
                statusDiv.setAttribute('aria-busy', 'false');
                pauseResumeBtn.disabled = true;
            } else {
                throw new Error('Invalid response');
            }
        } catch (error) {
            console.error('Error:', error);
            statusDiv.textContent = 'Status: Upload failed';
            statusDiv.classList.add('error');
            statusDiv.setAttribute('aria-busy', 'false');
        }
    });

    pauseResumeBtn.onclick = () => {
        if (currentTaskId) {
            console.log('Toggling pause for task:', currentTaskId);
            socket.emit('toggle_pause', { task_id: currentTaskId });
        }
    };

    toggleHeatmapBtn.onclick = () => {
        if (currentTaskId) {
            console.log('Toggling heatmap for task:', currentTaskId);
            socket.emit('toggle_heatmap', { task_id: currentTaskId });
        }
    };

    function resetToInitialState() {
        currentTaskId = null;
        streamImg.classList.remove('active');
        outputVideo.classList.remove('active');
        processedImage.classList.remove('active');
        placeholder.style.display = 'block';
        statusDiv.textContent = 'Status: Waiting for upload...';
        statusDiv.classList.remove('streaming', 'completed', 'error', 'stopped');
        statusDiv.setAttribute('aria-busy', 'false');
        pauseResumeBtn.disabled = true;
        toggleHeatmapBtn.disabled = true;
        showClassCheckbox.disabled = true;
        showConfidenceCheckbox.disabled = true;
        showBboxCheckbox.disabled = true;
        showClassCheckbox.checked = true;
        showConfidenceCheckbox.checked = false;
        showBboxCheckbox.checked = true;
        confThresholdSlider.disabled = true;
        confThresholdSlider.value = 0.25;
        confValueSpan.textContent = '0.25';
        statsContent.textContent = 'No objects detected yet';
        toggleHeatmapBtn.textContent = 'Turn On Heatmap';
        toggleHeatmapBtn.classList.remove('active');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        zoomLevel = 1.0;
        lastRoi = null;
        roi = null;
        videoWidth = 0;
        videoHeight = 0;
        form.reset();
    }

    resetBtn.onclick = () => {
        console.log('Resetting and cleaning all tasks');
        socket.emit('reset_task', {});
        resetToInitialState();
    };
});