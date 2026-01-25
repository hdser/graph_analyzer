/**
 * Time Zoom Bar Component
 * Timeline UI component for rapid temporal navigation through snapshots
 * Features: scrubbing, zoom levels, preloading, playback controls
 */
const TimeZoomBar = (function() {
    // State
    let container = null;
    let snapshots = [];
    let currentIndex = 0;
    let isPlaying = false;
    let playbackInterval = null;
    let playbackSpeed = 1; // 1 = normal, 2 = 2x, etc.
    let zoomLevel = 'all'; // 'day', 'week', 'month', 'year', 'all'
    let preloadQueue = [];
    let preloadedSnapshots = new Map();
    let baseSqlFile = null;
    let isVisible = false;

    // DOM elements
    const elements = {};

    // Configuration
    const config = {
        preloadAhead: 3,         // Number of snapshots to preload ahead
        preloadBehind: 1,        // Number of snapshots to preload behind
        playbackDelays: {        // ms per frame at each speed
            0.5: 2000,
            1: 1000,
            2: 500,
            4: 250
        },
        scrubDebounceMs: 100     // Debounce for scrub preview
    };

    // Debounce timer for scrubbing
    let scrubDebounceTimer = null;

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    /**
     * Initialize the TimeZoomBar
     * @param {string} containerId - ID of the container element
     */
    function init(containerId = 'time-zoom-bar') {
        container = document.getElementById(containerId);
        if (!container) {
            console.warn('[TimeZoomBar] Container not found:', containerId);
            return false;
        }

        createDOM();
        bindEvents();
        hide(); // Start hidden

        console.log('[TimeZoomBar] Initialized');
        return true;
    }

    /**
     * Create DOM structure - compact dropdown layout
     */
    function createDOM() {
        container.innerHTML = `
            <div class="tzb-header">
                <span class="tzb-title">Timeline</span>
                <button class="tzb-toggle" title="Close">×</button>
            </div>

            <div class="tzb-content">
                <div class="tzb-main-row">
                    <div class="tzb-info">
                        <span class="tzb-current-date">--</span>
                        <span class="tzb-current-block">Block: --</span>
                        <span class="tzb-position">0 / 0</span>
                    </div>

                    <div class="tzb-track-container">
                        <div class="tzb-track">
                            <div class="tzb-markers"></div>
                            <div class="tzb-progress"></div>
                            <div class="tzb-scrubber"></div>
                        </div>
                        <input type="range" class="tzb-slider" min="0" max="0" value="0">
                    </div>
                </div>

                <div class="tzb-controls">
                    <div class="tzb-playback">
                        <button class="tzb-btn tzb-prev" title="Previous">⟨</button>
                        <button class="tzb-btn tzb-play" title="Play">▶</button>
                        <button class="tzb-btn tzb-next" title="Next">⟩</button>
                    </div>

                    <div class="tzb-speed">
                        <label>Speed:</label>
                        <select class="tzb-speed-select">
                            <option value="0.5">0.5x</option>
                            <option value="1" selected>1x</option>
                            <option value="2">2x</option>
                            <option value="4">4x</option>
                        </select>
                    </div>

                    <div class="tzb-zoom">
                        <label>Zoom:</label>
                        <select class="tzb-zoom-select">
                            <option value="all" selected>All</option>
                            <option value="year">Year</option>
                            <option value="month">Month</option>
                            <option value="week">Week</option>
                            <option value="day">Day</option>
                        </select>
                    </div>

                </div>
            </div>
        `;

        // Cache element references
        elements.header = container.querySelector('.tzb-header');
        elements.toggle = container.querySelector('.tzb-toggle');
        elements.content = container.querySelector('.tzb-content');
        elements.currentDate = container.querySelector('.tzb-current-date');
        elements.currentBlock = container.querySelector('.tzb-current-block');
        elements.position = container.querySelector('.tzb-position');
        elements.track = container.querySelector('.tzb-track');
        elements.markers = container.querySelector('.tzb-markers');
        elements.progress = container.querySelector('.tzb-progress');
        elements.scrubber = container.querySelector('.tzb-scrubber');
        elements.slider = container.querySelector('.tzb-slider');
        elements.prevBtn = container.querySelector('.tzb-prev');
        elements.playBtn = container.querySelector('.tzb-play');
        elements.nextBtn = container.querySelector('.tzb-next');
        elements.speedSelect = container.querySelector('.tzb-speed-select');
        elements.zoomSelect = container.querySelector('.tzb-zoom-select');
    }

    /**
     * Bind event handlers
     */
    function bindEvents() {
        // Toggle close (hide the dropdown)
        elements.toggle?.addEventListener('click', () => {
            hide();
        });

        // Bind toolbar toggle button
        const toolbarToggle = document.getElementById('timeline-toggle-btn');
        if (toolbarToggle) {
            toolbarToggle.addEventListener('click', () => {
                toggle();
            });
        }

        // Slider events
        elements.slider?.addEventListener('input', handleSliderInput);
        elements.slider?.addEventListener('change', handleSliderChange);

        // Playback controls
        elements.prevBtn?.addEventListener('click', handlePrev);
        elements.playBtn?.addEventListener('click', handlePlayPause);
        elements.nextBtn?.addEventListener('click', handleNext);

        // Speed and zoom
        elements.speedSelect?.addEventListener('change', handleSpeedChange);
        elements.zoomSelect?.addEventListener('change', handleZoomChange);

        // Keyboard shortcuts (when timeline is focused)
        container.addEventListener('keydown', handleKeydown);
    }

    // =========================================================================
    // PUBLIC METHODS
    // =========================================================================

    /**
     * Load snapshots into the timeline
     * @param {string} sqlFile - Base SQL file name
     * @param {Array} snapshotList - Array of snapshot objects
     */
    function loadSnapshots(sqlFile, snapshotList) {
        console.log('[TimeZoomBar] loadSnapshots called with', snapshotList?.length || 0, 'snapshots for', sqlFile);
        baseSqlFile = sqlFile;

        // Sort by block number ascending (oldest first)
        snapshots = [...snapshotList].sort((a, b) => a.block_number - b.block_number);

        currentIndex = 0;
        preloadedSnapshots.clear();

        if (snapshots.length === 0) {
            console.log('[TimeZoomBar] No snapshots, hiding');
            hide();
            hideToolbarToggle();
            return;
        }

        // Show the toolbar toggle button since we have snapshots
        showToolbarToggle();

        // Update slider range
        elements.slider.min = 0;
        elements.slider.max = snapshots.length - 1;
        elements.slider.value = 0;

        // Render markers
        renderMarkers();

        // Update display
        updateDisplay();

        // Start preloading
        schedulePreload();

        // Don't auto-show, just make the toggle button visible
        // User can click the toggle to show the timeline
        console.log('[TimeZoomBar] Loaded', snapshots.length, 'snapshots for', sqlFile);
    }

    /**
     * Set current position by index
     * @param {number} index - Snapshot index
     */
    function setPosition(index) {
        if (index < 0 || index >= snapshots.length) return;

        currentIndex = index;
        elements.slider.value = index;
        updateDisplay();
        schedulePreload();

        // Dispatch event for external listeners
        dispatchPositionChange();
    }

    /**
     * Set current position by snapshot ID
     * @param {string} snapshotId - Snapshot ID
     */
    function setPositionById(snapshotId) {
        const index = snapshots.findIndex(s => s.snapshot_id === snapshotId);
        if (index >= 0) {
            setPosition(index);
        }
    }

    /**
     * Get current snapshot
     * @returns {Object|null} Current snapshot object
     */
    function getCurrentSnapshot() {
        return snapshots[currentIndex] || null;
    }

    /**
     * Show the timeline
     */
    function show() {
        container.classList.add('visible');
        isVisible = true;
        updateToolbarToggle(true);
    }

    /**
     * Hide the timeline
     */
    function hide() {
        container.classList.remove('visible');
        isVisible = false;
        stopPlayback();
        updateToolbarToggle(false);
    }

    /**
     * Toggle the timeline visibility
     */
    function toggle() {
        if (isVisible) {
            hide();
        } else {
            show();
        }
    }

    /**
     * Update the toolbar toggle button state
     */
    function updateToolbarToggle(active) {
        const toolbarToggle = document.getElementById('timeline-toggle-btn');
        if (toolbarToggle) {
            if (active) {
                toolbarToggle.classList.add('active');
            } else {
                toolbarToggle.classList.remove('active');
            }
        }
    }

    /**
     * Show the toolbar toggle button (called when snapshots are available)
     */
    function showToolbarToggle() {
        const toolbarToggle = document.getElementById('timeline-toggle-btn');
        if (toolbarToggle) {
            toolbarToggle.style.display = '';
        }
    }

    /**
     * Hide the toolbar toggle button
     */
    function hideToolbarToggle() {
        const toolbarToggle = document.getElementById('timeline-toggle-btn');
        if (toolbarToggle) {
            toolbarToggle.style.display = 'none';
        }
    }

    /**
     * Toggle playback
     */
    function togglePlayback() {
        if (isPlaying) {
            stopPlayback();
        } else {
            startPlayback();
        }
    }

    /**
     * Check if preloaded data exists for a snapshot
     * @param {string} snapshotId - Snapshot ID
     * @returns {boolean}
     */
    function isPreloaded(snapshotId) {
        return preloadedSnapshots.has(snapshotId);
    }

    /**
     * Get preloaded data for a snapshot
     * @param {string} snapshotId - Snapshot ID
     * @returns {Object|null}
     */
    function getPreloadedData(snapshotId) {
        return preloadedSnapshots.get(snapshotId) || null;
    }

    // =========================================================================
    // INTERNAL METHODS
    // =========================================================================

    /**
     * Render timeline markers
     */
    function renderMarkers() {
        if (!elements.markers) return;
        elements.markers.innerHTML = '';

        if (snapshots.length === 0) return;

        // Determine marker density based on count
        let markerInterval = 1;
        if (snapshots.length > 100) markerInterval = Math.ceil(snapshots.length / 50);
        else if (snapshots.length > 50) markerInterval = 2;

        const filteredSnapshots = getFilteredSnapshots();
        const trackWidth = elements.track.offsetWidth || 300;

        filteredSnapshots.forEach((snapshot, i) => {
            if (i % markerInterval !== 0 && i !== filteredSnapshots.length - 1) return;

            const marker = document.createElement('div');
            marker.className = 'tzb-marker';
            marker.style.left = `${(i / (filteredSnapshots.length - 1)) * 100}%`;
            marker.title = formatSnapshotLabel(snapshot);
            marker.dataset.index = i;

            // Add click handler
            marker.addEventListener('click', () => {
                setPosition(snapshots.indexOf(snapshot));
                loadSnapshot(snapshot);
            });

            elements.markers.appendChild(marker);
        });
    }

    /**
     * Get filtered snapshots based on zoom level
     */
    function getFilteredSnapshots() {
        if (zoomLevel === 'all') return snapshots;

        // For zoom levels, filter to snapshots within the time range
        const now = new Date();
        let cutoff;

        switch (zoomLevel) {
            case 'day':
                cutoff = new Date(now - 24 * 60 * 60 * 1000);
                break;
            case 'week':
                cutoff = new Date(now - 7 * 24 * 60 * 60 * 1000);
                break;
            case 'month':
                cutoff = new Date(now - 30 * 24 * 60 * 60 * 1000);
                break;
            case 'year':
                cutoff = new Date(now - 365 * 24 * 60 * 60 * 1000);
                break;
            default:
                return snapshots;
        }

        return snapshots.filter(s => {
            const ts = new Date(s.block_timestamp);
            return ts >= cutoff;
        });
    }

    /**
     * Update the display (labels, progress, scrubber)
     */
    function updateDisplay() {
        const snapshot = snapshots[currentIndex];
        if (!snapshot) return;

        // Update info labels
        if (snapshot.block_timestamp) {
            const date = new Date(snapshot.block_timestamp);
            elements.currentDate.textContent = date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } else {
            elements.currentDate.textContent = '--';
        }

        elements.currentBlock.textContent = `Block: ${snapshot.block_number?.toLocaleString() || '--'}`;
        elements.position.textContent = `${currentIndex + 1} / ${snapshots.length}`;

        // Update progress bar
        const progress = snapshots.length > 1 ? (currentIndex / (snapshots.length - 1)) * 100 : 0;
        elements.progress.style.width = `${progress}%`;

        // Update scrubber position
        elements.scrubber.style.left = `${progress}%`;
    }

    /**
     * Format snapshot label for display
     */
    function formatSnapshotLabel(snapshot) {
        let label = `Block ${snapshot.block_number?.toLocaleString() || '?'}`;

        if (snapshot.block_timestamp) {
            const date = new Date(snapshot.block_timestamp);
            label = `${date.toLocaleDateString()} - ${label}`;
        }

        if (snapshot.node_count) {
            label += ` (${snapshot.node_count.toLocaleString()} nodes)`;
        }

        return label;
    }

    /**
     * Dispatch position change event
     */
    function dispatchPositionChange() {
        const snapshot = snapshots[currentIndex];
        if (!snapshot) return;

        const event = new CustomEvent('timeZoomBarPositionChange', {
            detail: {
                index: currentIndex,
                snapshot: snapshot,
                total: snapshots.length
            }
        });

        document.dispatchEvent(event);
    }

    // =========================================================================
    // EVENT HANDLERS
    // =========================================================================

    /**
     * Handle slider input (during drag)
     */
    function handleSliderInput(e) {
        const index = parseInt(e.target.value);

        // Update display immediately for responsiveness
        currentIndex = index;
        updateDisplay();

        // Debounce the actual load
        clearTimeout(scrubDebounceTimer);
        scrubDebounceTimer = setTimeout(() => {
            // Preview only - don't actually load yet
        }, config.scrubDebounceMs);
    }

    /**
     * Handle slider change (on release)
     */
    function handleSliderChange(e) {
        clearTimeout(scrubDebounceTimer);

        const index = parseInt(e.target.value);
        currentIndex = index;
        updateDisplay();

        // Actually load the snapshot
        const snapshot = snapshots[currentIndex];
        if (snapshot) {
            loadSnapshot(snapshot);
            schedulePreload();
        }
    }

    /**
     * Handle previous button
     */
    function handlePrev() {
        if (currentIndex > 0) {
            setPosition(currentIndex - 1);
            loadSnapshot(snapshots[currentIndex]);
        }
    }

    /**
     * Handle next button
     */
    function handleNext() {
        if (currentIndex < snapshots.length - 1) {
            setPosition(currentIndex + 1);
            loadSnapshot(snapshots[currentIndex]);
        }
    }

    /**
     * Handle play/pause button
     */
    function handlePlayPause() {
        togglePlayback();
    }

    /**
     * Handle speed change
     */
    function handleSpeedChange(e) {
        playbackSpeed = parseFloat(e.target.value);

        // Restart playback with new speed if playing
        if (isPlaying) {
            stopPlayback();
            startPlayback();
        }
    }

    /**
     * Handle zoom change
     */
    function handleZoomChange(e) {
        zoomLevel = e.target.value;
        renderMarkers();
    }

    /**
     * Handle keyboard shortcuts
     */
    function handleKeydown(e) {
        switch (e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                handlePrev();
                break;
            case 'ArrowRight':
                e.preventDefault();
                handleNext();
                break;
            case ' ':
                e.preventDefault();
                togglePlayback();
                break;
            case 'Home':
                e.preventDefault();
                setPosition(0);
                loadSnapshot(snapshots[0]);
                break;
            case 'End':
                e.preventDefault();
                setPosition(snapshots.length - 1);
                loadSnapshot(snapshots[snapshots.length - 1]);
                break;
        }
    }

    // =========================================================================
    // PLAYBACK
    // =========================================================================

    /**
     * Start playback
     */
    function startPlayback() {
        if (isPlaying || snapshots.length < 2) return;

        // If at end, restart from beginning
        if (currentIndex >= snapshots.length - 1) {
            setPosition(0);
        }

        isPlaying = true;
        elements.playBtn.textContent = '⏸';
        elements.playBtn.title = 'Pause';

        const delay = config.playbackDelays[playbackSpeed] || 1000;

        const playNext = () => {
            if (!isPlaying) return;

            if (currentIndex < snapshots.length - 1) {
                setPosition(currentIndex + 1);
                loadSnapshot(snapshots[currentIndex], true); // isPlayback = true to suppress toast
                playbackInterval = setTimeout(playNext, delay);
            } else {
                stopPlayback();
            }
        };

        // Start immediately
        playbackInterval = setTimeout(playNext, delay);
    }

    /**
     * Stop playback
     */
    function stopPlayback() {
        isPlaying = false;

        if (playbackInterval) {
            clearTimeout(playbackInterval);
            playbackInterval = null;
        }

        if (elements.playBtn) {
            elements.playBtn.textContent = '▶';
            elements.playBtn.title = 'Play';
        }
    }

    // =========================================================================
    // PRELOADING
    // =========================================================================

    /**
     * Schedule preloading of adjacent snapshots
     */
    function schedulePreload() {
        // Cancel any pending preloads
        preloadQueue = [];

        // Build preload list
        const indicesToPreload = [];

        // Preload ahead
        for (let i = 1; i <= config.preloadAhead; i++) {
            const idx = currentIndex + i;
            if (idx < snapshots.length) {
                indicesToPreload.push(idx);
            }
        }

        // Preload behind
        for (let i = 1; i <= config.preloadBehind; i++) {
            const idx = currentIndex - i;
            if (idx >= 0) {
                indicesToPreload.push(idx);
            }
        }

        // Filter out already preloaded
        const toPreload = indicesToPreload.filter(idx => {
            const snapshot = snapshots[idx];
            return snapshot && !preloadedSnapshots.has(snapshot.snapshot_id);
        });

        if (toPreload.length > 0) {
            preloadQueue = toPreload.map(idx => snapshots[idx]);
            processPreloadQueue();
        }
    }

    /**
     * Process the preload queue
     */
    async function processPreloadQueue() {
        if (preloadQueue.length === 0) {
            updatePreloadStatus('Ready');
            return;
        }

        const snapshot = preloadQueue.shift();
        if (!snapshot || preloadedSnapshots.has(snapshot.snapshot_id)) {
            // Already preloaded, continue with next
            processPreloadQueue();
            return;
        }

        updatePreloadStatus(`Preloading: ${snapshot.block_number}...`);

        try {
            // Use the preload endpoint
            const response = await fetch(`/api/snapshots/preload/${baseSqlFile}?snapshot_ids=${snapshot.snapshot_id}`);

            if (response.ok) {
                const data = await response.json();
                if (data.preloaded && data.preloaded[snapshot.snapshot_id]) {
                    preloadedSnapshots.set(snapshot.snapshot_id, data.preloaded[snapshot.snapshot_id]);
                }
            }
        } catch (error) {
            console.warn('[TimeZoomBar] Preload failed for', snapshot.snapshot_id, error);
        }

        // Continue with next in queue
        processPreloadQueue();
    }

    /**
     * Update preload status display (no-op, status removed for cleaner UI)
     */
    function updatePreloadStatus(text) {
        // Status display removed to prevent layout shifts
    }

    // =========================================================================
    // SNAPSHOT LOADING
    // =========================================================================

    /**
     * Load a snapshot (delegates to Snapshots module)
     * @param {Object} snapshot - The snapshot to load
     * @param {boolean} isPlayback - If true, this is during playback (suppress toast)
     */
    async function loadSnapshot(snapshot, isPlayback = false) {
        if (!snapshot) return;

        // Dispatch event - let Snapshots module handle the actual loading
        const event = new CustomEvent('timeZoomBarLoadSnapshot', {
            detail: {
                snapshotId: snapshot.snapshot_id,
                snapshot: snapshot,
                preloadedData: preloadedSnapshots.get(snapshot.snapshot_id),
                isPlayback: isPlayback
            }
        });

        document.dispatchEvent(event);
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================

    return {
        init,
        loadSnapshots,
        setPosition,
        setPositionById,
        getCurrentSnapshot,
        show,
        hide,
        toggle,
        togglePlayback,
        startPlayback: () => startPlayback(),
        stopPlayback,
        isPreloaded,
        getPreloadedData,
        showToolbarToggle,
        hideToolbarToggle,

        // State getters
        isVisible: () => isVisible,
        isPlaying: () => isPlaying,
        getSnapshots: () => [...snapshots],
        getCurrentIndex: () => currentIndex
    };
})();
