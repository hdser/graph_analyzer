/**
 * State Module
 * Global state management for the application
 */

const State = {
    // =========================================================================
    // RENDERER STATE
    // =========================================================================
    
    // Current renderer instance (GraphRendererInterface implementation)
    renderer: null,
    
    // Current renderer type ('cosmos' or 'cytoscape')
    rendererType: null,
    
    // User's renderer preference ('auto', 'cosmos', 'cytoscape')
    rendererPreference: 'auto',
    
    // WebGL capability information
    rendererCapabilities: null,
    
    // Legacy: Cytoscape instance (for backward compatibility)
    // Use getRenderer() or getCy() instead
    cy: null,
    
    // Current graph identifier
    currentGraph: null,
    
    // Current application state from server
    currentState: null,
    
    // Available configuration from server
    availableConfig: null,
    
    // Graph data cache
    graphData: {},
    
    // Neighbor highlight state (0=none, 1=in, 2=out, 3=all)
    neighborHighlightState: 0,
    
    // Performance mode flag
    performanceMode: true,
    
    // Edge loading in progress
    edgesLoading: false,
    
    // Distributions popup window reference
    distributionsWindow: null,
    
    // Style cache for metric ranges
    styleCache: {
        sizeRange: { min: 0, max: 1 },
        colorRange: { min: 0, max: 1 },
        widthRange: { min: 0, max: 1 }
    },
    
    // Current selected node data
    currentNodeData: null,
    
    // Current selected edge data
    currentEdgeData: null,
    
    // Auto-reload SSE connection
    autoReloadSSE: null,
    
    // Auto-reload enabled flag
    autoReloadEnabled: false,
    
    // ==========================================================================
    // SNAPSHOT STATE
    // ==========================================================================
    
    // Snapshot viewing state
    snapshot: {
        isActive: false,            // Currently viewing a snapshot?
        currentSnapshot: null,      // SnapshotInfo of current snapshot
        availableSnapshots: [],     // List of available snapshots
        isLoading: false,           // Loading a snapshot?
        error: null,                // Last error message
        batchProgress: null         // Batch creation progress
    },
    
    // ==========================================================================
    // RENDERER METHODS
    // ==========================================================================
    
    /**
     * Set the current renderer
     * @param {GraphRendererInterface} renderer
     */
    setRenderer(renderer) {
        this.renderer = renderer;
        this.rendererType = renderer?.getType() || null;
        
        // Maintain backward compatibility with cy reference
        if (renderer && renderer.getType() === 'cytoscape') {
            this.cy = renderer.getCy();
        } else {
            this.cy = null;
        }
        
        // Dispatch event for other modules
        document.dispatchEvent(new CustomEvent('rendererChanged', {
            detail: { 
                renderer, 
                type: this.rendererType 
            }
        }));
    },
    
    /**
     * Get the current renderer
     * @returns {GraphRendererInterface|null}
     */
    getRenderer() {
        return this.renderer;
    },
    
    /**
     * Get renderer type
     * @returns {string|null} 'cosmos' or 'cytoscape'
     */
    getRendererType() {
        return this.rendererType;
    },
    
    /**
     * Check if using cosmos.gl renderer
     * @returns {boolean}
     */
    isCosmosRenderer() {
        return this.rendererType === 'cosmos';
    },
    
    /**
     * Check if using Cytoscape.js renderer
     * @returns {boolean}
     */
    isCytoscapeRenderer() {
        return this.rendererType === 'cytoscape';
    },
    
    /**
     * Set renderer preference
     * @param {string} preference - 'auto', 'cosmos', 'cytoscape'
     */
    setRendererPreference(preference) {
        if (['auto', 'cosmos', 'cytoscape'].includes(preference)) {
            this.rendererPreference = preference;
            localStorage.setItem('rendererPreference', preference);
            
            document.dispatchEvent(new CustomEvent('rendererPreferenceChanged', {
                detail: { preference }
            }));
        }
    },
    
    /**
     * Load renderer preference from localStorage
     */
    loadRendererPreference() {
        const saved = localStorage.getItem('rendererPreference');
        if (saved && ['auto', 'cosmos', 'cytoscape'].includes(saved)) {
            this.rendererPreference = saved;
        }
    },
    
    // ==========================================================================
    // SNAPSHOT METHODS (on State object for easy access)
    // ==========================================================================
    
    /**
     * Set snapshot active state
     */
    setSnapshotActive(isActive, snapshotInfo = null) {
        this.snapshot.isActive = isActive;
        this.snapshot.currentSnapshot = snapshotInfo;
        
        // Dispatch event for UI updates
        document.dispatchEvent(new CustomEvent('snapshotStateChanged', { 
            detail: { isActive, snapshotInfo } 
        }));
    },
    
    /**
     * Set available snapshots list
     */
    setAvailableSnapshots(snapshots) {
        this.snapshot.availableSnapshots = snapshots;
    },
    
    /**
     * Get available snapshots
     */
    getAvailableSnapshots() {
        return this.snapshot.availableSnapshots;
    },
    
    /**
     * Check if currently viewing a snapshot
     */
    isViewingSnapshot() {
        return this.snapshot.isActive;
    },
    
    /**
     * Get current snapshot info
     */
    getCurrentSnapshot() {
        return this.snapshot.currentSnapshot;
    },
    
    /**
     * Set snapshot loading state
     */
    setSnapshotLoading(isLoading) {
        this.snapshot.isLoading = isLoading;
    },
    
    /**
     * Check if snapshot is loading
     */
    isSnapshotLoading() {
        return this.snapshot.isLoading;
    },
    
    /**
     * Set snapshot error
     */
    setSnapshotError(error) {
        this.snapshot.error = error;
    },
    
    /**
     * Set batch progress
     */
    setSnapshotBatchProgress(progress) {
        this.snapshot.batchProgress = progress;
    }
};

/**
 * DOM element cache for performance
 */
const DOMCache = {};

/**
 * Cache frequently accessed DOM elements
 */
function cacheDOMElements() {
    Object.assign(DOMCache, {
        // Node info
        nodeId: document.getElementById('node-id'),
        nodeCount: document.getElementById('node-count'),
        edgeCount: document.getElementById('edge-count'),
        
        // Panels
        infoPanel: document.getElementById('info-panel'),
        nodeInfo: document.getElementById('node-info'),
        edgeInfo: document.getElementById('edge-info'),
        multiInfo: document.getElementById('multi-info'),
        multiMetricsList: document.getElementById('multi-metrics-list'),
        
        // Metrics display
        allMetrics: document.getElementById('all-metrics'),
        edgeMetrics: document.getElementById('edge-metrics'),
        
        // Neighbors
        inCount: document.getElementById('in-count'),
        outCount: document.getElementById('out-count'),
        neighborInList: document.getElementById('neighbors-in-list'),
        neighborOutList: document.getElementById('neighbors-out-list'),
        
        // Status and loading
        status: document.getElementById('status'),
        loading: document.getElementById('loading'),
        
        // Main containers
        cyContainer: document.getElementById('cy'),
        toastContainer: document.getElementById('toast-container'),
        
        // Edge loading
        edgesProgress: document.getElementById('edges-progress'),
        loadEdgesBtn: document.getElementById('load-edges-btn'),
        
        // Renderer controls
        rendererIndicator: document.getElementById('renderer-indicator'),
        rendererPreferenceRadios: document.querySelectorAll('input[name="renderer-preference"]'),
        
        // Auto-reload controls
        autoReloadToggle: document.getElementById('auto-reload-toggle'),
        reloadInterval: document.getElementById('reload-interval'),
        reloadComputeMetrics: document.getElementById('reload-compute-metrics'),
        reloadStatusText: document.getElementById('reload-status-text'),
        lastReloadTime: document.getElementById('last-reload-time'),
        nextReloadTime: document.getElementById('next-reload-time'),
        lastReloadDiff: document.getElementById('last-reload-diff'),
        reloadIndicator: document.getElementById('reload-indicator'),
        
        // Composite metrics
        compositeMetric1: document.getElementById('composite-metric-1'),
        compositeMetric2: document.getElementById('composite-metric-2'),
        compositeOperation: document.getElementById('composite-operation'),
        compositeName: document.getElementById('composite-name'),
        compositeNormalize: document.getElementById('composite-normalize'),
        createCompositeBtn: document.getElementById('create-composite-btn'),
        savedCompositesList: document.getElementById('saved-composites-list'),
        refreshCompositesBtn: document.getElementById('refresh-composites-btn'),
        
        // Snapshot controls
        snapshotSection: document.getElementById('snapshots-section'),
        snapshotSelect: document.getElementById('snapshot-select'),
        loadSnapshotBtn: document.getElementById('load-snapshot-btn'),
        returnLiveBtn: document.getElementById('return-live-btn'),
        snapshotBlockInput: document.getElementById('snapshot-block-input'),
        createSnapshotBtn: document.getElementById('create-snapshot-btn'),
        batchSnapshotsBtn: document.getElementById('batch-snapshots-btn'),
        suggestBlocksBtn: document.getElementById('suggest-blocks-btn'),
        snapshotProgress: document.getElementById('snapshot-progress'),
        snapshotProgressBar: document.getElementById('snapshot-progress-bar'),
        snapshotProgressText: document.getElementById('snapshot-progress-text'),
        snapshotStatus: document.getElementById('snapshot-status')
    });
}

/**
 * Get Cytoscape instance (backward compatibility)
 * @returns {Object|null} Cytoscape instance or null if using cosmos
 */
function getCy() {
    // If using Cytoscape adapter, return the cy instance
    if (State.renderer && State.rendererType === 'cytoscape') {
        return State.renderer.getCy();
    }
    return State.cy;
}

/**
 * Set Cytoscape instance (backward compatibility)
 * @deprecated Use State.setRenderer() instead
 */
function setCy(cy) {
    State.cy = cy;
}

/**
 * Get the current renderer
 * @returns {GraphRendererInterface|null}
 */
function getRenderer() {
    return State.renderer;
}

/**
 * Update state property
 */
function updateState(key, value) {
    State[key] = value;
}

/**
 * Get state property
 */
function getState(key) {
    return State[key];
}