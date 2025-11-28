/**
 * State Module
 * Global state management for the application
 */

const State = {
    // Cytoscape instance
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
    autoReloadEnabled: false
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
        refreshCompositesBtn: document.getElementById('refresh-composites-btn')
    });
}

/**
 * Get Cytoscape instance
 */
function getCy() {
    return State.cy;
}

/**
 * Set Cytoscape instance
 */
function setCy(cy) {
    State.cy = cy;
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