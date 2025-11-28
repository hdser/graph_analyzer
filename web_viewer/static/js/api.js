/**
 * API Module
 * Backend API communication functions
 */

const API = {
    /**
     * Generic fetch wrapper with error handling
     */
    async fetch(url, options = {}) {
        const response = await fetch(url, {
            headers: { 
                'Content-Type': 'application/json', 
                ...options.headers 
            },
            ...options
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        return response.json();
    },

    // =========================================================================
    // Configuration
    // =========================================================================
    
    /**
     * Get application configuration
     */
    getConfig() {
        return this.fetch('/api/config');
    },

    /**
     * Get current application state
     */
    getState() {
        return this.fetch('/api/state');
    },

    // =========================================================================
    // Network Loading
    // =========================================================================
    
    /**
     * Load network from SQL files
     */
    loadNetwork(config) {
        return this.fetch('/api/load', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },

    /**
     * Get graph elements (nodes and/or edges)
     */
    getGraphElements(graphId, mode = 'full') {
        return this.fetch(`/api/graphs/${graphId}/elements?mode=${mode}`);
    },

    /**
     * Get graph edges with pagination
     */
    getGraphEdges(graphId, offset = 0, limit = 50000) {
        return this.fetch(`/api/graphs/${graphId}/edges?offset=${offset}&limit=${limit}`);
    },

    // =========================================================================
    // Metrics
    // =========================================================================
    
    /**
     * Compute metrics for graph
     */
    computeMetrics(config) {
        return this.fetch('/api/metrics', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },

    /**
     * Create composite metric
     */
    createCompositeMetric(config) {
        return this.fetch('/api/metrics/composite', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },

    /**
     * Get saved composite metrics
     */
    getSavedComposites() {
        return this.fetch('/api/metrics/composite/saved');
    },

    /**
     * Delete composite metric
     */
    deleteCompositeMetric(name) {
        return this.fetch(`/api/metrics/composite/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });
    },

    // =========================================================================
    // Anomaly Detection
    // =========================================================================
    
    /**
     * Detect anomalies in graph
     */
    detectAnomalies(config) {
        return this.fetch('/api/anomaly/detect', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },

    // =========================================================================
    // Auto-Reload
    // =========================================================================
    
    /**
     * Start auto-reload
     */
    startAutoReload(config) {
        return this.fetch('/api/auto-reload/start', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },

    /**
     * Stop auto-reload
     */
    stopAutoReload() {
        return this.fetch('/api/auto-reload/stop', {
            method: 'POST'
        });
    },

    /**
     * Get auto-reload status
     */
    getAutoReloadStatus() {
        return this.fetch('/api/auto-reload/status');
    },

    /**
     * Create SSE connection for auto-reload events
     */
    createAutoReloadSSE() {
        return new EventSource('/api/auto-reload/events');
    },

    // =========================================================================
    // Layouts
    // =========================================================================
    
    /**
     * Get cached layouts
     */
    getCachedLayouts() {
        return this.fetch('/api/layouts/cached');
    },

    /**
     * Clear cached layouts
     */
    clearCachedLayouts() {
        return this.fetch('/api/layouts/clear', {
            method: 'POST'
        });
    }
};