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

    /**
     * Get updated node data for incremental refresh
     * Used by auto-reload to update frontend without full reload
     */
    getNodeUpdates(graphId, nodeIds = null) {
        let url = `/api/graphs/${graphId}/node-updates`;
        if (nodeIds && nodeIds.length > 0) {
            url += `?node_ids=${nodeIds.join(',')}`;
        }
        return this.fetch(url);
    },

    /**
     * Get neighbors of specified nodes
     * @param {string} graphId - Graph ID
     * @param {string[]} nodeIds - List of node IDs
     * @param {string} direction - "in", "out", or "both"
     */
    getNeighbors(graphId, nodeIds, direction = 'both') {
        return this.fetch(`/api/network/graphs/${graphId}/neighbors`, {
            method: 'POST',
            body: JSON.stringify({ node_ids: nodeIds, direction: direction })
        });
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


// =============================================================================
// SNAPSHOT API
// =============================================================================

const SnapshotAPI = {
    /**
     * List available snapshots
     * @param {string} baseSqlFile - Optional filter by SQL file
     * @returns {Promise<{snapshots: Array, total_count: number}>}
     */
    async listSnapshots(baseSqlFile = null) {
        const params = baseSqlFile ? `?base_sql_file=${encodeURIComponent(baseSqlFile)}` : '';
        return API.fetch(`/api/snapshots${params}`);
    },
    
    /**
     * Get available SQL files with snapshot templates
     * @returns {Promise<{sql_files: string[]}>}
     */
    async getAvailableSqlFiles() {
        return API.fetch('/api/snapshots/available-sql-files');
    },
    
    /**
     * Get snapshot metadata
     * @param {string} snapshotId
     * @returns {Promise<SnapshotInfo>}
     */
    async getSnapshotInfo(snapshotId) {
        return API.fetch(`/api/snapshots/${encodeURIComponent(snapshotId)}`);
    },
    
    /**
     * Get full snapshot data (edges, layout, metrics)
     * NOTE: For large snapshots, use getSnapshotNodes + getSnapshotEdges instead
     * @param {string} snapshotId
     * @returns {Promise<SnapshotData>}
     */
    async getSnapshotData(snapshotId) {
        return API.fetch(`/api/snapshots/${encodeURIComponent(snapshotId)}/data`);
    },
    
    /**
     * Get snapshot nodes with positions and metrics (no edges)
     * Fast endpoint for initial render
     * @param {string} snapshotId
     * @returns {Promise<{elements: Array, metadata: Object}>}
     */
    async getSnapshotNodes(snapshotId) {
        return API.fetch(`/api/snapshots/${encodeURIComponent(snapshotId)}/nodes`);
    },
    
    /**
     * Get snapshot edges with pagination
     * @param {string} snapshotId
     * @param {number} offset - Starting offset
     * @param {number} limit - Max edges to return
     * @returns {Promise<{edges: Array, total: number, has_more: boolean}>}
     */
    async getSnapshotEdges(snapshotId, offset = 0, limit = 50000) {
        return API.fetch(
            `/api/snapshots/${encodeURIComponent(snapshotId)}/edges?offset=${offset}&limit=${limit}`
        );
    },
    
    /**
     * Create a new snapshot
     * @param {Object} params - {base_sql_file, block_number, label?, metrics_mode}
     * @returns {Promise<SnapshotInfo>}
     */
    async createSnapshot(params) {
        return API.fetch('/api/snapshots/create', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    },
    
    /**
     * Create batch snapshots with SSE progress
     * @param {Object} params - {base_sql_file, block_numbers, metrics_mode}
     * @param {Function} onProgress - Callback for progress events
     * @param {Function} onComplete - Callback for individual completion
     * @param {Function} onDone - Callback for batch completion
     * @param {Function} onError - Callback for errors
     * @returns {EventSource} - EventSource for manual close if needed
     */
    createBatch(params, onProgress, onComplete, onDone, onError) {
        const url = '/api/snapshots/create-batch';
        
        // For SSE, we need to POST then listen
        const eventSource = new EventSource(url + '?' + new URLSearchParams({
            base_sql_file: params.base_sql_file,
            block_numbers: params.block_numbers.join(','),
            metrics_mode: params.metrics_mode || 'standard'
        }));
        
        // Actually, SSE requires POST body, so we need a different approach
        // Use fetch with ReadableStream instead
        this._createBatchWithFetch(params, onProgress, onComplete, onDone, onError);
        
        return { close: () => {} }; // Placeholder
    },
    
    /**
     * Internal: Create batch using fetch and SSE
     */
    async _createBatchWithFetch(params, onProgress, onComplete, onDone, onError) {
        try {
            const response = await fetch('/api/snapshots/create-batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            // Check if SSE response
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('text/event-stream')) {
                // Parse SSE stream
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    
                    let currentEvent = null;
                    let currentData = '';
                    
                    for (const line of lines) {
                        if (line.startsWith('event:')) {
                            currentEvent = line.slice(6).trim();
                        } else if (line.startsWith('data:')) {
                            currentData = line.slice(5).trim();
                            
                            if (currentEvent && currentData) {
                                try {
                                    const data = JSON.parse(currentData);
                                    
                                    switch (currentEvent) {
                                        case 'progress':
                                            if (onProgress) onProgress(data);
                                            break;
                                        case 'complete':
                                            if (onComplete) onComplete(data);
                                            break;
                                        case 'done':
                                            if (onDone) onDone(data);
                                            break;
                                        case 'error':
                                            if (onError) onError(data);
                                            break;
                                    }
                                } catch (e) {
                                    console.error('Failed to parse SSE data:', e);
                                }
                                currentEvent = null;
                                currentData = '';
                            }
                        }
                    }
                }
            } else {
                // JSON response (fallback without SSE)
                const result = await response.json();
                if (onDone) {
                    onDone({ 
                        total_created: result.snapshots?.length || 0,
                        snapshots: result.snapshots || []
                    });
                }
            }
        } catch (error) {
            console.error('Batch creation error:', error);
            if (onError) onError({ error: error.message });
        }
    },
    
    /**
     * Get suggested block numbers for snapshot creation
     * @param {Object} params - {base_sql_file, interval, start_date?, end_date?, count?}
     * @returns {Promise<{suggestions: Array}>}
     */
    async suggestBlockNumbers(params) {
        return API.fetch('/api/snapshots/suggest', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    },
    
    /**
     * Delete a snapshot
     * @param {string} snapshotId
     * @returns {Promise<{success: boolean}>}
     */
    async deleteSnapshot(snapshotId) {
        return API.fetch(`/api/snapshots/${encodeURIComponent(snapshotId)}`, {
            method: 'DELETE'
        });
    },
    
    /**
     * Get storage statistics
     * @returns {Promise<StorageStats>}
     */
    async getStorageStats() {
        return API.fetch('/api/snapshots/storage-stats');
    },
    
    // =========================================================================
    // Comparison & Animation
    // =========================================================================
    
    /**
     * Compare two snapshots
     * @param {string} fromSnapshotId - Earlier snapshot
     * @param {string} toSnapshotId - Later snapshot
     * @returns {Promise<ComparisonResult>}
     */
    async compareSnapshots(fromSnapshotId, toSnapshotId) {
        return API.fetch(
            `/api/snapshots/compare/${encodeURIComponent(fromSnapshotId)}/${encodeURIComponent(toSnapshotId)}`
        );
    },
    
    /**
     * Get animation data for all snapshots
     * @param {string} baseSqlFile - SQL file to get snapshots for
     * @returns {Promise<AnimationData>}
     */
    async getAnimationData(baseSqlFile) {
        return API.fetch(`/api/snapshots/animation/${encodeURIComponent(baseSqlFile)}`);
    }
};