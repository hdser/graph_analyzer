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

    /**
     * Get available analysis algorithms
     */
    getAnalysisAlgorithms() {
        return this.fetch('/api/anomaly/algorithms');
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
    },

    // =========================================================================
    // Snapshot Analysis
    // =========================================================================
    
    /**
     * Run analysis on a snapshot
     * @param {string} snapshotId - Snapshot identifier
     * @param {Object} config - Analysis configuration
     * @returns {Promise<Object>} SnapshotAnalysisResult
     */
    analyzeSnapshot(snapshotId, config = {}) {
        return this.fetch(`/api/snapshots/${encodeURIComponent(snapshotId)}/analyze`, {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },
    
    /**
     * Get existing analysis results for a snapshot
     * @param {string} snapshotId - Snapshot identifier
     * @returns {Promise<Object|null>} SnapshotAnalysisResult or null if not found
     */
    async getSnapshotAnalysis(snapshotId) {
        try {
            return await this.fetch(`/api/snapshots/${encodeURIComponent(snapshotId)}/analysis`);
        } catch (e) {
            if (e.message.includes('404')) return null;
            throw e;
        }
    },
    
    /**
     * Get metric values from a snapshot analysis
     * @param {string} snapshotId - Snapshot identifier
     * @param {string} metricName - Metric name
     * @param {boolean} includeValues - Include per-node values
     * @returns {Promise<Object>} MetricValuesResponse
     */
    getSnapshotMetricValues(snapshotId, metricName, includeValues = true) {
        const url = `/api/snapshots/${encodeURIComponent(snapshotId)}/metrics/${encodeURIComponent(metricName)}?include_values=${includeValues}`;
        return this.fetch(url);
    },
    
    /**
     * List analyzed snapshots
     * @param {string} baseSqlFile - Base SQL file name
     * @returns {Promise<Object>} AnalyzedSnapshotsListResponse
     */
    listAnalyzedSnapshots(baseSqlFile) {
        return this.fetch(`/api/snapshots/analyzed?base_sql_file=${encodeURIComponent(baseSqlFile)}`);
    },
    
    /**
     * Check if snapshot has analysis
     * @param {string} snapshotId - Snapshot identifier
     * @returns {Promise<boolean>}
     */
    async hasSnapshotAnalysis(snapshotId) {
        try {
            const data = await this.fetch(`/api/snapshots/${encodeURIComponent(snapshotId)}/has-analysis`);
            return data.has_analysis;
        } catch (e) {
            return false;
        }
    },
    
    /**
     * Run batch analysis
     * @param {Object} config - Batch analysis configuration
     * @returns {Promise<Object>} BatchAnalysisResult
     */
    analyzeBatch(config) {
        return this.fetch('/api/snapshots/analyze/batch', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },
    
    // =========================================================================
    // Timeseries
    // =========================================================================
    
    /**
     * Get metric timeseries data
     * @param {string} baseSqlFile - Base SQL file
     * @param {string} metric - Metric name
     * @param {Object} options - Query options
     * @returns {Promise<Object>} TimeseriesData
     */
    getMetricTimeseries(baseSqlFile, metric, options = {}) {
        const params = new URLSearchParams({
            base_sql_file: baseSqlFile,
            metric: metric,
            aggregation: options.aggregation || 'mean',
            include_trend: options.includeTrend !== false
        });
        
        if (options.startBlock) params.append('start_block', options.startBlock);
        if (options.endBlock) params.append('end_block', options.endBlock);
        
        return this.fetch(`/api/timeseries/metric?${params}`);
    },
    
    /**
     * Get network summary timeseries
     * @param {string} baseSqlFile - Base SQL file
     * @returns {Promise<Object>} NetworkTimeseriesData
     */
    getNetworkTimeseries(baseSqlFile) {
        return this.fetch(`/api/timeseries/network?base_sql_file=${encodeURIComponent(baseSqlFile)}`);
    },
    
    /**
     * Get node trajectories
     * @param {string} baseSqlFile - Base SQL file
     * @param {string[]} nodeIds - Node IDs
     * @param {string} metric - Metric name
     * @param {Object} options - Query options
     * @returns {Promise<Object>} NodeTrajectoriesResponse
     */
    getNodeTrajectories(baseSqlFile, nodeIds, metric, options = {}) {
        return this.fetch('/api/timeseries/trajectories', {
            method: 'POST',
            body: JSON.stringify({
                base_sql_file: baseSqlFile,
                node_ids: nodeIds,
                metric: metric,
                include_statistics: options.includeStatistics !== false,
                include_trend: options.includeTrend || false
            })
        });
    },
    
    /**
     * Compare distributions between snapshots
     * @param {string} baseSqlFile - Base SQL file
     * @param {string} metric - Metric name
     * @param {number} fromBlock - Earlier block number
     * @param {number} toBlock - Later block number
     * @returns {Promise<Object>} DistributionComparison
     */
    compareDistributions(baseSqlFile, metric, fromBlock, toBlock) {
        return this.fetch('/api/timeseries/distributions/compare', {
            method: 'POST',
            body: JSON.stringify({
                base_sql_file: baseSqlFile,
                metric: metric,
                from_block: fromBlock,
                to_block: toBlock
            })
        });
    },
    
    /**
     * Get available metrics for timeseries
     * @param {string} baseSqlFile - Base SQL file
     * @returns {Promise<Object>} Available metrics
     */
    getTimeseriesMetrics(baseSqlFile) {
        return this.fetch(`/api/timeseries/available-metrics?base_sql_file=${encodeURIComponent(baseSqlFile)}`);
    },
    
    // =========================================================================
    // Temporal Composite
    // =========================================================================
    
    /**
     * Get available temporal operations
     * @returns {Promise<Object>} AvailableOperationsResponse
     */
    getTemporalOperations() {
        return this.fetch('/api/temporal/operations');
    },
    
    /**
     * Get temporal presets
     * @returns {Promise<Object>} TemporalPresetsResponse
     */
    getTemporalPresets() {
        return this.fetch('/api/temporal/presets');
    },
    
    /**
     * Compute temporal composite metric
     * @param {Object} config - TemporalCompositeConfig
     * @returns {Promise<Object>} TemporalCompositeResult
     */
    computeTemporalMetric(config) {
        return this.fetch('/api/temporal/compute', {
            method: 'POST',
            body: JSON.stringify({ config })
        });
    },
    
    /**
     * Preview temporal composite metric
     * @param {Object} config - TemporalCompositeConfig
     * @param {number} sampleSize - Number of top/bottom nodes
     * @returns {Promise<Object>} TemporalPreviewResult
     */
    previewTemporalMetric(config, sampleSize = 10) {
        return this.fetch('/api/temporal/preview', {
            method: 'POST',
            body: JSON.stringify({ config, sample_size: sampleSize })
        });
    },
    
    /**
     * Apply temporal preset
     * @param {string} presetId - Preset identifier
     * @param {string} baseSqlFile - Base SQL file
     * @param {number} targetBlock - Target block number
     * @param {Object} options - Options
     * @returns {Promise<Object>} TemporalCompositeResult
     */
    applyTemporalPreset(presetId, baseSqlFile, targetBlock, options = {}) {
        return this.fetch(`/api/temporal/presets/${encodeURIComponent(presetId)}/apply`, {
            method: 'POST',
            body: JSON.stringify({
                base_sql_file: baseSqlFile,
                target_block: targetBlock,
                window_blocks: options.windowBlocks,
                save: options.save !== false
            })
        });
    },
    
    /**
     * Get saved temporal metrics
     * @param {string} baseSqlFile - Base SQL file
     * @returns {Promise<Object>} List of saved temporal metrics
     */
    getSavedTemporalMetrics(baseSqlFile) {
        return this.fetch(`/api/temporal/saved?base_sql_file=${encodeURIComponent(baseSqlFile)}`);
    }
};


// =============================================================================
// SNAPSHOT API (Separate object for snapshot-specific operations)
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
        // Use fetch with ReadableStream for SSE
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