/**
 * Snapshot Analysis Module
 * 
 * Handles running analysis on historical snapshots including:
 * - Triggering metrics computation
 * - Running anomaly detection
 * - Displaying analysis results
 * - Batch analysis with progress
 */

// Inline SVG icons for snapshot analysis
const SnapshotIcons = {
    chart: `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M2 14V6"/><path d="M6 14V2"/><path d="M10 14V8"/><path d="M14 14V4"/></svg>`,
    warning: `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="#faad14" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M8 1.5l6.5 12H1.5L8 1.5z"/><path d="M8 6v3"/><circle cx="8" cy="11" r="0.5" fill="#faad14"/></svg>`,
    time: `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><circle cx="8" cy="8" r="6"/><path d="M8 4v4l2.5 2.5"/></svg>`
};

const SnapshotAnalysis = {
    // State
    currentAnalysis: null,
    isAnalyzing: false,
    batchEventSource: null,
    
    // Configuration
    config: {
        defaultMetricsMode: 'essential',
        defaultAnomalyAlgorithm: 'isolation_forest',
        defaultAnomalyMetrics: ['in_degree', 'out_degree', 'pagerank', 'clustering_coefficient']
    },
    
    // =========================================================================
    // Initialization
    // =========================================================================
    
    /**
     * Initialize the snapshot analysis module
     */
    init() {
        this.setupEventListeners();
        this.loadAvailableAlgorithms();
        console.log('[SNAPSHOT_ANALYSIS] Module initialized');
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Analyze button
        const analyzeBtn = document.getElementById('analyze-snapshot-btn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeCurrentSnapshot());
        }
        
        // Batch analyze button
        const batchBtn = document.getElementById('batch-analyze-btn');
        if (batchBtn) {
            batchBtn.addEventListener('click', () => this.analyzeBatch());
        }
        
        // Anomaly detection toggle
        const anomalyToggle = document.getElementById('run-anomaly');
        if (anomalyToggle) {
            anomalyToggle.addEventListener('change', (e) => {
                const anomalyOptions = document.getElementById('anomaly-options');
                if (anomalyOptions) {
                    anomalyOptions.style.display = e.target.checked ? 'block' : 'none';
                }
            });
        }
        
        // Listen for snapshot changes
        document.addEventListener('snapshot:loaded', (e) => {
            this.onSnapshotLoaded(e.detail);
        });
    },
    
    /**
     * Load available anomaly detection algorithms
     */
    async loadAvailableAlgorithms() {
        try {
            const data = await API.getAnalysisAlgorithms();
            
            if (data.available && data.algorithms) {
                const select = document.getElementById('anomaly-algorithm');
                if (select) {
                    select.innerHTML = '';
                    Object.entries(data.algorithms).forEach(([key, info]) => {
                        const option = document.createElement('option');
                        option.value = key;
                        option.textContent = info.display_name || info.name;
                        option.title = info.description || '';
                        select.appendChild(option);
                    });
                }
            }
        } catch (error) {
            console.warn('[SNAPSHOT_ANALYSIS] Failed to load algorithms:', error);
        }
    },
    
    // =========================================================================
    // Analysis Operations
    // =========================================================================
    
    /**
     * Analyze the currently loaded snapshot
     */
    async analyzeCurrentSnapshot() {
        const snapshotId = State.getCurrentSnapshotId();
        
        if (!snapshotId) {
            Toast.warning('No snapshot loaded');
            return;
        }
        
        if (this.isAnalyzing) {
            Toast.warning('Analysis already in progress');
            return;
        }
        
        const config = this.buildAnalysisConfig();
        
        this.isAnalyzing = true;
        this.showAnalysisProgress('Starting analysis...');
        
        try {
            const result = await API.analyzeSnapshot(snapshotId, config);
            
            this.currentAnalysis = result;
            this.displayAnalysisResults(result);
            
            if (result.status === 'completed') {
                Toast.success(`Analysis complete: ${result.metrics_computed.length} metrics computed`);
                
                // Emit event for other modules
                document.dispatchEvent(new CustomEvent('snapshot:analyzed', {
                    detail: { snapshotId, result }
                }));
            } else if (result.status === 'failed') {
                Toast.error('Analysis failed: ' + (result.error_message || 'Unknown error'));
            }
            
        } catch (error) {
            console.error('[SNAPSHOT_ANALYSIS] Analysis failed:', error);
            Toast.error('Analysis failed: ' + error.message);
        } finally {
            this.isAnalyzing = false;
            this.hideAnalysisProgress();
        }
    },
    
    /**
     * Run batch analysis on multiple snapshots
     */
    async analyzeBatch() {
        const baseSqlFile = State.getBaseSqlFile();
        
        if (!baseSqlFile) {
            Toast.warning('No data loaded');
            return;
        }
        
        // Get selected snapshots or all snapshots
        const snapshots = await API.listSnapshots(baseSqlFile);
        
        if (!snapshots || snapshots.length === 0) {
            Toast.warning('No snapshots available');
            return;
        }
        
        const blockNumbers = snapshots.map(s => s.block_number);
        const config = this.buildAnalysisConfig();
        
        this.isAnalyzing = true;
        this.showBatchProgress(0, blockNumbers.length);
        
        try {
            // Use SSE for batch analysis
            await this.runBatchAnalysisSSE(baseSqlFile, blockNumbers, config);
            
        } catch (error) {
            console.error('[SNAPSHOT_ANALYSIS] Batch analysis failed:', error);
            Toast.error('Batch analysis failed: ' + error.message);
        } finally {
            this.isAnalyzing = false;
            this.hideBatchProgress();
        }
    },
    
    /**
     * Run batch analysis with SSE progress
     */
    async runBatchAnalysisSSE(baseSqlFile, blockNumbers, config) {
        return new Promise((resolve, reject) => {
            const batchConfig = {
                base_sql_file: baseSqlFile,
                block_numbers: blockNumbers,
                config: config
            };
            
            // Check if SSE is available
            if (typeof EventSource === 'undefined') {
                // Fallback to regular POST
                API.analyzeBatch(batchConfig)
                    .then(resolve)
                    .catch(reject);
                return;
            }
            
            // Use fetch with streaming
            fetch('/api/snapshots/analyze-batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(batchConfig)
            }).then(response => {
                if (!response.ok) {
                    throw new Error('Batch analysis request failed');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                const processEvents = (text) => {
                    buffer += text;
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                this.handleBatchEvent(data);
                            } catch (e) {
                                console.warn('Failed to parse SSE data:', e);
                            }
                        }
                    }
                };
                
                const readStream = () => {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            resolve();
                            return;
                        }
                        processEvents(decoder.decode(value));
                        readStream();
                    }).catch(reject);
                };
                
                readStream();
            }).catch(reject);
        });
    },
    
    /**
     * Handle SSE events during batch analysis
     */
    handleBatchEvent(data) {
        if (data.event === 'progress') {
            this.showBatchProgress(data.current, data.total);
        } else if (data.event === 'complete') {
            console.log(`[SNAPSHOT_ANALYSIS] Completed: ${data.snapshot_id}`);
        } else if (data.event === 'error') {
            console.error(`[SNAPSHOT_ANALYSIS] Error: ${data.snapshot_id}`, data.error);
        } else if (data.event === 'done') {
            Toast.success(`Batch analysis complete: ${data.total_completed} snapshots`);
        }
    },
    
    /**
     * Build analysis configuration from UI
     */
    buildAnalysisConfig() {
        const metricsMode = document.getElementById('analysis-metrics-mode')?.value || 
                          this.config.defaultMetricsMode;
        const runAnomaly = document.getElementById('run-anomaly')?.checked || false;
        const algorithm = document.getElementById('anomaly-algorithm')?.value ||
                         this.config.defaultAnomalyAlgorithm;
        
        return {
            metrics_mode: metricsMode,
            recompute_metrics: false,
            run_anomaly_detection: runAnomaly,
            anomaly_algorithm: algorithm,
            anomaly_metrics: this.config.defaultAnomalyMetrics,
            anomaly_parameters: {},
            save_results: true,
            save_per_node_data: true
        };
    },
    
    // =========================================================================
    // Results Display
    // =========================================================================
    
    /**
     * Display analysis results in UI
     * @param {Object} result - SnapshotAnalysisResult from API
     */
    displayAnalysisResults(result) {
        const container = document.getElementById('analysis-results');
        if (!container) return;
        
        const metricsCount = result.metrics_computed?.length || 0;
        const anomalyCount = result.anomaly_results?.anomaly_count || 0;
        const hasAnomaly = result.anomaly_results !== null;
        
        container.innerHTML = `
            <div class="analysis-summary">
                <div class="analysis-stat">
                    <span class="stat-icon">${SnapshotIcons.chart}</span>
                    <span class="stat-value">${metricsCount}</span>
                    <span class="stat-label">metrics</span>
                </div>
                ${hasAnomaly ? `
                <div class="analysis-stat">
                    <span class="stat-icon">${SnapshotIcons.warning}</span>
                    <span class="stat-value">${anomalyCount}</span>
                    <span class="stat-label">anomalies</span>
                </div>
                ` : ''}
                <div class="analysis-stat">
                    <span class="stat-icon">${SnapshotIcons.time}</span>
                    <span class="stat-value">${result.computation_time_seconds.toFixed(1)}s</span>
                    <span class="stat-label">time</span>
                </div>
            </div>
            <div class="analysis-actions">
                <button class="btn-small" onclick="SnapshotAnalysis.showMetricsList()">
                    View Metrics
                </button>
                ${hasAnomaly ? `
                <button class="btn-small" onclick="SnapshotAnalysis.highlightAnomalies()">
                    Highlight Anomalies
                </button>
                ` : ''}
            </div>
        `;
        
        container.style.display = 'block';
    },
    
    /**
     * Show list of computed metrics
     */
    showMetricsList() {
        if (!this.currentAnalysis) return;
        
        const metrics = this.currentAnalysis.metrics_computed || [];
        const stats = this.currentAnalysis.metric_statistics || {};
        
        let html = '<div class="metrics-list"><h4>Computed Metrics</h4><ul>';
        
        metrics.forEach(metric => {
            const stat = stats[metric];
            const meanStr = stat ? ` (mean: ${stat.mean.toFixed(4)})` : '';
            html += `<li>${metric}${meanStr}</li>`;
        });
        
        html += '</ul></div>';
        
        // Show in modal or panel
        this.showModal('Analysis Metrics', html);
    },
    
    /**
     * Highlight anomalous nodes on the graph
     */
    highlightAnomalies() {
        if (!this.currentAnalysis?.anomaly_results) return;
        
        const anomalyIds = this.currentAnalysis.anomaly_results.top_anomaly_ids || [];
        
        if (anomalyIds.length === 0) {
            Toast.info('No anomalies to highlight');
            return;
        }
        
        // Emit event for cytoscape manager to handle
        document.dispatchEvent(new CustomEvent('analysis:highlightNodes', {
            detail: {
                nodeIds: anomalyIds,
                style: {
                    'background-color': '#ef4444',
                    'border-width': 3,
                    'border-color': '#dc2626'
                }
            }
        }));
        
        Toast.info(`Highlighted ${anomalyIds.length} anomalous nodes`);
    },
    
    /**
     * Load and display existing analysis for a snapshot
     */
    async loadExistingAnalysis(snapshotId) {
        try {
            const result = await API.getSnapshotAnalysis(snapshotId);
            
            if (result) {
                this.currentAnalysis = result;
                this.displayAnalysisResults(result);
                return true;
            }
        } catch (error) {
            // No existing analysis - that's OK
            console.log('[SNAPSHOT_ANALYSIS] No existing analysis for', snapshotId);
        }
        
        // Clear results display
        const container = document.getElementById('analysis-results');
        if (container) {
            container.innerHTML = '<div class="no-analysis">No analysis available</div>';
        }
        
        return false;
    },
    
    // =========================================================================
    // Event Handlers
    // =========================================================================
    
    /**
     * Handle snapshot loaded event
     */
    onSnapshotLoaded(detail) {
        const snapshotId = detail.snapshotId;
        if (snapshotId) {
            this.loadExistingAnalysis(snapshotId);
        }
    },
    
    // =========================================================================
    // UI Helpers
    // =========================================================================
    
    /**
     * Show analysis progress
     */
    showAnalysisProgress(message) {
        const btn = document.getElementById('analyze-snapshot-btn');
        if (btn) {
            btn.disabled = true;
            btn.textContent = message;
        }
        
        const progress = document.getElementById('analysis-progress');
        if (progress) {
            progress.style.display = 'block';
            progress.querySelector('.progress-text')?.textContent = message;
        }
    },
    
    /**
     * Hide analysis progress
     */
    hideAnalysisProgress() {
        const btn = document.getElementById('analyze-snapshot-btn');
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Run Analysis';
        }
        
        const progress = document.getElementById('analysis-progress');
        if (progress) {
            progress.style.display = 'none';
        }
    },
    
    /**
     * Show batch progress
     */
    showBatchProgress(current, total) {
        const progress = document.getElementById('batch-progress');
        if (progress) {
            progress.style.display = 'block';
            const pct = Math.round((current / total) * 100);
            progress.querySelector('.progress-bar')?.style.width = `${pct}%`;
            progress.querySelector('.progress-text')?.textContent = 
                `Analyzing ${current} of ${total} snapshots...`;
        }
    },
    
    /**
     * Hide batch progress
     */
    hideBatchProgress() {
        const progress = document.getElementById('batch-progress');
        if (progress) {
            progress.style.display = 'none';
        }
    },
    
    /**
     * Show a simple modal dialog
     */
    showModal(title, content) {
        // Check if modal exists
        let modal = document.getElementById('analysis-modal');
        
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'analysis-modal';
            modal.className = 'modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h3 class="modal-title"></h3>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body"></div>
                </div>
            `;
            document.body.appendChild(modal);
            
            modal.querySelector('.modal-close').addEventListener('click', () => {
                modal.style.display = 'none';
            });
            
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.style.display = 'none';
                }
            });
        }
        
        modal.querySelector('.modal-title').textContent = title;
        modal.querySelector('.modal-body').innerHTML = content;
        modal.style.display = 'flex';
    },
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    /**
     * Clean up resources
     */
    destroy() {
        if (this.batchEventSource) {
            this.batchEventSource.close();
            this.batchEventSource = null;
        }
        this.currentAnalysis = null;
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    SnapshotAnalysis.init();
});

// Export for use in other modules
window.SnapshotAnalysis = SnapshotAnalysis;