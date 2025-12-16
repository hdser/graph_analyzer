/**
 * Temporal Composite Metrics Module
 * 
 * Handles creation and visualization of temporal composite metrics including:
 * - Velocity (rate of change)
 * - Stability (consistency over time)
 * - Momentum (trend strength)
 * - Age-weighted metrics
 */

const TemporalComposite = {
    // State
    availableOperations: [],
    presets: [],
    currentResult: null,
    
    // Configuration
    config: {
        defaultWindow: 5,
        defaultDecay: 0.9,
        defaultAgeWeight: 0.1
    },
    
    // =========================================================================
    // Initialization
    // =========================================================================
    
    /**
     * Initialize the temporal composite module
     */
    init() {
        this.setupEventListeners();
        this.loadOperations();
        this.loadPresets();
        console.log('[TEMPORAL] Module initialized');
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Operation selector
        const opSelect = document.getElementById('temporal-operation');
        if (opSelect) {
            opSelect.addEventListener('change', (e) => {
                this.onOperationChanged(e.target.value);
            });
        }
        
        // Preview button
        const previewBtn = document.getElementById('preview-temporal');
        if (previewBtn) {
            previewBtn.addEventListener('click', () => this.previewTemporal());
        }
        
        // Create button
        const createBtn = document.getElementById('create-temporal');
        if (createBtn) {
            createBtn.addEventListener('click', () => this.createTemporal());
        }
        
        // Preset buttons
        document.querySelectorAll('.temporal-preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const presetId = e.target.dataset.presetId;
                this.applyPreset(presetId);
            });
        });
    },
    
    /**
     * Load available temporal operations
     */
    async loadOperations() {
        try {
            const data = await API.getTemporalOperations();
            this.availableOperations = data.operations || [];
            this.populateOperationSelect();
        } catch (error) {
            console.error('[TEMPORAL] Failed to load operations:', error);
        }
    },
    
    /**
     * Load preset temporal metrics
     */
    async loadPresets() {
        try {
            const data = await API.getTemporalPresets();
            this.presets = data.presets || [];
            this.populatePresetsUI();
        } catch (error) {
            console.error('[TEMPORAL] Failed to load presets:', error);
        }
    },
    
    /**
     * Populate operation selector
     */
    populateOperationSelect() {
        const select = document.getElementById('temporal-operation');
        if (!select) return;
        
        select.innerHTML = '';
        
        // Group operations by category
        const categories = {
            'Rate of Change': ['velocity', 'acceleration', 'momentum'],
            'Stability': ['stability', 'volatility'],
            'Age-Based': ['age', 'age_weighted', 'tenure_ratio']
        };
        
        Object.entries(categories).forEach(([category, ops]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = category;
            
            ops.forEach(opId => {
                const op = this.availableOperations.find(o => o.operation === opId);
                if (op) {
                    const option = document.createElement('option');
                    option.value = op.operation;
                    option.textContent = op.name;
                    option.title = op.description;
                    optgroup.appendChild(option);
                }
            });
            
            if (optgroup.children.length > 0) {
                select.appendChild(optgroup);
            }
        });
    },
    
    /**
     * Populate presets UI
     */
    populatePresetsUI() {
        const container = document.getElementById('temporal-presets');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.presets.forEach(preset => {
            const btn = document.createElement('button');
            btn.className = 'preset-btn';
            btn.dataset.presetId = preset.preset_id;
            btn.innerHTML = `
                <span class="preset-name">${preset.display_name}</span>
                <span class="preset-desc">${preset.description.substring(0, 50)}...</span>
            `;
            btn.title = preset.description;
            btn.addEventListener('click', () => this.applyPreset(preset.preset_id));
            container.appendChild(btn);
        });
    },
    
    // =========================================================================
    // Temporal Metric Operations
    // =========================================================================
    
    /**
     * Preview a temporal metric
     */
    async previewTemporal() {
        const config = this.buildConfig();
        
        if (!config) {
            Toast.warning('Please select a metric and operation');
            return;
        }
        
        this.showLoading('Computing preview...');
        
        try {
            const result = await API.previewTemporalMetric(config);
            this.displayPreview(result);
        } catch (error) {
            console.error('[TEMPORAL] Preview failed:', error);
            Toast.error('Preview failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Create and apply a temporal metric
     */
    async createTemporal() {
        const config = this.buildConfig();
        
        if (!config) {
            Toast.warning('Please select a metric and operation');
            return;
        }
        
        this.showLoading('Creating temporal metric...');
        
        try {
            const result = await API.computeTemporalMetric(config);
            this.currentResult = result;
            
            // Apply to graph visualization
            this.applyToGraph(result);
            
            Toast.success(`Created ${result.name}`);
            
            // Emit event
            document.dispatchEvent(new CustomEvent('temporal:created', {
                detail: { result }
            }));
            
        } catch (error) {
            console.error('[TEMPORAL] Creation failed:', error);
            Toast.error('Failed to create metric: ' + error.message);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Apply a preset temporal metric
     */
    async applyPreset(presetId) {
        const baseSqlFile = State.getBaseSqlFile();
        const targetBlock = State.getCurrentBlockNumber();
        
        if (!baseSqlFile || !targetBlock) {
            Toast.warning('No snapshot loaded');
            return;
        }
        
        this.showLoading('Applying preset...');
        
        try {
            const result = await API.applyTemporalPreset(presetId, {
                base_sql_file: baseSqlFile,
                target_block: targetBlock,
                save: true
            });
            
            this.currentResult = result;
            this.applyToGraph(result);
            
            Toast.success(`Applied ${result.name}`);
            
        } catch (error) {
            console.error('[TEMPORAL] Preset application failed:', error);
            Toast.error('Failed to apply preset: ' + error.message);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Build configuration from UI
     */
    buildConfig() {
        const baseMetric = document.getElementById('temporal-base-metric')?.value;
        const operation = document.getElementById('temporal-operation')?.value;
        const window = parseInt(document.getElementById('temporal-window')?.value) || 
                      this.config.defaultWindow;
        const normalize = document.getElementById('temporal-normalize')?.checked ?? true;
        
        const baseSqlFile = State.getBaseSqlFile();
        const targetBlock = State.getCurrentBlockNumber();
        
        if (!baseMetric || !operation || !baseSqlFile || !targetBlock) {
            return null;
        }
        
        // Get operation-specific parameters
        const decayFactor = parseFloat(document.getElementById('temporal-decay')?.value) ||
                          this.config.defaultDecay;
        const ageWeight = parseFloat(document.getElementById('temporal-age-weight')?.value) ||
                         this.config.defaultAgeWeight;
        
        return {
            name: `${baseMetric}_${operation}`,
            base_metric: baseMetric,
            temporal_config: {
                operation: operation,
                window_blocks: window,
                decay_factor: decayFactor,
                age_weight: ageWeight,
                normalize_output: normalize
            },
            base_sql_file: baseSqlFile,
            target_block: targetBlock,
            save: true
        };
    },
    
    // =========================================================================
    // Visualization
    // =========================================================================
    
    /**
     * Display preview results
     */
    displayPreview(result) {
        const container = document.getElementById('temporal-preview');
        if (!container) return;
        
        container.innerHTML = `
            <div class="preview-header">
                <h5>${result.name}</h5>
                <p class="formula">${result.formula_description}</p>
            </div>
            
            <div class="preview-stats">
                <div class="stat">
                    <span class="label">Min:</span>
                    <span class="value">${result.statistics.min.toFixed(4)}</span>
                </div>
                <div class="stat">
                    <span class="label">Max:</span>
                    <span class="value">${result.statistics.max.toFixed(4)}</span>
                </div>
                <div class="stat">
                    <span class="label">Mean:</span>
                    <span class="value">${result.statistics.mean.toFixed(4)}</span>
                </div>
                <div class="stat">
                    <span class="label">Nodes:</span>
                    <span class="value">${result.statistics.nodes_with_history}</span>
                </div>
            </div>
            
            <div class="preview-correlation">
                <span class="label">Correlation with base:</span>
                <span class="value">${result.correlation_with_base.toFixed(3)}</span>
            </div>
            
            <div class="preview-histogram" id="temporal-histogram"></div>
            
            <div class="preview-samples">
                <div class="top-nodes">
                    <h6>Top Nodes</h6>
                    <ul>
                        ${result.top_nodes.slice(0, 5).map(n => `
                            <li>
                                <span class="node-id">${this.formatNodeId(n.node_id)}</span>
                                <span class="node-value">${n.value.toFixed(4)}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                <div class="bottom-nodes">
                    <h6>Bottom Nodes</h6>
                    <ul>
                        ${result.bottom_nodes.slice(0, 5).map(n => `
                            <li>
                                <span class="node-id">${this.formatNodeId(n.node_id)}</span>
                                <span class="node-value">${n.value.toFixed(4)}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            </div>
        `;
        
        container.style.display = 'block';
        
        // Render histogram
        this.renderHistogram(result.histogram_bins, result.histogram_counts);
    },
    
    /**
     * Render histogram chart
     */
    renderHistogram(bins, counts) {
        const container = document.getElementById('temporal-histogram');
        if (!container || !bins || !counts) return;
        
        const width = container.clientWidth || 300;
        const height = 100;
        const maxCount = Math.max(...counts);
        const barWidth = width / counts.length;
        
        let svg = `<svg width="${width}" height="${height}" style="background: #f9fafb;">`;
        
        counts.forEach((count, i) => {
            const barHeight = (count / maxCount) * (height - 10);
            const x = i * barWidth;
            const y = height - barHeight;
            
            svg += `<rect x="${x}" y="${y}" width="${barWidth - 1}" height="${barHeight}" 
                         fill="#3b82f6" opacity="0.8"/>`;
        });
        
        svg += '</svg>';
        container.innerHTML = svg;
    },
    
    /**
     * Apply temporal metric to graph visualization
     */
    applyToGraph(result) {
        if (!result.values) return;
        
        // Emit event for cytoscape manager to handle
        document.dispatchEvent(new CustomEvent('temporal:applyMetric', {
            detail: {
                name: result.name,
                values: result.values,
                statistics: result.statistics
            }
        }));
        
        // Update metric selector if it exists
        this.addToMetricSelector(result.name);
    },
    
    /**
     * Add new metric to metric selector
     */
    addToMetricSelector(metricName) {
        const selectors = document.querySelectorAll('.metric-select, #color-metric, #size-metric');
        
        selectors.forEach(select => {
            // Check if already exists
            const existing = select.querySelector(`option[value="${metricName}"]`);
            if (!existing) {
                const option = document.createElement('option');
                option.value = metricName;
                option.textContent = `⏱️ ${metricName}`;
                option.dataset.temporal = 'true';
                select.appendChild(option);
            }
        });
    },
    
    // =========================================================================
    // Event Handlers
    // =========================================================================
    
    /**
     * Handle operation change
     */
    onOperationChanged(operation) {
        // Show/hide relevant parameter inputs
        const decayGroup = document.getElementById('temporal-decay-group');
        const ageWeightGroup = document.getElementById('temporal-age-weight-group');
        const windowGroup = document.getElementById('temporal-window-group');
        
        // Momentum needs decay
        if (decayGroup) {
            decayGroup.style.display = operation === 'momentum' ? 'block' : 'none';
        }
        
        // Age-weighted needs age weight
        if (ageWeightGroup) {
            ageWeightGroup.style.display = operation === 'age_weighted' ? 'block' : 'none';
        }
        
        // Age doesn't need window
        if (windowGroup) {
            windowGroup.style.display = operation === 'age' ? 'none' : 'block';
        }
        
        // Update window minimum based on operation
        const windowInput = document.getElementById('temporal-window');
        if (windowInput) {
            const op = this.availableOperations.find(o => o.operation === operation);
            if (op) {
                windowInput.min = op.min_window;
                windowInput.value = Math.max(parseInt(windowInput.value), op.min_window);
            }
        }
    },
    
    // =========================================================================
    // UI Helpers
    // =========================================================================
    
    /**
     * Format node ID for display
     */
    formatNodeId(nodeId) {
        if (nodeId.length > 12) {
            return nodeId.substring(0, 6) + '...' + nodeId.substring(nodeId.length - 4);
        }
        return nodeId;
    },
    
    /**
     * Show loading indicator
     */
    showLoading(message) {
        const btns = document.querySelectorAll('#preview-temporal, #create-temporal');
        btns.forEach(btn => {
            btn.disabled = true;
        });
        
        const loader = document.getElementById('temporal-loader');
        if (loader) {
            loader.textContent = message;
            loader.style.display = 'block';
        }
    },
    
    /**
     * Hide loading indicator
     */
    hideLoading() {
        const btns = document.querySelectorAll('#preview-temporal, #create-temporal');
        btns.forEach(btn => {
            btn.disabled = false;
        });
        
        const loader = document.getElementById('temporal-loader');
        if (loader) {
            loader.style.display = 'none';
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    TemporalComposite.init();
});

// Export for use in other modules
window.TemporalComposite = TemporalComposite;