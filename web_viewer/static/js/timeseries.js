/**
 * Timeseries Module
 * 
 * Handles visualization of metric evolution across historical snapshots including:
 * - Network-level metric timeseries charts
 * - Node trajectory tracking
 * - Trend analysis display
 * - Integration with animation timeline
 */

const Timeseries = {
    // State
    currentMetric: null,
    currentAggregation: 'mean',
    timeseriesData: null,
    networkSummaryData: null,
    nodeTrajectories: {},
    chartInstance: null,
    trajectoryChartInstance: null,
    
    // Configuration
    config: {
        chartHeight: 200,
        maxTrajectoryNodes: 10,
        colors: {
            primary: '#3b82f6',
            secondary: '#10b981',
            tertiary: '#f59e0b',
            grid: '#e5e7eb',
            text: '#374151',
            highlight: '#ef4444'
        }
    },
    
    // =========================================================================
    // Initialization
    // =========================================================================
    
    /**
     * Initialize the timeseries module
     */
    init() {
        this.setupEventListeners();
        this.createChartContainers();
        console.log('[TIMESERIES] Module initialized');
    },
    
    /**
     * Setup event listeners for timeseries controls
     */
    setupEventListeners() {
        // Metric selector
        const metricSelect = document.getElementById('timeseries-metric');
        if (metricSelect) {
            metricSelect.addEventListener('change', (e) => {
                this.loadMetricTimeseries(e.target.value);
            });
        }
        
        // Aggregation selector
        const aggSelect = document.getElementById('timeseries-aggregation');
        if (aggSelect) {
            aggSelect.addEventListener('change', (e) => {
                this.currentAggregation = e.target.value;
                if (this.currentMetric) {
                    this.loadMetricTimeseries(this.currentMetric);
                }
            });
        }
        
        // Track selected nodes button
        const trackBtn = document.getElementById('track-selected-nodes');
        if (trackBtn) {
            trackBtn.addEventListener('click', () => this.trackSelectedNodes());
        }
        
        // Network summary button
        const summaryBtn = document.getElementById('load-network-summary');
        if (summaryBtn) {
            summaryBtn.addEventListener('click', () => this.loadNetworkSummary());
        }
    },
    
    /**
     * Create chart container elements if they don't exist
     */
    createChartContainers() {
        // Main timeseries chart
        if (!document.getElementById('timeseries-chart')) {
            const container = document.getElementById('timeseries-panel');
            if (container) {
                const chartDiv = document.createElement('div');
                chartDiv.id = 'timeseries-chart';
                chartDiv.style.height = `${this.config.chartHeight}px`;
                chartDiv.style.width = '100%';
                container.appendChild(chartDiv);
            }
        }
        
        // Trajectory chart
        if (!document.getElementById('trajectory-chart')) {
            const container = document.getElementById('timeseries-panel');
            if (container) {
                const chartDiv = document.createElement('div');
                chartDiv.id = 'trajectory-chart';
                chartDiv.style.height = `${this.config.chartHeight}px`;
                chartDiv.style.width = '100%';
                chartDiv.style.display = 'none';
                container.appendChild(chartDiv);
            }
        }
    },
    
    // =========================================================================
    // Data Loading
    // =========================================================================
    
    /**
     * Load timeseries data for a metric
     * @param {string} metric - Metric name
     * @param {string} aggregation - Aggregation method (mean, median, etc.)
     */
    async loadMetricTimeseries(metric, aggregation = null) {
        if (!metric) return;
        
        const agg = aggregation || this.currentAggregation;
        const baseSqlFile = State.getBaseSqlFile();
        
        if (!baseSqlFile) {
            Toast.warning('No data loaded');
            return;
        }
        
        this.showLoading('Loading timeseries...');
        
        try {
            const data = await API.getMetricTimeseries(baseSqlFile, metric, agg);
            
            this.currentMetric = metric;
            this.timeseriesData = data;
            
            this.renderTimeseriesChart(data);
            this.updateTrendDisplay(data.trend);
            this.updateStatsDisplay(data.statistics);
            
            Toast.success(`Loaded ${metric} timeseries`);
            
        } catch (error) {
            console.error('[TIMESERIES] Error loading timeseries:', error);
            Toast.error('Failed to load timeseries: ' + error.message);
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Load network summary timeseries
     */
    async loadNetworkSummary() {
        const baseSqlFile = State.getBaseSqlFile();
        
        if (!baseSqlFile) {
            Toast.warning('No data loaded');
            return;
        }
        
        this.showLoading('Loading network summary...');
        
        try {
            const data = await API.getNetworkSummaryTimeseries(baseSqlFile);
            
            this.networkSummaryData = data;
            this.renderNetworkSummaryChart(data);
            
            Toast.success('Loaded network summary');
            
        } catch (error) {
            console.error('[TIMESERIES] Error loading network summary:', error);
            Toast.error('Failed to load network summary');
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Load trajectories for selected nodes
     */
    async trackSelectedNodes() {
        const selectedNodes = State.getSelectedNodes();
        
        if (!selectedNodes || selectedNodes.length === 0) {
            Toast.warning('No nodes selected');
            return;
        }
        
        if (selectedNodes.length > this.config.maxTrajectoryNodes) {
            Toast.warning(`Maximum ${this.config.maxTrajectoryNodes} nodes for trajectory tracking`);
            return;
        }
        
        const metric = this.currentMetric || 'in_degree';
        const baseSqlFile = State.getBaseSqlFile();
        
        if (!baseSqlFile) {
            Toast.warning('No data loaded');
            return;
        }
        
        this.showLoading('Loading trajectories...');
        
        try {
            const data = await API.getNodeTrajectories(baseSqlFile, selectedNodes, metric);
            
            this.nodeTrajectories = data.trajectories;
            this.renderTrajectoryChart(data);
            
            // Show trajectory chart
            const trajectoryChart = document.getElementById('trajectory-chart');
            if (trajectoryChart) {
                trajectoryChart.style.display = 'block';
            }
            
            Toast.success(`Tracking ${Object.keys(data.trajectories).length} nodes`);
            
        } catch (error) {
            console.error('[TIMESERIES] Error loading trajectories:', error);
            Toast.error('Failed to load trajectories');
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Load available metrics for timeseries
     */
    async loadAvailableMetrics() {
        const baseSqlFile = State.getBaseSqlFile();
        if (!baseSqlFile) return;
        
        try {
            const data = await API.getTimeseriesAvailableMetrics(baseSqlFile);
            
            const select = document.getElementById('timeseries-metric');
            if (select && data.metrics) {
                select.innerHTML = '<option value="">Select metric...</option>';
                data.metrics.forEach(metric => {
                    const option = document.createElement('option');
                    option.value = metric;
                    option.textContent = metric;
                    select.appendChild(option);
                });
            }
            
        } catch (error) {
            console.error('[TIMESERIES] Error loading available metrics:', error);
        }
    },
    
    // =========================================================================
    // Chart Rendering
    // =========================================================================
    
    /**
     * Render the main timeseries chart
     * @param {Object} data - TimeseriesData from API
     */
    renderTimeseriesChart(data) {
        const container = document.getElementById('timeseries-chart');
        if (!container) return;
        
        const points = data.data_points || [];
        if (points.length === 0) {
            container.innerHTML = '<div class="no-data">No data available</div>';
            return;
        }
        
        // Extract data for chart
        const labels = points.map(p => this.formatBlockLabel(p.block_number, p.timestamp));
        const values = points.map(p => p.value);
        const blockNumbers = points.map(p => p.block_number);
        
        // Create canvas if needed
        let canvas = container.querySelector('canvas');
        if (!canvas) {
            container.innerHTML = '';
            canvas = document.createElement('canvas');
            container.appendChild(canvas);
        }
        
        // Destroy existing chart
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }
        
        // Create chart using Chart.js (assumed to be loaded)
        if (typeof Chart !== 'undefined') {
            this.chartInstance = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: `${data.metric} (${data.aggregation})`,
                        data: values,
                        borderColor: this.config.colors.primary,
                        backgroundColor: this.config.colors.primary + '20',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            callbacks: {
                                title: (items) => {
                                    const idx = items[0].dataIndex;
                                    return `Block ${blockNumbers[idx]}`;
                                },
                                label: (item) => {
                                    return `${data.metric}: ${item.raw.toFixed(4)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Snapshot'
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 0
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: data.metric
                            }
                        }
                    },
                    onClick: (event, elements) => {
                        if (elements.length > 0) {
                            const idx = elements[0].index;
                            const blockNumber = blockNumbers[idx];
                            this.onChartPointClick(blockNumber);
                        }
                    }
                }
            });
        } else {
            // Fallback: simple SVG chart
            this.renderSimpleChart(container, points, data.metric);
        }
    },
    
    /**
     * Render network summary chart (node/edge counts over time)
     * @param {Object} data - NetworkTimeseriesData from API
     */
    renderNetworkSummaryChart(data) {
        const container = document.getElementById('timeseries-chart');
        if (!container) return;
        
        const points = data.data_points || [];
        if (points.length === 0) {
            container.innerHTML = '<div class="no-data">No data available</div>';
            return;
        }
        
        const labels = points.map(p => this.formatBlockLabel(p.block_number, p.timestamp));
        const nodeCounts = points.map(p => p.node_count);
        const edgeCounts = points.map(p => p.edge_count);
        const densities = points.map(p => p.density * 1000); // Scale for visibility
        
        let canvas = container.querySelector('canvas');
        if (!canvas) {
            container.innerHTML = '';
            canvas = document.createElement('canvas');
            container.appendChild(canvas);
        }
        
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }
        
        if (typeof Chart !== 'undefined') {
            this.chartInstance = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Nodes',
                            data: nodeCounts,
                            borderColor: this.config.colors.primary,
                            backgroundColor: 'transparent',
                            yAxisID: 'y'
                        },
                        {
                            label: 'Edges',
                            data: edgeCounts,
                            borderColor: this.config.colors.secondary,
                            backgroundColor: 'transparent',
                            yAxisID: 'y'
                        },
                        {
                            label: 'Density (×1000)',
                            data: densities,
                            borderColor: this.config.colors.tertiary,
                            backgroundColor: 'transparent',
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: { display: true, text: 'Snapshot' }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Count' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Density (×1000)' },
                            grid: { drawOnChartArea: false }
                        }
                    }
                }
            });
        }
    },
    
    /**
     * Render trajectory chart for multiple nodes
     * @param {Object} data - NodeTrajectoriesResponse from API
     */
    renderTrajectoryChart(data) {
        const container = document.getElementById('trajectory-chart');
        if (!container) return;
        
        const trajectories = data.trajectories || {};
        const nodeIds = Object.keys(trajectories);
        
        if (nodeIds.length === 0) {
            container.innerHTML = '<div class="no-data">No trajectories available</div>';
            return;
        }
        
        const blockNumbers = data.block_numbers || [];
        const labels = blockNumbers.map(b => `Block ${b}`);
        
        // Build datasets for each node
        const colors = this.generateColors(nodeIds.length);
        const datasets = nodeIds.map((nodeId, idx) => {
            const trajectory = trajectories[nodeId];
            const values = trajectory.values.map(v => v.exists ? v.value : null);
            
            return {
                label: this.formatNodeLabel(nodeId),
                data: values,
                borderColor: colors[idx],
                backgroundColor: 'transparent',
                tension: 0.2,
                spanGaps: true
            };
        });
        
        let canvas = container.querySelector('canvas');
        if (!canvas) {
            container.innerHTML = '';
            canvas = document.createElement('canvas');
            container.appendChild(canvas);
        }
        
        if (this.trajectoryChartInstance) {
            this.trajectoryChartInstance.destroy();
        }
        
        if (typeof Chart !== 'undefined') {
            this.trajectoryChartInstance = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: { display: true, text: 'Snapshot' }
                        },
                        y: {
                            display: true,
                            title: { display: true, text: data.metric }
                        }
                    }
                }
            });
        }
    },
    
    /**
     * Simple SVG fallback chart
     */
    renderSimpleChart(container, points, metric) {
        const width = container.clientWidth || 400;
        const height = this.config.chartHeight;
        const padding = 40;
        
        const values = points.map(p => p.value);
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const range = maxVal - minVal || 1;
        
        const xScale = (width - padding * 2) / (points.length - 1);
        const yScale = (height - padding * 2) / range;
        
        const pathPoints = points.map((p, i) => {
            const x = padding + i * xScale;
            const y = height - padding - (p.value - minVal) * yScale;
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        }).join(' ');
        
        container.innerHTML = `
            <svg width="${width}" height="${height}" style="background: #f9fafb;">
                <path d="${pathPoints}" fill="none" stroke="${this.config.colors.primary}" stroke-width="2"/>
                <text x="${padding}" y="20" font-size="12">${metric}</text>
            </svg>
        `;
    },
    
    // =========================================================================
    // UI Updates
    // =========================================================================
    
    /**
     * Update trend display with analysis results
     * @param {Object} trend - TrendAnalysis from API
     */
    updateTrendDisplay(trend) {
        const trendEl = document.getElementById('ts-trend');
        const changeEl = document.getElementById('ts-change');
        
        if (!trend) {
            if (trendEl) trendEl.textContent = '-';
            if (changeEl) changeEl.textContent = '-';
            return;
        }
        
        if (trendEl) {
            const direction = trend.trend_direction;
            const icons = {
                'increasing': '↗',
                'decreasing': '↘',
                'stable': '→',
                'volatile': '↕'
            };
            trendEl.textContent = `${icons[direction] || ''} ${direction}`;
            trendEl.className = `trend-${direction}`;
        }
        
        if (changeEl) {
            const pct = trend.percent_change.toFixed(1);
            const sign = trend.percent_change >= 0 ? '+' : '';
            changeEl.textContent = `${sign}${pct}%`;
            changeEl.className = trend.percent_change >= 0 ? 'positive' : 'negative';
        }
    },
    
    /**
     * Update statistics display
     * @param {Object} stats - TimeseriesStatistics from API
     */
    updateStatsDisplay(stats) {
        const container = document.getElementById('timeseries-stats');
        if (!container || !stats) return;
        
        container.innerHTML = `
            <div class="stat-row">
                <span class="stat-label">Min:</span>
                <span class="stat-value">${stats.min.toFixed(4)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Max:</span>
                <span class="stat-value">${stats.max.toFixed(4)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Mean:</span>
                <span class="stat-value">${stats.mean.toFixed(4)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Std:</span>
                <span class="stat-value">${stats.std.toFixed(4)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Points:</span>
                <span class="stat-value">${stats.count}</span>
            </div>
        `;
    },
    
    /**
     * Highlight a specific point on the chart (for animation sync)
     * @param {number} blockNumber - Block number to highlight
     */
    highlightBlock(blockNumber) {
        if (!this.chartInstance || !this.timeseriesData) return;
        
        const points = this.timeseriesData.data_points || [];
        const idx = points.findIndex(p => p.block_number === blockNumber);
        
        if (idx >= 0) {
            // Update point styling
            this.chartInstance.setActiveElements([
                { datasetIndex: 0, index: idx }
            ]);
            this.chartInstance.update('none');
        }
    },
    
    /**
     * Sync with animation frame
     * @param {number} frameIndex - Current animation frame index
     * @param {number} blockNumber - Current block number
     */
    syncWithAnimation(frameIndex, blockNumber) {
        this.highlightBlock(blockNumber);
    },
    
    // =========================================================================
    // Event Handlers
    // =========================================================================
    
    /**
     * Handle click on chart point - jump to that snapshot
     * @param {number} blockNumber - Clicked block number
     */
    onChartPointClick(blockNumber) {
        console.log('[TIMESERIES] Chart point clicked:', blockNumber);
        
        // Emit event for snapshots module to handle
        const event = new CustomEvent('timeseries:blockSelected', {
            detail: { blockNumber }
        });
        document.dispatchEvent(event);
        
        // If animation is available, jump to that frame
        if (typeof Snapshots !== 'undefined' && Snapshots.jumpToBlock) {
            Snapshots.jumpToBlock(blockNumber);
        }
    },
    
    // =========================================================================
    // Utility Methods
    // =========================================================================
    
    /**
     * Format block number for chart label
     */
    formatBlockLabel(blockNumber, timestamp) {
        if (timestamp) {
            const date = new Date(timestamp);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        }
        return `#${blockNumber}`;
    },
    
    /**
     * Format node ID for display
     */
    formatNodeLabel(nodeId) {
        if (nodeId.length > 10) {
            return nodeId.substring(0, 6) + '...' + nodeId.substring(nodeId.length - 4);
        }
        return nodeId;
    },
    
    /**
     * Generate distinct colors for multiple series
     */
    generateColors(count) {
        const baseColors = [
            '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
            '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
        ];
        
        const colors = [];
        for (let i = 0; i < count; i++) {
            colors.push(baseColors[i % baseColors.length]);
        }
        return colors;
    },
    
    /**
     * Show loading indicator
     */
    showLoading(message = 'Loading...') {
        const container = document.getElementById('timeseries-panel');
        if (container) {
            let loader = container.querySelector('.ts-loader');
            if (!loader) {
                loader = document.createElement('div');
                loader.className = 'ts-loader';
                container.insertBefore(loader, container.firstChild);
            }
            loader.textContent = message;
            loader.style.display = 'block';
        }
    },
    
    /**
     * Hide loading indicator
     */
    hideLoading() {
        const loader = document.querySelector('.ts-loader');
        if (loader) {
            loader.style.display = 'none';
        }
    },
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    
    /**
     * Destroy charts and clean up state
     */
    destroy() {
        if (this.chartInstance) {
            this.chartInstance.destroy();
            this.chartInstance = null;
        }
        if (this.trajectoryChartInstance) {
            this.trajectoryChartInstance.destroy();
            this.trajectoryChartInstance = null;
        }
        this.timeseriesData = null;
        this.nodeTrajectories = {};
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    Timeseries.init();
});

// Export for use in other modules
window.Timeseries = Timeseries;