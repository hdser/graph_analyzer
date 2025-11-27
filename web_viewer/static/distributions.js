/**
 * Distributions Analysis Window
 * Standalone window for analyzing metric distributions
 * Features: Histograms with stats, Scatter plots with zoom and selection
 */

// Global state
let nodeData = [];
let allMetrics = [];
let selectedMetrics = new Set();
let histogramCharts = {};
let scatterChart = null;
let useSelectedOnly = false;
let currentView = 'histograms';
let selectedScatterPoints = [];
let scatterNodeData = []; // Store node data for scatter plot selection

// Anomaly detection state
let anomalyAlgorithms = {};
let lastAnomalyResult = null;
let anomalyInitialized = false;
let anomalyHistogramChart = null;

// Communication with parent window
window.addEventListener('message', (event) => {
    console.log('[Distributions] Received message:', event.data.type);
    
    if (event.data.type === 'DISTRIBUTION_DATA') {
        // Data is nested under event.data.data
        const payload = event.data.data || event.data;
        nodeData = payload.nodes || [];
        
        console.log('[Distributions] Received', nodeData.length, 'nodes');
        if (nodeData.length > 0) {
            console.log('[Distributions] Sample node keys:', Object.keys(nodeData[0]));
        }
        
        // Mark selected nodes
        const selectedIds = new Set(payload.selectedIds || []);
        nodeData.forEach(node => {
            node._selected = selectedIds.has(node.id);
        });
        
        initializeMetrics();
        updateNodeInfo();
    } else if (event.data.type === 'SELECTION_UPDATE') {
        // Update selection state - data is nested under event.data.data
        const payload = event.data.data || event.data;
        const selectedIds = new Set(payload.selectedIds || []);
        nodeData.forEach(node => {
            node._selected = selectedIds.has(node.id);
        });
        
        if (useSelectedOnly) {
            refreshAllCharts();
        }
        updateNodeInfo();
    }
});

// Request data from parent on load
window.addEventListener('load', () => {
    if (window.opener) {
        window.opener.postMessage({ type: 'REQUEST_DISTRIBUTION_DATA' }, '*');
    }
    setupEventListeners();
});

function setupEventListeners() {
    // Selected only filter
    document.getElementById('selected-only').addEventListener('change', (e) => {
        useSelectedOnly = e.target.checked;
        refreshAllCharts();
        updateNodeInfo();
    });

    // View tabs
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const view = tab.dataset.view;
            switchView(view);
        });
    });

    // Clear all button
    document.getElementById('clear-all-btn').addEventListener('click', () => {
        selectedMetrics.clear();
        Object.values(histogramCharts).forEach(chart => chart.destroy());
        histogramCharts = {};
        updateMetricsList();
        updateChartsArea();
    });

    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', () => {
        if (window.opener) {
            window.opener.postMessage({ type: 'REQUEST_DISTRIBUTION_DATA' }, '*');
        }
    });

    // Render scatter plot button
    document.getElementById('render-scatter-btn').addEventListener('click', renderScatterPlot);

    // Reset zoom button
    document.getElementById('reset-zoom-btn').addEventListener('click', () => {
        if (scatterChart) {
            scatterChart.resetZoom();
        }
    });

    // Toggle info panel button
    document.getElementById('toggle-info-btn').addEventListener('click', () => {
        const panel = document.getElementById('scatter-info-panel');
        const btn = document.getElementById('toggle-info-btn');
        if (panel.classList.contains('active')) {
            panel.classList.remove('active');
            btn.textContent = 'Show Selection';
        } else {
            panel.classList.add('active');
            btn.textContent = 'Hide Selection';
        }
    });

    // Anomaly detection event listeners
    document.getElementById('anomaly-algorithm')?.addEventListener('change', updateAlgorithmUI);
    document.getElementById('run-anomaly-btn')?.addEventListener('click', runAnomalyDetection);
    document.getElementById('apply-anomaly-btn')?.addEventListener('click', applyAnomalyToGraph);
    document.getElementById('highlight-anomalies-btn')?.addEventListener('click', highlightAnomalies);
    document.getElementById('export-anomalies-btn')?.addEventListener('click', exportAnomaliesCSV);
}

function switchView(view) {
    currentView = view;
    
    // Update tabs
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });

    // Update sidebar content
    document.getElementById('metrics-list').style.display = view === 'histograms' ? 'block' : 'none';
    document.getElementById('scatter-config').classList.toggle('active', view === 'scatter');
    document.getElementById('anomaly-config').classList.toggle('active', view === 'anomaly');

    // Update main content
    document.getElementById('charts-area').style.display = view === 'histograms' ? 'flex' : 'none';
    document.getElementById('scatter-area').classList.toggle('active', view === 'scatter');
    document.getElementById('anomaly-area').classList.toggle('active', view === 'anomaly');

    // Update toolbar
    if (view === 'histograms') {
        document.getElementById('chart-label').textContent = 'metrics selected';
    } else if (view === 'scatter') {
        document.getElementById('chart-label').textContent = 'scatter plot';
        document.getElementById('chart-count').textContent = scatterChart ? '1' : '0';
    } else if (view === 'anomaly') {
        document.getElementById('chart-label').textContent = 'anomaly detection';
        document.getElementById('chart-count').textContent = lastAnomalyResult ? '1' : '0';
    }
    
    // Initialize anomaly tab if switching to it for the first time
    if (view === 'anomaly' && !anomalyInitialized) {
        initializeAnomalyTab();
    }
}

function initializeMetrics() {
    if (nodeData.length === 0) {
        document.getElementById('metrics-list').innerHTML = '<div class="no-data">No data available</div>';
        return;
    }

    // Extract numeric metrics from first node
    const firstNode = nodeData[0];
    allMetrics = Object.keys(firstNode).filter(key => {
        if (['id', 'label', 'isNew', 'x', 'y'].includes(key)) return false;
        return typeof firstNode[key] === 'number';
    }).sort();

    updateMetricsList();
    populateScatterSelects();
    
    // Also update anomaly metrics if tab is active
    if (anomalyInitialized) {
        populateAnomalyMetrics();
    }
}

function updateMetricsList() {
    const container = document.getElementById('metrics-list');
    
    if (allMetrics.length === 0) {
        container.innerHTML = '<div class="no-data">No numeric metrics found</div>';
        return;
    }

    container.innerHTML = allMetrics.map(metric => `
        <div class="metric-item ${selectedMetrics.has(metric) ? 'selected' : ''}" data-metric="${metric}">
            <input type="checkbox" ${selectedMetrics.has(metric) ? 'checked' : ''}>
            <span class="metric-name">${metric.replace(/_/g, ' ')}</span>
        </div>
    `).join('');

    // Add click handlers
    container.querySelectorAll('.metric-item').forEach(item => {
        item.addEventListener('click', (e) => {
            const metric = item.dataset.metric;
            const checkbox = item.querySelector('input[type="checkbox"]');
            
            if (e.target !== checkbox) {
                checkbox.checked = !checkbox.checked;
            }

            if (checkbox.checked) {
                selectedMetrics.add(metric);
                item.classList.add('selected');
                createHistogramChart(metric);
            } else {
                selectedMetrics.delete(metric);
                item.classList.remove('selected');
                removeHistogramChart(metric);
            }
            
            updateChartsArea();
        });
    });
}

function populateScatterSelects() {
    const xSelect = document.getElementById('scatter-x');
    const ySelect = document.getElementById('scatter-y');
    const colorSelect = document.getElementById('scatter-color');

    const options = allMetrics.map(m => `<option value="${m}">${m.replace(/_/g, ' ')}</option>`).join('');
    
    xSelect.innerHTML = options;
    ySelect.innerHTML = options;
    colorSelect.innerHTML = '<option value="">None (uniform color)</option>' + options;

    // Set defaults if available
    if (allMetrics.includes('in_degree')) xSelect.value = 'in_degree';
    else if (allMetrics.length > 0) xSelect.value = allMetrics[0];

    if (allMetrics.includes('out_degree')) ySelect.value = 'out_degree';
    else if (allMetrics.length > 1) ySelect.value = allMetrics[1];
    else if (allMetrics.length > 0) ySelect.value = allMetrics[0];

    if (allMetrics.includes('pagerank')) colorSelect.value = 'pagerank';
}

function updateNodeInfo() {
    const data = getFilteredData();
    const total = nodeData.length;
    const filtered = data.length;
    
    const infoEl = document.getElementById('node-info');
    if (useSelectedOnly && filtered < total) {
        infoEl.textContent = `Analyzing ${filtered} of ${total} nodes (selected)`;
    } else {
        infoEl.textContent = `Analyzing ${total} nodes`;
    }
}

function getFilteredData() {
    if (!useSelectedOnly) return nodeData;
    return nodeData.filter(n => n._selected);
}

function updateChartsArea() {
    const container = document.getElementById('charts-area');
    const count = selectedMetrics.size;
    
    document.getElementById('chart-count').textContent = count;
    
    if (count === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>Select Metrics to Analyze</h3>
                <p>Choose one or more metrics from the sidebar to view their distribution histograms and statistics.</p>
            </div>
        `;
    }
}

function createHistogramChart(metric) {
    const data = getFilteredData();
    const values = data.map(n => n[metric]).filter(v => v !== undefined && v !== null && !isNaN(v));
    
    if (values.length === 0) return;

    // Remove empty state if present
    const emptyState = document.querySelector('#charts-area .empty-state');
    if (emptyState) emptyState.remove();

    // Create chart card
    const cardId = `chart-${metric}`;
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.id = cardId;
    
    const stats = calculateStats(values);
    const histogram = calculateHistogram(values, 30);
    
    card.innerHTML = `
        <div class="chart-header">
            <span class="chart-title">${metric.replace(/_/g, ' ')}</span>
            <button class="chart-close" data-metric="${metric}">&times;</button>
        </div>
        <div class="chart-body">
            <div class="chart-container">
                <canvas id="canvas-${metric}"></canvas>
            </div>
            <div class="stats-panel">
                <div class="stats-section">
                    <div class="stats-section-title">Basic Statistics</div>
                    <div class="stat-row">
                        <span class="stat-label">Count</span>
                        <span class="stat-value">${stats.count.toLocaleString()}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Min</span>
                        <span class="stat-value">${formatNumber(stats.min)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max</span>
                        <span class="stat-value">${formatNumber(stats.max)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Mean</span>
                        <span class="stat-value">${formatNumber(stats.mean)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Median</span>
                        <span class="stat-value">${formatNumber(stats.median)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Std Dev</span>
                        <span class="stat-value">${formatNumber(stats.stdDev)}</span>
                    </div>
                </div>
                <div class="stats-section">
                    <div class="stats-section-title">Percentiles</div>
                    <div class="stat-row">
                        <span class="stat-label">25th</span>
                        <span class="stat-value">${formatNumber(stats.p25)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">50th</span>
                        <span class="stat-value">${formatNumber(stats.p50)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">75th</span>
                        <span class="stat-value">${formatNumber(stats.p75)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">90th</span>
                        <span class="stat-value">${formatNumber(stats.p90)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">95th</span>
                        <span class="stat-value">${formatNumber(stats.p95)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">99th</span>
                        <span class="stat-value">${formatNumber(stats.p99)}</span>
                    </div>
                </div>
                <div class="stats-section">
                    <div class="stats-section-title">IQR Range (25th - 75th)</div>
                    <div class="percentile-bar">
                        <div class="percentile-fill" style="width: ${((stats.p75 - stats.p25) / (stats.max - stats.min) * 100) || 0}%; margin-left: ${((stats.p25 - stats.min) / (stats.max - stats.min) * 100) || 0}%"></div>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById('charts-area').appendChild(card);

    // Add close handler
    card.querySelector('.chart-close').addEventListener('click', () => {
        selectedMetrics.delete(metric);
        removeHistogramChart(metric);
        updateMetricsList();
        updateChartsArea();
    });

    // Create Chart.js histogram
    const ctx = document.getElementById(`canvas-${metric}`).getContext('2d');
    histogramCharts[metric] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: histogram.labels,
            datasets: [{
                label: 'Frequency',
                data: histogram.counts,
                backgroundColor: 'rgba(74, 144, 226, 0.7)',
                borderColor: 'rgba(74, 144, 226, 1)',
                borderWidth: 1,
                barPercentage: 1.0,
                categoryPercentage: 1.0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 300
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            if (items.length > 0) {
                                const idx = items[0].dataIndex;
                                return `Range: ${histogram.ranges[idx]}`;
                            }
                            return '';
                        },
                        label: (item) => `Count: ${item.raw.toLocaleString()}`
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: metric.replace(/_/g, ' '),
                        color: '#808080',
                        font: { size: 11 }
                    },
                    ticks: {
                        color: '#808080',
                        font: { size: 9 },
                        maxRotation: 45,
                        minRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 10
                    },
                    grid: {
                        color: '#2a2a2a'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency (count)',
                        color: '#808080',
                        font: { size: 11 }
                    },
                    ticks: {
                        color: '#808080',
                        font: { size: 10 },
                        precision: 0,  // Force integer ticks
                        callback: function(value) {
                            if (Number.isInteger(value)) {
                                return value.toLocaleString();
                            }
                            return null;
                        }
                    },
                    grid: {
                        color: '#2a2a2a'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

function removeHistogramChart(metric) {
    if (histogramCharts[metric]) {
        histogramCharts[metric].destroy();
        delete histogramCharts[metric];
    }
    const card = document.getElementById(`chart-${metric}`);
    if (card) card.remove();
}

function refreshAllCharts() {
    // Refresh histograms
    selectedMetrics.forEach(metric => {
        removeHistogramChart(metric);
        createHistogramChart(metric);
    });
    
    // Refresh scatter if visible
    if (currentView === 'scatter' && scatterChart) {
        renderScatterPlot();
    }
}

function calculateStats(values) {
    const sorted = Float64Array.from(values).sort();
    const n = sorted.length;
    
    const sum = sorted.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    
    const squaredDiffs = sorted.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / n;
    const stdDev = Math.sqrt(variance);
    
    const percentile = (p) => {
        const idx = (p / 100) * (n - 1);
        const lower = Math.floor(idx);
        const upper = Math.ceil(idx);
        if (lower === upper) return sorted[lower];
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower);
    };
    
    return {
        count: n,
        min: sorted[0],
        max: sorted[n - 1],
        mean,
        median: percentile(50),
        stdDev,
        p25: percentile(25),
        p50: percentile(50),
        p75: percentile(75),
        p90: percentile(90),
        p95: percentile(95),
        p99: percentile(99)
    };
}

function calculateHistogram(values, numBins = 30) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    // Handle edge case where all values are the same
    if (min === max) {
        return {
            labels: [formatNumber(min)],
            counts: [values.length],
            ranges: [`${formatNumber(min)}`]
        };
    }
    
    const binWidth = (max - min) / numBins;
    const counts = new Array(numBins).fill(0);
    const labels = [];
    const ranges = [];
    
    // Calculate bin edges
    for (let i = 0; i < numBins; i++) {
        const binStart = min + i * binWidth;
        const binEnd = min + (i + 1) * binWidth;
        labels.push(formatBinLabel(binStart, binEnd));
        ranges.push(`${formatNumber(binStart)} - ${formatNumber(binEnd)}`);
    }
    
    // Count values in each bin
    values.forEach(v => {
        let binIndex = Math.floor((v - min) / binWidth);
        if (binIndex >= numBins) binIndex = numBins - 1; // Handle max value
        if (binIndex < 0) binIndex = 0;
        counts[binIndex]++;
    });
    
    return { labels, counts, ranges };
}

function formatBinLabel(start, end) {
    // Show the midpoint of the bin
    const mid = (start + end) / 2;
    if (Math.abs(mid) >= 1000) {
        return mid.toExponential(1);
    } else if (Math.abs(mid) >= 1) {
        return Math.round(mid).toString();
    } else if (mid === 0) {
        return '0';
    } else {
        return mid.toFixed(2);
    }
}

function formatNumber(n) {
    if (n === undefined || n === null || isNaN(n)) return '-';
    if (Number.isInteger(n)) return n.toLocaleString();
    if (Math.abs(n) >= 1000) return n.toExponential(3);
    if (Math.abs(n) >= 1) return n.toFixed(2);
    if (Math.abs(n) >= 0.01) return n.toFixed(4);
    return n.toExponential(3);
}

// Scatter Plot Functions
function renderScatterPlot() {
    const xMetric = document.getElementById('scatter-x').value;
    const yMetric = document.getElementById('scatter-y').value;
    const colorMetric = document.getElementById('scatter-color').value;
    const xScale = document.getElementById('scatter-x-scale').value;
    const yScale = document.getElementById('scatter-y-scale').value;

    if (!xMetric || !yMetric) {
        alert('Please select both X and Y metrics');
        return;
    }

    const data = getFilteredData();
    
    // Clear previous selection
    selectedScatterPoints = [];
    updateSelectedPointsList();
    
    // Prepare scatter data and store node info for selection
    scatterNodeData = [];
    const scatterData = [];
    let colorValues = [];
    
    data.forEach(node => {
        const x = node[xMetric];
        const y = node[yMetric];
        
        if (x !== undefined && y !== undefined && !isNaN(x) && !isNaN(y)) {
            // Skip zero/negative values for log scale
            if (xScale === 'logarithmic' && x <= 0) return;
            if (yScale === 'logarithmic' && y <= 0) return;
            
            const point = { x, y };
            scatterData.push(point);
            scatterNodeData.push({
                id: node.id,
                x, y,
                xMetric, yMetric,
                colorValue: colorMetric ? node[colorMetric] : null,
                allData: node
            });
            
            if (colorMetric && node[colorMetric] !== undefined) {
                colorValues.push(node[colorMetric]);
            }
        }
    });

    if (scatterData.length === 0) {
        alert('No valid data points for the selected metrics and scale');
        return;
    }

    // Calculate colors
    let colors;
    if (colorMetric && colorValues.length > 0) {
        const colorMin = Math.min(...colorValues);
        const colorMax = Math.max(...colorValues);
        colors = scatterNodeData.map(n => {
            const v = n.colorValue;
            if (v === undefined || v === null) return 'rgba(128, 128, 128, 0.6)';
            const norm = colorMax > colorMin ? (v - colorMin) / (colorMax - colorMin) : 0.5;
            return getViridisColor(norm, 0.7);
        });
    } else {
        colors = scatterData.map(() => 'rgba(74, 144, 226, 0.6)');
    }

    // Calculate correlation
    const correlation = calculateCorrelation(
        scatterNodeData.map(n => n.x),
        scatterNodeData.map(n => n.y)
    );

    // Update stats
    document.getElementById('scatter-points').textContent = scatterData.length.toLocaleString();
    document.getElementById('scatter-correlation').textContent = isNaN(correlation) ? '-' : correlation.toFixed(4);
    document.getElementById('scatter-r2').textContent = isNaN(correlation) ? '-' : (correlation * correlation).toFixed(4);

    // Update title
    document.getElementById('scatter-title').textContent = 
        `${xMetric.replace(/_/g, ' ')} vs ${yMetric.replace(/_/g, ' ')}`;

    // Destroy existing chart
    if (scatterChart) {
        scatterChart.destroy();
    }

    // Point size based on dataset size
    const pointRadius = scatterData.length > 5000 ? 2 : scatterData.length > 1000 ? 3 : 4;

    // Create new chart
    const ctx = document.getElementById('scatter-chart').getContext('2d');
    scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Nodes',
                data: scatterData,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.6', '1').replace('0.7', '1')),
                borderWidth: 1,
                pointRadius: pointRadius,
                pointHoverRadius: pointRadius + 3,
                pointHoverBorderWidth: 2,
                pointHoverBorderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 300
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const idx = context.dataIndex;
                            const node = scatterNodeData[idx];
                            const lines = [
                                `ID: ${node.id.substring(0, 20)}...`,
                                `${xMetric}: ${formatNumber(node.x)}`,
                                `${yMetric}: ${formatNumber(node.y)}`
                            ];
                            if (colorMetric && node.colorValue !== null) {
                                lines.push(`${colorMetric}: ${formatNumber(node.colorValue)}`);
                            }
                            return lines;
                        }
                    }
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy',
                        modifierKey: null
                    },
                    zoom: {
                        wheel: {
                            enabled: true
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy',
                        drag: {
                            enabled: false
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: xScale,
                    title: {
                        display: true,
                        text: xMetric.replace(/_/g, ' '),
                        color: '#808080',
                        font: { size: 12 }
                    },
                    ticks: {
                        color: '#808080',
                        font: { size: 10 },
                        callback: function(value) {
                            if (xScale === 'logarithmic') {
                                return formatNumber(value);
                            }
                            return formatNumber(value);
                        }
                    },
                    grid: {
                        color: '#2a2a2a'
                    }
                },
                y: {
                    type: yScale,
                    title: {
                        display: true,
                        text: yMetric.replace(/_/g, ' '),
                        color: '#808080',
                        font: { size: 12 }
                    },
                    ticks: {
                        color: '#808080',
                        font: { size: 10 },
                        callback: function(value) {
                            if (yScale === 'logarithmic') {
                                return formatNumber(value);
                            }
                            return formatNumber(value);
                        }
                    },
                    grid: {
                        color: '#2a2a2a'
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    const node = scatterNodeData[idx];
                    togglePointSelection(node, idx);
                }
            },
            onHover: (event, elements) => {
                event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
            }
        }
    });

    // Update chart count
    document.getElementById('chart-count').textContent = '1';

    // Double-click to reset zoom
    document.getElementById('scatter-chart').addEventListener('dblclick', () => {
        if (scatterChart) {
            scatterChart.resetZoom();
        }
    });
}

function togglePointSelection(node, idx) {
    const existingIdx = selectedScatterPoints.findIndex(p => p.id === node.id);
    
    if (existingIdx >= 0) {
        // Remove from selection
        selectedScatterPoints.splice(existingIdx, 1);
    } else {
        // Add to selection (limit to 20 for performance)
        if (selectedScatterPoints.length >= 20) {
            selectedScatterPoints.shift(); // Remove oldest
        }
        selectedScatterPoints.push(node);
    }
    
    updateSelectedPointsList();
    
    // Show the info panel if we have selections
    if (selectedScatterPoints.length > 0) {
        document.getElementById('scatter-info-panel').classList.add('active');
        document.getElementById('toggle-info-btn').textContent = 'Hide Selection';
    }
}

function updateSelectedPointsList() {
    const container = document.getElementById('selected-points-list');
    
    if (selectedScatterPoints.length === 0) {
        container.innerHTML = '<p style="color: #808080; font-size: 11px;">Click points on the chart to see details</p>';
        return;
    }
    
    const xMetric = document.getElementById('scatter-x').value;
    const yMetric = document.getElementById('scatter-y').value;
    const colorMetric = document.getElementById('scatter-color').value;
    
    container.innerHTML = selectedScatterPoints.map(node => {
        // Get additional metrics to display
        const additionalMetrics = Object.entries(node.allData)
            .filter(([k, v]) => {
                if (['id', 'label', 'isNew', 'x', 'y', '_selected'].includes(k)) return false;
                if (k === xMetric || k === yMetric || k === colorMetric) return false;
                return typeof v === 'number';
            })
            .slice(0, 5) // Limit to 5 additional metrics
            .map(([k, v]) => `
                <div class="stat-row">
                    <span class="stat-label">${k.replace(/_/g, ' ')}</span>
                    <span class="stat-value">${formatNumber(v)}</span>
                </div>
            `).join('');
        
        return `
            <div class="selected-node-card">
                <div class="node-id">${node.id}</div>
                <div class="stat-row">
                    <span class="stat-label">${xMetric.replace(/_/g, ' ')}</span>
                    <span class="stat-value">${formatNumber(node.x)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">${yMetric.replace(/_/g, ' ')}</span>
                    <span class="stat-value">${formatNumber(node.y)}</span>
                </div>
                ${colorMetric && node.colorValue !== null ? `
                <div class="stat-row">
                    <span class="stat-label">${colorMetric.replace(/_/g, ' ')}</span>
                    <span class="stat-value">${formatNumber(node.colorValue)}</span>
                </div>
                ` : ''}
                ${additionalMetrics}
            </div>
        `;
    }).join('');
}

function calculateCorrelation(x, y) {
    const n = x.length;
    if (n === 0) return NaN;
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
    const sumY2 = y.reduce((acc, yi) => acc + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

function getViridisColor(t, alpha = 1) {
    // Simplified viridis-like gradient
    const colors = [
        [68, 1, 84],      // 0.0 - dark purple
        [59, 82, 139],    // 0.25 - blue
        [33, 145, 140],   // 0.5 - teal
        [94, 201, 98],    // 0.75 - green
        [253, 231, 37]    // 1.0 - yellow
    ];
    
    const idx = Math.min(Math.floor(t * 4), 3);
    const localT = (t * 4) - idx;
    
    const c1 = colors[idx];
    const c2 = colors[idx + 1];
    
    const r = Math.round(c1[0] + (c2[0] - c1[0]) * localT);
    const g = Math.round(c1[1] + (c2[1] - c1[1]) * localT);
    const b = Math.round(c1[2] + (c2[2] - c1[2]) * localT);
    
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}


// =============================================================================
// ANOMALY DETECTION
// =============================================================================

async function initializeAnomalyTab() {
    if (anomalyInitialized) return;
    
    console.log('[Anomaly] Initializing anomaly detection tab...');
    
    try {
        // Fetch available algorithms from backend
        const response = await fetch('/api/anomaly/algorithms');
        if (!response.ok) {
            throw new Error('Failed to load algorithms');
        }
        
        anomalyAlgorithms = await response.json();
        console.log('[Anomaly] Loaded algorithms:', Object.keys(anomalyAlgorithms));
        
        populateAlgorithmSelect();
        populateAnomalyMetrics();
        anomalyInitialized = true;
        
    } catch (error) {
        console.error('[Anomaly] Failed to initialize:', error);
        document.getElementById('algorithm-description').innerHTML = 
            '<div style="color: #ff4d4f;">Failed to load algorithms. Is the server running?</div>';
    }
}

function populateAlgorithmSelect() {
    const select = document.getElementById('anomaly-algorithm');
    if (!select || !anomalyAlgorithms) return;
    
    select.innerHTML = Object.entries(anomalyAlgorithms)
        .map(([key, info]) => `<option value="${key}">${info.name}</option>`)
        .join('');
    
    // Set default and trigger UI update
    select.value = 'isolation_forest';
    updateAlgorithmUI();
}

function updateAlgorithmUI() {
    const algorithm = document.getElementById('anomaly-algorithm')?.value;
    if (!algorithm || !anomalyAlgorithms) return;
    
    const info = anomalyAlgorithms[algorithm];
    if (!info) return;
    
    // Update description
    document.getElementById('algorithm-description').innerHTML = `
        <p>${info.description}</p>
        <p class="complexity">Complexity: ${info.complexity}</p>
        <p class="multivariate">${info.multivariate ? '✓ Supports multiple metrics' : '○ Single metric recommended'}</p>
    `;
    
    // Update parameters
    const paramsContainer = document.getElementById('anomaly-parameters');
    if (!paramsContainer) return;
    
    paramsContainer.innerHTML = Object.entries(info.parameters)
        .map(([paramName, paramInfo]) => {
            const inputType = paramInfo.type === 'bool' ? 'checkbox' : 'number';
            const step = paramInfo.type === 'int' ? '1' : '0.01';
            
            if (paramInfo.type === 'bool') {
                return `
                    <div class="config-row">
                        <label style="display: flex; align-items: center; gap: 8px;">
                            <input type="checkbox" id="param-${paramName}" ${paramInfo.default ? 'checked' : ''}>
                            ${paramName.replace(/_/g, ' ')}
                        </label>
                        <div style="font-size: 10px; color: #606060; margin-top: 2px;">${paramInfo.description}</div>
                    </div>
                `;
            }
            
            return `
                <div class="config-row">
                    <label title="${paramInfo.description}">${paramName.replace(/_/g, ' ')}</label>
                    <input type="${inputType}" 
                           class="param-input"
                           id="param-${paramName}"
                           value="${paramInfo.default}"
                           min="${paramInfo.min}"
                           max="${paramInfo.max}"
                           step="${step}">
                </div>
            `;
        }).join('');
    
    // Update metrics list based on multivariate support
    populateAnomalyMetrics();
}

function populateAnomalyMetrics() {
    const container = document.getElementById('anomaly-metrics-list');
    if (!container) return;
    
    const algorithm = document.getElementById('anomaly-algorithm')?.value;
    const info = anomalyAlgorithms[algorithm];
    
    // Filter to numeric metrics
    const numericMetrics = allMetrics.filter(m => {
        const sample = nodeData[0]?.[m];
        return typeof sample === 'number';
    });
    
    if (numericMetrics.length === 0) {
        container.innerHTML = '<div style="color: #808080; font-size: 11px; padding: 10px;">No numeric metrics available</div>';
        return;
    }
    
    // For multivariate algorithms, use checkboxes; for single-metric, use radio buttons
    const inputType = info?.multivariate !== false ? 'checkbox' : 'radio';
    
    container.innerHTML = numericMetrics.map(metric => `
        <label class="metric-checkbox">
            <input type="${inputType}" name="anomaly-metric" value="${metric}">
            <span>${metric.replace(/_/g, ' ')}</span>
        </label>
    `).join('');
    
    // Pre-select reasonable defaults
    const defaults = ['pagerank', 'total_degree', 'clustering_coefficient', 'in_degree', 'out_degree'];
    let selectedCount = 0;
    defaults.forEach(d => {
        if (selectedCount < 3) {
            const input = container.querySelector(`input[value="${d}"]`);
            if (input) {
                input.checked = true;
                selectedCount++;
            }
        }
    });
    
    // If no defaults found, select first metric
    if (selectedCount === 0 && numericMetrics.length > 0) {
        const firstInput = container.querySelector('input');
        if (firstInput) firstInput.checked = true;
    }
}

async function runAnomalyDetection() {
    const algorithm = document.getElementById('anomaly-algorithm')?.value;
    if (!algorithm) {
        showAnomalyError('Please select an algorithm');
        return;
    }
    
    const info = anomalyAlgorithms[algorithm];
    if (!info) return;
    
    // Get selected metrics
    const selectedMetrics = Array.from(
        document.querySelectorAll('#anomaly-metrics-list input:checked')
    ).map(input => input.value);
    
    if (selectedMetrics.length === 0) {
        showAnomalyError('Please select at least one metric');
        return;
    }
    
    // Get parameters
    const parameters = {};
    Object.keys(info.parameters).forEach(paramName => {
        const input = document.getElementById(`param-${paramName}`);
        if (input) {
            if (info.parameters[paramName].type === 'bool') {
                parameters[paramName] = input.checked;
            } else if (info.parameters[paramName].type === 'int') {
                parameters[paramName] = parseInt(input.value);
            } else {
                parameters[paramName] = parseFloat(input.value);
            }
        }
    });
    
    const name = document.getElementById('anomaly-name')?.value || 'anomaly_score';
    
    // Show progress
    document.getElementById('anomaly-progress').style.display = 'block';
    document.getElementById('run-anomaly-btn').disabled = true;
    
    try {
        console.log('[Anomaly] Running detection:', { algorithm, metrics: selectedMetrics, parameters, name });
        
        const response = await fetch('/api/anomaly/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                metrics: selectedMetrics,
                algorithm: algorithm,
                parameters: parameters,
                apply_to_graph: false  // Don't apply yet, let user decide
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Detection failed');
        }
        
        lastAnomalyResult = await response.json();
        console.log('[Anomaly] Detection complete:', lastAnomalyResult);
        
        displayAnomalyResults(lastAnomalyResult);
        
        // Enable action buttons
        document.getElementById('apply-anomaly-btn').disabled = false;
        document.getElementById('highlight-anomalies-btn').disabled = false;
        document.getElementById('export-anomalies-btn').disabled = false;
        
        // Update toolbar counter
        document.getElementById('chart-count').textContent = '1';
        
    } catch (error) {
        console.error('[Anomaly] Detection failed:', error);
        showAnomalyError(error.message);
    } finally {
        document.getElementById('anomaly-progress').style.display = 'none';
        document.getElementById('run-anomaly-btn').disabled = false;
    }
}

function showAnomalyError(message) {
    // Simple toast-like notification
    const progress = document.getElementById('anomaly-progress');
    if (progress) {
        progress.style.display = 'block';
        progress.style.color = '#ff4d4f';
        progress.textContent = `Error: ${message}`;
        setTimeout(() => {
            progress.style.display = 'none';
            progress.style.color = '#808080';
            progress.textContent = 'Running analysis...';
        }, 3000);
    }
}

function displayAnomalyResults(result) {
    // Hide empty state, show results
    document.getElementById('anomaly-empty-state').style.display = 'none';
    document.getElementById('anomaly-summary').classList.add('visible');
    document.getElementById('anomaly-chart-container').style.display = 'block';
    document.getElementById('anomaly-table-container').style.display = 'block';
    
    // Update summary stats
    document.getElementById('result-algorithm').textContent = result.algorithm.replace(/_/g, ' ');
    document.getElementById('result-count').textContent = `${result.n_anomalies} / ${result.n_total}`;
    document.getElementById('result-percentage').textContent = `${result.anomaly_percentage.toFixed(1)}%`;
    document.getElementById('result-time').textContent = `${result.computation_time.toFixed(2)}s`;
    
    // Render histogram of anomaly scores
    renderAnomalyHistogram(result);
    
    // Populate top anomalies table
    const tbody = document.getElementById('anomaly-table-body');
    tbody.innerHTML = result.top_anomalies.map((node, i) => {
        const nodeIdDisplay = node.id.length > 20 ? node.id.substring(0, 18) + '...' : node.id;
        const scoreClass = node.score > 0.7 ? 'style="color: #ff4d4f; font-weight: bold;"' : 
                          node.score > 0.5 ? 'style="color: #faad14;"' : '';
        
        return `
            <tr>
                <td>${i + 1}</td>
                <td class="node-id-cell" title="${node.id}">${nodeIdDisplay}</td>
                <td ${scoreClass}>${node.score.toFixed(4)}</td>
                <td>
                    <button class="btn-tiny" onclick="locateNode('${node.id}')">Locate</button>
                </td>
            </tr>
        `;
    }).join('');
}

function renderAnomalyHistogram(result) {
    const canvas = document.getElementById('anomaly-histogram');
    if (!canvas) return;
    
    // Destroy existing chart
    if (anomalyHistogramChart) {
        anomalyHistogramChart.destroy();
    }
    
    // Build histogram bins from scores
    const scores = Object.values(result.top_anomalies).map(n => n.score);
    
    // If we have full score statistics, use them to compute distribution
    const stats = result.score_statistics;
    const bins = 20;
    const binWidth = 1.0 / bins;
    const counts = new Array(bins).fill(0);
    
    // Count scores in each bin
    result.top_anomalies.forEach(node => {
        const binIdx = Math.min(Math.floor(node.score / binWidth), bins - 1);
        counts[binIdx]++;
    });
    
    // Create labels
    const labels = [];
    for (let i = 0; i < bins; i++) {
        labels.push(((i + 0.5) * binWidth).toFixed(2));
    }
    
    anomalyHistogramChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Anomaly Score Distribution',
                data: counts,
                backgroundColor: labels.map((_, i) => {
                    const t = i / (bins - 1);
                    // Red for high scores, blue for low
                    return t > 0.7 ? 'rgba(255, 77, 79, 0.8)' :
                           t > 0.5 ? 'rgba(250, 173, 20, 0.8)' :
                           'rgba(74, 144, 226, 0.8)';
                }),
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Anomaly Score Distribution (Top 20 Nodes)',
                    color: '#e0e0e0',
                    font: { size: 14 }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Anomaly Score',
                        color: '#808080'
                    },
                    ticks: { color: '#808080' },
                    grid: { color: '#2a2a2a' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Count',
                        color: '#808080'
                    },
                    ticks: { color: '#808080' },
                    grid: { color: '#2a2a2a' },
                    beginAtZero: true
                }
            }
        }
    });
}

function locateNode(nodeId) {
    if (window.opener) {
        window.opener.postMessage({
            type: 'LOCATE_NODE',
            data: { nodeId }
        }, '*');
    }
}

async function applyAnomalyToGraph() {
    if (!lastAnomalyResult) return;
    
    try {
        document.getElementById('apply-anomaly-btn').disabled = true;
        document.getElementById('apply-anomaly-btn').textContent = 'Applying...';
        
        const response = await fetch('/api/anomaly/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: lastAnomalyResult.metric_name,
                metrics: lastAnomalyResult.metrics_used || [],
                algorithm: lastAnomalyResult.algorithm,
                parameters: lastAnomalyResult.parameters_used || {},
                apply_to_graph: true
            })
        });
        
        if (!response.ok) throw new Error('Failed to apply');
        
        const result = await response.json();
        
        // Notify parent window to update
        if (window.opener) {
            window.opener.postMessage({
                type: 'ANOMALY_APPLIED',
                data: {
                    metric_name: result.metric_name,
                    node_updates: result.node_updates
                }
            }, '*');
        }
        
        showAnomalySuccess(`Applied ${result.metric_name} to ${result.node_updates?.length || 0} nodes`);
        
    } catch (error) {
        showAnomalyError(error.message);
    } finally {
        document.getElementById('apply-anomaly-btn').disabled = false;
        document.getElementById('apply-anomaly-btn').textContent = 'Apply to Graph';
    }
}

function highlightAnomalies() {
    if (!lastAnomalyResult || !lastAnomalyResult.top_anomalies) return;
    
    // Get all anomaly node IDs (those with is_anomaly = true)
    const anomalyIds = lastAnomalyResult.top_anomalies
        .filter(n => n.is_anomaly)
        .map(n => n.id);
    
    if (window.opener && window.opener.highlightAnomalies) {
        window.opener.highlightAnomalies(anomalyIds);
    }
}

function exportAnomaliesCSV() {
    if (!lastAnomalyResult || !lastAnomalyResult.top_anomalies) return;
    
    // Build CSV content
    const headers = ['rank', 'node_id', 'anomaly_score', 'is_anomaly'];
    const metricsUsed = lastAnomalyResult.metrics_used || [];
    headers.push(...metricsUsed);
    
    const rows = lastAnomalyResult.top_anomalies.map((node, i) => {
        const row = [
            i + 1,
            node.id,
            node.score.toFixed(6),
            node.is_anomaly ? 'true' : 'false'
        ];
        metricsUsed.forEach(m => {
            row.push(node[m] !== undefined ? node[m].toFixed(6) : '');
        });
        return row.join(',');
    });
    
    const csv = [headers.join(','), ...rows].join('\n');
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `anomalies_${lastAnomalyResult.algorithm}_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showAnomalySuccess(message) {
    const progress = document.getElementById('anomaly-progress');
    if (progress) {
        progress.style.display = 'block';
        progress.style.color = '#52c41a';
        progress.textContent = message;
        setTimeout(() => {
            progress.style.display = 'none';
            progress.style.color = '#808080';
            progress.textContent = 'Running analysis...';
        }, 2000);
    }
}