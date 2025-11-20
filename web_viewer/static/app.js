/**
 * Graph Analyzer Web Viewer - Ultra-Light with Dynamic Style Toggle
 */

// Performance utilities
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Global state
let cy = null;
let currentGraph = null;
let currentState = null;
let availableConfig = null;
let currentStyle = null;
let graphData = {};
let neighborHighlightState = 0;
let performanceMode = true; // Start in performance mode
let styleCache = {
    sizeRange: { min: 0, max: 1 },
    colorRange: { min: 0, max: 1 },
    widthRange: { min: 0, max: 1 }
};

// Cache DOM elements
let domCache = {};

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing Graph Analyzer (Ultra-Light Mode)...');
    cacheDOMElements();
    await loadAvailableConfig();
    setupEventListeners();
    setupDropdownLogic();
    initializeDefaultStyle();
    addPerformanceToggle();
});

function cacheDOMElements() {
    domCache = {
        nodeId: document.getElementById('node-id'),
        nodeCount: document.getElementById('node-count'),
        edgeCount: document.getElementById('edge-count'),
        infoPanel: document.getElementById('info-panel'),
        nodeInfo: document.getElementById('node-info'),
        edgeInfo: document.getElementById('edge-info'),
        multiInfo: document.getElementById('multi-info'),
        allMetrics: document.getElementById('all-metrics'),
        edgeMetrics: document.getElementById('edge-metrics'),
        inCount: document.getElementById('in-count'),
        outCount: document.getElementById('out-count'),
        neighborInList: document.getElementById('neighbors-in-list'),
        neighborOutList: document.getElementById('neighbors-out-list'),
        status: document.getElementById('status'),
        loading: document.getElementById('loading'),
        cyContainer: document.getElementById('cy')
    };
}

function addPerformanceToggle() {
    // Add radio buttons to toolbar
    const toolbar = document.querySelector('.toolbar-actions');
    if (toolbar) {
        const toggleDiv = document.createElement('div');
        toggleDiv.style.cssText = 'display: flex; gap: 10px; align-items: center; margin-left: 10px; padding: 5px 10px; background: #2a2a2a; border-radius: 4px;';
        toggleDiv.innerHTML = `
            <label style="display: flex; align-items: center; gap: 5px; cursor: pointer; color: #e0e0e0; font-size: 12px;">
                <input type="radio" name="render-mode" value="performance" checked> Performance
            </label>
            <label style="display: flex; align-items: center; gap: 5px; cursor: pointer; color: #e0e0e0; font-size: 12px;">
                <input type="radio" name="render-mode" value="style"> Style
            </label>
        `;
        toolbar.appendChild(toggleDiv);
        
        // Add event listeners
        toggleDiv.querySelectorAll('input[name="render-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) {
                    toggleRenderMode(e.target.value === 'performance');
                }
            });
        });
    }
}

function toggleRenderMode(toPerformance) {
    performanceMode = toPerformance;
    
    if (!cy) return;
    
    if (performanceMode) {
        // CRITICAL FIX: Remove all bypass styles (colors applied directly to nodes)
        // This ensures we return to the clean, gray performance state
        cy.elements().removeStyle();

        // Complete removal of all styles - bare minimum
        cy.style()
            .selector('node')
            .style({
                'background-color': '#666',
                'width': 10,
                'height': 10,
                'label': '',
                'border-width': 0
            })
            .selector('edge')
            .style({
                'line-color': '#333',
                'width': 1,
                'opacity': 0.3,
                'curve-style': 'straight', // Straight is fastest for WebGL
                'target-arrow-shape': 'none'
            })
            .selector('node:selected')
            .style({
                'background-color': '#FF0000',
                'width': 20,
                'height': 20,
                'border-width': 0,
                'z-index': 999
            })
            .selector('edge:selected')
            .style({
                'line-color': '#FF0000',
                'width': 2,
                'opacity': 1,
                'z-index': 999
            })
            .selector('.highlighted')
            .style({
                'background-color': '#FFA500',
                'line-color': '#FFA500',
                'opacity': 0.8,
                'z-index': 998
            })
            .selector('.searched')
            .style({
                'background-color': '#00FF00',
                'border-width': 2,
                'border-color': '#00FF00',
                'z-index': 997
            })
            .update();
            
        updateStatus('Performance mode: styles removed', 'info');
    } else {
        updateCytoscapeStyle();
        updateStatus('Style mode: visual customization enabled', 'success');
    }
}

function initializeDefaultStyle() {
    currentStyle = {
        node: {
            sizeMetric: 'fixed', 
            sizeMin: 20, 
            sizeMax: 60,
            colorMetric: 'fixed', 
            colorFixed: '#4A90E2', 
            colorGradient: 'spectral',
            colorSelected: '#FF0000'
        },
        edge: {
            widthMetric: 'fixed', 
            widthMin: 2, 
            widthMax: 5,
            color: '#fcfafa', 
            colorSelected: '#FF0000', 
            opacity: 0.2
        }
    };
}

// Color gradients
const COLOR_GRADIENTS = {
    viridis: [
        { stop: 0.0,  color: '#440154' },
        { stop: 0.11, color: '#3C2F6E' },
        { stop: 0.22, color: '#335D88' },
        { stop: 0.33, color: '#328287' },
        { stop: 0.44, color: '#34A57E' },
        { stop: 0.56, color: '#49BE6D' },
        { stop: 0.67, color: '#71CC56' },
        { stop: 0.78, color: '#9BD940' },
        { stop: 0.89, color: '#CCE032' },
        { stop: 1.0,  color: '#FDE724' }
    ],

    plasma: [
        { stop: 0.0,  color: '#0D0887' },
        { stop: 0.11, color: '#3F0696' },
        { stop: 0.22, color: '#7104A4' },
        { stop: 0.33, color: '#981A98' },
        { stop: 0.44, color: '#BB3883' },
        { stop: 0.56, color: '#D6586C' },
        { stop: 0.67, color: '#E97B53' },
        { stop: 0.78, color: '#F7A03D' },
        { stop: 0.89, color: '#F4CD2F' },
        { stop: 1.0,  color: '#F0F921' }
    ],

    inferno: [
        { stop: 0.0,  color: '#000004' },
        { stop: 0.11, color: '#1D0430' },
        { stop: 0.22, color: '#3B095D' },
        { stop: 0.33, color: '#5D1368' },
        { stop: 0.44, color: '#812067' },
        { stop: 0.56, color: '#A3305D' },
        { stop: 0.67, color: '#C44349' },
        { stop: 0.78, color: '#E06446' },
        { stop: 0.89, color: '#EEB275' },
        { stop: 1.0,  color: '#FCFFA4' }
    ],

    magma: [
        { stop: 0.0,  color: '#000004' },
        { stop: 0.11, color: '#1A0734' },
        { stop: 0.22, color: '#340D64' },
        { stop: 0.33, color: '#561876' },
        { stop: 0.44, color: '#7A237D' },
        { stop: 0.56, color: '#9E307B' },
        { stop: 0.67, color: '#C33E70' },
        { stop: 0.78, color: '#E25369' },
        { stop: 0.89, color: '#F0796B' },
        { stop: 1.0,  color: '#FE9F6D' }
    ],

    turbo: [
        { stop: 0.0,  color: '#23171B' },
        { stop: 0.11, color: '#3F7BF3' },
        { stop: 0.22, color: '#30ADDD' },
        { stop: 0.33, color: '#2FD7C1' },
        { stop: 0.44, color: '#4EF097' },
        { stop: 0.56, color: '#8AE86B' },
        { stop: 0.67, color: '#E3BF3D' },
        { stop: 0.78, color: '#F79C3D' },
        { stop: 0.89, color: '#EE7C51' },
        { stop: 1.0,  color: '#900C00' }
    ],

    rainbow: [
        { stop: 0.0,  color: '#FF0000' },
        { stop: 0.11, color: '#FF5900' },
        { stop: 0.22, color: '#FFAF00' },
        { stop: 0.33, color: '#FAFF00' },
        { stop: 0.44, color: '#53FF00' },
        { stop: 0.56, color: '#00FF53' },
        { stop: 0.67, color: '#00FFFA' },
        { stop: 0.78, color: '#0053FF' },
        { stop: 0.89, color: '#5800FF' },
        { stop: 1.0,  color: '#FF00FF' }
    ],

    spectral: [
        { stop: 0.0,  color: '#5E4FA2' },
        { stop: 0.11, color: '#466FB1' },
        { stop: 0.22, color: '#388EBA' },
        { stop: 0.33, color: '#55AFAD' },
        { stop: 0.44, color: '#9FD99F' },
        { stop: 0.56, color: '#F3E991' },
        { stop: 0.67, color: '#FBBA73' },
        { stop: 0.78, color: '#F57A4B' },
        { stop: 0.89, color: '#CE3D43' },
        { stop: 1.0,  color: '#9E0142' }
    ],

    coolwarm: [
        { stop: 0.0,  color: '#3B4CC0' },
        { stop: 0.11, color: '#526CCB' },
        { stop: 0.22, color: '#698BD6' },
        { stop: 0.33, color: '#94ACDA' },
        { stop: 0.44, color: '#C5CDDC' },
        { stop: 0.56, color: '#DFC8C0' },
        { stop: 0.67, color: '#E97B53' },
        { stop: 0.78, color: '#E07055' },
        { stop: 0.89, color: '#CA3A3E' },
        { stop: 1.0,  color: '#B40426' }
    ]
};

function getColorFromGradient(value, gradientName, minVal, maxVal) {
    const gradient = COLOR_GRADIENTS[gradientName] || COLOR_GRADIENTS.spectral;
    let norm = (value - minVal) / (maxVal - minVal);
    norm = Math.max(0, Math.min(1, norm));
    
    let lower = gradient[0];
    let upper = gradient[gradient.length - 1];
    
    for (let i = 0; i < gradient.length - 1; i++) {
        if (norm >= gradient[i].stop && norm <= gradient[i + 1].stop) {
            lower = gradient[i];
            upper = gradient[i + 1];
            break;
        }
    }
    
    const range = upper.stop - lower.stop;
    const ratio = range > 0 ? (norm - lower.stop) / range : 0;
    
    const c1 = hexToRgb(lower.color);
    const c2 = hexToRgb(upper.color);
    
    return rgbToHex(
        Math.round(c1.r + (c2.r - c1.r) * ratio),
        Math.round(c1.g + (c2.g - c1.g) * ratio),
        Math.round(c1.b + (c2.b - c1.b) * ratio)
    );
}

function hexToRgb(hex) {
    const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return r ? {r: parseInt(r[1], 16), g: parseInt(r[2], 16), b: parseInt(r[3], 16)} : null;
}

function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function setupDropdownLogic() {
    const dropdown = document.getElementById('sql-files-dropdown');
    const header = document.getElementById('sql-dropdown-header');
    const list = document.getElementById('sql-files-list');

    header.addEventListener('click', () => {
        list.style.display = list.style.display === 'block' ? 'none' : 'block';
    });

    document.addEventListener('click', (e) => {
        if (!dropdown.contains(e.target)) list.style.display = 'none';
    });

    list.addEventListener('change', () => {
        const checked = list.querySelectorAll('input[type="checkbox"]:checked');
        if (checked.length === 0) header.textContent = 'Select files...';
        else if (checked.length === 1) header.textContent = checked[0].parentNode.textContent.trim();
        else header.textContent = `${checked.length} files selected`;
    });
}

async function loadAvailableConfig() {
    try {
        const response = await fetch('/api/config');
        availableConfig = await response.json();
        
        // Build SQL files list
        const list = document.getElementById('sql-files-list');
        const fragment = document.createDocumentFragment();
        
        availableConfig.sql_files.forEach(file => {
            const div = document.createElement('div');
            div.className = 'dropdown-item';
            div.innerHTML = `<label><input type="checkbox" name="sql-file" value="${file.filename}"
                ${['crc_v1_trusts','crc_v2_invites','crc_v2_trusts','crc_v2_flows'].some(x => file.filename.includes(x)) ? 'checked' : ''}>
                ${file.graph_id}</label>`;
            fragment.appendChild(div);
        });
        
        list.innerHTML = '';
        list.appendChild(fragment);
        list.dispatchEvent(new Event('change'));

        // Populate other dropdowns
        const metricsGraphSelect = document.getElementById('metrics-graph');
        metricsGraphSelect.innerHTML = '<option value="">Auto (first selected)</option>' + 
            availableConfig.sql_files.map(file => 
                `<option value="${file.graph_id}">${file.graph_id}</option>`
            ).join('');
        metricsGraphSelect.value = 'crc_v2_invites';

        // Custom metrics
        if (availableConfig.metric_modes?.categories) {
            const customDiv = document.getElementById('custom-metrics');
            customDiv.innerHTML = Object.entries(availableConfig.metric_modes.categories)
                .map(([key, desc]) => 
                    `<label><input type="checkbox" name="custom-metric" value="${key}"> ${key}: ${desc}</label>`
                ).join('');
        }

        // Gradients
        const gradientSelect = document.getElementById('node-color-gradient');
        gradientSelect.innerHTML = Object.keys(COLOR_GRADIENTS)
            .map(name => `<option value="${name}" ${name === 'spectral' ? 'selected' : ''}>${name.charAt(0).toUpperCase() + name.slice(1)}</option>`)
            .join('');

    } catch (error) {
        console.error('Error config:', error);
        updateStatus('Config error', 'error');
    }
}

function setupEventListeners() {
    // Core buttons
    document.getElementById('load-btn').addEventListener('click', loadGraphs);
    document.getElementById('metrics-btn').addEventListener('click', runMetrics);
    document.getElementById('filter-btn').addEventListener('click', filterNodes);
    document.getElementById('reset-filter-btn').addEventListener('click', () => {
        if(cy) {
            cy.elements().unselect();
            cy.elements().removeClass('highlighted');
        }
        updateStatus('Selection reset', 'info');
    });
    
    document.getElementById('neighbor-toggle-btn').addEventListener('click', toggleNeighborHighlight);
    
    document.getElementById('metrics-mode').addEventListener('change', (e) => {
        document.getElementById('custom-metrics').style.display = e.target.value === 'custom' ? 'block' : 'none';
    });

    document.getElementById('graph-select').addEventListener('change', (e) => {
        if (e.target.value) displayGraph(e.target.value);
    });

    // Toolbar
    document.getElementById('fit-btn')?.addEventListener('click', () => cy?.fit());
    document.getElementById('center-btn')?.addEventListener('click', () => cy?.center());
    document.getElementById('search-btn')?.addEventListener('click', searchNode);
    document.getElementById('node-search')?.addEventListener('keypress', (e) => { 
        if(e.key === 'Enter') searchNode(); 
    });
    document.getElementById('clear-search-btn')?.addEventListener('click', clearSearch);
    
    // Info Panel
    document.querySelector('.close-btn')?.addEventListener('click', () => {
        domCache.infoPanel.style.display = 'none';
    });

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => switchTab(e.target.dataset.tab));
    });

    // Collapsibles
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.collapse-icon');
            const isHidden = content.style.display === 'none';
            content.style.display = isHidden ? 'block' : 'none';
            icon.textContent = isHidden ? '▲' : '▼';
        });
    });

    // Style controls
    document.getElementById('apply-style-btn')?.addEventListener('click', applyStyle);
    document.getElementById('edge-opacity')?.addEventListener('input', (e) => {
        document.getElementById('edge-opacity-value').textContent = e.target.value + '%';
    });
    
    document.getElementById('node-color-metric')?.addEventListener('change', (e) => {
        const isFixed = e.target.value === 'fixed';
        document.getElementById('gradient-selector').style.display = isFixed ? 'none' : 'flex';
        document.getElementById('fixed-color-selector').style.display = isFixed ? 'flex' : 'none';
    });

    document.getElementById('edge-width-min')?.addEventListener('input', (e) => {
        document.getElementById('edge-width-min-value').textContent = e.target.value;
    });
    document.getElementById('edge-width-max')?.addEventListener('input', (e) => {
        document.getElementById('edge-width-max-value').textContent = e.target.value;
    });
}

async function loadGraphs() {
    const selectedFiles = Array.from(document.querySelectorAll('input[name="sql-file"]:checked')).map(cb => cb.value);
    if (selectedFiles.length === 0) return updateStatus('Select at least one SQL file', 'error');

    const config = {
        sql_files: selectedFiles,
        use_cached_layout: document.getElementById('use-cached-layout').checked
    };

    const btn = document.getElementById('load-btn');
    const status = document.getElementById('load-status');
    btn.disabled = true;
    btn.textContent = "Loading...";
    status.style.display = 'block';
    
    try {
        const response = await fetch('/api/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });

        if (!response.ok) throw new Error((await response.json()).detail || 'Failed');

        currentState = await response.json();
        updateStatus(`Loaded ${currentState.loaded_graphs.length} graphs`, 'success');

        document.getElementById('graph-selector').style.display = 'block';
        const select = document.getElementById('graph-select');
        select.innerHTML = currentState.loaded_graphs
            .map(id => `<option value="${id}">${id}</option>`)
            .join('');

        document.getElementById('metrics-btn').disabled = false;

        if (currentState.loaded_graphs.length > 0) {
            select.value = currentState.loaded_graphs[0];
            await displayGraph(currentState.loaded_graphs[0]);
        }

    } catch (err) {
        console.error(err);
        updateStatus(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = "Load Networks";
        status.style.display = 'none';
    }
}

async function runMetrics() {
    let metricsMode = document.getElementById('metrics-mode').value;
    if (metricsMode === 'custom') {
        const sels = Array.from(document.querySelectorAll('input[name="custom-metric"]:checked')).map(cb => cb.value);
        if (sels.length === 0) return updateStatus('Select metric category', 'error');
        metricsMode = sels.join(',');
    }

    const config = {
        metrics_mode: metricsMode,
        metrics_graph_id: document.getElementById('metrics-graph').value || null
    };

    const btn = document.getElementById('metrics-btn');
    const status = document.getElementById('metrics-status');
    btn.disabled = true;
    btn.textContent = "Running...";
    status.style.display = 'block';

    try {
        const response = await fetch('/api/metrics', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });

        if (!response.ok) throw new Error((await response.json()).detail || 'Failed');
        
        const result = await response.json();
        updateStatus(`Computed ${result.metrics_computed.length} metrics`, 'success');
        
        if (cy && result.node_data) {
            cy.batch(() => {
                result.node_data.forEach(data => {
                    const node = cy.getElementById(data.id);
                    if (node.length > 0) {
                        node.data(data);
                    }
                });
            });
            
            const nodes = cy.nodes().map(n => ({ data: n.data() }));
            populateMetricDropdowns(nodes, null);
            
            // Clear cache
            styleCache = {
                sizeRange: { min: 0, max: 1 },
                colorRange: { min: 0, max: 1 },
                widthRange: { min: 0, max: 1 }
            };
            
            if (!performanceMode) {
                updateCytoscapeStyle();
            }
        }

    } catch (err) {
        console.error(err);
        updateStatus(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Metrics";
        status.style.display = 'none';
    }
}

async function displayGraph(graphId) {
    currentGraph = graphId;
    showLoading(true);
    updateStatus(`Loading ${graphId}...`, 'info');

    try {
        const res = await fetch(`/api/graphs/${graphId}/elements`);
        if (!res.ok) throw new Error('Failed elements');
        const data = await res.json();
        graphData[graphId] = data.elements;

        const nodes = data.elements.filter(e => e.group === 'nodes');
        const edges = data.elements.filter(e => e.group === 'edges');
        domCache.nodeCount.textContent = `${nodes.length} nodes`;
        domCache.edgeCount.textContent = `${edges.length} edges`;

        if (cy) {
            cy.destroy();
            cy = null;
        }
        
        // Clear style cache
        styleCache = {
            sizeRange: { min: 0, max: 1 },
            colorRange: { min: 0, max: 1 },
            widthRange: { min: 0, max: 1 }
        };
        
        // Create Cytoscape instance with WebGL enabled
        cy = cytoscape({
            container: domCache.cyContainer,
            elements: graphData[graphId],
            style: getPerformanceStyle(), 
            
            // NEW: Enable WebGL Renderer
            renderer: {
                name: 'canvas',
                webgl: true,           // Turn on experimental WebGL
                webglTexSize: 4096,    // Optional: larger texture size for clearer nodes
                showFps: false         // Set to true if you want to debug performance
            },

            layout: { name: 'preset' },
            minZoom: 0.1,
            maxZoom: 10,
            wheelSensitivity: 0.3,
            boxSelectionEnabled: false, // Disable box selection for better performance
            autounselectify: false,
            autoungrabify: false,
            textureOnViewport: true,    // Use texture during pan/zoom (smoother)
            hideEdgesOnViewport: true,  // Hide edges while moving (huge fps boost)
            hideLabelsOnViewport: true, // Hide labels while moving
            pixelRatio: 1,              // Force 1x resolution (saves GPU on retina screens)
            motionBlur: false
        });
        
        populateMetricDropdowns(nodes, edges);
        setupCyListeners();
        cy.fit();
        
        // Apply styles if in style mode
        if (!performanceMode) {
            updateCytoscapeStyle();
        }
        
        updateStatus(`Displayed ${graphId}`, 'success');

    } catch (err) {
        updateStatus(err.message, 'error');
    } finally {
        showLoading(false);
    }
}

function getPerformanceStyle() {
    return [
        {
            selector: 'node',
            style: {
                'background-color': '#666',
                'width': 10,
                'height': 10,
                'label': '',
                'border-width': 0
            }
        },
        {
            selector: 'edge',
            style: {
                'line-color': '#333',
                'width': 1,
                'opacity': 0.3,
                'curve-style': 'straight',
                'target-arrow-shape': 'none'
            }
        },
        {
            selector: 'node:selected',
            style: {
                'background-color': '#FF0000',
                'width': 20,
                'height': 20,
                'border-width': 0,
                'z-index': 999
            }
        },
        {
            selector: 'edge:selected',
            style: {
                'line-color': '#FF0000',
                'width': 2,
                'opacity': 1,
                'z-index': 999
            }
        },
        {
            selector: '.highlighted',
            style: {
                'background-color': '#FFA500',
                'line-color': '#FFA500',
                'opacity': 0.8,
                'z-index': 998
            }
        },
        {
            selector: '.searched',
            style: {
                'background-color': '#00FF00',
                'border-width': 2,
                'border-color': '#00FF00',
                'z-index': 997
            }
        }
    ];
}

function populateMetricDropdowns(nodes, edges) {
    if (!nodes || nodes.length === 0) return;
    
    const firstNodeData = nodes[0].data;
    const nodeMetrics = Object.keys(firstNodeData).filter(k => 
        !['id', 'label', 'isNew', 'x', 'y', 'source', 'target'].includes(k) &&
        typeof firstNodeData[k] === 'number'
    );
    
    let edgeMetrics = [];
    if (edges && edges.length > 0) {
        const firstEdgeData = edges[0].data;
        edgeMetrics = Object.keys(firstEdgeData).filter(k => 
            !['id', 'source', 'target'].includes(k) &&
            typeof firstEdgeData[k] === 'number'
        );
    }
    
    const rebuildOptions = (select, metrics, defaultVal) => {
        const currentVal = select.value;
        select.innerHTML = '<option value="fixed">Fixed / Select...</option>';
        metrics.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m.replace(/_/g, ' ');
            select.appendChild(opt);
        });
        
        if (metrics.includes(currentVal)) select.value = currentVal;
        else if (metrics.includes(defaultVal)) select.value = defaultVal;
    };

    rebuildOptions(document.getElementById('node-size-metric'), nodeMetrics, 'total_degree');
    rebuildOptions(document.getElementById('node-color-metric'), nodeMetrics, 'community_id');
    
    const filterSelect = document.getElementById('filter-metric');
    const currentFilter = filterSelect.value;
    filterSelect.innerHTML = '<option value="">Select Metric...</option>';
    nodeMetrics.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m.replace(/_/g, ' ');
        filterSelect.appendChild(opt);
    });
    if (nodeMetrics.includes(currentFilter)) filterSelect.value = currentFilter;

    if (edges) {
        const edgeWidthSelect = document.getElementById('edge-width-metric');
        rebuildOptions(edgeWidthSelect, edgeMetrics, 'amount');
    }
    
    const colorElem = document.getElementById('node-color-metric');
    if(colorElem) colorElem.dispatchEvent(new Event('change'));
}

function setupCyListeners() {
    const handleNodeClick = debounce((node) => {
        neighborHighlightState = 0;
        updateNeighborButtonText();
        cy.elements().removeClass('highlighted');
        showNodeInfo(node);
    }, 50);
    
    cy.on('tap', 'node', evt => handleNodeClick(evt.target));
    
    cy.on('tap', 'edge', evt => {
        hideAllInfo();
        showEdgeInfo(evt.target);
    });
    
    cy.on('select unselect', debounce(() => {
        const selected = cy.$(':selected');
        if (selected.length > 1) {
            hideAllInfo();
            showMultiInfo(selected);
        } else if (selected.length === 1) {
            if (selected.isNode()) {
                showNodeInfo(selected);
            } else {
                showEdgeInfo(selected);
            }
        }
    }, 100));
}

function filterNodes() {
    if (!cy) return;
    const metric = document.getElementById('filter-metric').value;
    const op = document.getElementById('filter-operator').value;
    const rawVal = document.getElementById('filter-value').value;
    const val = parseFloat(rawVal);

    if (!metric || isNaN(val)) {
        return updateStatus('Please select a metric and enter a numeric value', 'error');
    }

    cy.batch(() => {
        cy.elements().unselect();
        const matches = cy.nodes().filter(n => {
            const d = n.data(metric);
            if (d === undefined) return false;
            switch(op) {
                case 'gt': return d > val;
                case 'lt': return d < val;
                case 'eq': return d == val;
                case 'gte': return d >= val;
                case 'lte': return d <= val;
                default: return false;
            }
        });
        
        if (matches.length > 0) {
            matches.select();
            updateStatus(`Selected ${matches.length} matching nodes`, 'success');
        } else {
            updateStatus('No nodes match criteria', 'info');
        }
    });
}

function toggleNeighborHighlight() {
    if (!cy) return;
    const selected = cy.$('node:selected');
    if (selected.length === 0) return;

    neighborHighlightState = (neighborHighlightState + 1) % 4;
    updateNeighborButtonText();

    cy.batch(() => {
        cy.elements().removeClass('highlighted');

        if (neighborHighlightState === 0) return;

        let neighbors = cy.collection();
        if (neighborHighlightState === 1 || neighborHighlightState === 3) {
            neighbors = neighbors.union(selected.outgoers());
        }
        if (neighborHighlightState === 2 || neighborHighlightState === 3) {
            neighbors = neighbors.union(selected.incomers());
        }

        if (neighbors.length > 0) {
            neighbors.addClass('highlighted');
            updateStatus(`Highlighted ${neighbors.length} items`, 'info');
        }
    });
}

function updateNeighborButtonText() {
    const btn = document.getElementById('neighbor-toggle-btn');
    const texts = ["Highlight Neighbors: Off", "Highlight: Outgoing", "Highlight: Incoming", "Highlight: Both"];
    btn.textContent = texts[neighborHighlightState];
}

function hideAllInfo() {
    domCache.nodeInfo.style.display = 'none';
    domCache.edgeInfo.style.display = 'none';
    domCache.multiInfo.style.display = 'none';
}

function showNodeInfo(node) {
    hideAllInfo();
    domCache.infoPanel.style.display = 'block';
    domCache.nodeInfo.style.display = 'block';
    
    const data = node.data();
    domCache.nodeId.textContent = data.id;
    switchTab('metrics');

    const metricsHtml = Object.entries(data)
        .filter(([k, v]) => !['id','label','isNew'].includes(k) && typeof v !== 'object')
        .map(([k, v]) => {
            const val = typeof v === 'number' ? 
                (Number.isInteger(v) ? v : v.toFixed(4)) : v;
            return `<div class="metric-row">
                <span class="metric-label">${k.replace(/_/g, ' ')}</span>
                <span class="metric-value">${val}</span>
            </div>`;
        }).join('');
    
    domCache.allMetrics.innerHTML = metricsHtml;

    updateNeighborList(node.incomers().edges(), 'in-count', 'neighbors-in-list', 'source');
    updateNeighborList(node.outgoers().edges(), 'out-count', 'neighbors-out-list', 'target');
}

function showEdgeInfo(edge) {
    hideAllInfo();
    domCache.infoPanel.style.display = 'block';
    domCache.edgeInfo.style.display = 'block';
    
    const data = edge.data();
    document.getElementById('edge-source').textContent = data.source;
    document.getElementById('edge-target').textContent = data.target;
    
    const metricsHtml = Object.entries(data)
        .filter(([k]) => !['id','source','target'].includes(k))
        .map(([k, v]) => `<div class="metric-row">
            <span class="metric-label">${k.replace(/_/g, ' ')}</span>
            <span class="metric-value">${v}</span>
        </div>`).join('');
    
    domCache.edgeMetrics.innerHTML = metricsHtml;
}

function showMultiInfo(collection) {
    hideAllInfo();
    domCache.infoPanel.style.display = 'block';
    domCache.multiInfo.style.display = 'block';
    
    const nodes = collection.nodes();
    const edges = collection.edges();
    
    document.getElementById('multi-node-count').textContent = nodes.length;
    document.getElementById('multi-edge-count').textContent = edges.length;
    
    const metricsList = document.getElementById('multi-metrics-list');
    let html = '';

    if (edges.length > 0) {
        const sources = new Set(edges.map(e => e.data('source')));
        const targets = new Set(edges.map(e => e.data('target')));
        
        html += `
            <div class="metric-row"><span class="metric-label">Distinct Sources</span><span class="metric-value">${sources.size}</span></div>
            <div class="metric-row"><span class="metric-label">Distinct Targets</span><span class="metric-value">${targets.size}</span></div>
            <div class="metric-row" style="border-bottom: 1px dashed #333; margin: 5px 0;"></div>
        `;
    }

    if (nodes.length > 0) {
        const firstData = nodes[0].data();
        const numericKeys = Object.keys(firstData).filter(k => 
            typeof firstData[k] === 'number' && !['x','y','id'].includes(k)
        );

        numericKeys.forEach(key => {
            const values = nodes.map(n => n.data(key));
            const sum = values.reduce((a, b) => a + b, 0);
            const avg = sum / values.length;
            const max = Math.max(...values);
            
            html += `
                <div style="margin-bottom:8px; border-bottom:1px solid #222; padding-bottom:4px;">
                    <div style="color:#4A90E2; font-size:11px; margin-bottom:2px;">${key.replace(/_/g, ' ')}</div>
                    <div style="display:flex; justify-content:space-between; font-size:10px; color:#bbb;">
                        <span>Avg: ${avg.toFixed(2)}</span>
                        <span>Max: ${max.toFixed(2)}</span>
                        <span>Sum: ${sum.toFixed(0)}</span>
                    </div>
                </div>
            `;
        });
    }
    
    metricsList.innerHTML = html;
}

function updateNeighborList(edges, countId, listId, type) {
    document.getElementById(countId).textContent = edges.length;
    const list = document.getElementById(listId);
    
    const MAX_INITIAL = 50;
    let html = '';
    
    edges.slice(0, MAX_INITIAL).forEach(edge => {
        const target = type === 'source' ? edge.source() : edge.target();
        html += `<div class="neighbor-item" onclick="selectNode('${target.id()}')">${target.id()}</div>`;
    });
    
    if (edges.length > MAX_INITIAL) {
        html += `<div class="neighbor-more" onclick="loadMoreNeighbors('${listId}', '${type}', ${MAX_INITIAL})">
            Load ${edges.length - MAX_INITIAL} more...</div>`;
    }
    
    list.innerHTML = html;
}

// Global function for onclick
window.selectNode = function(nodeId) {
    if (!cy) return;
    cy.elements().unselect();
    const node = cy.getElementById(nodeId);
    if (node.length > 0) {
        node.select();
        showNodeInfo(node);
    }
};

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => 
        btn.classList.toggle('active', btn.dataset.tab === tabName));
    document.querySelectorAll('.tab-content').forEach(c => 
        c.style.display = c.id === `${tabName}-tab` ? 'block' : 'none');
}

function updateStatus(msg, type) {
    domCache.status.textContent = msg;
    domCache.status.className = `status status-${type}`;
    domCache.status.style.display = 'block';
    if (type !== 'error') {
        setTimeout(() => domCache.status.style.display = 'none', 4000);
    }
}

function showLoading(show) {
    domCache.loading.style.display = show ? 'flex' : 'none';
}

function searchNode() {
    if (!cy) return updateStatus('Please load a graph first', 'error');
    const term = document.getElementById('node-search').value.trim().toLowerCase();
    if (!term) return;
    
    cy.batch(() => {
        cy.elements().removeClass('searched');
        const matches = cy.nodes().filter(n => n.id().toLowerCase().includes(term));
        
        if (matches.length > 0) {
            matches.addClass('searched');
            if (matches.length === 1) {
                const n = matches[0];
                cy.animate({ center: { eles: n }, zoom: 2 }, { duration: 500 });
                showNodeInfo(n);
            } else {
                cy.fit(matches, 50);
            }
            updateStatus(`Found ${matches.length} node(s)`, 'success');
            document.getElementById('clear-search-btn').style.display = 'inline-block';
        } else {
            updateStatus(`No nodes found`, 'error');
        }
    });
}

function clearSearch() {
    if (cy) cy.elements().removeClass('searched');
    document.getElementById('node-search').value = '';
    document.getElementById('clear-search-btn').style.display = 'none';
    updateStatus('Search cleared', 'info');
}

function applyStyle() {
    if (!cy) return;
    
    currentStyle = {
        node: {
            sizeMetric: document.getElementById('node-size-metric').value,
            sizeMin: parseFloat(document.getElementById('node-size-min').value),
            sizeMax: parseFloat(document.getElementById('node-size-max').value),
            colorMetric: document.getElementById('node-color-metric').value,
            colorFixed: document.getElementById('node-color-fixed').value,
            colorGradient: document.getElementById('node-color-gradient').value,
            colorSelected: '#FF0000'
        },
        edge: {
            widthMetric: document.getElementById('edge-width-metric').value,
            widthMin: parseFloat(document.getElementById('edge-width-min').value),
            widthMax: parseFloat(document.getElementById('edge-width-max').value),
            color: document.getElementById('edge-color').value,
            colorSelected: '#FF0000',
            opacity: parseFloat(document.getElementById('edge-opacity').value) / 100
        }
    };
    
    // Switch to style mode
    performanceMode = false;
    document.querySelector('input[name="render-mode"][value="style"]').checked = true;
    
    updateCytoscapeStyle();
    updateStatus('Style applied', 'success');
}

function updateCytoscapeStyle() {
    if (!cy || performanceMode) return;
    
    const style = currentStyle;
    
    // Calculate ranges
    if (style.node.sizeMetric !== 'fixed' && !styleCache.sizeRange.calculated) {
        let min = Infinity, max = -Infinity;
        cy.nodes().forEach(n => {
            const v = n.data(style.node.sizeMetric);
            if (typeof v === 'number') {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        });
        styleCache.sizeRange = { min, max, calculated: true };
    }
    
    if (style.node.colorMetric !== 'fixed' && !styleCache.colorRange.calculated) {
        let min = Infinity, max = -Infinity;
        cy.nodes().forEach(n => {
            const v = n.data(style.node.colorMetric);
            if (typeof v === 'number') {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        });
        styleCache.colorRange = { min, max, calculated: true };
    }
    
    if (style.edge.widthMetric !== 'fixed' && !styleCache.widthRange.calculated) {
        let min = Infinity, max = -Infinity;
        cy.edges().forEach(e => {
            const v = e.data(style.edge.widthMetric);
            if (typeof v === 'number') {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        });
        styleCache.widthRange = { min, max, calculated: true };
    }
    
    // Build node style
    const nodeStyle = { 
        'border-width': 0, 
        'label': '' 
    };
    
    if (style.node.sizeMetric === 'fixed') {
        nodeStyle['width'] = style.node.sizeMin;
        nodeStyle['height'] = style.node.sizeMin;
    } else {
        const { min, max } = styleCache.sizeRange;
        if (min !== Infinity && min !== max) {
            nodeStyle['width'] = `mapData(${style.node.sizeMetric}, ${min}, ${max}, ${style.node.sizeMin}, ${style.node.sizeMax})`;
            nodeStyle['height'] = nodeStyle['width'];
        } else {
            nodeStyle['width'] = style.node.sizeMin;
            nodeStyle['height'] = style.node.sizeMin;
        }
    }
    
    if (style.node.colorMetric === 'fixed') {
        nodeStyle['background-color'] = style.node.colorFixed;
    }
    
    // Build edge style
    const edgeStyle = {
        'line-color': style.edge.color,
        'target-arrow-color': style.edge.color,
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'opacity': style.edge.opacity
    };
    
    if (style.edge.widthMetric === 'fixed') {
        edgeStyle['width'] = style.edge.widthMin;
    } else {
        const { min, max } = styleCache.widthRange;
        if (min !== Infinity && min !== max) {
            edgeStyle['width'] = `mapData(${style.edge.widthMetric}, ${min}, ${max}, ${style.edge.widthMin}, ${style.edge.widthMax})`;
        } else {
            edgeStyle['width'] = style.edge.widthMin;
        }
    }
    
    // Apply styles
    cy.style()
        .selector('node').style(nodeStyle)
        .selector('node:selected').style({
            'border-width': 3,
            'border-color': style.node.colorSelected,
            'z-index': 999
        })
        .selector('node.searched').style({
            'border-width': 3,
            'border-color': '#00FF00',
            'z-index': 998
        })
        .selector('.highlighted').style({
            'border-width': 2,
            'border-color': '#FF0000',
            'shadow-blur': 10,
            'shadow-color': '#FF0000',
            'z-index': 995
        })
        .selector('edge').style(edgeStyle)
        .selector('edge:selected').style({
            'line-color': style.edge.colorSelected,
            'target-arrow-color': style.edge.colorSelected,
            'opacity': 1,
            'width': Math.max(4, style.edge.widthMax),
            'z-index': 999
        })
        .selector('edge.highlighted').style({
            'line-color': '#FF0000',
            'target-arrow-color': '#FF0000',
            'opacity': 0.8,
            'z-index': 995
        })
        .update();
    
    // Apply gradient colors if needed
    if (style.node.colorMetric !== 'fixed') {
        const { min: colorMin, max: colorMax } = styleCache.colorRange;
        if (colorMin !== Infinity) {
            cy.batch(() => {
                cy.nodes().forEach(node => {
                    const val = node.data(style.node.colorMetric);
                    const color = (typeof val === 'number') 
                        ? getColorFromGradient(val, style.node.colorGradient, colorMin, 
                            colorMin === colorMax ? colorMax + 1 : colorMax)
                        : '#888888';
                    node.style('background-color', color);
                });
            });
        }
    }
}