/**
 * Graph Analyzer Web Viewer
 */

let cy = null;
let currentGraph = null;
let currentState = null;
let availableConfig = null;
let currentStyle = null;
let graphData = {};
let neighborHighlightState = 0; // 0: Off, 1: Out, 2: In, 3: Both

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing Graph Analyzer...');
    await loadAvailableConfig();
    setupEventListeners();
    setupDropdownLogic();
    initializeDefaultStyle();
});

function initializeDefaultStyle() {
    currentStyle = {
        node: {
            sizeMetric: 'fixed', sizeMin: 20, sizeMax: 60,
            colorMetric: 'fixed', colorFixed: '#4A90E2', colorGradient: 'spectral', colorSelected: '#FF0000'
        },
        edge: {
            widthMetric: 'fixed', widthMin: 2, widthMax: 5,
            color: '#fcfafa', colorSelected: '#FF0000', opacity: 0.2
        }
    };
}

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
    
    let lower = gradient[0], upper = gradient[gradient.length-1];
    for(let i=0; i<gradient.length-1; i++) {
        if(norm >= gradient[i].stop && norm <= gradient[i+1].stop) {
            lower = gradient[i]; upper = gradient[i+1]; break;
        }
    }
    const range = upper.stop - lower.stop;
    const ratio = (norm - lower.stop) / range;
    
    const c1 = hexToRgb(lower.color), c2 = hexToRgb(upper.color);
    return rgbToHex(
        Math.round(c1.r + (c2.r - c1.r) * ratio),
        Math.round(c1.g + (c2.g - c1.g) * ratio),
        Math.round(c1.b + (c2.b - c1.b) * ratio)
    );
}

function hexToRgb(hex) {
    const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return r ? {r:parseInt(r[1],16), g:parseInt(r[2],16), b:parseInt(r[3],16)} : null;
}
function rgbToHex(r,g,b) { return "#"+((1<<24)+(r<<16)+(g<<8)+b).toString(16).slice(1); }

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
        
        const list = document.getElementById('sql-files-list');
        list.innerHTML = '';
        availableConfig.sql_files.forEach(file => {
            const div = document.createElement('div');
            div.className = 'dropdown-item';
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'sql-file';
            checkbox.value = file.filename;
            
            if (['crc_v1_trusts','crc_v2_invites','crc_v2_trusts','crc_v2_flows'].some(x => file.filename.includes(x))) {
                checkbox.checked = true;
            }
            
            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(` ${file.graph_id}`));
            div.appendChild(label);
            list.appendChild(div);
        });
        list.dispatchEvent(new Event('change'));

        const metricsGraphSelect = document.getElementById('metrics-graph');
        metricsGraphSelect.innerHTML = '<option value="">Auto (first selected)</option>';
        availableConfig.sql_files.forEach(file => {
            const option = document.createElement('option');
            option.value = file.graph_id;
            option.textContent = file.graph_id;
            metricsGraphSelect.appendChild(option);
        });
        metricsGraphSelect.value = 'crc_v2_invites';

        if (availableConfig.metric_modes && availableConfig.metric_modes.categories) {
            const customDiv = document.getElementById('custom-metrics');
            customDiv.innerHTML = '';
            Object.entries(availableConfig.metric_modes.categories).forEach(([key, desc]) => {
                const label = document.createElement('label');
                const cb = document.createElement('input');
                cb.type = 'checkbox'; cb.name = 'custom-metric'; cb.value = key;
                
                label.appendChild(cb);
                label.appendChild(document.createTextNode(` ${key}: ${desc}`));
                customDiv.appendChild(label);
            });
        }

        const gradientSelect = document.getElementById('node-color-gradient');
        Object.keys(COLOR_GRADIENTS).forEach(name => {
            const opt = document.createElement('option');
            opt.value = name; opt.textContent = name.charAt(0).toUpperCase() + name.slice(1);
            if(name==='spectral') opt.selected = true;
            gradientSelect.appendChild(opt);
        });

        const edgeWidthSelect = document.getElementById('edge-width-metric');
        edgeWidthSelect.innerHTML = '<option value="fixed">Fixed</option>';

    } catch (error) {
        console.error('Error config:', error);
        updateStatus('Config error', 'error');
    }
}

function setupEventListeners() {
    document.getElementById('load-btn').addEventListener('click', loadGraphs);
    document.getElementById('metrics-btn').addEventListener('click', runMetrics);

    // Filter Buttons
    document.getElementById('filter-btn').addEventListener('click', filterNodes);
    document.getElementById('reset-filter-btn').addEventListener('click', () => {
        if(cy) cy.elements().unselect();
        updateStatus('Selection reset', 'info');
    });

    // Neighbor Toggle
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
    document.getElementById('node-search')?.addEventListener('keypress', (e) => { if(e.key==='Enter') searchNode(); });
    document.getElementById('clear-search-btn')?.addEventListener('click', clearSearch);

    // Info Panel
    document.querySelector('.close-btn')?.addEventListener('click', () => {
        document.getElementById('info-panel').style.display = 'none';
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => switchTab(e.target.dataset.tab));
    });

    // Collapsible
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.collapse-icon');
            if (content.style.display === 'none') {
                content.style.display = 'block';
                icon.textContent = '▲';
            } else {
                content.style.display = 'none';
                icon.textContent = '▼';
            }
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

    // Edge width range inputs
    document.getElementById('edge-width-min')?.addEventListener('input', (e) => {
        document.getElementById('edge-width-min-value').textContent = e.target.value;
    });
    document.getElementById('edge-width-max')?.addEventListener('input', (e) => {
        document.getElementById('edge-width-max-value').textContent = e.target.value;
    });

    document.querySelector('.close-btn')?.addEventListener('click', () => {
        document.getElementById('info-panel').style.display = 'none';
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
        select.innerHTML = '';
        currentState.loaded_graphs.forEach(id => {
            const opt = document.createElement('option');
            opt.value = id; opt.textContent = id;
            select.appendChild(opt);
        });

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
            updateCytoscapeStyle();
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
    updateStatus(`Switching to ${graphId}...`, 'info');

    try {
        const res = await fetch(`/api/graphs/${graphId}/elements`);
        if (!res.ok) throw new Error('Failed elements');
        const data = await res.json();
        graphData[graphId] = data.elements;

        const nodes = data.elements.filter(e => e.group === 'nodes');
        const edges = data.elements.filter(e => e.group === 'edges');
        document.getElementById('node-count').textContent = `${nodes.length} nodes`;
        document.getElementById('edge-count').textContent = `${edges.length} edges`;

        if (cy) cy.destroy();
        
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: graphData[graphId],
            style: [], 
            layout: { name: 'preset' },
            minZoom: 0.1, maxZoom: 10,
            hideEdgesOnViewport: true, 
            textureOnViewport: true,
            pixelRatio: 1,
            motionBlur: false
        });
        
        populateMetricDropdowns(nodes, edges);
        updateCytoscapeStyle();
        setupCyListeners();
        cy.fit();
        updateStatus(`Displayed ${graphId}`, 'success');

    } catch (err) {
        updateStatus(err.message, 'error');
    } finally {
        showLoading(false);
    }
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

    // Populate Style Dropdowns
    rebuildOptions(document.getElementById('node-size-metric'), nodeMetrics, 'total_degree');
    rebuildOptions(document.getElementById('node-color-metric'), nodeMetrics, 'community_id');
    
    // Populate Filter Dropdown
    const filterSelect = document.getElementById('filter-metric');
    // Keep "Select Metric..." prompt at top
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
        const currentEdgeVal = edgeWidthSelect.value;
        edgeWidthSelect.innerHTML = '<option value="fixed">Fixed</option>';
        edgeMetrics.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m.replace(/_/g, ' ');
            edgeWidthSelect.appendChild(opt);
        });
        if (edgeMetrics.includes(currentEdgeVal)) edgeWidthSelect.value = currentEdgeVal;
        else if (edgeMetrics.includes('amount')) edgeWidthSelect.value = 'amount';
    }
    
    const colorElem = document.getElementById('node-color-metric');
    if(colorElem) colorElem.dispatchEvent(new Event('change'));
}

function setupCyListeners() {
    cy.on('tap', 'node', evt => {
        const node = evt.target;
        neighborHighlightState = 0; // Reset neighbor state on new click
        updateNeighborButtonText();
        cy.elements().removeClass('highlighted-neighbor');
        
        hideAllInfo();
        showNodeInfo(node);
    });

    cy.on('tap', 'edge', evt => {
        const edge = evt.target;
        hideAllInfo();
        showEdgeInfo(edge);
    });

    cy.on('select unselect', () => {
        setTimeout(() => {
            const selected = cy.$(':selected');
            if (selected.length > 1) {
                hideAllInfo();
                showMultiInfo(selected);
            } else if (selected.length === 1) {
                if (selected.isNode()) {
                    hideAllInfo(); showNodeInfo(selected);
                } else {
                    hideAllInfo(); showEdgeInfo(selected);
                }
            }
        }, 50);
    });
}

// ==========================================
// NEW: Filter Logic
// ==========================================
function filterNodes() {
    if (!cy) return;
    const metric = document.getElementById('filter-metric').value;
    const op = document.getElementById('filter-operator').value;
    const rawVal = document.getElementById('filter-value').value;
    const val = parseFloat(rawVal);

    if (!metric || isNaN(val)) {
        return updateStatus('Please select a metric and enter a numeric value', 'error');
    }

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
}

// ==========================================
// NEW: Neighbor Highlight Logic
// ==========================================
function toggleNeighborHighlight() {
    if (!cy) return;
    const selected = cy.$('node:selected');
    if (selected.length === 0) return;

    // Cycle state: 0(Off) -> 1(Out) -> 2(In) -> 3(Both) -> 0(Off)
    neighborHighlightState = (neighborHighlightState + 1) % 4;
    updateNeighborButtonText();

    cy.elements().removeClass('highlighted-neighbor');

    if (neighborHighlightState === 0) return;

    let neighbors = cy.collection();
    if (neighborHighlightState === 1 || neighborHighlightState === 3) {
        neighbors = neighbors.union(selected.outgoers());
    }
    if (neighborHighlightState === 2 || neighborHighlightState === 3) {
        neighbors = neighbors.union(selected.incomers());
    }

    if (neighbors.length > 0) {
        neighbors.addClass('highlighted-neighbor');
        updateStatus(`Highlighted ${neighbors.length} items`, 'info');
    }
}

function updateNeighborButtonText() {
    const btn = document.getElementById('neighbor-toggle-btn');
    const texts = ["Highlight Neighbors: Off", "Highlight: Outgoing", "Highlight: Incoming", "Highlight: Both"];
    btn.textContent = texts[neighborHighlightState];
}

function hideAllInfo() {
    document.getElementById('node-info').style.display = 'none';
    document.getElementById('edge-info').style.display = 'none';
    document.getElementById('multi-info').style.display = 'none';
}

function showNodeInfo(node) {
    hideAllInfo();
    document.getElementById('info-panel').style.display = 'block';
    document.getElementById('node-info').style.display = 'block';
    
    const data = node.data();
    document.getElementById('node-id').textContent = data.id;
    switchTab('metrics');

    const div = document.getElementById('all-metrics');
    div.innerHTML = '';
    Object.keys(data).forEach(k => {
        if (['id','label','isNew'].includes(k)) return;
        if (typeof data[k] === 'object') return;
        const val = typeof data[k] === 'number' ? (Number.isInteger(data[k]) ? data[k] : data[k].toFixed(4)) : data[k];
        div.innerHTML += `<div class="metric-row"><span class="metric-label">${k.replace(/_/g, ' ')}</span><span class="metric-value">${val}</span></div>`;
    });

    updateNeighborList(node.incomers().edges(), 'in-count', 'neighbors-in-list', 'source');
    updateNeighborList(node.outgoers().edges(), 'out-count', 'neighbors-out-list', 'target');
}

function showEdgeInfo(edge) {
    hideAllInfo();
    document.getElementById('info-panel').style.display = 'block';
    document.getElementById('edge-info').style.display = 'block';
    
    const data = edge.data();
    // Apply updated address styling classes via HTML in index.html or here
    document.getElementById('edge-source').textContent = data.source;
    document.getElementById('edge-target').textContent = data.target;
    
    const metricsDiv = document.getElementById('edge-metrics');
    metricsDiv.innerHTML = '';
    Object.keys(data).forEach(k => {
        if (['id','source','target'].includes(k)) return;
        metricsDiv.innerHTML += `<div class="metric-row"><span class="metric-label">${k.replace(/_/g, ' ')}</span><span class="metric-value">${data[k]}</span></div>`;
    });
}

function showMultiInfo(collection) {
    hideAllInfo();
    document.getElementById('info-panel').style.display = 'block';
    document.getElementById('multi-info').style.display = 'block';
    
    const nodes = collection.nodes();
    const edges = collection.edges();
    
    document.getElementById('multi-node-count').textContent = nodes.length;
    document.getElementById('multi-edge-count').textContent = edges.length;
    
    const metricsList = document.getElementById('multi-metrics-list');
    metricsList.innerHTML = '';

    // 1. Detailed Counts for Edges
    if (edges.length > 0) {
        const sources = new Set(edges.map(e => e.data('source')));
        const targets = new Set(edges.map(e => e.data('target')));
        
        metricsList.innerHTML += `
            <div class="metric-row"><span class="metric-label">Distinct Sources</span><span class="metric-value">${sources.size}</span></div>
            <div class="metric-row"><span class="metric-label">Distinct Targets</span><span class="metric-value">${targets.size}</span></div>
            <div class="metric-row" style="border-bottom: 1px dashed #333; margin: 5px 0;"></div>
        `;
    }

    // 2. Aggregate Numeric Metrics for Nodes
    if (nodes.length > 0) {
        const firstData = nodes[0].data();
        const numericKeys = Object.keys(firstData).filter(k => 
            typeof firstData[k] === 'number' && !['x','y','id'].includes(k)
        );

        if (numericKeys.length === 0) {
            metricsList.innerHTML += '<div style="padding:10px; color:#808080; font-style:italic;">No numeric metrics to aggregate.</div>';
        } else {
            numericKeys.forEach(key => {
                const values = nodes.map(n => n.data(key));
                const sum = values.reduce((a, b) => a + b, 0);
                const avg = sum / values.length;
                const min = Math.min(...values);
                const max = Math.max(...values);
                
                metricsList.innerHTML += `
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
    }
}

function updateNeighborList(edges, countId, listId, type) {
    document.getElementById(countId).textContent = edges.length;
    const list = document.getElementById(listId);
    list.innerHTML = '';
    edges.slice(0,50).forEach(edge => {
        const div = document.createElement('div');
        div.className = 'neighbor-item';
        const target = type==='source' ? edge.source() : edge.target();
        div.textContent = target.id();
        div.onclick = () => {
            cy.elements().unselect(); target.select(); showNodeInfo(target);
        };
        list.appendChild(div);
    });
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabName));
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = c.id === `${tabName}-tab` ? 'block' : 'none');
}

function updateStatus(msg, type) {
    const el = document.getElementById('status');
    el.textContent = msg; el.className = `status status-${type}`; el.style.display = 'block';
    if(type!=='error') setTimeout(()=>el.style.display='none', 4000);
}

function showLoading(show) { document.getElementById('loading').style.display = show ? 'flex' : 'none'; }

function searchNode() {
    if (!cy) return updateStatus('Please load a graph first', 'error');
    const term = document.getElementById('node-search').value.trim().toLowerCase();
    if (!term) return;
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
    
    updateCytoscapeStyle();
    updateStatus('Style applied', 'success');
}

function updateCytoscapeStyle() {
    if (!cy) return;
    const style = currentStyle;
    
    const nodeStyle = { 
        'border-width': 0, 
        'label': ''
    };
    
    if (style.node.sizeMetric === 'fixed') {
        nodeStyle['width'] = style.node.sizeMin;
        nodeStyle['height'] = style.node.sizeMin;
    } else {
        let sizeMin = Infinity, sizeMax = -Infinity;
        cy.nodes().forEach(n => {
            const v = n.data(style.node.sizeMetric);
            if(typeof v === 'number') { 
                sizeMin = Math.min(sizeMin, v); 
                sizeMax = Math.max(sizeMax, v); 
            }
        });
        if (sizeMin !== Infinity && sizeMin !== sizeMax) {
            nodeStyle['width'] = `mapData(${style.node.sizeMetric}, ${sizeMin}, ${sizeMax}, ${style.node.sizeMin}, ${style.node.sizeMax})`;
            nodeStyle['height'] = `mapData(${style.node.sizeMetric}, ${sizeMin}, ${sizeMax}, ${style.node.sizeMin}, ${style.node.sizeMax})`;
        } else {
            nodeStyle['width'] = style.node.sizeMin;
            nodeStyle['height'] = style.node.sizeMin;
        }
    }
    
    if (style.node.colorMetric === 'fixed') {
        nodeStyle['background-color'] = style.node.colorFixed;
    }
    
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
        let edgeMin = Infinity, edgeMax = -Infinity;
        cy.edges().forEach(e => {
            const v = e.data(style.edge.widthMetric);
            if(typeof v === 'number') { 
                edgeMin = Math.min(edgeMin, v); 
                edgeMax = Math.max(edgeMax, v); 
            }
        });
        if (edgeMin !== Infinity && edgeMin !== edgeMax) {
            edgeStyle['width'] = `mapData(${style.edge.widthMetric}, ${edgeMin}, ${edgeMax}, ${style.edge.widthMin}, ${style.edge.widthMax})`;
        } else {
            edgeStyle['width'] = style.edge.widthMin;
        }
    }
    
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
        // NEW: Highlight neighbor style
        .selector('.highlighted-neighbor').style({
            'border-width': 2,
            'border-color': '#FF0000', // Orange glow
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
        .selector('edge.highlighted-neighbor').style({
            'line-color': '#FF0000',
            'target-arrow-color': '#FF0000',
            'opacity': 0.8,
            'z-index': 995
        })
        .update();
    
    if (style.node.colorMetric !== 'fixed') {
        let colorMin = Infinity, colorMax = -Infinity;
        cy.nodes().forEach(node => {
            const val = node.data(style.node.colorMetric);
            if (typeof val === 'number') {
                if (val < colorMin) colorMin = val;
                if (val > colorMax) colorMax = val;
            }
        });
        
        if (colorMin === Infinity || colorMin === colorMax) { 
            colorMin = 0; 
            colorMax = 1; 
        }
        
        cy.batch(() => {
            cy.nodes().forEach(node => {
                const val = node.data(style.node.colorMetric);
                const color = (typeof val === 'number') 
                    ? getColorFromGradient(val, style.node.colorGradient, colorMin, colorMax)
                    : '#888888';
                node.style('background-color', color);
            });
        });
    } else {
        cy.batch(() => {
            cy.nodes().removeStyle('background-color');
        });
    }
}