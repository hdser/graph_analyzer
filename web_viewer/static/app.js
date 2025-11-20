/**
 * Graph Analyzer Web Viewer
 */

let cy = null;
let currentGraph = null;
let currentState = null;
let availableConfig = null;
let currentStyle = null;
let graphData = {};

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
            sizeMetric: 'fixed', sizeMin: 10, sizeMax: 40,
            colorMetric: 'fixed', colorFixed: '#4A90E2', colorGradient: 'spectral', colorSelected: '#FFD700'
        },
        edge: {
            widthMetric: 'fixed', widthMin: 1, widthMax: 5,
            color: '#999999', colorSelected: '#FF0000', opacity: 0.3
        }
    };
}

const COLOR_GRADIENTS = {
    viridis: [
        { stop: 0, color: '#440154' },
        { stop: 0.25, color: '#31688E' },
        { stop: 0.5, color: '#35B779' },
        { stop: 0.75, color: '#8FD744' },
        { stop: 1, color: '#FDE724' }
    ],
    plasma: [
        { stop: 0, color: '#0D0887' },
        { stop: 0.25, color: '#7E03A8' },
        { stop: 0.5, color: '#CC4778' },
        { stop: 0.75, color: '#F89540' },
        { stop: 1, color: '#F0F921' }
    ],
    inferno: [
        { stop: 0, color: '#000004' },
        { stop: 0.25, color: '#420A68' },
        { stop: 0.5, color: '#932667' },
        { stop: 0.75, color: '#DD513A' },
        { stop: 1, color: '#FCFFA4' }
    ],
    magma: [
        { stop: 0, color: '#000004' },
        { stop: 0.25, color: '#3B0F70' },
        { stop: 0.5, color: '#8C2981' },
        { stop: 0.75, color: '#DE4968' },
        { stop: 1, color: '#FE9F6D' }
    ],
    turbo: [
        { stop: 0, color: '#23171B' },
        { stop: 0.1, color: '#4076F5' },
        { stop: 0.3, color: '#26D0CE' },
        { stop: 0.5, color: '#5EFC82' },
        { stop: 0.7, color: '#FDB32F' },
        { stop: 0.9, color: '#ED7953' },
        { stop: 1, color: '#900C00' }
    ],
    rainbow: [
        { stop: 0, color: '#FF0000' },
        { stop: 0.17, color: '#FF8800' },
        { stop: 0.33, color: '#FFFF00' },
        { stop: 0.5, color: '#00FF00' },
        { stop: 0.67, color: '#00FFFF' },
        { stop: 0.83, color: '#0000FF' },
        { stop: 1, color: '#FF00FF' }
    ],
    spectral: [
        { stop: 0, color: '#5E4FA2' },
        { stop: 0.2, color: '#3288BD' },
        { stop: 0.4, color: '#66C2A5' },
        { stop: 0.5, color: '#E6F598' },
        { stop: 0.6, color: '#FEE08B' },
        { stop: 0.8, color: '#F46D43' },
        { stop: 1, color: '#9E0142' }
    ],
    coolwarm: [
        { stop: 0, color: '#3B4CC0' },
        { stop: 0.25, color: '#6F93D9' },
        { stop: 0.5, color: '#DDDDDD' },
        { stop: 0.75, color: '#E67E5B' },
        { stop: 1, color: '#B40426' }
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

        // Also populate edge width dropdown with numeric metrics placeholder
        const edgeWidthSelect = document.getElementById('edge-width-metric');
        edgeWidthSelect.innerHTML = '<option value="fixed">Fixed</option>';

    } catch (error) {
        console.error('Error config:', error);
        updateStatus('Config error', 'error');
    }
}

function setupEventListeners() {
    document.getElementById('load-btn').addEventListener('click', loadGraphs);

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

    // Close button for info panel (RIGHT PANEL)
    document.querySelector('.close-btn')?.addEventListener('click', () => {
        document.getElementById('info-panel').style.display = 'none';
    });
}

async function loadGraphs() {
    const selectedFiles = Array.from(document.querySelectorAll('input[name="sql-file"]:checked')).map(cb => cb.value);
    if (selectedFiles.length === 0) return updateStatus('Select at least one SQL file', 'error');

    let metricsMode = document.getElementById('metrics-mode').value;
    if (metricsMode === 'custom') {
        const sels = Array.from(document.querySelectorAll('input[name="custom-metric"]:checked')).map(cb => cb.value);
        if (sels.length === 0) return updateStatus('Select metric category', 'error');
        metricsMode = sels.join(',');
    }

    const config = {
        sql_files: selectedFiles,
        metrics_mode: metricsMode,
        metrics_graph_id: document.getElementById('metrics-graph').value || null,
        layout_algorithm: document.getElementById('layout-algorithm').value,
        use_cached_layout: document.getElementById('use-cached-layout').checked
    };

    const btn = document.getElementById('load-btn');
    const status = document.getElementById('load-status');
    btn.disabled = true;
    btn.textContent = "Computing...";
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

        if (currentState.loaded_graphs.length > 0) {
            select.value = currentState.loaded_graphs[0];
            await displayGraph(currentState.loaded_graphs[0]);
        }

    } catch (err) {
        console.error(err);
        updateStatus(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = "Load Graphs";
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
            style: [], // Start with empty style
            layout: { name: 'preset' },
            minZoom: 0.1, maxZoom: 10,
            hideEdgesOnViewport: true, 
            textureOnViewport: true
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
    
    // Extract numeric metrics from nodes
    const firstNodeData = nodes[0].data;
    const nodeMetrics = Object.keys(firstNodeData).filter(k => 
        !['id', 'label', 'isNew', 'x', 'y', 'source', 'target'].includes(k) &&
        typeof firstNodeData[k] === 'number'
    );
    
    // Extract numeric metrics from edges if available
    let edgeMetrics = [];
    if (edges && edges.length > 0) {
        const firstEdgeData = edges[0].data;
        edgeMetrics = Object.keys(firstEdgeData).filter(k => 
            !['id', 'source', 'target'].includes(k) &&
            typeof firstEdgeData[k] === 'number'
        );
    }
    
    // Populate node size dropdown
    const sizeSelect = document.getElementById('node-size-metric');
    const rebuildNodeOptions = (select, metrics, defaultVal) => {
        const currentVal = select.value;
        select.innerHTML = '<option value="fixed">Fixed</option>';
        metrics.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m.replace(/_/g, ' ');
            select.appendChild(opt);
        });
        if (metrics.includes(currentVal)) select.value = currentVal;
        else if (metrics.includes(defaultVal)) select.value = defaultVal;
    };

    rebuildNodeOptions(sizeSelect, nodeMetrics, 'total_degree');
    
    // Populate node color dropdown
    const colorSelect = document.getElementById('node-color-metric');
    rebuildNodeOptions(colorSelect, nodeMetrics, 'community_id');
    
    // Populate edge width dropdown
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
    else if (edgeMetrics.includes('weight')) edgeWidthSelect.value = 'weight';
    
    // Trigger change events to update UI
    colorSelect.dispatchEvent(new Event('change'));
}

function setupCyListeners() {
    cy.on('tap', 'node', evt => {
        const node = evt.target;
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
    document.getElementById('multi-node-count').textContent = collection.nodes().length;
    document.getElementById('multi-edge-count').textContent = collection.edges().length;
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
            colorSelected: '#FFD700'
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
    
    // Build base style object for nodes
    const nodeStyle = { 
        'border-width': 0, 
        'label': ''
    };
    
    // Handle node sizing
    if (style.node.sizeMetric === 'fixed') {
        nodeStyle['width'] = style.node.sizeMin;
        nodeStyle['height'] = style.node.sizeMin;
    } else {
        // Calculate min/max for size mapping
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
            // Fallback to fixed if no valid range
            nodeStyle['width'] = style.node.sizeMin;
            nodeStyle['height'] = style.node.sizeMin;
        }
    }
    
    // Handle node color
    if (style.node.colorMetric === 'fixed') {
        nodeStyle['background-color'] = style.node.colorFixed;
    }
    // If not fixed, we'll apply colors after setting the stylesheet
    
    // Build edge style
    const edgeStyle = {
        'line-color': style.edge.color,
        'target-arrow-color': style.edge.color,
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'opacity': style.edge.opacity
    };
    
    // Handle edge width
    if (style.edge.widthMetric === 'fixed') {
        edgeStyle['width'] = style.edge.widthMin;
    } else {
        // Calculate min/max for edge width mapping
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
    
    // Apply the base stylesheet
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
        .selector('edge').style(edgeStyle)
        .selector('edge:selected').style({
            'line-color': style.edge.colorSelected,
            'target-arrow-color': style.edge.colorSelected,
            'opacity': 1,
            'width': Math.max(4, style.edge.widthMax),
            'z-index': 999
        })
        .update();
    
    // Now apply gradient colors as bypass styles if needed
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
            // No valid range, use default
            colorMin = 0; 
            colorMax = 1; 
        }
        
        // Apply colors as bypass styles
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
        // Clear bypass colors to let stylesheet color take effect
        cy.batch(() => {
            cy.nodes().removeStyle('background-color');
        });
    }
}