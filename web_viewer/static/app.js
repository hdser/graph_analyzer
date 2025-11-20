/**
 * Graph Analyzer Web Viewer - Fixed with Multi-Color Gradients and Smart Position Inference
 */

let cy = null;
let currentGraph = null;
let currentState = null;
let availableConfig = null;
let currentStyle = null;
let graphData = {}; // Store graph data separately to avoid mixing

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing Graph Analyzer...');
    await loadAvailableConfig();
    setupEventListeners();
    initializeDefaultStyle();
});

/**
 * Initialize default style configuration
 */
function initializeDefaultStyle() {
    currentStyle = {
        node: {
            sizeMetric: 'total_degree',
            sizeMin: 5,
            sizeMax: 30,
            colorMetric: 'community_id',
            colorFixed: '#FFFFFF',
            colorGradient: 'viridis', // Use predefined gradient
            colorSelected: '#FF0000'
        },
        edge: {
            widthMetric: 'fixed',
            widthMin: 1,
            widthMax: 3,
            color: '#FFFFFF',
            colorSelected: '#FF0000',
            opacity: 0.3
        }
    };
}

/**
 * Define color gradients with multiple stops
 */
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

/**
 * Interpolate color in a multi-stop gradient
 */
function getColorFromGradient(value, gradientName, minVal = 0, maxVal = 100) {
    const gradient = COLOR_GRADIENTS[gradientName] || COLOR_GRADIENTS.viridis;
    
    // Normalize value to 0-1
    let normalizedValue = (value - minVal) / (maxVal - minVal);
    normalizedValue = Math.max(0, Math.min(1, normalizedValue)); // Clamp to [0,1]
    
    // Find the two stops this value falls between
    let lowerStop = gradient[0];
    let upperStop = gradient[gradient.length - 1];
    
    for (let i = 0; i < gradient.length - 1; i++) {
        if (normalizedValue >= gradient[i].stop && normalizedValue <= gradient[i + 1].stop) {
            lowerStop = gradient[i];
            upperStop = gradient[i + 1];
            break;
        }
    }
    
    // Interpolate between the two stops
    const range = upperStop.stop - lowerStop.stop;
    const valueInRange = (normalizedValue - lowerStop.stop) / range;
    
    // Parse colors
    const color1 = hexToRgb(lowerStop.color);
    const color2 = hexToRgb(upperStop.color);
    
    // Interpolate RGB values
    const r = Math.round(color1.r + (color2.r - color1.r) * valueInRange);
    const g = Math.round(color1.g + (color2.g - color1.g) * valueInRange);
    const b = Math.round(color1.b + (color2.b - color1.b) * valueInRange);
    
    return rgbToHex(r, g, b);
}

/**
 * Convert hex to RGB
 */
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

/**
 * Convert RGB to hex
 */
function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/**
 * Calculate position for nodes without positions based on their neighbors
 * This handles cases where cached layouts don't include new nodes
 */
function inferMissingPositions(elements) {
    // Build complete position map from elements that have positions
    const positionMap = {};
    const nodeMap = {};
    let nodesWithoutPositions = [];
    let totalNodes = 0;
    
    // First, collect all positions and identify nodes without positions
    elements.forEach(el => {
        if (el.group === 'nodes') {
            totalNodes++;
            nodeMap[el.data.id] = el;
            
            if (el.position && el.position.x !== undefined && el.position.y !== undefined) {
                positionMap[el.data.id] = {
                    x: el.position.x,
                    y: el.position.y
                };
            } else {
                nodesWithoutPositions.push(el.data.id);
            }
        }
    });
    
    // If no nodes need positions, we're done
    if (nodesWithoutPositions.length === 0) {
        console.log('All nodes have positions from cache');
        return false;
    }
    
    console.log(`Found ${nodesWithoutPositions.length} nodes without positions out of ${totalNodes} total nodes`);
    console.log(`Nodes with positions: ${Object.keys(positionMap).length}`);
    
    // Build adjacency lists for the graph
    const adjacency = {};
    elements.forEach(el => {
        if (el.group === 'edges') {
            const source = el.data.source;
            const target = el.data.target;
            
            if (!adjacency[source]) adjacency[source] = new Set();
            if (!adjacency[target]) adjacency[target] = new Set();
            
            adjacency[source].add(target);
            adjacency[target].add(source);
        }
    });
    
    // Calculate bounds of existing positions for better placement
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    let hasPositions = false;
    
    Object.values(positionMap).forEach(pos => {
        hasPositions = true;
        minX = Math.min(minX, pos.x);
        maxX = Math.max(maxX, pos.x);
        minY = Math.min(minY, pos.y);
        maxY = Math.max(maxY, pos.y);
    });
    
    const centerX = hasPositions ? (minX + maxX) / 2 : 0;
    const centerY = hasPositions ? (minY + maxY) / 2 : 0;
    const spreadX = hasPositions ? Math.max(100, (maxX - minX) / 4) : 200;
    const spreadY = hasPositions ? Math.max(100, (maxY - minY) / 4) : 200;
    
    console.log(`Graph bounds: X[${minX.toFixed(0)}, ${maxX.toFixed(0)}], Y[${minY.toFixed(0)}, ${maxY.toFixed(0)}]`);
    console.log(`Center: (${centerX.toFixed(0)}, ${centerY.toFixed(0)}), Spread: (${spreadX.toFixed(0)}, ${spreadY.toFixed(0)})`);
    
    // Process nodes without positions in multiple passes
    let maxIterations = 10;
    let iteration = 0;
    let placedInIteration;
    const newlyPlaced = new Set();
    
    do {
        placedInIteration = 0;
        iteration++;
        
        // Try to place nodes based on their neighbors
        nodesWithoutPositions = nodesWithoutPositions.filter(nodeId => {
            if (positionMap[nodeId]) {
                return false; // Already placed in previous iteration
            }
            
            const neighbors = adjacency[nodeId] || new Set();
            
            if (neighbors.size === 0) {
                // Isolated node - will handle later
                return true;
            }
            
            // Find neighbors with positions (either original or newly placed)
            const positionedNeighbors = [];
            neighbors.forEach(neighborId => {
                if (positionMap[neighborId]) {
                    positionedNeighbors.push(positionMap[neighborId]);
                }
            });
            
            if (positionedNeighbors.length > 0) {
                // Calculate centroid of positioned neighbors
                let sumX = 0, sumY = 0;
                positionedNeighbors.forEach(pos => {
                    sumX += pos.x;
                    sumY += pos.y;
                });
                
                const baseX = sumX / positionedNeighbors.length;
                const baseY = sumY / positionedNeighbors.length;
                
                // Add position with small offset to avoid exact overlap
                const newPos = {
                    x: baseX + (Math.random() - 0.5) * 50,
                    y: baseY + (Math.random() - 0.5) * 50
                };
                
                positionMap[nodeId] = newPos;
                nodeMap[nodeId].position = newPos;
                newlyPlaced.add(nodeId);
                placedInIteration++;
                
                console.log(`Placed node ${nodeId.substring(0, 10)}... near ${positionedNeighbors.length} neighbors at (${newPos.x.toFixed(0)}, ${newPos.y.toFixed(0)})`);
                
                return false; // Successfully placed
            }
            
            return true; // Still needs placement
        });
        
        console.log(`Iteration ${iteration}: Placed ${placedInIteration} nodes, ${nodesWithoutPositions.length} remaining`);
        
    } while (placedInIteration > 0 && iteration < maxIterations);
    
    // Handle remaining isolated nodes or nodes with no positioned neighbors
    if (nodesWithoutPositions.length > 0) {
        console.log(`Placing ${nodesWithoutPositions.length} isolated/orphan nodes`);
        
        // If we have some positioned nodes, place orphans around the periphery
        if (hasPositions) {
            let angleStep = (2 * Math.PI) / nodesWithoutPositions.length;
            nodesWithoutPositions.forEach((nodeId, index) => {
                const angle = index * angleStep;
                const radius = Math.max(spreadX, spreadY) * 1.5;
                
                const newPos = {
                    x: centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 50,
                    y: centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 50
                };
                
                positionMap[nodeId] = newPos;
                nodeMap[nodeId].position = newPos;
                newlyPlaced.add(nodeId);
                
                console.log(`Placed orphan node ${nodeId.substring(0, 10)}... at periphery (${newPos.x.toFixed(0)}, ${newPos.y.toFixed(0)})`);
            });
        } else {
            // No existing positions at all - arrange in a circle
            let angleStep = (2 * Math.PI) / nodesWithoutPositions.length;
            nodesWithoutPositions.forEach((nodeId, index) => {
                const angle = index * angleStep;
                const radius = 300;
                
                const newPos = {
                    x: radius * Math.cos(angle),
                    y: radius * Math.sin(angle)
                };
                
                positionMap[nodeId] = newPos;
                nodeMap[nodeId].position = newPos;
                newlyPlaced.add(nodeId);
            });
        }
    }
    
    console.log(`Position inference complete: ${newlyPlaced.size} new positions assigned`);
    
    // Return true if we placed any new nodes (suggests refinement might help)
    return newlyPlaced.size > 0;
}

/**
 * Load available configuration from server
 */
async function loadAvailableConfig() {
    try {
        const response = await fetch('/api/config');
        availableConfig = await response.json();
        
        // Populate SQL files
        const sqlFilesDiv = document.getElementById('sql-files');
        sqlFilesDiv.innerHTML = '';
        
        availableConfig.sql_files.forEach(file => {
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'sql-file';
            checkbox.value = file.filename;
            
            // Default selections
            if (file.filename.includes('crc_v1_trusts') ||
                file.filename.includes('crc_v2_invites') ||
                file.filename.includes('crc_v2_trusts') ||
                file.filename.includes('crc_v2_flows')) {
                checkbox.checked = true;
            }
            
            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(` ${file.graph_id}`));
            sqlFilesDiv.appendChild(label);
        });
        
        // Populate metrics graph selector
        const metricsGraphSelect = document.getElementById('metrics-graph');
        metricsGraphSelect.innerHTML = '<option value="">Auto (first selected)</option>';
        
        availableConfig.sql_files.forEach(file => {
            const option = document.createElement('option');
            option.value = file.graph_id;
            option.textContent = file.graph_id;
            metricsGraphSelect.appendChild(option);
        });
        
        // Set default to crc_v2_invites
        metricsGraphSelect.value = 'crc_v2_invites';
        
        // Populate custom metrics categories
        if (availableConfig.metric_modes && availableConfig.metric_modes.categories) {
            const customMetricsDiv = document.getElementById('custom-metrics');
            customMetricsDiv.innerHTML = '';
            
            Object.entries(availableConfig.metric_modes.categories).forEach(([key, description]) => {
                const label = document.createElement('label');
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'custom-metric';
                checkbox.value = key;
                checkbox.id = `metric-${key}`;
                
                // Don't pre-check any boxes
                checkbox.checked = false;
                
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(` ${key}: ${description}`));
                customMetricsDiv.appendChild(label);
            });
        }
        
        // Populate gradient selector
        const gradientSelect = document.getElementById('node-color-gradient');
        if (gradientSelect) {
            gradientSelect.innerHTML = '';
            Object.keys(COLOR_GRADIENTS).forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name.charAt(0).toUpperCase() + name.slice(1);
                if (name === 'viridis') option.selected = true;
                gradientSelect.appendChild(option);
            });
        }
        
    } catch (error) {
        console.error('Error loading config:', error);
        updateStatus('Error loading configuration', 'error');
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Load button
    document.getElementById('load-btn').addEventListener('click', loadGraphs);
    
    // Metrics mode selector
    document.getElementById('metrics-mode').addEventListener('change', (e) => {
        const customDiv = document.getElementById('custom-metrics');
        if (e.target.value === 'custom') {
            customDiv.style.display = 'block';
        } else {
            customDiv.style.display = 'none';
        }
    });
    
    // Graph selector
    document.getElementById('graph-select').addEventListener('change', (e) => {
        if (e.target.value) {
            displayGraph(e.target.value);
        }
    });
    
    // Toolbar buttons
    document.getElementById('fit-btn')?.addEventListener('click', () => {
        if (cy) cy.fit();
    });
    
    document.getElementById('center-btn')?.addEventListener('click', () => {
        if (cy) cy.center();
    });
    
    // Search functionality
    document.getElementById('search-btn')?.addEventListener('click', searchNode);
    document.getElementById('node-search')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchNode();
    });
    document.getElementById('clear-search-btn')?.addEventListener('click', clearSearch);
    
    // Close button for info panel
    document.querySelector('.close-btn')?.addEventListener('click', () => {
        document.getElementById('node-info').style.display = 'none';
    });
    
    // Tab buttons in info panel
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchTab(e.target.dataset.tab);
        });
    });
    
    // Collapsible sections
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.collapse-icon');
            if (content.style.display === 'none' || content.style.display === '') {
                content.style.display = 'block';
                icon.textContent = '▲';
            } else {
                content.style.display = 'none';
                icon.textContent = '▼';
            }
        });
    });
    
    // Apply style button
    document.getElementById('apply-style-btn')?.addEventListener('click', applyStyle);
    
    // Edge opacity slider
    document.getElementById('edge-opacity')?.addEventListener('input', (e) => {
        document.getElementById('edge-opacity-value').textContent = e.target.value + '%';
    });
    
    // Color metric change
    document.getElementById('node-color-metric')?.addEventListener('change', (e) => {
        const gradientDiv = document.getElementById('gradient-selector');
        const fixedColorDiv = document.getElementById('fixed-color-selector');
        if (e.target.value === 'fixed') {
            gradientDiv.style.display = 'none';
            fixedColorDiv.style.display = 'block';
        } else {
            gradientDiv.style.display = 'block';
            fixedColorDiv.style.display = 'none';
        }
    });
}

/**
 * Search for a node by ID
 */
function searchNode() {
    if (!cy) {
        updateStatus('Please load a graph first', 'error');
        return;
    }
    
    const searchTerm = document.getElementById('node-search').value.trim().toLowerCase();
    if (!searchTerm) return;
    
    // Clear previous selection
    cy.elements().removeClass('searched');
    
    // Find and select matching nodes
    const matchingNodes = cy.nodes().filter(node => {
        return node.id().toLowerCase().includes(searchTerm);
    });
    
    if (matchingNodes.length > 0) {
        matchingNodes.addClass('searched');
        
        // If single match, center on it and show info
        if (matchingNodes.length === 1) {
            const node = matchingNodes[0];
            cy.animate({
                center: { eles: node },
                zoom: 2
            }, {
                duration: 500
            });
            showNodeInfo(node);
        } else {
            // Multiple matches - fit them in view
            cy.fit(matchingNodes, 50);
        }
        
        updateStatus(`Found ${matchingNodes.length} node(s) matching "${searchTerm}"`, 'success');
        document.getElementById('clear-search-btn').style.display = 'inline-block';
    } else {
        updateStatus(`No nodes found matching "${searchTerm}"`, 'error');
    }
}

/**
 * Clear search results
 */
function clearSearch() {
    if (cy) {
        cy.elements().removeClass('searched');
    }
    document.getElementById('node-search').value = '';
    document.getElementById('clear-search-btn').style.display = 'none';
    updateStatus('Search cleared', 'info');
}

/**
 * Apply style configuration
 */
function applyStyle() {
    if (!cy) {
        updateStatus('Please load a graph first', 'error');
        return;
    }
    
    // Read style configuration from UI
    currentStyle = {
        node: {
            sizeMetric: document.getElementById('node-size-metric').value,
            sizeMin: parseFloat(document.getElementById('node-size-min').value),
            sizeMax: parseFloat(document.getElementById('node-size-max').value),
            colorMetric: document.getElementById('node-color-metric').value,
            colorFixed: document.getElementById('node-color-fixed').value,
            colorGradient: document.getElementById('node-color-gradient').value,
            colorSelected: document.getElementById('node-color-selected').value
        },
        edge: {
            widthMetric: document.getElementById('edge-width-metric').value,
            widthMin: parseFloat(document.getElementById('edge-width-min').value),
            widthMax: parseFloat(document.getElementById('edge-width-max').value),
            color: document.getElementById('edge-color').value,
            colorSelected: document.getElementById('edge-color-selected').value,
            opacity: parseFloat(document.getElementById('edge-opacity').value) / 100
        }
    };
    
    // Apply the new style
    updateCytoscapeStyle();
    updateStatus('Style applied', 'success');
}

/**
 * Update Cytoscape style based on current configuration
 */
function updateCytoscapeStyle() {
    if (!cy) return;
    
    const style = currentStyle;
    
    // For gradient colors, we need to manually set colors per node based on metric value
    if (style.node.colorMetric !== 'fixed') {
        // Get min and max values for the metric
        let minVal = Infinity;
        let maxVal = -Infinity;
        
        cy.nodes().forEach(node => {
            const val = node.data(style.node.colorMetric) || 0;
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        });
        
        // Apply gradient colors to each node
        cy.nodes().forEach(node => {
            const val = node.data(style.node.colorMetric) || 0;
            const color = getColorFromGradient(val, style.node.colorGradient, minVal, maxVal);
            node.style('background-color', color);
        });
    }
    
    // Build node style
    const nodeStyle = {
        'border-width': 0,
        'label': '' // No labels by default
    };
    
    // Node size
    if (style.node.sizeMetric === 'fixed') {
        nodeStyle['width'] = style.node.sizeMin;
        nodeStyle['height'] = style.node.sizeMin;
    } else {
        nodeStyle['width'] = `mapData(${style.node.sizeMetric}, 0, 100, ${style.node.sizeMin}, ${style.node.sizeMax})`;
        nodeStyle['height'] = `mapData(${style.node.sizeMetric}, 0, 100, ${style.node.sizeMin}, ${style.node.sizeMax})`;
    }
    
    // Node color (only for fixed color mode)
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
    
    // Edge width
    if (style.edge.widthMetric === 'fixed') {
        edgeStyle['width'] = style.edge.widthMin;
    } else {
        edgeStyle['width'] = `mapData(${style.edge.widthMetric}, 0, 100, ${style.edge.widthMin}, ${style.edge.widthMax})`;
    }
    
    // Apply the style
    cy.style()
        .selector('node')
            .style(nodeStyle)
        .selector('node:selected')
            .style({
                'border-width': 2,
                'border-color': style.node.colorSelected,
                'background-color': style.node.colorSelected,
                'label': 'data(id)'
            })
        .selector('node.searched')
            .style({
                'border-width': 3,
                'border-color': '#00FF00',
                'z-index': 999
            })
        .selector('edge')
            .style(edgeStyle)
        .selector('edge:selected')
            .style({
                'line-color': style.edge.colorSelected,
                'target-arrow-color': style.edge.colorSelected,
                'opacity': 1
            })
        .update();
}

/**
 * Switch tabs in info panel
 */
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });
    
    const tabContent = document.getElementById(`${tabName}-tab`);
    if (tabContent) {
        tabContent.style.display = 'block';
    }
}

/**
 * Load graphs from server
 */
async function loadGraphs() {
    // Get selected SQL files
    const selectedFiles = Array.from(document.querySelectorAll('input[name="sql-file"]:checked'))
        .map(cb => cb.value);
    
    if (selectedFiles.length === 0) {
        updateStatus('Please select at least one SQL file', 'error');
        return;
    }
    
    // Get metrics mode
    const metricsMode = document.getElementById('metrics-mode').value;
    let finalMetricsMode = metricsMode;
    
    console.log('Metrics mode selected:', metricsMode);
    
    if (metricsMode === 'custom') {
        const selectedMetrics = Array.from(document.querySelectorAll('input[name="custom-metric"]:checked'))
            .map(cb => cb.value);
        
        console.log('Selected custom metrics:', selectedMetrics);
        
        if (selectedMetrics.length === 0) {
            updateStatus('Please select at least one metric category', 'error');
            return;
        }
        
        finalMetricsMode = selectedMetrics.join(',');
    }
    
    // Get configuration
    const config = {
        sql_files: selectedFiles,
        metrics_mode: finalMetricsMode,
        metrics_graph_id: document.getElementById('metrics-graph').value || null,
        layout_algorithm: document.getElementById('layout-algorithm').value,
        use_cached_layout: document.getElementById('use-cached-layout').checked
    };
        
    console.log('Configuration:', config);
    
    // Show loading
    showLoading(true);
    updateStatus('Loading graphs...', 'info');
    
    try {
        // Load network
        const response = await fetch('/api/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load graphs');
        }
        
        currentState = await response.json();
        console.log('Network loaded:', currentState);
        
        // Update UI
        updateStatus(
            `Loaded ${currentState.loaded_graphs.length} graph(s) with ${currentState.node_count} nodes in ${currentState.computation_time.toFixed(1)}s`, 
            'success'
        );
        
        // Update graph selector
        const graphSelector = document.getElementById('graph-selector');
        graphSelector.style.display = 'block';
        
        const graphSelect = document.getElementById('graph-select');
        graphSelect.innerHTML = '';
        
        currentState.loaded_graphs.forEach(graphId => {
            const option = document.createElement('option');
            option.value = graphId;
            option.textContent = graphId;
            graphSelect.appendChild(option);
        });
        
        // Display first graph
        if (currentState.loaded_graphs.length > 0) {
            graphSelect.value = currentState.loaded_graphs[0];
            await displayGraph(currentState.loaded_graphs[0]);
        }
        
        // Show stats
        showStats();
        
    } catch (error) {
        console.error('Error loading graphs:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Display a specific graph
 */
async function displayGraph(graphId) {
    console.log(`Displaying graph: ${graphId}`);
    currentGraph = graphId;
    
    showLoading(true);
    updateStatus(`Loading ${graphId}...`, 'info');
    
    try {
        // Fetch graph elements
        const response = await fetch(`/api/graphs/${graphId}/elements`);
        
        if (!response.ok) {
            throw new Error('Failed to load graph elements');
        }
        
        const data = await response.json();
        console.log(`Loaded ${data.count} elements for ${graphId}`);
        
        // Store this graph's data separately to avoid mixing
        graphData[graphId] = data.elements;
        
        // IMPORTANT: Infer positions for nodes without cached positions
        const needsRefinement = inferMissingPositions(data.elements);
        
        // Count nodes and edges
        const nodes = data.elements.filter(el => el.group === 'nodes');
        const edges = data.elements.filter(el => el.group === 'edges');
        
        // Update toolbar
        document.getElementById('node-count').textContent = `${nodes.length} nodes`;
        document.getElementById('edge-count').textContent = `${edges.length} edges`;
        
        // Destroy previous cytoscape instance completely
        if (cy) {
            cy.destroy();
            cy = null;
        }
        
        // Create new cytoscape instance with ONLY this graph's data
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: graphData[graphId], // Use stored data for this specific graph
            
            style: [], // Will be set by updateCytoscapeStyle
            
            layout: {
                name: 'preset'  // Use positions from server (or inferred)
            },
            
            // Performance settings
            minZoom: 0.1,
            maxZoom: 10,
            wheelSensitivity: 0.1,
            hideEdgesOnViewport: true,
            textureOnViewport: true,
            motionBlur: true,
            motionBlurOpacity: 0.2
        });
        
        // OPTIONAL: Run quick layout refinement if we inferred new positions
        if (needsRefinement && document.getElementById('use-cached-layout').checked) {
            console.log('Running quick layout refinement for new nodes...');
            
            updateStatus('Refining layout for new nodes...', 'info');
            
            // Run a quick force-directed layout to settle new nodes
            const layout = cy.layout({
                name: 'cose',
                animate: false,
                randomize: false,  // Don't randomize, use current positions as starting point
                nodeRepulsion: function(node) { return 400; },
                idealEdgeLength: function(edge) { return 50; },
                nodeOverlap: 20,
                numIter: 20,  // Just 20 iterations for quick refinement
                initialTemp: 200, // Lower initial temperature for gentler refinement
                coolingFactor: 0.95,
                minTemp: 1.0,
                gravity: 0.25,
                fit: false, // Don't fit to viewport, maintain current zoom/pan
                padding: 30
            });
            
            layout.run();
            
            console.log('Layout refinement complete');
        }
        
        // Apply current style
        updateCytoscapeStyle();
        
        // Add event listeners
        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            showNodeInfo(node);
        });
        
        // Fit to viewport
        cy.fit();
        
        updateStatus(`Displayed ${graphId}`, 'success');
        
    } catch (error) {
        console.error('Error displaying graph:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Show node information - Enhanced version
 */
function showNodeInfo(node) {
    const data = node.data();
    const panel = document.getElementById('node-info');
    
    // Update node ID
    document.getElementById('node-id').textContent = data.id;
    
    // Switch to metrics tab by default
    switchTab('metrics');
    
    // Populate all metrics
    const metricsDiv = document.getElementById('all-metrics');
    metricsDiv.innerHTML = '';
    
    // Sort metrics alphabetically, but put key metrics first
    const keyMetrics = ['in_degree', 'out_degree', 'total_degree', 'community_id', 'pagerank'];
    const allMetrics = Object.keys(data).filter(key => 
        key !== 'id' && key !== 'label' && typeof data[key] !== 'object'
    );
    
    const sortedMetrics = [
        ...keyMetrics.filter(k => allMetrics.includes(k)),
        ...allMetrics.filter(k => !keyMetrics.includes(k)).sort()
    ];
    
    sortedMetrics.forEach(key => {
        const value = data[key];
        const metricRow = document.createElement('div');
        metricRow.className = 'metric-row';
        
        const label = document.createElement('span');
        label.className = 'metric-label';
        label.textContent = key.replace(/_/g, ' ');
        
        const val = document.createElement('span');
        val.className = 'metric-value';
        val.textContent = typeof value === 'number' ? 
            (Number.isInteger(value) ? value : value.toFixed(4)) : value;
        
        metricRow.appendChild(label);
        metricRow.appendChild(val);
        metricsDiv.appendChild(metricRow);
    });
    
    // Populate neighbors - IMPORTANT: Only from current graph
    const incomers = node.incomers().edges();
    const outgoers = node.outgoers().edges();
    
    // In neighbors
    document.getElementById('in-count').textContent = incomers.length;
    const inList = document.getElementById('neighbors-in-list');
    inList.innerHTML = '';
    
    incomers.forEach((edge, index) => {
        if (index < 100) { // Limit to first 100
            const neighborDiv = document.createElement('div');
            neighborDiv.className = 'neighbor-item';
            neighborDiv.textContent = edge.source().id();
            neighborDiv.addEventListener('click', () => {
                cy.elements().unselect();
                edge.source().select();
                showNodeInfo(edge.source());
            });
            inList.appendChild(neighborDiv);
        }
    });
    
    if (incomers.length > 100) {
        const moreDiv = document.createElement('div');
        moreDiv.className = 'neighbor-more';
        moreDiv.textContent = `... and ${incomers.length - 100} more`;
        inList.appendChild(moreDiv);
    }
    
    // Out neighbors
    document.getElementById('out-count').textContent = outgoers.length;
    const outList = document.getElementById('neighbors-out-list');
    outList.innerHTML = '';
    
    outgoers.forEach((edge, index) => {
        if (index < 100) { // Limit to first 100
            const neighborDiv = document.createElement('div');
            neighborDiv.className = 'neighbor-item';
            neighborDiv.textContent = edge.target().id();
            neighborDiv.addEventListener('click', () => {
                cy.elements().unselect();
                edge.target().select();
                showNodeInfo(edge.target());
            });
            outList.appendChild(neighborDiv);
        }
    });
    
    if (outgoers.length > 100) {
        const moreDiv = document.createElement('div');
        moreDiv.className = 'neighbor-more';
        moreDiv.textContent = `... and ${outgoers.length - 100} more`;
        outList.appendChild(moreDiv);
    }
    
    panel.style.display = 'block';
}

/**
 * Show statistics
 */
function showStats() {
    if (!currentState) return;
    
    const statsDiv = document.getElementById('stats');
    statsDiv.style.display = 'block';
    
    statsDiv.innerHTML = `
        <h3>Statistics</h3>
        <div>Nodes: ${currentState.node_count.toLocaleString()}</div>
        <div>Edges: ${currentState.edge_count.toLocaleString()}</div>
        <div>Metrics: ${currentState.metrics_computed.length}</div>
        <div>Layout: ${currentState.layout_computation_time.toFixed(1)}s</div>
        <div>Total: ${currentState.computation_time.toFixed(1)}s</div>
    `;
}

/**
 * Update status message
 */
function updateStatus(message, type = 'info') {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status status-${type}`;
    statusDiv.style.display = 'block';
    
    // Auto-hide after 5 seconds for success/info messages
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    const loadingDiv = document.getElementById('loading');
    loadingDiv.style.display = show ? 'flex' : 'none';
}