/**
 * Graph Analyzer Web Viewer
 */

let cy = null;
let currentGraph = null;
let currentState = null;
let availableConfig = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing Graph Analyzer...');
    await loadAvailableConfig();
    setupEventListeners();
});

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
                
                // Default selections for essential mode
                if (['topology', 'centrality', 'clustering', 'community'].includes(key)) {
                    checkbox.checked = true;
                }
                
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(` ${key}: ${description}`));
                customMetricsDiv.appendChild(label);
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
    
    // Close button for info panel
    document.querySelector('.close-btn')?.addEventListener('click', () => {
        document.getElementById('node-info').style.display = 'none';
    });
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
    
    if (metricsMode === 'custom') {
        const selectedMetrics = Array.from(document.querySelectorAll('input[name="custom-metric"]:checked'))
            .map(cb => cb.value);
        
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
        
    console.log('Loading graphs with config:', config);
    
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
        
        // Update toolbar
        document.getElementById('node-count').textContent = `${data.count} nodes`;
        document.getElementById('edge-count').textContent = `${data.count} edges`;
        
        // Initialize or update Cytoscape
        if (cy) {
            cy.destroy();
        }
        
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: data.elements,
            
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': 'mapData(community_id, 0, 100, #FBE723, #440256)',
                        'label': '',  // Hide labels by default for performance
                        'width': 'mapData(total_degree, 0, 100, 5, 30)',
                        'height': 'mapData(total_degree, 0, 100, 5, 30)',
                        'border-width': 0
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'border-width': 2,
                        'border-color': '#FFFF00',
                        'label': 'data(id)'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 1,
                        'line-color': '#323232',
                        'target-arrow-shape': 'triangle',
                        'target-arrow-color': '#323232',
                        'curve-style': 'bezier',
                        'opacity': 0.5
                    }
                }
            ],
            
            layout: {
                name: 'preset'  // Use positions from server
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
 * Show node information
 */
function showNodeInfo(node) {
    const data = node.data();
    const panel = document.getElementById('node-info');
    
    // Update node ID
    document.getElementById('node-id').textContent = data.id;
    
    // Update key metrics
    const metricsHtml = `
        <div>in_degree: ${data.in_degree || 0}</div>
        <div>out_degree: ${data.out_degree || 0}</div>
        <div>community_id: ${data.community_id || 0}</div>
        <div>degree: ${data.total_degree || 0}</div>
    `;
    document.getElementById('key-metrics').innerHTML = metricsHtml;
    
    // Update connections
    const inEdges = node.incomers().edges().length;
    const outEdges = node.outgoers().edges().length;
    const connectionsHtml = `
        <div>In: ${inEdges}</div>
        <div>Out: ${outEdges}</div>
        <div>Mutual: 0</div>
    `;
    document.getElementById('connections').innerHTML = connectionsHtml;
    
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
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    const loadingDiv = document.getElementById('loading');
    loadingDiv.style.display = show ? 'flex' : 'none';
}