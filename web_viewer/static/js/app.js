/**
 * App Module
 * Main entry point and initialization
 */

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing Graph Analyzer...');
    
    // Inject SVG icons
    Icons.inject();
    
    // Cache DOM elements
    cacheDOMElements();
    
    // Load configuration
    await loadAvailableConfig();
    
    // Setup event listeners
    setupEventListeners();
    setupDropdownLogic();
    setupCollapsibleSections();
    
    // Initialize features
    initializeDefaultStyle();
    addPerformanceToggle();
    
    // Setup modules
    DistributionsComm.setup();
    AutoReload.setup();
    CompositeMetrics.setup();
    InfoPanel.setupNeighborClicks();
    Metrics.initFilterUI();
    
    console.log('Graph Analyzer initialized');
});

// =============================================================================
// CONFIGURATION
// =============================================================================

async function loadAvailableConfig() {
    try {
        const config = await API.getConfig();
        State.availableConfig = config;
        
        // Build SQL files list
        const list = document.getElementById('sql-files-list');
        const fragment = document.createDocumentFragment();
        
        config.sql_files.forEach(file => {
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

        // Build Node Properties files list (if available)
        const propertiesContainer = document.getElementById('properties-dropdown-container');
        const propertiesList = document.getElementById('properties-files-list');
        
        if (config.node_properties_files && config.node_properties_files.length > 0) {
            const propertiesFragment = document.createDocumentFragment();
            
            config.node_properties_files.forEach(file => {
                const div = document.createElement('div');
                div.className = 'dropdown-item';
                const displayName = file.name || file.filename.replace('.sql', '');
                div.innerHTML = `<label><input type="checkbox" name="properties-file" value="${file.filename}">
                    ${displayName}</label>`;
                propertiesFragment.appendChild(div);
            });
            
            propertiesList.innerHTML = '';
            propertiesList.appendChild(propertiesFragment);
            propertiesList.dispatchEvent(new Event('change'));
            
            // Show the properties dropdown container
            propertiesContainer.style.display = '';
            
            console.log(`Loaded ${config.node_properties_files.length} properties files`);
        } else {
            // Hide if no files available
            propertiesContainer.style.display = 'none';
        }

        // Populate metrics target graph dropdown
        const metricsGraphSelect = document.getElementById('metrics-graph');
        metricsGraphSelect.innerHTML = '<option value="">Auto (first selected)</option>' + 
            config.sql_files.map(file => `<option value="${file.graph_id}">${file.graph_id}</option>`).join('');
        metricsGraphSelect.value = 'crc_v2_invites';

        // Populate custom metrics checkboxes
        if (config.metric_modes?.categories) {
            const customDiv = document.getElementById('custom-metrics');
            customDiv.innerHTML = Object.entries(config.metric_modes.categories)
                .map(([key, desc]) => 
                    `<label title="${desc}">
                        <input type="checkbox" name="custom-metric" value="${key}" 
                            ${['topology', 'clustering'].includes(key) ? 'checked' : ''}>
                        <span style="font-weight:500;">${key}</span>
                        <span style="color:#808080; font-size:11px; display:block; margin-left:20px; margin-bottom:4px;">${desc}</span>
                    </label>`
                ).join('');
        }

        // Hide data source UI elements in production mode
        if (config.hide_data_source_ui) {
            console.log('[CONFIG] Production mode - hiding admin UI sections');
            
            // Hide entire sections using their IDs
            // Note: metrics-section stays visible - users can still run metrics
            const sectionsToHide = [
                'data-source-section',   // Data Source (Load) section
                'auto-reload-section',   // Auto Reload section  
            ];
            
            sectionsToHide.forEach(id => {
                const section = document.getElementById(id);
                if (section) {
                    section.style.display = 'none';
                    console.log(`[CONFIG] Hidden section: ${id}`);
                } else {
                    console.warn(`[CONFIG] Section not found: ${id}`);
                }
            });
            
            // Show loading indicator and start polling for data
            DOMCache.loading.style.display = 'flex';
            updateStatus('Loading network data in background...', 'info');
            
            // Poll for data ready state
            pollForDataReady();
        }

        // Populate color gradients
        const gradientSelect = document.getElementById('node-color-gradient');
        gradientSelect.innerHTML = Object.keys(COLOR_GRADIENTS)
            .map(name => `<option value="${name}" ${name === 'spectral' ? 'selected' : ''}>${name.charAt(0).toUpperCase() + name.slice(1)}</option>`)
            .join('');

    } catch (error) {
        console.error('Error loading config:', error);
        updateStatus('Config error: ' + error.message, 'error');
    }
}

/**
 * Wait for background data load to complete using SSE.
 * Used in production mode when HIDE_DATA_SOURCE_UI is true.
 * SSE is much more efficient than polling - single connection, server pushes updates.
 */
async function pollForDataReady() {
    console.log('[STARTUP] Connecting to startup events...');
    
    // First check if data is already ready
    try {
        const statusResponse = await fetch('/api/startup-status');
        const status = await statusResponse.json();
        
        if (status.status === 'ready') {
            console.log('[STARTUP] Data already ready, displaying graph...');
            await displayLoadedGraph(status);
            return;
        }
    } catch (err) {
        console.error('[STARTUP] Error checking initial status:', err);
    }
    
    // Connect to SSE for updates
    const eventSource = new EventSource('/api/startup-events');
    
    eventSource.onopen = () => {
        console.log('[STARTUP] SSE connected, waiting for data load...');
    };
    
    eventSource.addEventListener('status', async (event) => {
        const status = JSON.parse(event.data);
        console.log('[STARTUP] Status update:', status.status, status.message);
        
        if (status.status === 'loading') {
            updateStatus(`Loading: ${status.message}`, 'info');
        } else if (status.status === 'ready') {
            console.log('[STARTUP] ✓ Data ready!');
            eventSource.close();
            await displayLoadedGraph(status);
        } else if (status.status === 'error') {
            console.error('[STARTUP] ✗ Load failed:', status.message);
            eventSource.close();
            updateStatus(`Load failed: ${status.message}`, 'error');
            DOMCache.loading.style.display = 'none';
        }
    });
    
    eventSource.addEventListener('ping', () => {
        // Keepalive, ignore
    });
    
    eventSource.onerror = (err) => {
        console.error('[STARTUP] SSE error:', err);
        eventSource.close();
        
        // Fallback to simple polling if SSE fails
        console.log('[STARTUP] Falling back to polling...');
        fallbackPoll();
    };
}

/**
 * Display the loaded graph after startup completes.
 */
async function displayLoadedGraph(status) {
    updateStatus('Data loaded, displaying graph...', 'info');
    
    try {
        // Get the full network state
        const stateResponse = await fetch('/api/state');
        const state = await stateResponse.json();
        
        if (state.loaded_graphs && state.loaded_graphs.length > 0) {
            State.currentState = state;
            
            // Populate graph selector
            const graphSelect = document.getElementById('graph-select');
            const graphSelector = document.getElementById('graph-selector');
            
            graphSelect.innerHTML = '<option value="">Select graph...</option>' +
                state.loaded_graphs.map(g => `<option value="${g}">${g}</option>`).join('');
            
            if (graphSelector) {
                graphSelector.style.display = 'block';
            }
            
            // Auto-select and display first graph
            const firstGraph = state.loaded_graphs[0];
            graphSelect.value = firstGraph;
            console.log('[STARTUP] Displaying graph:', firstGraph);
            await GraphLoader.displayGraph(firstGraph);
            
            updateStatus(`Loaded ${status.node_count || state.node_count} nodes`, 'success');
        }
    } catch (err) {
        console.error('[STARTUP] Error displaying graph:', err);
        updateStatus('Error loading graph data', 'error');
    }
    
    DOMCache.loading.style.display = 'none';
}

/**
 * Fallback polling if SSE is not available.
 * Polls every 5 seconds (much less frequent than before).
 */
async function fallbackPoll() {
    const maxAttempts = 60;  // 5 minutes max
    let attempts = 0;
    
    const poll = async () => {
        attempts++;
        
        try {
            const response = await fetch('/api/startup-status');
            const status = await response.json();
            
            console.log(`[POLL #${attempts}] status=${status.status}`);
            
            if (status.status === 'ready') {
                await displayLoadedGraph(status);
                return;
            } else if (status.status === 'error') {
                updateStatus(`Load failed: ${status.message}`, 'error');
                DOMCache.loading.style.display = 'none';
                return;
            }
            
            updateStatus(`Loading... (${attempts * 5}s)`, 'info');
            
            if (attempts < maxAttempts) {
                setTimeout(poll, 5000);  // Poll every 5 seconds
            } else {
                updateStatus('Load timeout - please refresh', 'error');
                DOMCache.loading.style.display = 'none';
            }
        } catch (err) {
            console.error('[POLL] Error:', err);
            if (attempts < maxAttempts) {
                setTimeout(poll, 5000);
            }
        }
    };
    
    setTimeout(poll, 2000);
}

// =============================================================================
// COLLAPSIBLE SECTIONS
// =============================================================================

function setupCollapsibleSections() {
    document.querySelectorAll('.section.collapsible .collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            header.classList.toggle('collapsed');
            
            // Find the content element (next sibling with class style-subsection)
            const content = header.nextElementSibling;
            if (content) {
                if (header.classList.contains('collapsed')) {
                    content.style.display = 'none';
                } else {
                    content.style.display = '';
                }
            }
            
            // Rotate chevron icon
            const icon = header.querySelector('.collapse-icon');
            if (icon) {
                icon.style.transform = header.classList.contains('collapsed') ? 'rotate(-90deg)' : '';
            }
        });
    });
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // Core buttons
    document.getElementById('load-btn').addEventListener('click', () => GraphLoader.loadGraphs());
    document.getElementById('metrics-btn').addEventListener('click', () => Metrics.run());
    document.getElementById('filter-btn').addEventListener('click', () => Metrics.filter());
    document.getElementById('reset-filter-btn').addEventListener('click', () => Metrics.reset());
    document.getElementById('neighbor-toggle-btn').addEventListener('click', () => CytoscapeManager.toggleNeighborHighlight());

    // Graph selector
    document.getElementById('graph-select').addEventListener('change', (e) => {
        if (e.target.value) GraphLoader.displayGraph(e.target.value);
    });

    // Distributions button
    document.getElementById('distributions-btn')?.addEventListener('click', () => DistributionsComm.open());

    // Toolbar buttons
    document.getElementById('fit-btn')?.addEventListener('click', () => State.cy?.fit());
    document.getElementById('center-btn')?.addEventListener('click', () => State.cy?.center());
    document.getElementById('search-btn')?.addEventListener('click', () => Search.search());
    document.getElementById('node-search')?.addEventListener('keypress', (e) => { 
        if (e.key === 'Enter') Search.search(); 
    });
    document.getElementById('clear-search-btn')?.addEventListener('click', () => Search.clear());

    // Edge loader
    document.getElementById('load-edges-btn')?.addEventListener('click', () => {
        if (State.currentGraph) GraphLoader.loadEdgesIncrementally(State.currentGraph);
    });
    
    // Info Panel
    document.querySelector('.close-btn')?.addEventListener('click', () => InfoPanel.close());

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => InfoPanel.switchTab(e.target.dataset.tab));
    });

    // Copy buttons
    document.getElementById('copy-node-btn')?.addEventListener('click', () => Export.copyNodeData());
    document.getElementById('copy-id-btn')?.addEventListener('click', () => Export.copyNodeId());
    document.getElementById('copy-selected-btn')?.addEventListener('click', () => Export.copySelectedIds());
    document.getElementById('export-csv-btn')?.addEventListener('click', () => Export.exportSelectedAsCsv());

    // Style controls (debounced)
    const styleHandler = Utils.debounce(() => {
        if (!State.performanceMode) CytoscapeManager.updateStyle();
    }, 150);
    
    document.getElementById('node-size-metric')?.addEventListener('change', styleHandler);
    document.getElementById('node-color-metric')?.addEventListener('change', styleHandler);
    document.getElementById('node-color-gradient')?.addEventListener('change', styleHandler);
    
    // Apply Style button - applies all style settings including edge styles
    document.getElementById('apply-style-btn')?.addEventListener('click', () => {
        if (!State.cy) {
            updateStatus('Load a graph first', 'error');
            return;
        }
        CytoscapeManager.applyFullStyle();
    });
    
    // Edge opacity slider - update display value
    document.getElementById('edge-opacity')?.addEventListener('input', (e) => {
        const valueDisplay = document.getElementById('edge-opacity-value');
        if (valueDisplay) {
            valueDisplay.textContent = `${e.target.value}%`;
        }
    });
}

// =============================================================================
// DROPDOWN LOGIC
// =============================================================================

function setupDropdownLogic() {
    // SQL files dropdown
    setupSingleDropdown(
        'sql-files-dropdown',
        'sql-dropdown-header', 
        'sql-files-list',
        'Select files...'
    );
    
    // Properties files dropdown
    setupSingleDropdown(
        'properties-files-dropdown',
        'properties-dropdown-header',
        'properties-files-list',
        'Select properties...'
    );
}

/**
 * Setup a single dropdown with header, list, and toggle behavior
 */
function setupSingleDropdown(dropdownId, headerId, listId, placeholder) {
    const dropdown = document.getElementById(dropdownId);
    const header = document.getElementById(headerId);
    const list = document.getElementById(listId);
    
    if (!dropdown || !header || !list) return;

    header.addEventListener('click', () => {
        list.style.display = list.style.display === 'block' ? 'none' : 'block';
    });

    document.addEventListener('click', (e) => {
        if (!dropdown.contains(e.target)) {
            list.style.display = 'none';
        }
    });

    list.addEventListener('change', () => {
        const checked = list.querySelectorAll('input[type="checkbox"]:checked');
        if (checked.length === 0) {
            header.textContent = placeholder;
        } else if (checked.length === 1) {
            header.textContent = checked[0].parentNode.textContent.trim();
        } else {
            header.textContent = `${checked.length} files selected`;
        }
    });
}

// =============================================================================
// STYLE INITIALIZATION
// =============================================================================

function initializeDefaultStyle() {
    const sizeMetric = document.getElementById('node-size-metric');
    const colorMetric = document.getElementById('node-color-metric');
    
    if (sizeMetric) sizeMetric.value = '';
    if (colorMetric) colorMetric.value = '';
}

function addPerformanceToggle() {
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
        
        toggleDiv.querySelectorAll('input[name="render-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) {
                    CytoscapeManager.toggleRenderMode(e.target.value === 'performance');
                }
            });
        });
    }
}