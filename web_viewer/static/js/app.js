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
    setupPanelNavigation();  // NEW: Setup button-based panel system
    setupSubsectionCollapsibles();
    
    // Initialize features
    initializeDefaultStyle();
    addPerformanceToggle();
    
    // Setup modules
    DistributionsComm.setup();
    AutoReload.setup();
    CompositeMetrics.setup();
    InfoPanel.setupNeighborClicks();
    Snapshots.init();

    // Initialize Embedding Panel (Deep Learning)
    try {
        await EmbeddingPanel.init();
        console.log('[App] EmbeddingPanel initialized');
    } catch (error) {
        console.warn('[App] EmbeddingPanel init failed:', error);
    }
    
    // Initialize Metrics module
    try {
        await Metrics.init();
        console.log('[App] Metrics module initialized');
    } catch (error) {
        console.error('[App] Failed to initialize Metrics:', error);
    }
    
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
        if (metricsGraphSelect) {
            metricsGraphSelect.innerHTML = '<option value="">Auto (first selected)</option>' + 
                config.sql_files.map(file => `<option value="${file.graph_id}">${file.graph_id}</option>`).join('');
            metricsGraphSelect.value = 'crc_v2_invites';
        }

        // Metrics UI is now initialized by Metrics.init()
        // (Old metrics-graph and custom-metrics elements removed)


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
            console.log('[STARTUP] ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ Data ready!');
            eventSource.close();
            await displayLoadedGraph(status);
        } else if (status.status === 'error') {
            console.error('[STARTUP] ÃƒÂ¢Ã…â€œÃ¢â‚¬â€ Error:', status.message);
            eventSource.close();
            updateStatus(`Error: ${status.message}`, 'error');
            DOMCache.loading.style.display = 'none';
        }
    });
    
    eventSource.onerror = (error) => {
        console.error('[STARTUP] SSE error:', error);
        eventSource.close();
        
        // Fallback to polling if SSE fails
        console.log('[STARTUP] Falling back to polling...');
        startPollingFallback();
    };
}

/**
 * Fallback polling in case SSE doesn't work
 */
async function startPollingFallback() {
    const maxAttempts = 60;
    let attempts = 0;
    
    const poll = async () => {
        attempts++;
        try {
            const response = await fetch('/api/startup-status');
            const status = await response.json();
            
            if (status.status === 'ready') {
                await displayLoadedGraph(status);
                return;
            } else if (status.status === 'error') {
                updateStatus(`Error: ${status.message}`, 'error');
                DOMCache.loading.style.display = 'none';
                return;
            }
        } catch (err) {
            console.error('[STARTUP] Poll error:', err);
        }
        
        if (attempts < maxAttempts) {
            setTimeout(poll, 2000);
        } else {
            updateStatus('Timeout waiting for data', 'error');
            DOMCache.loading.style.display = 'none';
        }
    };
    
    poll();
}

/**
 * Display graph after background load completes
 */
async function displayLoadedGraph(status) {
    console.log('[STARTUP] Displaying loaded graph:', status);
    
    // The graphs are already loaded in State.graphs by the background loader
    // We just need to fetch them via the API and display
    const graphs = status.graphs || [];
    
    if (graphs.length === 0) {
        updateStatus('No graphs loaded', 'error');
        DOMCache.loading.style.display = 'none';
        return;
    }
    
    // Fetch each graph's data
    for (const graphId of graphs) {
        try {
            const graphData = await API.getGraph(graphId);
            State.graphs[graphId] = graphData;
            console.log(`[STARTUP] Loaded ${graphId}: ${graphData.nodes?.length || 0} nodes`);
        } catch (err) {
            console.error(`[STARTUP] Error loading ${graphId}:`, err);
        }
    }
    
    // Setup graph selector if multiple graphs
    if (graphs.length > 1) {
        const selector = document.getElementById('graph-selector');
        const select = document.getElementById('graph-select');
        
        select.innerHTML = graphs.map(id => 
            `<option value="${id}">${id}</option>`
        ).join('');
        
        selector.style.display = 'block';
    }
    
    // Display first graph
    const firstGraph = graphs[0];
    GraphLoader.displayGraph(firstGraph);
    
    // Enable metrics button
    const metricsBtn = document.getElementById('compute-metrics-btn');
    if (metricsBtn) {
        metricsBtn.disabled = false;
    }
    
    DOMCache.loading.style.display = 'none';
    updateStatus(`Loaded ${graphs.length} graph(s)`, 'success');
}

// =============================================================================
// PANEL NAVIGATION SYSTEM
// =============================================================================

/**
 * Setup the new button-based panel navigation system
 */
function setupPanelNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn[data-panel]');
    const panels = document.querySelectorAll('.panel[data-panel]');
    const panelCloseButtons = document.querySelectorAll('.panel-close');
    
    console.log('[Panel Nav] Found', navButtons.length, 'buttons and', panels.length, 'panels');
    
    // Store active panel
    let activePanel = null;
    
    /**
     * Close all panels
     */
    function closeAllPanels() {
        panels.forEach(panel => {
            panel.classList.remove('active');
        });
        navButtons.forEach(btn => {
            btn.classList.remove('active');
        });
        activePanel = null;
        console.log('[Panel Nav] All panels closed');
    }
    
    /**
     * Open a specific panel
     */
    function openPanel(panelName) {
        console.log('[Panel Nav] Opening panel:', panelName);
        const panel = document.getElementById(`panel-${panelName}`);
        const button = document.querySelector(`.nav-btn[data-panel="${panelName}"]`);
        
        if (!panel) {
            console.error('[Panel Nav] Panel not found:', `panel-${panelName}`);
            return;
        }
        if (!button) {
            console.error('[Panel Nav] Button not found for panel:', panelName);
            return;
        }
        
        // If clicking the same button, close the panel
        if (activePanel === panelName) {
            console.log('[Panel Nav] Closing active panel:', panelName);
            closeAllPanels();
            return;
        }
        
        // Close all panels first
        closeAllPanels();
        
        // Open the requested panel
        panel.classList.add('active');
        button.classList.add('active');
        activePanel = panelName;
        
        console.log('[Panel Nav] Panel opened:', panelName);
        
        // Inject icons in the panel content
        setTimeout(() => {
            Icons.inject();
            console.log('[Panel Nav] Icons injected');
        }, 50);
    }
    
    // Setup button click handlers
    navButtons.forEach(btn => {
        const panelName = btn.dataset.panel;
        console.log('[Panel Nav] Setting up button for panel:', panelName);
        
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('[Panel Nav] Button clicked:', panelName);
            openPanel(panelName);
        });
    });
    
    // Setup panel close buttons
    panelCloseButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('[Panel Nav] Close button clicked');
            closeAllPanels();
        });
    });
    
    // Setup ESC key to close panels
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && activePanel) {
            console.log('[Panel Nav] ESC pressed, closing panel');
            closeAllPanels();
        }
    });
    
    console.log('[Panel Nav] Setup complete');
    
    // Inject icons in buttons
    Icons.inject();
}

/**
 * Setup collapsible subsections within panels
 */
function setupSubsectionCollapsibles() {
    document.querySelectorAll('.collapsible-sub').forEach(header => {
        header.addEventListener('click', () => {
            header.classList.toggle('collapsed');
            
            // Find the target content
            const targetId = header.dataset.target;
            const content = document.getElementById(targetId);
            
            if (content) {
                if (header.classList.contains('collapsed')) {
                    content.style.display = 'none';
                } else {
                    content.style.display = '';
                    // Inject icons when content is shown
                    Icons.inject();
                }
            }
            
            // Rotate icon
            const icon = header.querySelector('.collapse-icon-small');
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
    // Metrics button handled by Metrics module itself
    document.getElementById('neighbor-toggle-btn')?.addEventListener('click', () => CytoscapeManager.toggleNeighborHighlight());

    // Graph selector
    document.getElementById('graph-select')?.addEventListener('change', (e) => {
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

    // Edge loader - check if snapshot is active first
    document.getElementById('load-edges-btn')?.addEventListener('click', async () => {
        // If Snapshots module is available and handles it, we're done
        if (typeof Snapshots !== 'undefined' && Snapshots.loadEdges) {
            const handled = await Snapshots.loadEdges();
            if (handled) return;
        }
        // Otherwise use normal graph loader
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

    // Data Explorer
    document.getElementById('data-explorer-btn')?.addEventListener('click', () => {
        if (typeof DataExplorer !== 'undefined') {
            DataExplorer.open();
        }
    });
    
    // Filter nodes
    document.getElementById('filter-btn')?.addEventListener('click', () => Metrics.filter());
    document.getElementById('reset-filter-btn')?.addEventListener('click', () => Metrics.reset());
    
    // Node visibility controls
    document.getElementById('show-only-selected-btn')?.addEventListener('click', () => Metrics.showOnlySelected());
    document.getElementById('hide-selected-btn')?.addEventListener('click', () => Metrics.hideSelected());
    document.getElementById('show-all-nodes-btn')?.addEventListener('click', () => Metrics.showAllNodes());
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