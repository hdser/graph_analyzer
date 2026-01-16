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
    
    // Load configuration (includes renderer config)
    await loadAvailableConfig();
    
    // Initialize renderer settings and load user preference
    await initializeRenderer();
    
    // Setup event listeners
    setupEventListeners();
    setupDropdownLogic();
    setupPanelNavigation();  // Setup button-based panel system
    setupSubsectionCollapsibles();
    setupRendererPreferenceControls();
    
    // Initialize features
    initializeDefaultStyle();
    
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
// RENDERER INITIALIZATION
// =============================================================================

/**
 * Initialize renderer settings and load user preference
 */
async function initializeRenderer() {
    try {
        // Load renderer config from server
        const rendererConfig = await API.getRendererConfig();
        
        // Initialize RendererSettings with server config
        if (typeof RendererSettings !== 'undefined') {
            RendererSettings.init(rendererConfig);
            console.log('[App] RendererSettings initialized');
        }
        
        // Load user's saved preference
        State.loadRendererPreference();
        console.log(`[App] Renderer preference: ${State.rendererPreference}`);
        
        // Update UI to show capability summary
        updateRendererCapabilityDisplay();
        
    } catch (error) {
        console.warn('[App] Failed to load renderer config, using defaults:', error);
        // Continue with defaults
        if (typeof RendererSettings !== 'undefined') {
            RendererSettings.init({});
        }
    }
}

/**
 * Update the renderer capability display in the UI
 */
function updateRendererCapabilityDisplay() {
    const capabilityInfo = document.getElementById('renderer-capability-info');
    if (!capabilityInfo) return;
    
    if (typeof RendererFactory === 'undefined' || typeof WebGLDetector === 'undefined') {
        capabilityInfo.innerHTML = '<span class="warning">Renderer modules not loaded</span>';
        return;
    }
    
    const summary = RendererFactory.getCapabilitySummary();
    const maxNodesFormatted = summary.maxNodes >= 1000000 
        ? `${(summary.maxNodes / 1000000).toFixed(1)}M` 
        : `${Math.round(summary.maxNodes / 1000)}k`;
    
    // SVG icons for status indicators
    const checkIcon = Icons.get('check', { size: 14 });
    const warningIcon = Icons.get('warning', { size: 14 });
    const infoIcon = Icons.get('info', { size: 14 });
    
    let html = `<span class="capability-tier capability-${summary.tier}" style="display: flex; align-items: center; gap: 4px;">`;
    
    if (summary.cosmosAvailable && summary.cosmosLibraryLoaded) {
        html += `<span style="color: #52c41a;">${checkIcon}</span> cosmos.gl available (max ~${maxNodesFormatted} nodes)`;
    } else if (summary.cosmosLibraryLoaded && !summary.cosmosAvailable) {
        html += `<span style="color: #faad14;">${warningIcon}</span> cosmos.gl library loaded but WebGL limited: ${summary.reason}`;
    } else {
        html += `<span style="color: #4A90E2;">${infoIcon}</span> Cytoscape.js only (cosmos.gl library not loaded)`;
    }
    
    html += `</span>`;
    
    // Add GPU info if available
    if (summary.gpuInfo && summary.gpuInfo !== 'Unknown') {
        html += `<br><span class="gpu-info" title="GPU: ${summary.gpuInfo}">${summary.gpuInfo}</span>`;
    }
    
    capabilityInfo.innerHTML = html;
}

/**
 * Setup renderer preference radio controls
 */
function setupRendererPreferenceControls() {
    const radios = document.querySelectorAll('input[name="renderer-preference"]');
    if (radios.length === 0) return;
    
    // Set initial value
    radios.forEach(radio => {
        if (radio.value === State.rendererPreference) {
            radio.checked = true;
        }
        
        radio.addEventListener('change', async (e) => {
            if (e.target.checked) {
                const preference = e.target.value;
                console.log(`[App] Renderer preference changed to: ${preference}`);
                
                // Update state and reload graph if one is loaded
                await GraphLoader.switchRenderer(preference);
                
                Toast.show(`Renderer preference set to: ${preference}`, 'success');
            }
        });
    });
    
    // Listen for external preference changes
    document.addEventListener('rendererPreferenceChanged', (e) => {
        radios.forEach(radio => {
            radio.checked = (radio.value === e.detail.preference);
        });
    });
    
    // Listen for renderer changes to show/hide cosmos simulation controls
    document.addEventListener('rendererChanged', (e) => {
        const isCosmos = e.detail.type === 'cosmos';
        const controls = document.getElementById('cosmos-simulation-controls');
        if (controls) {
            controls.style.display = isCosmos ? 'block' : 'none';
        }
        updateSimulationButtons(State.cosmosSimulationPaused);
    });
    
    // Setup cosmos.gl simulation control buttons
    setupSimulationControls();
    
    // Setup all enhanced cosmos simulation controls (presets, sliders, snapshots, etc.)
    setupEnhancedSimulationControls();
}

/**
 * Setup cosmos.gl simulation control buttons and all enhanced controls
 */
function setupSimulationControls() {
    const pauseBtn = document.getElementById('cosmos-pause-btn');
    const startBtn = document.getElementById('cosmos-start-btn');
    const fitBtn = document.getElementById('cosmos-fit-btn');
    const stepBtn = document.getElementById('cosmos-step-btn');
    
    // Start/Resume simulation
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            if (State.rendererType === 'cosmos' && State.renderer) {
                State.renderer.startSimulation(0.5);
                updateSimulationButtons(false);
                Toast.show('Simulation started', 'info');
            }
        });
    }
    
    // Pause simulation
    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
            GraphLoader.pauseSimulation();
            updateSimulationButtons(true);
            Toast.show('Simulation paused', 'info');
        });
    }
    
    // Step simulation (single frame)
    if (stepBtn) {
        stepBtn.addEventListener('click', () => {
            if (State.rendererType === 'cosmos' && State.renderer) {
                State.renderer.stepSimulation();
            }
        });
    }
    
    // Fit view
    if (fitBtn) {
        fitBtn.addEventListener('click', () => {
            const renderer = State.renderer;
            if (renderer && renderer.fitView) {
                renderer.fitView();
            }
        });
    }
}

/**
 * Update simulation button visibility based on pause state
 */
function updateSimulationButtons(isPaused) {
    const pauseBtn = document.getElementById('cosmos-pause-btn');
    const startBtn = document.getElementById('cosmos-start-btn');
    
    if (pauseBtn && startBtn) {
        pauseBtn.style.display = isPaused ? 'none' : 'inline-flex';
        startBtn.style.display = isPaused ? 'inline-flex' : 'none';
    }
    
    State.cosmosSimulationPaused = isPaused;
}

/**
 * Setup cosmos simulation parameter controls
 * Sliders update display, Apply button applies and restarts simulation
 */
function setupEnhancedSimulationControls() {
    // Slider value display updates
    const sliders = [
        { id: 'cosmos-repulsion', valueId: 'cosmos-repulsion-value' },
        { id: 'cosmos-gravity', valueId: 'cosmos-gravity-value' },
        { id: 'cosmos-link-distance', valueId: 'cosmos-link-distance-value' },
        { id: 'cosmos-friction', valueId: 'cosmos-friction-value' },
        { id: 'cosmos-link-spring', valueId: 'cosmos-link-spring-value' },
        { id: 'cosmos-decay', valueId: 'cosmos-decay-value' }
    ];
    
    sliders.forEach(({ id, valueId }) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(valueId);
        if (slider && display) {
            slider.addEventListener('input', () => {
                display.textContent = slider.value;
            });
        }
    });
    
    // Apply button - apply parameters and restart simulation
    document.getElementById('cosmos-apply-params')?.addEventListener('click', () => {
        applyCosmosSimulationParams();
    });
    
    // Reset button - reset to defaults and apply
    document.getElementById('cosmos-reset-params')?.addEventListener('click', () => {
        resetCosmosSimulationParams();
    });
    
    // Layout snapshot controls
    document.getElementById('cosmos-snapshot-btn')?.addEventListener('click', () => {
        if (State.rendererType === 'cosmos' && State.renderer) {
            const name = `snapshot_${Date.now()}`;
            if (State.renderer.createSnapshot(name)) {
                State._lastSnapshotName = name;
                Toast.show('Layout saved', 'success');
            } else {
                Toast.show('Failed to save layout', 'error');
            }
        }
    });
    
    document.getElementById('cosmos-restore-btn')?.addEventListener('click', () => {
        if (State.rendererType === 'cosmos' && State.renderer) {
            const name = State._lastSnapshotName || 'default';
            if (State.renderer.restoreSnapshot(name)) {
                Toast.show('Layout restored', 'success');
            } else {
                Toast.show('No saved layout to restore', 'warning');
            }
        }
    });
    
    // Position export
    document.getElementById('cosmos-export-pos-btn')?.addEventListener('click', () => {
        if (State.rendererType === 'cosmos' && State.renderer) {
            const positions = State.renderer.exportPositions();
            const blob = new Blob([JSON.stringify(positions, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `layout_${State.currentGraph || 'graph'}_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            Toast.show('Layout exported', 'success');
        }
    });
    
    // Position import
    document.getElementById('cosmos-import-pos-btn')?.addEventListener('click', () => {
        document.getElementById('cosmos-import-file')?.click();
    });
    
    document.getElementById('cosmos-import-file')?.addEventListener('change', async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        
        try {
            const text = await file.text();
            const positions = JSON.parse(text);
            
            if (State.rendererType === 'cosmos' && State.renderer) {
                State.renderer.importPositions(positions, true);
                Toast.show(`Imported layout for ${Object.keys(positions).length} nodes`, 'success');
            }
        } catch (err) {
            console.error('Import error:', err);
            Toast.show('Failed to import: Invalid file', 'error');
        }
        
        e.target.value = '';
    });
}


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

        // Apply UI mode configuration
        const uiMode = config.ui_mode || {};
        const hiddenPanels = uiMode.hidden_panels || [];
        const productionMode = uiMode.production_mode || false;
        const autoLoadOnStartup = uiMode.auto_load_on_startup || false;
        
        if (productionMode && hiddenPanels.length > 0) {
            console.log('[CONFIG] Production mode - hiding panels:', hiddenPanels);
            
            // Hide navigation buttons and panels for hidden panels
            hiddenPanels.forEach(panelName => {
                // Hide nav button
                const navBtn = document.querySelector(`.nav-btn[data-panel="${panelName}"]`);
                if (navBtn) {
                    navBtn.style.display = 'none';
                    console.log(`[CONFIG] Hidden nav button: ${panelName}`);
                }
                
                // Hide panel
                const panel = document.getElementById(`panel-${panelName}`);
                if (panel) {
                    panel.style.display = 'none';
                    console.log(`[CONFIG] Hidden panel: ${panelName}`);
                }
            });
        }
        
        // Handle auto-load on startup
        if (autoLoadOnStartup) {
            console.log('[CONFIG] Auto-load enabled - waiting for data...');
            
            // Show loading indicator and start polling for data
            DOMCache.loading.style.display = 'flex';
            updateStatus('Loading network data in background...', 'info');
            
            // Poll for data ready state
            pollForDataReady();
        }

        // Populate color gradients from ColorGradients module if available
        const gradientSelect = document.getElementById('node-color-gradient');
        if (gradientSelect && typeof ColorGradients !== 'undefined') {
            // Use module names, preserving HTML-defined options if present
            const existingOptions = gradientSelect.querySelectorAll('option').length;
            if (existingOptions <= 1) {
                gradientSelect.innerHTML = ColorGradients.getNames()
                    .map(name => `<option value="${name}" ${name === 'turbo' ? 'selected' : ''}>${name.charAt(0).toUpperCase() + name.slice(1)}</option>`)
                    .join('');
            }
        }

    } catch (error) {
        console.error('Error loading config:', error);
        updateStatus('Config error: ' + error.message, 'error');
    }
}

/**
 * Wait for background data load to complete using SSE.
 * Used in production mode when AUTO_LOAD_ON_STARTUP is true.
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
            console.log('[STARTUP] [OK] Data ready!');
            eventSource.close();
            await displayLoadedGraph(status);
        } else if (status.status === 'error') {
            console.error('[STARTUP] [ERROR] Error:', status.message);
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
    console.log('[STARTUP] displayLoadedGraph called with status:', JSON.stringify(status, null, 2));
    
    // The backend returns 'loaded_graphs' - check both for compatibility
    const graphs = status.loaded_graphs || status.graphs || [];
    
    console.log('[STARTUP] Extracted graphs array:', graphs);
    console.log('[STARTUP] graphs.length:', graphs.length);
    
    if (!graphs || graphs.length === 0) {
        console.error('[STARTUP] No graphs in status! Full status object:', status);
        updateStatus('No graphs loaded - check server logs', 'error');
        DOMCache.loading.style.display = 'none';
        return;
    }
    
    // Fetch each graph's data from the API
    console.log(`[STARTUP] Fetching ${graphs.length} graphs from API...`);
    for (const graphId of graphs) {
        try {
            console.log(`[STARTUP] Fetching graph: ${graphId}`);
            const graphData = await API.getGraph(graphId);
            State.graphs[graphId] = graphData;
            console.log(`[STARTUP] Loaded ${graphId}: ${graphData.nodes?.length || 0} nodes, ${graphData.edges?.length || 0} edges`);
        } catch (err) {
            console.error(`[STARTUP] Error loading ${graphId}:`, err);
        }
    }
    
    // Setup graph selector if multiple graphs
    if (graphs.length > 1) {
        const selector = document.getElementById('graph-selector');
        const select = document.getElementById('graph-select');
        
        if (select) {
            select.innerHTML = graphs.map(id => 
                `<option value="${id}">${id}</option>`
            ).join('');
        }
        
        if (selector) {
            selector.style.display = 'block';
        }
    }
    
    // Display first graph
    const firstGraph = graphs[0];
    console.log(`[STARTUP] Displaying first graph: ${firstGraph}`);
    GraphLoader.displayGraph(firstGraph);
    
    // Enable metrics button
    const metricsBtn = document.getElementById('compute-metrics-btn');
    if (metricsBtn) {
        metricsBtn.disabled = false;
    }
    
    DOMCache.loading.style.display = 'none';
    updateStatus(`Loaded ${graphs.length} graph(s): ${graphs.join(', ')}`, 'success');
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

    // Toolbar buttons - work with both Cytoscape and cosmos.gl
    document.getElementById('fit-btn')?.addEventListener('click', () => {
        if (State.rendererType === 'cosmos' && State.renderer) {
            State.renderer.fitView();
        } else if (State.cy) {
            State.cy.fit(50);
        }
    });
    document.getElementById('center-btn')?.addEventListener('click', () => {
        if (State.rendererType === 'cosmos' && State.renderer) {
            State.renderer.fitView();
        } else if (State.cy) {
            State.cy.center();
        }
    });
    
    // Renderer switch button - opens modal to select renderer
    document.getElementById('renderer-switch-btn')?.addEventListener('click', () => {
        showRendererModal();
    });
    
    // Simulation control button (for cosmos.gl)
    document.getElementById('toolbar-sim-btn')?.addEventListener('click', () => {
        if (State.rendererType === 'cosmos' && State.renderer) {
            const btn = document.getElementById('toolbar-sim-btn');
            const isCurrentlyRunning = btn?.classList.contains('running');
            
            if (isCurrentlyRunning) {
                State.renderer.pauseSimulation();
                updateSimulationButton(false); // paused
            } else {
                State.renderer.startSimulation();
                updateSimulationButton(true); // running
            }
        }
    });
    
    // Note: Cosmos simulation controls are set up in setupEnhancedSimulationControls()
    
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
        const renderer = State.renderer;
        if (!renderer) {
            updateStatus('Load a graph first', 'error');
            return;
        }
        
        // Get style settings
        const sizeMetric = document.getElementById('node-size-metric')?.value;
        const colorMetric = document.getElementById('node-color-metric')?.value;
        const gradientName = document.getElementById('node-color-gradient')?.value || 'spectral';
        const sizeMin = parseFloat(document.getElementById('node-size-min')?.value) || 8;
        const sizeMax = parseFloat(document.getElementById('node-size-max')?.value) || 25;
        const edgeOpacity = (parseFloat(document.getElementById('edge-opacity')?.value) || 20) / 100;
        const edgeColor = document.getElementById('edge-color')?.value || '#fcfafa';
        const edgeWidthMin = parseFloat(document.getElementById('edge-width-min')?.value) || 1;
        
        // Apply styles based on renderer type
        if (State.rendererType === 'cytoscape') {
            CytoscapeManager.applyFullStyle();
        } else if (State.rendererType === 'cosmos') {
            // Apply node colors to cosmos renderer
            if (colorMetric) {
                // Pass gradient name, not gradient object - cosmos-adapter uses ColorGradients.get()
                renderer.applyNodeColors(colorMetric, { gradient: gradientName });
            }
            // Apply node sizes
            if (sizeMetric) {
                renderer.applyNodeSizes(sizeMetric, { min: sizeMin, max: sizeMax });
            }
            // Apply edge styles
            renderer.setEdgeStyle({
                color: edgeColor,
                opacity: edgeOpacity,
                width: edgeWidthMin
            });
            showToast('Style applied', 'success');
        } else {
            updateStatus('Unknown renderer type', 'error');
        }
    });
    
    // Clear Style button - resets to default styling
    document.getElementById('clear-style-btn')?.addEventListener('click', () => {
        const renderer = State.renderer;
        if (!renderer) {
            updateStatus('Load a graph first', 'error');
            return;
        }
        
        // Reset form fields to defaults
        const sizeMetric = document.getElementById('node-size-metric');
        const colorMetric = document.getElementById('node-color-metric');
        const sizeMin = document.getElementById('node-size-min');
        const sizeMax = document.getElementById('node-size-max');
        const edgeOpacity = document.getElementById('edge-opacity');
        const edgeOpacityValue = document.getElementById('edge-opacity-value');
        const edgeColor = document.getElementById('edge-color');
        const edgeWidthMin = document.getElementById('edge-width-min');
        const edgeWidthMax = document.getElementById('edge-width-max');
        
        if (sizeMetric) sizeMetric.value = '';
        if (colorMetric) colorMetric.value = '';
        if (sizeMin) sizeMin.value = '8';
        if (sizeMax) sizeMax.value = '25';
        if (edgeOpacity) edgeOpacity.value = '20';
        if (edgeOpacityValue) edgeOpacityValue.textContent = '20%';
        if (edgeColor) edgeColor.value = '#fcfafa';
        if (edgeWidthMin) edgeWidthMin.value = '2';
        if (edgeWidthMax) edgeWidthMax.value = '5';
        
        // Reset renderer style
        if (State.rendererType === 'cytoscape') {
            CytoscapeManager.resetStyle();
        } else if (State.rendererType === 'cosmos') {
            renderer.resetStyle();
            // Also reset edge style to defaults
            renderer.setEdgeStyle({
                color: '#fcfafa',
                opacity: 0.2,
                width: 1
            });
        }
        
        showToast('Style reset to defaults', 'success');
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
    
    // Listen for graph loaded events to update UI
    document.addEventListener('graphLoaded', () => {
        updateRendererSwitchButton();
        updateSimulationControls();
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

// =============================================================================
// RENDERER SWITCH MODAL
// =============================================================================

/**
 * Show the renderer selection modal
 */
function showRendererModal() {
    // Remove any existing modal
    const existing = document.querySelector('.modal-overlay');
    if (existing) existing.remove();
    
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>Select Renderer</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div class="renderer-options">
                    <label class="renderer-option">
                        <input type="radio" name="renderer-choice" value="cosmos" ${State.rendererPreference === 'cosmos' ? 'checked' : ''}>
                        <div class="renderer-option-content">
                            <div class="renderer-option-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="12" cy="12" r="3"/>
                                    <path d="M12 2v4m0 12v4M2 12h4m12 0h4"/>
                                    <path d="m4.93 4.93 2.83 2.83m8.48 8.48 2.83 2.83m0-14.14-2.83 2.83m-8.48 8.48-2.83 2.83"/>
                                </svg>
                            </div>
                            <div class="renderer-option-text">
                                <strong>cosmos.gl (GPU)</strong>
                                <span>WebGL-accelerated, best for large graphs (10K+ nodes)</span>
                            </div>
                        </div>
                    </label>
                    <label class="renderer-option">
                        <input type="radio" name="renderer-choice" value="cytoscape" ${State.rendererPreference === 'cytoscape' ? 'checked' : ''}>
                        <div class="renderer-option-content">
                            <div class="renderer-option-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="5" cy="6" r="2"/>
                                    <circle cx="12" cy="18" r="2"/>
                                    <circle cx="19" cy="6" r="2"/>
                                    <path d="M5 8v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8"/>
                                    <path d="M12 14v2"/>
                                </svg>
                            </div>
                            <div class="renderer-option-text">
                                <strong>Cytoscape.js (Canvas)</strong>
                                <span>Full-featured, rich interactivity, best for smaller graphs</span>
                            </div>
                        </div>
                    </label>
                    <label class="renderer-option">
                        <input type="radio" name="renderer-choice" value="auto" ${State.rendererPreference === 'auto' ? 'checked' : ''}>
                        <div class="renderer-option-content">
                            <div class="renderer-option-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M12 2v4m0 12v4M2 12h4m12 0h4"/>
                                    <circle cx="12" cy="12" r="5"/>
                                </svg>
                            </div>
                            <div class="renderer-option-text">
                                <strong>Auto-select</strong>
                                <span>Automatically choose based on graph size</span>
                            </div>
                        </div>
                    </label>
                </div>
                <p style="margin-top: 16px; font-size: 11px; color: #888;">
                    The current graph will keep its renderer until you load a new graph.
                </p>
                <button id="reload-with-new-renderer" class="secondary-btn" style="width: 100%; margin-top: 8px;">
                    Reload Current Graph with New Renderer
                </button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close handlers
    modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
    
    // Reload with new renderer button
    modal.querySelector('#reload-with-new-renderer')?.addEventListener('click', () => {
        if (State.currentGraph) {
            modal.remove();
            // Reload the current graph with the new renderer preference
            GraphLoader.displayGraph(State.currentGraph);
        } else {
            updateStatus('No graph loaded', 'warning');
        }
    });
    
    // Radio change handler
    modal.querySelectorAll('input[name="renderer-choice"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            State.rendererPreference = e.target.value;
            localStorage.setItem('rendererPreference', e.target.value);
            updateRendererSwitchButton();
            updateStatus(`Renderer preference set to: ${e.target.value}`, 'success');
        });
    });
}

/**
 * Update the renderer switch button appearance
 */
function updateRendererSwitchButton() {
    const btn = document.getElementById('renderer-switch-btn');
    if (!btn) return;
    
    const currentRenderer = State.rendererType || 'cytoscape';
    const label = btn.querySelector('.renderer-label') || btn;
    
    if (currentRenderer === 'cosmos') {
        btn.classList.remove('cytoscape-active');
        btn.classList.add('cosmos-active');
        if (label.tagName !== 'BUTTON') label.textContent = 'cosmos.gl';
    } else {
        btn.classList.remove('cosmos-active');
        btn.classList.add('cytoscape-active');
        if (label.tagName !== 'BUTTON') label.textContent = 'Cytoscape';
    }
}

/**
 * Update simulation button state
 */
function updateSimulationButton(isRunning) {
    const btn = document.getElementById('toolbar-sim-btn');
    const icon = document.getElementById('toolbar-sim-icon');
    if (!btn) return;
    
    if (isRunning) {
        btn.classList.remove('paused');
        btn.classList.add('running');
        btn.title = 'Pause simulation';
        if (icon) icon.setAttribute('data-icon', 'pause');
    } else {
        btn.classList.remove('running');
        btn.classList.add('paused');
        btn.title = 'Resume simulation';
        if (icon) icon.setAttribute('data-icon', 'play');
    }
    
    // Re-inject icons if needed
    Icons.inject();
}

/**
 * Show/hide simulation controls based on renderer type
 */
function updateSimulationControls() {
    const simBtn = document.getElementById('toolbar-sim-btn');
    const simControls = document.getElementById('cosmos-simulation-controls');
    
    if (State.rendererType === 'cosmos') {
        if (simBtn) simBtn.style.display = 'flex';
        if (simControls) simControls.style.display = 'block';
        // cosmos.gl pauses after layout stabilizes, so start button as paused
        updateSimulationButton(false);
    } else {
        if (simBtn) simBtn.style.display = 'none';
        if (simControls) simControls.style.display = 'none';
    }
}

/**
 * Apply cosmos simulation parameters from sliders
 */
function applyCosmosSimulationParams() {
    if (State.rendererType !== 'cosmos' || !State.renderer) {
        Toast.show('Cosmos renderer not active', 'error');
        return;
    }
    
    // Read values from sliders (using cosmos.gl defaults if not found)
    const repulsion = parseFloat(document.getElementById('cosmos-repulsion')?.value) || 1.0;
    const gravity = parseFloat(document.getElementById('cosmos-gravity')?.value) || 0.25;
    const linkDistance = parseFloat(document.getElementById('cosmos-link-distance')?.value) || 10;
    const linkSpring = parseFloat(document.getElementById('cosmos-link-spring')?.value) || 1.0;
    const friction = parseFloat(document.getElementById('cosmos-friction')?.value) || 0.85;
    const decay = parseFloat(document.getElementById('cosmos-decay')?.value) || 5000;
    
    const params = {
        repulsion,
        gravity,
        linkDistance,
        linkSpring,
        friction,
        decay
    };
    
    console.log('[App] Applying cosmos simulation params:', params);
    
    // Apply parameters to cosmos renderer
    if (typeof State.renderer.setSimulationParams === 'function') {
        const success = State.renderer.setSimulationParams(params);
        
        if (success) {
            // Restart simulation so changes take effect
            State.renderer.startSimulation(0.5);
            Toast.show('Parameters applied', 'success');
        } else {
            Toast.show('Failed to apply parameters', 'error');
        }
    } else {
        Toast.show('setSimulationParams not available', 'warning');
    }
}

/**
 * Reset simulation parameters to cosmos.gl defaults
 */
function resetCosmosSimulationParams() {
    // cosmos.gl default values from documentation
    const defaults = {
        repulsion: 1.0,
        gravity: 0.25,
        linkDistance: 10,
        linkSpring: 1.0,
        friction: 0.85,
        decay: 5000
    };
    
    // Update sliders and displays
    const setValue = (id, value) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(id + '-value');
        if (slider) slider.value = value;
        if (display) display.textContent = value;
    };
    
    setValue('cosmos-repulsion', defaults.repulsion);
    setValue('cosmos-gravity', defaults.gravity);
    setValue('cosmos-link-distance', defaults.linkDistance);
    setValue('cosmos-link-spring', defaults.linkSpring);
    setValue('cosmos-friction', defaults.friction);
    setValue('cosmos-decay', defaults.decay);
    
    // Apply the defaults
    if (State.rendererType === 'cosmos' && State.renderer) {
        State.renderer.setSimulationParams(defaults);
        State.renderer.startSimulation(0.5);
        Toast.show('Reset to defaults', 'success');
    }
}