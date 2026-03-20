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
    AnalysisPanel.init();    // IDE-style bottom panel (Snapshots, Metrics, Embeddings, Query)
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
    console.log('[App] setupRendererPreferenceControls called');
    
    // Listen for renderer changes to show/hide cosmos simulation controls - ALWAYS register this
    document.addEventListener('rendererChanged', (e) => {
        console.log('[App] rendererChanged event:', e.detail.type);
        const isCosmos = e.detail.type === 'cosmos';
        const controls = document.getElementById('cosmos-simulation-controls');
        if (controls) {
            controls.style.display = isCosmos ? 'block' : 'none';
        }
    });
    
    // Setup cosmos simulation parameter controls (sliders, presets, snapshots)
    setupEnhancedSimulationControls();
    
    const radios = document.querySelectorAll('input[name="renderer-preference"]');
    if (radios.length === 0) {
        console.log('[App] No renderer preference radios found, skipping radio setup');
        return;
    }
    
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
}

/**
 * Setup cosmos simulation parameter controls
 * Sliders auto-apply on change, presets apply immediately
 */
function setupEnhancedSimulationControls() {
    console.log('[App] setupEnhancedSimulationControls called');
    
    // Slider configuration - maps slider ID to cosmos parameter name
    const sliderConfig = [
        { id: 'cosmos-repulsion', param: 'repulsion' },
        { id: 'cosmos-gravity', param: 'gravity' },
        { id: 'cosmos-center', param: 'center' },
        { id: 'cosmos-cluster', param: 'cluster' },
        { id: 'cosmos-link-distance', param: 'linkDistance' },
        { id: 'cosmos-link-spring', param: 'linkSpring' },
        { id: 'cosmos-friction', param: 'friction' },
        { id: 'cosmos-decay', param: 'decay' }
    ];
    
    // Setup sliders - update display AND auto-apply parameter on change
    sliderConfig.forEach(({ id, param }) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(id + '-value');
        
        if (slider) {
            slider.addEventListener('input', () => {
                // Update display value
                if (display) {
                    display.textContent = slider.value;
                }
                
                // Auto-apply this single parameter to renderer
                const renderer = State.renderer;
                if (renderer && typeof renderer.setSimulationParams === 'function') {
                    const value = parseFloat(slider.value);
                    const params = { [param]: value };
                    renderer.setSimulationParams(params, { restart: true, alpha: 0.3 });
                }
            });
        }
    });
    
    // Preset selector - apply preset on change
    const presetSelect = document.getElementById('cosmos-preset-select');
    console.log('[App] Preset select element:', presetSelect);
    if (presetSelect) {
        presetSelect.addEventListener('change', (e) => {
            const presetId = e.target.value;
            console.log('[App] Preset selected:', presetId);
            if (presetId) {
                applyPreset(presetId);
            }
        });
        console.log('[App] Preset change listener attached');
    }
    
    // Reset button - reset sliders and apply defaults
    const resetBtn = document.getElementById('cosmos-reset-params');
    console.log('[App] Reset button element:', resetBtn);
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            console.log('[App] Reset params clicked');
            resetCosmosSimulationParams();
        });
        console.log('[App] Reset button listener attached');
    }
    
    // Layout snapshot controls
    const snapshotBtn = document.getElementById('cosmos-snapshot-btn');
    console.log('[App] Snapshot button element:', snapshotBtn);
    if (snapshotBtn) {
        snapshotBtn.addEventListener('click', () => {
            console.log('[App] Snapshot save clicked');
            const renderer = State.renderer;
            if (renderer && typeof renderer.createSnapshot === 'function') {
                const name = `snapshot_${Date.now()}`;
                if (renderer.createSnapshot(name)) {
                    State._lastSnapshotName = name;
                    Toast.show('Layout saved', 'success');
                } else {
                    Toast.show('Failed to save layout', 'error');
                }
            } else {
                Toast.show('Renderer not available', 'error');
            }
        });
        console.log('[App] Snapshot button listener attached');
    }

    const restoreBtn = document.getElementById('cosmos-restore-btn');
    if (restoreBtn) {
        restoreBtn.addEventListener('click', () => {
            console.log('[App] Snapshot restore clicked');
            const renderer = State.renderer;
            if (renderer && typeof renderer.restoreSnapshot === 'function') {
                const name = State._lastSnapshotName || 'default';
                if (renderer.restoreSnapshot(name)) {
                    Toast.show('Layout restored', 'success');
                } else {
                    Toast.show('No saved layout to restore', 'warning');
                }
            } else {
                Toast.show('Renderer not available', 'error');
            }
        });
    }
    
    // Position export
    const exportBtn = document.getElementById('cosmos-export-pos-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            console.log('[App] Export positions clicked');
            const renderer = State.renderer;
            if (renderer && typeof renderer.exportPositions === 'function') {
                const positions = renderer.exportPositions();
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
            } else {
                Toast.show('Renderer not available', 'error');
            }
        });
    }
    
    // Save to Server button (saves to parquet cache)
    const saveServerBtn = document.getElementById('cosmos-save-server-btn');
    if (saveServerBtn) {
        saveServerBtn.addEventListener('click', async () => {
            console.log('[App] Save to server clicked');
            const renderer = State.renderer;
            const graphId = State.currentGraph;
            
            if (!graphId) {
                Toast.show('No graph loaded', 'error');
                return;
            }
            
            if (!renderer || typeof renderer.exportPositions !== 'function') {
                Toast.show('Renderer not available', 'error');
                return;
            }
            
            const positions = renderer.exportPositions();
            if (!positions || Object.keys(positions).length === 0) {
                Toast.show('No positions to save', 'error');
                return;
            }
            
            const saveAsBase = document.getElementById('cosmos-save-as-base')?.checked || false;
            
            try {
                saveServerBtn.disabled = true;
                saveServerBtn.innerHTML = '<span data-icon="loading"></span> Saving...';
                Icons.inject();
                
                const result = await API.saveLayoutToServer(graphId, positions, {
                    name: 'cosmos',
                    saveAsBase: saveAsBase
                });
                
                console.log('[App] Layout saved to server:', result);
                
                if (saveAsBase) {
                    Toast.show(`Layout saved as default (${result.node_count} nodes)`, 'success');
                } else {
                    Toast.show(`Layout saved to server (${result.node_count} nodes)`, 'success');
                }
                
                // Refresh the saved layouts list
                if (typeof Layout !== 'undefined' && Layout.loadSavedLayouts) {
                    Layout.loadSavedLayouts();
                }
                
            } catch (error) {
                console.error('[App] Failed to save layout:', error);
                Toast.show(`Failed to save: ${error.message}`, 'error');
            } finally {
                saveServerBtn.disabled = false;
                saveServerBtn.innerHTML = '<span data-icon="save"></span> Save to Server';
                Icons.inject();
            }
        });
    }
    
    // Position import
    const importBtn = document.getElementById('cosmos-import-pos-btn');
    if (importBtn) {
        importBtn.addEventListener('click', () => {
            document.getElementById('cosmos-import-file')?.click();
        });
    }
    
    const importFile = document.getElementById('cosmos-import-file');
    if (importFile) {
        importFile.addEventListener('change', async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            
            try {
                const text = await file.text();
                const positions = JSON.parse(text);
                
                const renderer = State.renderer;
                if (renderer && typeof renderer.importPositions === 'function') {
                    renderer.importPositions(positions, true);
                    Toast.show(`Imported layout for ${Object.keys(positions).length} nodes`, 'success');
                }
            } catch (err) {
                console.error('Import error:', err);
                Toast.show('Failed to import: Invalid file', 'error');
            }
            
            e.target.value = '';
        });
    }
    
    // Setup simulation progress monitoring
    setupSimulationProgressMonitor();
    
    console.log('[App] setupEnhancedSimulationControls complete');
}

/**
 * Apply preset - NO separate function, directly defined here
 */
function applyPreset(presetId) {
    const presets = {
        default: { repulsion: 1.0, gravity: 0.25, center: 0, cluster: 0.1, linkDistance: 10, linkSpring: 1.0, friction: 0.85, decay: 5000 },
        dense: { repulsion: 1.5, gravity: 0.1, center: 0, cluster: 0.2, linkDistance: 5, linkSpring: 1.2, friction: 0.9, decay: 3000 },
        sparse: { repulsion: 0.5, gravity: 0.15, center: 0.1, cluster: 0.05, linkDistance: 30, linkSpring: 0.5, friction: 0.8, decay: 8000 },
        clustered: { repulsion: 1.0, gravity: 0.15, center: 0.2, cluster: 0.8, linkDistance: 15, linkSpring: 0.8, friction: 0.85, decay: 5000 },
        hierarchical: { repulsion: 0.8, gravity: 0.2, center: 0.1, cluster: 0.1, linkDistance: 50, linkSpring: 1.5, friction: 0.9, decay: 6000 },
        fast: { repulsion: 1.0, gravity: 0.3, center: 0, cluster: 0.1, linkDistance: 10, linkSpring: 1.0, friction: 0.6, decay: 2000 },
        quality: { repulsion: 1.0, gravity: 0.2, center: 0, cluster: 0.15, linkDistance: 15, linkSpring: 0.8, friction: 0.95, decay: 10000 }
    };
    
    const preset = presets[presetId];
    if (!preset) {
        console.error('[App] Unknown preset:', presetId);
        Toast.show('Unknown preset', 'error');
        return;
    }
    
    console.log('[App] Applying preset:', presetId, preset);
    
    // Update slider values in UI
    const sliderMap = {
        repulsion: 'cosmos-repulsion',
        gravity: 'cosmos-gravity',
        center: 'cosmos-center',
        cluster: 'cosmos-cluster',
        linkDistance: 'cosmos-link-distance',
        linkSpring: 'cosmos-link-spring',
        friction: 'cosmos-friction',
        decay: 'cosmos-decay'
    };
    
    for (const [param, sliderId] of Object.entries(sliderMap)) {
        const value = preset[param];
        if (value !== undefined) {
            const slider = document.getElementById(sliderId);
            const display = document.getElementById(sliderId + '-value');
            if (slider) slider.value = value;
            if (display) display.textContent = value;
        }
    }
    
    // Apply to renderer
    const renderer = State.renderer;
    if (renderer && typeof renderer.setSimulationParams === 'function') {
        renderer.setSimulationParams(preset, { restart: true, alpha: 0.5 });
        Toast.show(`Applied: ${presetId}`, 'success');
    } else {
        console.error('[App] Cannot apply preset - renderer not available');
        Toast.show('Renderer not ready', 'error');
    }
}

/**
 * Setup simulation progress monitoring UI
 */
function setupSimulationProgressMonitor() {
    // Register callbacks when renderer is available
    document.addEventListener('rendererChanged', (e) => {
        if (e.detail.type === 'cosmos' && State.renderer) {
            // Register simulation callbacks
            State.renderer.onSimulationTick((data) => {
                const alphaText = document.getElementById('cosmos-alpha-text');
                if (alphaText) {
                    alphaText.textContent = data.alpha?.toFixed(3) || '0.000';
                }
            });
            
            State.renderer.onSimulationStart(() => {
                const statusText = document.getElementById('cosmos-status-text');
                if (statusText) statusText.textContent = 'Running';
                State.cosmosSimulationPaused = false;
            });
            
            State.renderer.onSimulationEnd(() => {
                const statusText = document.getElementById('cosmos-status-text');
                if (statusText) statusText.textContent = 'Stopped';
                State.cosmosSimulationPaused = true;
            });
            
            State.renderer.onSimulationPause(() => {
                const statusText = document.getElementById('cosmos-status-text');
                if (statusText) statusText.textContent = 'Paused';
                State.cosmosSimulationPaused = true;
            });
        }
    });
}

/**
 * Sync slider values with current renderer parameters
 */
function syncSlidersWithRenderer() {
    if (State.rendererType !== 'cosmos' || !State.renderer) return;
    
    const params = State.renderer.getSimulationParams?.() || {};
    
    const simMapping = {
        repulsion: 'cosmos-repulsion',
        gravity: 'cosmos-gravity',
        center: 'cosmos-center',
        cluster: 'cosmos-cluster',
        linkDistance: 'cosmos-link-distance',
        linkSpring: 'cosmos-link-spring',
        friction: 'cosmos-friction',
        decay: 'cosmos-decay'
    };
    
    // Sync simulation params
    Object.entries(simMapping).forEach(([param, sliderId]) => {
        if (params[param] !== undefined) {
            const slider = document.getElementById(sliderId);
            const display = document.getElementById(sliderId + '-value');
            if (slider) slider.value = params[param];
            if (display) display.textContent = params[param];
        }
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
 * Display graph after background load completes.
 * Called when PRODUCTION_MODE auto-load finishes.
 * 
 * Note: This function does NOT fetch graph data via API.getGraph() because
 * GraphLoader.displayGraph() handles all data fetching internally via
 * API.getGraphElements(). This is the correct flow for production mode.
 */
async function displayLoadedGraph(status) {
    console.log('[STARTUP] displayLoadedGraph called with status:', status);
    
    // The backend returns 'loaded_graphs' - check both for compatibility
    const graphs = status.loaded_graphs || status.graphs || [];
    
    if (!graphs || graphs.length === 0) {
        console.error('[STARTUP] No graphs in status:', status);
        updateStatus('No graphs loaded - check server logs', 'error');
        DOMCache.loading.style.display = 'none';
        return;
    }
    
    // Update State with startup info
    State.currentState = {
        loaded_graphs: graphs,
        node_count: status.node_count || 0,
        edge_count: status.edge_count || 0
    };
    
    // Initialize graph data cache entries
    graphs.forEach(g => {
        State.graphData[g] = { loaded: false };
    });
    
    // Setup graph selector
    const graphSelector = document.getElementById('graph-selector');
    const graphSelect = document.getElementById('graph-select');
    
    if (graphSelect) {
        graphSelect.innerHTML = '<option value="">Select graph...</option>' +
            graphs.map(id => `<option value="${id}">${id}</option>`).join('');
    }
    
    // Show selector if we have graphs
    if (graphSelector && graphs.length > 0) {
        graphSelector.style.display = 'block';
    }
    
    // Auto-select and display first graph
    if (graphs.length > 0) {
        const firstGraph = graphs[0];
        console.log(`[STARTUP] Displaying first graph: ${firstGraph}`);
        
        if (graphSelect) {
            graphSelect.value = firstGraph;
        }
        
        // GraphLoader.displayGraph() handles all data fetching internally
        // via API.getGraphElements() - no need to call API.getGraph()
        try {
            await GraphLoader.displayGraph(firstGraph);
        } catch (err) {
            console.error(`[STARTUP] Error displaying ${firstGraph}:`, err);
            updateStatus(`Error displaying graph: ${err.message}`, 'error');
        }
    }
    
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
                State.cosmosSimulationPaused = true;
                updateSimulationButton(false); // paused
            } else {
                State.renderer.startSimulation();
                State.cosmosSimulationPaused = false;
                updateSimulationButton(true); // running
            }
            
            // Update renderer indicator to reflect layout mode change
            if (typeof GraphLoader !== 'undefined' && GraphLoader.updateRendererIndicator) {
                GraphLoader.updateRendererIndicator();
            }
        }
    });
    
    // Note: Cosmos simulation controls are set up in setupEnhancedSimulationControls()
    
    document.getElementById('search-btn')?.addEventListener('click', () => Search.search());
    document.getElementById('node-search')?.addEventListener('keypress', (e) => { 
        if (e.key === 'Enter') Search.search(); 
    });
    document.getElementById('clear-search-btn')?.addEventListener('click', () => Search.clear());

    // Edge toggle button - load/clear edges
    document.getElementById('edges-toggle-btn')?.addEventListener('click', async () => {
        const btn = document.getElementById('edges-toggle-btn');
        const isLoaded = btn?.classList.contains('edges-loaded');

        if (isLoaded) {
            // Clear edges
            if (typeof Snapshots !== 'undefined' && Snapshots.clearEdges) {
                Snapshots.clearEdges();
            }
            btn?.classList.remove('edges-loaded');
            if (btn) btn.title = 'Load edges';
        } else {
            // Load edges
            if (typeof Snapshots !== 'undefined' && Snapshots.loadEdges) {
                const handled = await Snapshots.loadEdges();
                if (handled) {
                    btn?.classList.add('edges-loaded');
                    if (btn) btn.title = 'Clear edges';
                    return;
                }
            }
            // Otherwise use normal graph loader
            if (State.currentGraph) {
                GraphLoader.loadEdgesIncrementally(State.currentGraph);
                btn?.classList.add('edges-loaded');
                if (btn) btn.title = 'Clear edges';
            }
        }
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

    // Style controls (debounced) - ONLY for Cytoscape, and ONLY for size/color metric changes
    // Gradient changes should NOT auto-apply - user must click Apply Style
    const styleHandler = Utils.debounce(() => {
        if (!State.performanceMode && State.rendererType === 'cytoscape') {
            CytoscapeManager.updateStyle();
        }
    }, 150);
    
    document.getElementById('node-size-metric')?.addEventListener('change', styleHandler);
    document.getElementById('node-color-metric')?.addEventListener('change', styleHandler);
    // REMOVED: gradient auto-apply - user must click Apply Style button
    // document.getElementById('node-color-gradient')?.addEventListener('change', styleHandler);
    
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
    
    // Background color change listener
    document.getElementById('background-color')?.addEventListener('change', (e) => {
        const color = e.target.value;
        const renderer = State.renderer;
        const container = document.getElementById('cy');
        
        // Always update the container background for both renderers
        if (container) {
            container.style.backgroundColor = color;
        }
        
        // Update cosmos.gl specific elements
        if (State.rendererType === 'cosmos' && renderer) {
            if (typeof renderer.setBackgroundColor === 'function') {
                renderer.setBackgroundColor(color);
            }
            // Also try to update canvas directly
            const canvas = container?.querySelector('canvas');
            if (canvas) {
                canvas.style.backgroundColor = color;
            }
        }
        
        // Also update main content area background
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.style.backgroundColor = color;
        }
        
        showToast('Background color updated', 'success');
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
        const backgroundColor = document.getElementById('background-color');
        
        if (sizeMetric) sizeMetric.value = '';
        if (colorMetric) colorMetric.value = '';
        if (sizeMin) sizeMin.value = '8';
        if (sizeMax) sizeMax.value = '25';
        if (edgeOpacity) edgeOpacity.value = '50';
        if (edgeOpacityValue) edgeOpacityValue.textContent = '50%';
        if (edgeColor) edgeColor.value = '#ffffff';
        if (edgeWidthMin) edgeWidthMin.value = '2';
        if (edgeWidthMax) edgeWidthMax.value = '5';
        if (backgroundColor) backgroundColor.value = '#1a1a1a';
        
        // Reset background color
        const container = document.getElementById('cy');
        const mainContent = document.querySelector('.main-content');
        if (container) container.style.backgroundColor = '#1a1a1a';
        if (mainContent) mainContent.style.backgroundColor = '#0a0a0a';
        
        // Reset renderer style
        if (State.rendererType === 'cytoscape') {
            CytoscapeManager.resetStyle();
        } else if (State.rendererType === 'cosmos') {
            renderer.resetStyle();
            // Also reset edge style to defaults (white)
            renderer.setEdgeStyle({
                color: '#ffffff',
                opacity: 0.5,
                width: 1
            });
            // Reset background
            if (typeof renderer.setBackgroundColor === 'function') {
                renderer.setBackgroundColor('#1a1a1a');
            }
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
        
        // Check config for default simulation state
        const simulationOnLoad = RendererSettings.getValue('cosmos.simulationOnLoad', false);
        
        if (simulationOnLoad) {
            // Simulation runs automatically
            updateSimulationButton(true);
            State.cosmosSimulationPaused = false;
        } else {
            // Simulation paused by default (static mode)
            updateSimulationButton(false);
            State.cosmosSimulationPaused = true;
        }
        
        // Update renderer indicator to show layout mode
        if (typeof GraphLoader !== 'undefined' && GraphLoader.updateRendererIndicator) {
            GraphLoader.updateRendererIndicator();
        }
    } else {
        if (simBtn) simBtn.style.display = 'none';
        if (simControls) simControls.style.display = 'none';
    }
}

/**
 * Reset simulation parameters to cosmos.gl defaults
 */
function resetCosmosSimulationParams() {
    console.log('[App] resetCosmosSimulationParams called');
    
    // cosmos.gl default values
    const defaults = {
        repulsion: 1.0,
        gravity: 0.25,
        center: 0,
        cluster: 0.1,
        linkDistance: 10,
        linkSpring: 1.0,
        friction: 0.85,
        decay: 5000
    };
    
    // Update sliders and displays
    const setSliderValue = (id, value) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(id + '-value');
        if (slider) slider.value = value;
        if (display) display.textContent = value;
    };
    
    setSliderValue('cosmos-repulsion', defaults.repulsion);
    setSliderValue('cosmos-gravity', defaults.gravity);
    setSliderValue('cosmos-center', defaults.center);
    setSliderValue('cosmos-cluster', defaults.cluster);
    setSliderValue('cosmos-link-distance', defaults.linkDistance);
    setSliderValue('cosmos-link-spring', defaults.linkSpring);
    setSliderValue('cosmos-friction', defaults.friction);
    setSliderValue('cosmos-decay', defaults.decay);
    
    // Reset preset dropdown
    const presetSelect = document.getElementById('cosmos-preset-select');
    if (presetSelect) presetSelect.value = '';
    
    // Apply the defaults
    const renderer = State.renderer;
    if (renderer && typeof renderer.setSimulationParams === 'function') {
        renderer.setSimulationParams(defaults, { restart: true, alpha: 0.5 });
        Toast.show('Reset to defaults', 'success');
    } else {
        console.error('[App] Cannot reset - no renderer');
        Toast.show('Renderer not available', 'error');
    }
}