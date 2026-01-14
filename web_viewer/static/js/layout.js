/**
 * Layout Module
 * 
 * Handles layout backend selection, recomputation, saved layouts management,
 * warm start (using existing layout as starting point), and setting default layouts.
 */

const Layout = (function() {
    'use strict';
    
    // State
    let availableBackends = [];
    let savedLayouts = [];
    let currentBackend = 'auto';
    let currentAlgorithm = 'auto';
    
    /**
     * Initialize the layout module
     */
    async function init() {
        console.log('[Layout] Initializing...');
        
        // Load available backends
        await loadBackends();
        
        // Setup event listeners
        setupEventListeners();
        
        // Populate backend info display
        populateBackendInfo();
        
        // Load saved layouts for current graph (if any)
        await loadSavedLayouts();
        
        // Also listen for panel open to refresh layouts
        const layoutPanel = document.getElementById('panel-layout');
        if (layoutPanel) {
            // Use MutationObserver to detect when panel becomes visible
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.attributeName === 'class' || mutation.attributeName === 'style') {
                        const isVisible = layoutPanel.classList.contains('active') || 
                                         layoutPanel.style.display !== 'none';
                        if (isVisible) {
                            console.log('[Layout] Panel opened, refreshing layouts...');
                            loadSavedLayouts();
                        }
                    }
                });
            });
            observer.observe(layoutPanel, { attributes: true });
        }
        
        // Listen for nav button click to refresh
        const layoutNavBtn = document.querySelector('[data-panel="layout"]');
        if (layoutNavBtn) {
            layoutNavBtn.addEventListener('click', () => {
                console.log('[Layout] Nav button clicked, refreshing...');
                setTimeout(loadSavedLayouts, 100);
            });
        }
        
        console.log('[Layout] Initialized');
    }
    
    /**
     * Load available layout backends from API
     */
    async function loadBackends() {
        try {
            const response = await fetch('/api/layout/backends');
            if (!response.ok) {
                console.warn('[Layout] Failed to load backends:', response.status);
                return;
            }
            const data = await response.json();
            
            availableBackends = data.backends || [];
            const priority = data.priority || [];
            
            console.log('[Layout] Available backends:', availableBackends.map(b => b.id));
            
            // Populate backend selector
            populateBackendSelector();
            
        } catch (error) {
            console.error('[Layout] Failed to load backends:', error);
        }
    }
    
    /**
     * Load saved layouts for the current graph
     */
    async function loadSavedLayouts() {
        const graphId = getCurrentGraphId();
        console.log('[Layout] loadSavedLayouts called, graphId:', graphId);
        
        if (!graphId) {
            console.log('[Layout] No graph ID found, clearing layouts');
            savedLayouts = [];
            populateSavedLayoutsSelector();
            return;
        }
        
        try {
            const url = `/api/layout/saved/${graphId}`;
            console.log('[Layout] Fetching:', url);
            
            const response = await fetch(url);
            if (!response.ok) {
                console.warn('[Layout] Failed to load saved layouts:', response.status);
                savedLayouts = [];
            } else {
                const data = await response.json();
                savedLayouts = data.layouts || [];
                console.log('[Layout] Loaded', savedLayouts.length, 'saved layouts for', graphId);
                console.log('[Layout] Layouts:', savedLayouts);
            }
        } catch (error) {
            console.error('[Layout] Error loading saved layouts:', error);
            savedLayouts = [];
        }
        
        populateSavedLayoutsSelector();
    }
    
    /**
     * Populate the backend selector dropdown
     */
    function populateBackendSelector() {
        const select = document.getElementById('layout-backend-select');
        if (!select) return;
        
        // Clear existing options except 'auto'
        select.innerHTML = '<option value="auto">Auto (Best Available)</option>';
        
        // Add available backends
        availableBackends.forEach(backend => {
            const option = document.createElement('option');
            option.value = backend.id;
            option.textContent = backend.name;
            
            if (!backend.available) {
                option.disabled = true;
                option.textContent += ' (unavailable)';
            }
            
            select.appendChild(option);
        });
    }
    
    /**
     * Populate saved layouts selector
     */
    function populateSavedLayoutsSelector() {
        const select = document.getElementById('layout-saved-select');
        if (!select) {
            console.warn('[Layout] layout-saved-select element not found');
            return;
        }
        
        console.log('[Layout] Populating selector with', savedLayouts.length, 'layouts');
        
        select.innerHTML = '<option value="">Select saved layout...</option>';
        
        savedLayouts.forEach(layout => {
            const option = document.createElement('option');
            option.value = layout.filename;
            
            let label = layout.display_name;
            if (layout.is_base) {
                label += ' â˜…';  // Star for default
            }
            label += ` (${layout.node_count} nodes)`;
            
            option.textContent = label;
            option.dataset.isBase = layout.is_base;
            
            select.appendChild(option);
        });
        
        // Enable/disable related buttons
        const loadBtn = document.getElementById('load-layout-btn');
        const setDefaultBtn = document.getElementById('set-default-layout-btn');
        const deleteBtn = document.getElementById('delete-layout-btn');
        
        const hasLayouts = savedLayouts.length > 0;
        if (loadBtn) loadBtn.disabled = !hasLayouts;
        if (setDefaultBtn) setDefaultBtn.disabled = !hasLayouts;
        if (deleteBtn) deleteBtn.disabled = !hasLayouts;
        
        console.log('[Layout] Selector populated, buttons enabled:', hasLayouts);
    }
    
    /**
     * Populate backend info display
     */
    function populateBackendInfo() {
        const infoDiv = document.getElementById('layout-backends-info');
        if (!infoDiv) return;
        
        if (availableBackends.length === 0) {
            infoDiv.innerHTML = '<div style="color: #888;">Loading...</div>';
            return;
        }
        
        let html = '';
        availableBackends.forEach(backend => {
            const statusClass = backend.available ? 'available' : 'unavailable';
            const statusIcon = backend.available ? Icons.get('check') : Icons.get('close');
            html += `
                <div class="backend-item">
                    <span class="backend-name">${backend.name}</span>
                    <span class="backend-status ${statusClass}">${statusIcon}</span>
                </div>
            `;
        });
        
        infoDiv.innerHTML = html;
    }
    
    /**
     * Setup event listeners
     */
    function setupEventListeners() {
        // Backend selector change
        const backendSelect = document.getElementById('layout-backend-select');
        if (backendSelect) {
            backendSelect.addEventListener('change', handleBackendChange);
        }
        
        // Algorithm selector change
        const algorithmSelect = document.getElementById('layout-algorithm-select');
        if (algorithmSelect) {
            algorithmSelect.addEventListener('change', (e) => {
                currentAlgorithm = e.target.value;
            });
        }
        
        // Recompute button
        const recomputeBtn = document.getElementById('recompute-layout-btn');
        if (recomputeBtn) {
            recomputeBtn.addEventListener('click', handleRecomputeLayout);
        }
        
        // Saved layouts selector
        const savedSelect = document.getElementById('layout-saved-select');
        if (savedSelect) {
            savedSelect.addEventListener('change', handleSavedLayoutChange);
        }
        
        // Load saved layout button
        const loadBtn = document.getElementById('load-layout-btn');
        if (loadBtn) {
            loadBtn.addEventListener('click', handleLoadLayout);
        }
        
        // Set as default button
        const setDefaultBtn = document.getElementById('set-default-layout-btn');
        if (setDefaultBtn) {
            setDefaultBtn.addEventListener('click', handleSetDefault);
        }
        
        // Delete layout button
        const deleteBtn = document.getElementById('delete-layout-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', handleDeleteLayout);
        }
        
        // Refresh saved layouts button
        const refreshBtn = document.getElementById('refresh-layouts-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', loadSavedLayouts);
        }
        
        // Listen for graph load events to refresh saved layouts
        document.addEventListener('graphLoaded', () => {
            setTimeout(loadSavedLayouts, 500);
        });
    }
    
    /**
     * Handle backend selection change
     */
    function handleBackendChange(e) {
        currentBackend = e.target.value;
        
        // Show/hide algorithm options based on backend
        const algorithmGroup = document.getElementById('layout-algorithm-group');
        const algorithmSelect = document.getElementById('layout-algorithm-select');
        
        if (!algorithmGroup || !algorithmSelect) return;
        
        // Get selected backend info
        const backend = availableBackends.find(b => b.id === currentBackend);
        
        if (backend && backend.algorithms && backend.algorithms.length > 1) {
            // Populate algorithm options
            algorithmSelect.innerHTML = '';
            backend.algorithms.forEach(algo => {
                const option = document.createElement('option');
                option.value = algo;
                option.textContent = algo.charAt(0).toUpperCase() + algo.slice(1);
                algorithmSelect.appendChild(option);
            });
            algorithmGroup.style.display = 'flex';
            currentAlgorithm = backend.algorithms[0];
        } else {
            algorithmGroup.style.display = 'none';
            currentAlgorithm = 'auto';
        }
    }
    
    /**
     * Handle saved layout selection change
     */
    function handleSavedLayoutChange(e) {
        const filename = e.target.value;
        const setDefaultBtn = document.getElementById('set-default-layout-btn');
        const deleteBtn = document.getElementById('delete-layout-btn');
        
        if (setDefaultBtn) {
            // Disable if already the default
            const option = e.target.options[e.target.selectedIndex];
            setDefaultBtn.disabled = !filename || option?.dataset?.isBase === 'true';
        }
        if (deleteBtn) {
            // Disable delete for base layouts
            const option = e.target.options[e.target.selectedIndex];
            deleteBtn.disabled = !filename || option?.dataset?.isBase === 'true';
        }
    }
    
    /**
     * Get current graph ID from various sources
     */
    function getCurrentGraphId() {
        // Try State.currentGraph first (main source)
        if (typeof State !== 'undefined' && State.currentGraph) {
            return State.currentGraph;
        }
        
        // Try graph selector dropdown
        const graphSelect = document.getElementById('graph-select');
        if (graphSelect && graphSelect.value) {
            return graphSelect.value;
        }
        
        // Try to get from State.graphData keys
        if (typeof State !== 'undefined' && State.graphData && Object.keys(State.graphData).length > 0) {
            return Object.keys(State.graphData)[0];
        }
        
        // Try to get from currentState
        if (typeof State !== 'undefined' && State.currentState && State.currentState.loaded_graphs && State.currentState.loaded_graphs.length > 0) {
            return State.currentState.loaded_graphs[0];
        }
        
        return null;
    }
    
    /**
     * Handle recompute layout button click
     */
    async function handleRecomputeLayout() {
        const graphId = getCurrentGraphId();
        if (!graphId) {
            showToast('No graph loaded', 'error');
            console.error('[Layout] No graph loaded');
            return;
        }
        
        console.log('[Layout] Recomputing layout for graph:', graphId);
        
        const btn = document.getElementById('recompute-layout-btn');
        const originalText = btn.textContent;
        
        try {
            btn.disabled = true;
            btn.textContent = 'Computing...';
            showToast('Recomputing layout...', 'info');
            
            // Build request
            const request = {
                from_scratch: !document.getElementById('layout-warm-start')?.checked
            };
            
            if (currentBackend !== 'auto') {
                request.backend = currentBackend;
            }
            if (currentAlgorithm !== 'auto') {
                request.algorithm = currentAlgorithm;
            }
            
            // Get optional parameters
            const iterationsInput = document.getElementById('layout-iterations');
            if (iterationsInput && iterationsInput.value) {
                request.iterations = parseInt(iterationsInput.value);
            }
            
            const saveAsBase = document.getElementById('layout-save-as-base');
            if (saveAsBase && saveAsBase.checked) {
                request.save_as_base = true;
            }
            
            console.log('[Layout] Request:', request);
            
            // Call API
            const response = await fetch(`/api/layout/recompute/${graphId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Layout computation failed');
            }
            
            const result = await response.json();
            console.log('[Layout] Result:', result);
            
            // Reload graph to apply new layout
            await reloadGraphWithNewLayout(graphId);
            
            // Refresh saved layouts list
            await loadSavedLayouts();
            
            const warmStartText = result.warm_start ? ' (warm start)' : '';
            showToast(
                `Layout: ${result.backend}/${result.algorithm}${warmStartText} (${result.computation_time.toFixed(2)}s)`,
                'success'
            );
            
        } catch (error) {
            console.error('[Layout] Recompute error:', error);
            showToast(`Layout failed: ${error.message}`, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }
    
    /**
     * Handle load saved layout button
     */
    async function handleLoadLayout() {
        const graphId = getCurrentGraphId();
        const select = document.getElementById('layout-saved-select');
        const filename = select?.value;
        
        if (!graphId || !filename) {
            showToast('Select a layout to load', 'warning');
            return;
        }
        
        try {
            const response = await fetch(`/api/layout/load/${graphId}/${filename}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to load layout');
            }
            
            const result = await response.json();
            
            // Reload graph to apply layout
            await reloadGraphWithNewLayout(graphId);
            
            showToast(`Loaded layout: ${filename}`, 'success');
            
        } catch (error) {
            console.error('[Layout] Load error:', error);
            showToast(`Load failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Handle set as default button
     */
    async function handleSetDefault() {
        const graphId = getCurrentGraphId();
        const select = document.getElementById('layout-saved-select');
        const filename = select?.value;
        
        if (!graphId || !filename) {
            showToast('Select a layout first', 'warning');
            return;
        }
        
        try {
            const response = await fetch(`/api/layout/set-default/${graphId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to set default');
            }
            
            // Refresh saved layouts list
            await loadSavedLayouts();
            
            showToast(`Set default layout: ${filename}`, 'success');
            
        } catch (error) {
            console.error('[Layout] Set default error:', error);
            showToast(`Failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Handle delete layout button
     */
    async function handleDeleteLayout() {
        const graphId = getCurrentGraphId();
        const select = document.getElementById('layout-saved-select');
        const filename = select?.value;
        
        if (!graphId || !filename) {
            showToast('Select a layout first', 'warning');
            return;
        }
        
        if (!confirm(`Delete layout: ${filename}?`)) {
            return;
        }
        
        try {
            const response = await fetch(`/api/layout/${graphId}/${filename}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete');
            }
            
            // Refresh saved layouts list
            await loadSavedLayouts();
            
            showToast(`Deleted: ${filename}`, 'success');
            
        } catch (error) {
            console.error('[Layout] Delete error:', error);
            showToast(`Delete failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Reload graph with new layout positions
     * Supports both Cytoscape.js and cosmos.gl
     */
    async function reloadGraphWithNewLayout(graphId) {
        try {
            // Fetch updated graph elements
            const response = await fetch(`/api/graphs/${graphId}/elements?mode=nodes_only`);
            if (!response.ok) throw new Error('Failed to fetch updated positions');
            
            const data = await response.json();
            const elements = data.elements || [];
            
            // Build positions map
            const positions = {};
            elements.forEach(el => {
                if (el.group === 'nodes' && el.position) {
                    positions[el.data.id] = el.position;
                }
            });
            
            // Check renderer type and apply positions accordingly
            const renderer = (typeof State !== 'undefined' && State.renderer);
            const rendererType = (typeof State !== 'undefined' && State.rendererType);
            
            if (rendererType === 'cosmos' && renderer) {
                // cosmos.gl: pause simulation and update positions
                renderer.pauseSimulation();
                renderer.updatePositions(positions);
                renderer.fitView();
                State.cosmosSimulationPaused = true;
                console.log('[Layout] Updated', Object.keys(positions).length, 'node positions (cosmos.gl)');
                
            } else {
                // Cytoscape.js: batch update positions
                const cy = (typeof State !== 'undefined' && State.cy) || window.cy;
                if (cy) {
                    cy.batch(() => {
                        elements.forEach(el => {
                            if (el.group === 'nodes' && el.position) {
                                const cyNode = cy.getElementById(el.data.id);
                                if (cyNode.length > 0) {
                                    cyNode.position(el.position);
                                }
                            }
                        });
                    });
                    
                    // Fit view to new layout
                    cy.fit(50);
                    console.log('[Layout] Updated', elements.length, 'node positions (Cytoscape.js)');
                }
            }
            
        } catch (error) {
            console.error('[Layout] Failed to reload graph:', error);
            throw error;
        }
    }
    
    /**
     * Handle test backend button click
     */
    async function handleTestBackend() {
        const backend = currentBackend === 'auto' ? 'igraph' : currentBackend;
        const nodes = parseInt(document.getElementById('layout-test-nodes')?.value || '500');
        
        const btn = document.getElementById('test-layout-btn');
        const resultDiv = document.getElementById('layout-test-result');
        
        try {
            btn.disabled = true;
            btn.textContent = 'Testing...';
            
            const params = new URLSearchParams({ nodes });
            if (currentAlgorithm !== 'auto') {
                params.append('algorithm', currentAlgorithm);
            }
            
            const response = await fetch(`/api/layout/test/${backend}?${params}`);
            const result = await response.json();
            
            if (resultDiv) {
                if (result.success) {
                    resultDiv.innerHTML = `
                        <span class="success">${Icons.get("check")} ${result.algorithm}</span><br>
                        ${result.node_count} nodes in ${result.computation_time.toFixed(3)}s
                    `;
                    resultDiv.className = 'test-result success';
                } else {
                    resultDiv.innerHTML = `<span class="error">${Icons.get("close")} ${result.error || 'Failed'}</span>`;
                    resultDiv.className = 'test-result error';
                }
            }
            
        } catch (error) {
            if (resultDiv) {
                resultDiv.innerHTML = `<span class="error">${Icons.get("close")} ${error.message}</span>`;
                resultDiv.className = 'test-result error';
            }
        } finally {
            btn.disabled = false;
            btn.textContent = 'Test Backend';
        }
    }
    
    /**
     * Show toast notification
     */
    function showToast(message, type = 'info') {
        if (window.Toast && Toast.show) {
            Toast.show(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
    
    /**
     * Get current layout settings
     */
    function getSettings() {
        return {
            backend: currentBackend,
            algorithm: currentAlgorithm,
            availableBackends: availableBackends,
            savedLayouts: savedLayouts
        };
    }
    
    // Public API
    return {
        init,
        loadBackends,
        loadSavedLayouts,
        getSettings,
        recomputeLayout: handleRecomputeLayout,
        getCurrentGraphId
    };
})();

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Delay init to ensure other modules are ready
    setTimeout(() => Layout.init(), 500);
});

// Also refresh layouts when a graph is loaded
document.addEventListener('graphLoaded', (event) => {
    console.log('[Layout] graphLoaded event received:', event.detail);
    setTimeout(() => Layout.loadSavedLayouts(), 100);
});