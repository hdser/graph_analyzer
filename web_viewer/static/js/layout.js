/**
 * Layout Module
 * 
 * Handles layout backend selection and recomputation.
 * Integrates with the sidebar panel system.
 */

const Layout = (function() {
    'use strict';
    
    // State
    let availableBackends = [];
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
            const statusText = backend.available ? '✓' : '✗';
            html += `
                <div class="backend-item">
                    <span class="backend-name">${backend.name}</span>
                    <span class="backend-status ${statusClass}">${statusText}</span>
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
        
        // Test backend button
        const testBtn = document.getElementById('test-layout-btn');
        if (testBtn) {
            testBtn.addEventListener('click', handleTestBackend);
        }
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
            algorithmGroup.style.display = 'block';
            currentAlgorithm = backend.algorithms[0];
        } else {
            algorithmGroup.style.display = 'none';
            currentAlgorithm = 'auto';
        }
    }
    
    /**
     * Get current graph ID from various sources
     */
    function getCurrentGraphId() {
        // Try State.currentGraph first (main source)
        if (State.currentGraph) {
            return State.currentGraph;
        }
        
        // Try graph selector dropdown
        const graphSelect = document.getElementById('graph-select');
        if (graphSelect && graphSelect.value) {
            return graphSelect.value;
        }
        
        // Try to get from State.graphData keys
        if (State.graphData && Object.keys(State.graphData).length > 0) {
            return Object.keys(State.graphData)[0];
        }
        
        // Try to get from currentState
        if (State.currentState && State.currentState.loaded_graphs && State.currentState.loaded_graphs.length > 0) {
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
            console.error('[Layout] No graph loaded - State.currentGraph:', State.currentGraph, 
                          'State.graphData:', Object.keys(State.graphData || {}));
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
            const request = {};
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
            
            showToast(
                `Layout computed: ${result.algorithm} (${result.computation_time.toFixed(2)}s)`,
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
     * Reload graph with new layout positions
     */
    async function reloadGraphWithNewLayout(graphId) {
        try {
            // Fetch updated graph elements
            const response = await fetch(`/api/graphs/${graphId}/elements?mode=nodes_only`);
            if (!response.ok) throw new Error('Failed to fetch updated positions');
            
            const data = await response.json();
            const elements = data.elements || [];
            
            // Update Cytoscape positions
            const cy = State.cy || window.cy;
            if (cy) {
                elements.forEach(el => {
                    if (el.group === 'nodes' && el.position) {
                        const cyNode = cy.getElementById(el.data.id);
                        if (cyNode.length > 0) {
                            cyNode.position(el.position);
                        }
                    }
                });
                
                // Fit view to new layout
                cy.fit(50);
                console.log('[Layout] Updated', elements.length, 'node positions');
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
                        <span class="success">✓ ${result.algorithm}</span><br>
                        ${result.node_count} nodes in ${result.computation_time.toFixed(3)}s
                    `;
                    resultDiv.className = 'test-result success';
                } else {
                    resultDiv.innerHTML = `<span class="error">✗ ${result.error || 'Failed'}</span>`;
                    resultDiv.className = 'test-result error';
                }
            }
            
        } catch (error) {
            if (resultDiv) {
                resultDiv.innerHTML = `<span class="error">✗ ${error.message}</span>`;
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
            availableBackends: availableBackends
        };
    }
    
    // Public API
    return {
        init,
        loadBackends,
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