/**
 * Path Analysis Module
 * 
 * Interactive path finding and analysis for the Graph Analyzer.
 * Uses State.cy for Cytoscape access.
 * 
 * Location: web_viewer/static/js/path-analysis.js
 */

const PathAnalysis = (function() {
    'use strict';
    
    // State
    let state = {
        sourceNode: null,
        targetNode: null,
        pickMode: null,
        lastResult: null,
        selectedPathIndex: -1,
        selectedPaths: new Set(),  // Multiple selection
        originalStyles: new Map(),
        isIsolated: false,  // Track if we're in isolated view
        hiddenNodes: [],    // Nodes hidden during isolation
    };
    
    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
    
    function init() {
        console.log('[PathAnalysis] Initializing...');
        
        const panel = document.getElementById('panel-paths');
        if (!panel) {
            console.warn('[PathAnalysis] Panel not found');
            return;
        }
        
        // Source/Target inputs
        const sourceInput = document.getElementById('path-source-input');
        const targetInput = document.getElementById('path-target-input');
        
        if (sourceInput) {
            sourceInput.addEventListener('input', (e) => {
                state.sourceNode = e.target.value.trim() || null;
            });
        }
        
        if (targetInput) {
            targetInput.addEventListener('input', (e) => {
                state.targetNode = e.target.value.trim() || null;
            });
        }
        
        // Pick buttons
        document.getElementById('pick-source-btn')?.addEventListener('click', () => startPicking('source'));
        document.getElementById('pick-target-btn')?.addEventListener('click', () => startPicking('target'));
        
        // Swap button
        document.getElementById('swap-nodes-btn')?.addEventListener('click', swapNodes);
        
        // Algorithm select
        document.getElementById('path-algorithm-select')?.addEventListener('change', onAlgorithmChange);
        
        // Find Paths button
        document.getElementById('find-paths-btn')?.addEventListener('click', findPaths);
        
        // Clear button
        document.getElementById('clear-paths-btn')?.addEventListener('click', clearResults);
        
        // Subgraph tools
        document.getElementById('extract-neighborhood-btn')?.addEventListener('click', extractNeighborhood);
        document.getElementById('extract-component-btn')?.addEventListener('click', extractComponent);
        
        // Flow analysis
        document.getElementById('max-flow-btn')?.addEventListener('click', computeMaxFlow);
        document.getElementById('min-cut-btn')?.addEventListener('click', computeMinCut);
        
        // Initial params visibility
        onAlgorithmChange();
        
        console.log('[PathAnalysis] Initialized');
    }
    
    // ==========================================================================
    // RENDERER ACCESS
    // ==========================================================================
    
    function getCytoscape() {
        // Try multiple ways to get Cytoscape instance
        if (typeof State !== 'undefined' && State.cy) {
            return State.cy;
        }
        if (typeof getCy === 'function') {
            return getCy();
        }
        if (typeof window.cy !== 'undefined') {
            return window.cy;
        }
        return null;
    }
    
    /**
     * Get the current renderer (either Cytoscape or Cosmos)
     * @returns {Object|null} The renderer instance
     */
    function getRenderer() {
        if (typeof State !== 'undefined' && State.renderer) {
            return State.renderer;
        }
        if (typeof window.getRenderer === 'function') {
            return window.getRenderer();
        }
        return null;
    }
    
    /**
     * Check if using CosmosGL renderer
     * @returns {boolean}
     */
    function isCosmosRenderer() {
        if (typeof State !== 'undefined') {
            return State.rendererType === 'cosmos';
        }
        const renderer = getRenderer();
        return renderer && typeof renderer.getType === 'function' && renderer.getType() === 'cosmos';
    }
    
    // ==========================================================================
    // NODE PICKING
    // ==========================================================================
    
    function startPicking(mode) {
        state.pickMode = mode;
        
        const btn = mode === 'source' 
            ? document.getElementById('pick-source-btn')
            : document.getElementById('pick-target-btn');
        
        if (btn) btn.classList.add('picking');
        
        showToast(`Click a node to set as ${mode}`, 'info');
        
        const cyDiv = document.getElementById('cy');
        if (cyDiv) cyDiv.style.cursor = 'crosshair';
    }
    
    function onNodeClick(nodeId) {
        if (!state.pickMode) return false;
        
        if (state.pickMode === 'source') {
            setSource(nodeId);
        } else {
            setTarget(nodeId);
        }
        
        cancelPicking();
        return true;
    }
    
    function cancelPicking() {
        state.pickMode = null;
        document.querySelectorAll('.mini-icon-btn.picking').forEach(btn => btn.classList.remove('picking'));
        const cyDiv = document.getElementById('cy');
        if (cyDiv) cyDiv.style.cursor = '';
    }
    
    function setSource(nodeId) {
        state.sourceNode = nodeId;
        const input = document.getElementById('path-source-input');
        if (input) input.value = nodeId;
    }
    
    function setTarget(nodeId) {
        state.targetNode = nodeId;
        const input = document.getElementById('path-target-input');
        if (input) input.value = nodeId;
    }
    
    function swapNodes() {
        const temp = state.sourceNode;
        setSource(state.targetNode || '');
        setTarget(temp || '');
    }
    
    // ==========================================================================
    // ALGORITHM SELECTION
    // ==========================================================================
    
    function onAlgorithmChange() {
        const select = document.getElementById('path-algorithm-select');
        if (!select) return;
        
        const algorithm = select.value;
        
        const kRow = document.getElementById('param-k-row');
        const cutoffRow = document.getElementById('param-cutoff-row');
        const weightRow = document.getElementById('param-weight-row');
        
        if (kRow) kRow.style.display = 'none';
        if (cutoffRow) cutoffRow.style.display = 'none';
        if (weightRow) weightRow.style.display = 'none';
        
        switch (algorithm) {
            case 'k_shortest_paths':
                if (kRow) kRow.style.display = 'flex';
                if (weightRow) weightRow.style.display = 'flex';
                break;
            case 'all_simple_paths':
                if (cutoffRow) cutoffRow.style.display = 'flex';
                break;
            case 'dijkstra':
                if (weightRow) weightRow.style.display = 'flex';
                break;
        }
    }
    
    // ==========================================================================
    // PATH FINDING
    // ==========================================================================
    
    function findPaths() {
        const sourceInput = document.getElementById('path-source-input');
        const targetInput = document.getElementById('path-target-input');
        
        state.sourceNode = sourceInput?.value?.trim() || null;
        state.targetNode = targetInput?.value?.trim() || null;
        
        if (!state.sourceNode || !state.targetNode) {
            showToast('Please enter both source and target nodes', 'error');
            return;
        }
        
        const algorithmSelect = document.getElementById('path-algorithm-select');
        const algorithm = algorithmSelect?.value || 'shortest_path';
        
        const directedCheckbox = document.getElementById('path-directed');
        const directed = directedCheckbox?.checked ?? true;
        
        const body = {
            source: state.sourceNode,
            target: state.targetNode,
            algorithm: algorithm,
            directed: directed,
            max_paths: 1000,  // Request up to 1000 paths
        };
        
        if (algorithm === 'k_shortest_paths') {
            body.k = parseInt(document.getElementById('path-k')?.value) || 5;
            const weight = document.getElementById('path-weight')?.value?.trim();
            if (weight) body.weight = weight;
        } else if (algorithm === 'all_simple_paths') {
            body.cutoff = parseInt(document.getElementById('path-cutoff')?.value) || 10;
        } else if (algorithm === 'dijkstra') {
            body.weight = document.getElementById('path-weight')?.value?.trim() || 'weight';
        }
        
        const graphSelect = document.getElementById('graph-select');
        const graphName = graphSelect?.value || null;
        
        let url = '/api/algorithms/paths';
        if (graphName) url += `?graph_name=${encodeURIComponent(graphName)}`;
        
        console.log('[PathAnalysis] Finding paths:', body);
        showLoading(true);
        
        // Non-blocking: yield to event loop, then fetch
        setTimeout(() => {
            fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.detail || 'Failed to find paths');
                    });
                }
                return response.json();
            })
            .then(result => {
                console.log('[PathAnalysis] Result:', result);
                
                state.lastResult = result;
                state.selectedPathIndex = -1;
                
                // Yield before updating UI
                setTimeout(() => {
                    displayResults(result);
                    if (result.paths && result.paths.length > 0) {
                        selectPath(0);
                    }
                    showLoading(false);
                }, 0);
            })
            .catch(error => {
                console.error('[PathAnalysis] Error:', error);
                showToast(error.message, 'error');
                displayResults({ paths: [], message: error.message, success: false });
                showLoading(false);
            });
        }, 0);
    }
    
    function displayResults(result) {
        const resultsSection = document.getElementById('path-results');
        const summaryEl = document.getElementById('path-results-summary');
        const timeEl = document.getElementById('path-results-time');
        const listEl = document.getElementById('path-results-list');
        
        if (!resultsSection || !listEl) return;
        
        resultsSection.style.display = 'block';
        
        // Reset multiple selection
        state.selectedPaths.clear();
        
        if (summaryEl) {
            if (result.paths?.length > 0) {
                summaryEl.textContent = `Found ${result.paths.length} path(s)${result.truncated ? ' (truncated)' : ''}`;
            } else {
                summaryEl.textContent = result.message || 'No paths found';
            }
        }
        
        if (timeEl && result.computation_time_ms !== undefined) {
            timeEl.textContent = `${result.computation_time_ms.toFixed(1)}ms`;
        }
        
        listEl.innerHTML = '';
        
        if (!result.paths?.length) {
            listEl.innerHTML = '<div class="path-item no-paths">No paths found</div>';
            return;
        }
        
        // Add select all checkbox if multiple paths
        if (result.paths.length > 1) {
            const selectAllDiv = document.createElement('div');
            selectAllDiv.className = 'path-select-all';
            selectAllDiv.innerHTML = `
                <label class="path-checkbox-label">
                    <input type="checkbox" id="select-all-paths" onchange="PathAnalysis.toggleSelectAll(this.checked)">
                    <span>Select All (${result.paths.length})</span>
                </label>
            `;
            listEl.appendChild(selectAllDiv);
        }
        
        result.paths.forEach((path, index) => {
            const item = document.createElement('div');
            item.className = 'path-item';
            item.dataset.index = index;
            
            const length = path.length ?? (path.nodes?.length - 1) ?? 0;
            const weight = path.weight !== undefined ? ` (${path.weight.toFixed(2)})` : '';
            
            item.innerHTML = `
                <div class="path-item-header">
                    <label class="path-checkbox-label" onclick="event.stopPropagation()">
                        <input type="checkbox" class="path-checkbox" data-index="${index}" 
                               onchange="PathAnalysis.togglePathSelection(${index}, this.checked)">
                    </label>
                    <span class="path-index">#${index + 1}</span>
                    <span class="path-length">Length: ${length}${weight}</span>
                </div>
                <div class="path-nodes">${formatPathNodes(path.nodes)}</div>
            `;
            
            item.addEventListener('click', (e) => {
                // Don't trigger if clicking checkbox
                if (e.target.type !== 'checkbox') {
                    selectPath(index);
                }
            });
            listEl.appendChild(item);
        });
    }
    
    /**
     * Toggle selection of a single path
     */
    function togglePathSelection(index, checked) {
        if (checked) {
            state.selectedPaths.add(index);
        } else {
            state.selectedPaths.delete(index);
        }
        
        updatePathSelectionUI();
        
        // If multiple selected, highlight all
        if (state.selectedPaths.size > 1) {
            highlightMultiplePaths();
        } else if (state.selectedPaths.size === 1) {
            const idx = [...state.selectedPaths][0];
            selectPath(idx);
        }
    }
    
    /**
     * Toggle select all paths
     */
    function toggleSelectAll(checked) {
        if (!state.lastResult?.paths) return;
        
        state.selectedPaths.clear();
        
        if (checked) {
            state.lastResult.paths.forEach((_, idx) => state.selectedPaths.add(idx));
        }
        
        // Update all checkboxes
        document.querySelectorAll('.path-checkbox').forEach((cb, idx) => {
            cb.checked = checked;
        });
        
        updatePathSelectionUI();
        
        if (state.selectedPaths.size > 0) {
            highlightMultiplePaths();
        } else {
            clearHighlights();
        }
    }
    
    /**
     * Update UI to reflect path selection
     */
    function updatePathSelectionUI() {
        document.querySelectorAll('.path-item').forEach((item, idx) => {
            if (item.classList.contains('path-select-all')) return;
            const index = parseInt(item.dataset.index);
            item.classList.toggle('selected', state.selectedPaths.has(index));
        });
        
        // Update select all checkbox
        const selectAllCb = document.getElementById('select-all-paths');
        if (selectAllCb && state.lastResult?.paths) {
            selectAllCb.checked = state.selectedPaths.size === state.lastResult.paths.length;
            selectAllCb.indeterminate = state.selectedPaths.size > 0 && 
                                        state.selectedPaths.size < state.lastResult.paths.length;
        }
    }
    
    /**
     * Highlight multiple paths at once
     */
    function highlightMultiplePaths() {
        clearHighlights();
        
        if (!state.lastResult?.paths) return;
        
        // Collect all nodes and edges from selected paths
        // Track which nodes are sources/targets
        const allPathNodes = new Map(); // nodeId -> 'source' | 'target' | 'intermediate'
        const pathEdges = [];
        
        state.selectedPaths.forEach(idx => {
            const path = state.lastResult.paths[idx];
            if (!path?.nodes) return;
            
            path.nodes.forEach((n, i) => {
                if (i === 0) {
                    // Source - only set if not already set as something else
                    if (!allPathNodes.has(n)) allPathNodes.set(n, 'source');
                } else if (i === path.nodes.length - 1) {
                    // Target
                    if (!allPathNodes.has(n) || allPathNodes.get(n) === 'intermediate') {
                        allPathNodes.set(n, 'target');
                    }
                } else {
                    // Intermediate - only set if not already source/target
                    if (!allPathNodes.has(n)) allPathNodes.set(n, 'intermediate');
                }
            });
            
            for (let i = 0; i < path.nodes.length - 1; i++) {
                pathEdges.push([path.nodes[i], path.nodes[i + 1]]);
            }
        });
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            highlightMultiplePathsCosmos(allPathNodes, pathEdges);
            return;
        }
        
        // Cytoscape implementation
        const cy = getCytoscape();
        if (!cy) return;
        
        cy.batch(() => {
            // Highlight all path nodes with correct colors
            allPathNodes.forEach((type, nodeId) => {
                const node = cy.getElementById(nodeId);
                if (node.length) {
                    let color, size;
                    if (type === 'source') {
                        color = '#22c55e'; // Green
                        size = 35;
                    } else if (type === 'target') {
                        color = '#ef4444'; // Red
                        size = 35;
                    } else {
                        color = '#00d4ff'; // Cyan
                        size = 25;
                    }
                    
                    node.style({
                        'background-color': color,
                        'border-color': '#ffffff',
                        'border-width': 3,
                        'width': size,
                        'height': size,
                        'z-index': 9999,
                    });
                    state.originalStyles.set(nodeId, node);
                }
            });
            
            // Highlight/create edges
            pathEdges.forEach(([sourceId, targetId], i) => {
                let edge = cy.edges(`[source="${sourceId}"][target="${targetId}"]`);
                if (!edge.length) {
                    edge = cy.edges(`[source="${targetId}"][target="${sourceId}"]`);
                }
                
                if (edge.length) {
                    edge.style({
                        'line-color': '#00d4ff',
                        'target-arrow-color': '#00d4ff',
                        'width': 4,
                        'opacity': 1,
                        'z-index': 9998,
                    });
                    state.originalStyles.set(edge.id(), edge);
                } else {
                    // Create temp edge
                    const tempEdgeId = `path-edge-multi-${i}`;
                    if (!cy.getElementById(tempEdgeId).length) {
                        cy.add({
                            group: 'edges',
                            data: { id: tempEdgeId, source: sourceId, target: targetId, _pathTemp: true }
                        });
                        cy.getElementById(tempEdgeId).style({
                            'line-color': '#00d4ff',
                            'width': 4,
                            'opacity': 1,
                            'z-index': 9998,
                        });
                        state.originalStyles.set(tempEdgeId, { _isTemp: true });
                    }
                }
            });
        });
        
        // Fit to all path nodes
        const pathNodes = cy.nodes().filter(n => allPathNodes.has(n.id()));
        if (pathNodes.length > 0) {
            cy.animate({
                fit: { eles: pathNodes, padding: 80 },
                duration: 500
            });
        }
        
        // Show multi-path details panel
        showMultiplePathDetails();
        
        showToast(`Highlighting ${state.selectedPaths.size} paths (${allPathNodes.size} nodes)`, 'info');
    }
    
    /**
     * Highlight multiple paths using CosmosGL renderer
     */
    function highlightMultiplePathsCosmos(allPathNodes, pathEdges) {
        const renderer = getRenderer();
        if (!renderer) return;
        
        console.log('[PathAnalysis] Highlighting multiple paths (CosmosGL):', allPathNodes.size, 'nodes');
        
        // Build node color map
        const nodeColorMap = new Map();
        allPathNodes.forEach((type, nodeId) => {
            let color;
            if (type === 'source') {
                color = '#22c55e'; // Green
            } else if (type === 'target') {
                color = '#ef4444'; // Red
            } else {
                color = '#00d4ff'; // Cyan
            }
            nodeColorMap.set(nodeId, { color, type });
        });
        
        // Build edge pairs
        const edgePairs = pathEdges.map(([source, target]) => ({ source, target }));
        
        // Apply highlighting
        renderer.highlightPathNodes(nodeColorMap);
        renderer.highlightPathEdges(edgePairs, '#00d4ff', 1.0);
        
        // Fit view to all path nodes
        renderer.fitView(Array.from(allPathNodes.keys()), 0.2);
        
        // Show multi-path details panel
        showMultiplePathDetails();
        
        showToast(`Highlighting ${state.selectedPaths.size} paths (${allPathNodes.size} nodes)`, 'info');
    }
    
    /**
     * Show details for multiple selected paths
     */
    function showMultiplePathDetails() {
        const infoPanel = document.getElementById('info-panel');
        if (!infoPanel) return;
        
        // Hide other sections
        const nodeInfo = document.getElementById('node-info');
        const edgeInfo = document.getElementById('edge-info');
        const multiInfo = document.getElementById('multi-info');
        if (nodeInfo) nodeInfo.style.display = 'none';
        if (edgeInfo) edgeInfo.style.display = 'none';
        if (multiInfo) multiInfo.style.display = 'none';
        
        // Update header
        const headerTitle = infoPanel.querySelector('.info-header h3');
        if (headerTitle) headerTitle.textContent = `${state.selectedPaths.size} Paths Selected`;
        
        // Collect stats
        let totalNodes = new Set();
        let totalHops = 0;
        
        let html = `<div id="path-detail-panel">`;
        
        // Summary stats
        state.selectedPaths.forEach(idx => {
            const path = state.lastResult?.paths[idx];
            if (path?.nodes) {
                path.nodes.forEach(n => totalNodes.add(n));
                totalHops += path.nodes.length - 1;
            }
        });
        
        html += `
            <div class="path-stats">
                <div class="path-stat">
                    <div class="path-stat-label">Paths</div>
                    <div class="path-stat-value">${state.selectedPaths.size}</div>
                </div>
                <div class="path-stat">
                    <div class="path-stat-label">Unique Nodes</div>
                    <div class="path-stat-value">${totalNodes.size}</div>
                </div>
            </div>
        `;
        
        // List each path
        html += `<div class="multi-path-list">`;
        
        [...state.selectedPaths].sort((a, b) => a - b).forEach(idx => {
            const path = state.lastResult?.paths[idx];
            if (!path?.nodes) return;
            
            const length = path.length ?? (path.nodes.length - 1);
            const weight = path.weight !== undefined ? path.weight.toFixed(2) : length;
            
            html += `
                <div class="multi-path-item" onclick="PathAnalysis.selectPath(${idx})">
                    <div class="multi-path-header">
                        <span class="path-index">#${idx + 1}</span>
                        <span class="path-length">${length} hops (${weight})</span>
                    </div>
                    <div class="multi-path-route">
                        <span class="node-badge source">${path.nodes[0].slice(0,8)}...</span>
                        <span class="route-arrow">-></span>
                        <span class="node-badge target">${path.nodes[path.nodes.length-1].slice(0,8)}...</span>
                    </div>
                </div>
            `;
        });
        
        html += `</div>`;
        
        // Actions
        html += `
            <div class="path-actions">
                <button onclick="PathAnalysis.fitToSelectedPaths()">Fit to All</button>
                <button onclick="PathAnalysis.copyAllPaths()">Copy All</button>
            </div>
            <div class="path-actions">
                <button onclick="PathAnalysis.isolatePath()" class="isolate-btn">Isolate Paths</button>
                <button onclick="PathAnalysis.showAllNodes()" class="show-all-btn">Show All</button>
            </div>
        </div>`;
        
        // Remove existing detail panel
        document.getElementById('path-detail-panel')?.remove();
        
        // Insert after header
        const infoHeader = infoPanel.querySelector('.info-header');
        if (infoHeader) {
            infoHeader.insertAdjacentHTML('afterend', html);
        } else {
            infoPanel.insertAdjacentHTML('afterbegin', html);
        }
        
        infoPanel.style.display = 'block';
    }
    
    /**
     * Fit view to all selected paths
     */
    function fitToSelectedPaths() {
        if (!state.lastResult?.paths) return;
        
        const allNodes = new Set();
        state.selectedPaths.forEach(idx => {
            const path = state.lastResult.paths[idx];
            if (path?.nodes) path.nodes.forEach(n => allNodes.add(n));
        });
        
        if (allNodes.size === 0) return;
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer && typeof renderer.fitView === 'function') {
                renderer.fitView(Array.from(allNodes), 0.2);
            }
            return;
        }
        
        // Cytoscape implementation
        const cy = getCytoscape();
        if (!cy) return;
        
        const pathNodes = cy.nodes().filter(n => allNodes.has(n.id()));
        if (pathNodes.length > 0) {
            cy.animate({
                fit: { eles: pathNodes, padding: 80 },
                duration: 500
            });
        }
    }
    
    /**
     * Copy all selected paths to clipboard
     */
    function copyAllPaths() {
        if (!state.lastResult?.paths || state.selectedPaths.size === 0) return;
        
        const text = [...state.selectedPaths]
            .sort((a, b) => a - b)
            .map(idx => {
                const path = state.lastResult.paths[idx];
                return `Path #${idx + 1}: ${path.nodes.join(' -> ')}`;
            })
            .join('\n\n');
        
        navigator.clipboard.writeText(text).then(() => {
            showToast(`${state.selectedPaths.size} paths copied!`, 'success');
        });
    }
    
    function formatPathNodes(nodes) {
        if (!nodes?.length) return '-';
        
        // Truncate node IDs for display in list (show full in details panel)
        const fmt = (id) => id.length > 12 ? id.slice(0, 6) + '...' + id.slice(-4) : id;
        
        // Show first 2 and last 2 nodes with ... in between for long paths
        if (nodes.length > 5) {
            const first = nodes.slice(0, 2).map(fmt);
            const last = nodes.slice(-2).map(fmt);
            return `${first.join(' -> ')} -> ... -> ${last.join(' -> ')}`;
        }
        
        return nodes.map(fmt).join(' -> ');
    }
    
    function selectPath(index) {
        if (!state.lastResult?.paths) return;
        
        const path = state.lastResult.paths[index];
        if (!path) return;
        
        state.selectedPathIndex = index;
        
        document.querySelectorAll('.path-item').forEach((item, i) => {
            item.classList.toggle('selected', i === index);
        });
        
        highlightPath(path);
        showPathDetails(path, index);
    }
    
    // ==========================================================================
    // PATH HIGHLIGHTING
    // ==========================================================================
    
    function highlightPath(path) {
        if (!path?.nodes) return;
        
        clearHighlights();
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            highlightPathCosmos(path);
            return;
        }
        
        // Cytoscape path highlighting
        const cy = getCytoscape();
        if (!cy) {
            console.error('[PathAnalysis] No renderer available');
            return;
        }
        
        console.log('[PathAnalysis] Highlighting path with', path.nodes.length, 'nodes (Cytoscape)');
        
        cy.batch(() => {
            // DON'T dim - just make path nodes really stand out
            
            // Highlight path nodes - make them BIG and colorful
            path.nodes.forEach((nodeId, i) => {
                const node = cy.getElementById(nodeId);
                if (node.length) {
                    let color, size;
                    if (i === 0) {
                        color = '#22c55e'; // Source - green
                        size = 40;
                    } else if (i === path.nodes.length - 1) {
                        color = '#ef4444'; // Target - red  
                        size = 40;
                    } else {
                        color = '#00d4ff'; // Intermediate - cyan
                        size = 30;
                    }
                    
                    node.style({
                        'background-color': color,
                        'border-color': '#ffffff',
                        'border-width': 4,
                        'width': size,
                        'height': size,
                        'z-index': 9999,
                    });
                    
                    state.originalStyles.set(nodeId, node);
                }
            });
            
            // Draw path edges - create temporary edges if they don't exist
            for (let i = 0; i < path.nodes.length - 1; i++) {
                const sourceId = path.nodes[i];
                const targetId = path.nodes[i + 1];
                
                // Check if edge exists
                let edge = cy.edges(`[source="${sourceId}"][target="${targetId}"]`);
                if (!edge.length) {
                    edge = cy.edges(`[source="${targetId}"][target="${sourceId}"]`);
                }
                
                if (edge.length) {
                    // Style existing edge
                    edge.style({
                        'line-color': '#00d4ff',
                        'target-arrow-color': '#00d4ff',
                        'source-arrow-color': '#00d4ff',
                        'width': 5,
                        'opacity': 1,
                        'z-index': 9998,
                    });
                    state.originalStyles.set(edge.id(), edge);
                } else {
                    // Create temporary edge for the path
                    const tempEdgeId = `path-edge-${i}`;
                    cy.add({
                        group: 'edges',
                        data: {
                            id: tempEdgeId,
                            source: sourceId,
                            target: targetId,
                            _pathTemp: true
                        }
                    });
                    
                    const newEdge = cy.getElementById(tempEdgeId);
                    newEdge.style({
                        'line-color': '#00d4ff',
                        'target-arrow-color': '#00d4ff',
                        'width': 5,
                        'curve-style': 'bezier',
                        'line-style': 'solid',
                        'opacity': 1,
                        'z-index': 9998,
                    });
                    
                    state.originalStyles.set(tempEdgeId, { _isTemp: true });
                }
            }
        });
        
        // Fit view to path
        const pathNodeIds = path.nodes;
        const pathNodes = cy.nodes().filter(n => pathNodeIds.includes(n.id()));
        
        if (pathNodes.length > 0) {
            cy.animate({
                fit: { eles: pathNodes, padding: 100 },
                duration: 500,
                easing: 'ease-out'
            });
        }
    }
    
    /**
     * Highlight path using CosmosGL renderer
     */
    function highlightPathCosmos(path) {
        const renderer = getRenderer();
        if (!renderer) {
            console.error('[PathAnalysis] CosmosGL renderer not available');
            return;
        }
        
        console.log('[PathAnalysis] Highlighting path with', path.nodes.length, 'nodes (CosmosGL)');
        
        // Build node color map
        const nodeColorMap = new Map();
        path.nodes.forEach((nodeId, i) => {
            let color, type;
            if (i === 0) {
                color = '#22c55e'; // Source - green
                type = 'source';
            } else if (i === path.nodes.length - 1) {
                color = '#ef4444'; // Target - red
                type = 'target';
            } else {
                color = '#00d4ff'; // Intermediate - cyan
                type = 'intermediate';
            }
            nodeColorMap.set(nodeId, { color, type });
        });
        
        // Build edge pairs
        const edgePairs = [];
        for (let i = 0; i < path.nodes.length - 1; i++) {
            edgePairs.push({
                source: path.nodes[i],
                target: path.nodes[i + 1]
            });
        }
        
        // Apply highlighting
        renderer.highlightPathNodes(nodeColorMap);
        renderer.highlightPathEdges(edgePairs, '#00d4ff', 1.0);
        
        // Fit view to path nodes
        renderer.fitView(path.nodes, 0.2);
        
        // Store for cleanup tracking
        state.originalStyles.set('_cosmosPath', { nodes: path.nodes, edges: edgePairs });
    }
    
    function clearHighlights() {
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer && typeof renderer.clearPathHighlights === 'function') {
                renderer.clearPathHighlights();
            }
            state.originalStyles.clear();
            return;
        }
        
        // Cytoscape cleanup
        const cy = getCytoscape();
        if (!cy) {
            state.originalStyles.clear();
            return;
        }
        
        cy.batch(() => {
            // Remove temporary path edges
            cy.edges('[_pathTemp]').remove();
            
            // Reset styled elements
            state.originalStyles.forEach((value, key) => {
                if (value._isTemp) return; // Already removed above
                if (key === '_cosmosPath') return; // Cosmos marker
                
                const ele = cy.getElementById(key);
                if (ele.length) {
                    if (ele.isNode()) {
                        ele.removeStyle('background-color border-color border-width width height z-index');
                    } else if (ele.isEdge()) {
                        ele.removeStyle('line-color target-arrow-color source-arrow-color width opacity z-index');
                    }
                }
            });
        });
        
        state.originalStyles.clear();
    }
    
    // ==========================================================================
    // PATH DETAILS PANEL
    // ==========================================================================
    
    function showPathDetails(path, index) {
        const infoPanel = document.getElementById('info-panel');
        if (!infoPanel) return;
        
        // Hide other sections
        const nodeInfo = document.getElementById('node-info');
        const edgeInfo = document.getElementById('edge-info');
        const multiInfo = document.getElementById('multi-info');
        if (nodeInfo) nodeInfo.style.display = 'none';
        if (edgeInfo) edgeInfo.style.display = 'none';
        if (multiInfo) multiInfo.style.display = 'none';
        
        // Update header
        const headerTitle = infoPanel.querySelector('.info-header h3');
        if (headerTitle) headerTitle.textContent = 'Path Details';
        
        const length = path.length ?? (path.nodes?.length - 1) ?? 0;
        const weight = path.weight !== undefined ? path.weight.toFixed(2) : length.toString();
        
        // Get weight attribute used
        const weightAttr = document.getElementById('path-weight')?.value?.trim() || 'hops';
        
        let html = `
            <div id="path-detail-panel">
                <div class="path-stats">
                    <div class="path-stat">
                        <div class="path-stat-label">Hops</div>
                        <div class="path-stat-value">${length}</div>
                    </div>
                    <div class="path-stat">
                        <div class="path-stat-label">Weight (${weightAttr})</div>
                        <div class="path-stat-value">${weight}</div>
                    </div>
                </div>
                
                <div class="path-node-list-header">Path Nodes (${path.nodes.length})</div>
                <div class="path-node-list">
        `;
        
        path.nodes.forEach((nodeId, i) => {
            let hopClass = '';
            let hopLabel = i;
            
            if (i === 0) {
                hopClass = 'source';
                hopLabel = 'S';
            } else if (i === path.nodes.length - 1) {
                hopClass = 'target';
                hopLabel = 'T';
            }
            
            html += `
                <div class="path-node-item" onclick="PathAnalysis.zoomToNode('${nodeId}')" title="Click to zoom">
                    <span class="path-node-hop ${hopClass}">${hopLabel}</span>
                    <span class="path-node-id">${nodeId}</span>
                </div>
            `;
            
            if (i < path.nodes.length - 1) {
                html += `<div class="path-edge-info">| hop ${i + 1}</div>`;
            }
        });
        
        html += `
                </div>
                <div class="path-actions">
                    <button onclick="PathAnalysis.fitToPath()">Fit to Path</button>
                    <button onclick="PathAnalysis.copyPath()">Copy Path</button>
                </div>
                <div class="path-actions">
                    <button onclick="PathAnalysis.isolatePath()" class="isolate-btn">Isolate Path</button>
                    <button onclick="PathAnalysis.showAllNodes()" class="show-all-btn">Show All</button>
                </div>
            </div>
        `;
        
        // Remove existing detail panel
        document.getElementById('path-detail-panel')?.remove();
        
        // Insert after header
        const infoHeader = infoPanel.querySelector('.info-header');
        if (infoHeader) {
            infoHeader.insertAdjacentHTML('afterend', html);
        } else {
            infoPanel.insertAdjacentHTML('afterbegin', html);
        }
        
        infoPanel.style.display = 'block';
    }
    
    function hidePathDetails() {
        const infoPanel = document.getElementById('info-panel');
        if (!infoPanel) return;
        
        document.getElementById('path-detail-panel')?.remove();
        
        // Restore original header title
        const headerTitle = infoPanel.querySelector('.info-header h3');
        if (headerTitle) headerTitle.textContent = 'Information';
        
        // Show the original info sections again
        const nodeInfo = document.getElementById('node-info');
        const edgeInfo = document.getElementById('edge-info');
        const multiInfo = document.getElementById('multi-info');
        if (nodeInfo) nodeInfo.style.display = '';
        if (edgeInfo) edgeInfo.style.display = '';
        if (multiInfo) multiInfo.style.display = '';
    }
    
    function zoomToNode(nodeId) {
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer && typeof renderer.zoomToNode === 'function') {
                renderer.zoomToNode(nodeId, 2, 400);
            }
            return;
        }
        
        // Cytoscape implementation
        const cy = getCytoscape();
        if (!cy) return;
        
        const node = cy.getElementById(nodeId);
        if (node.length) {
            cy.animate({
                center: { eles: node },
                zoom: 2,
                duration: 400,
            });
        }
    }
    
    function fitToPath() {
        if (!state.lastResult || state.selectedPathIndex < 0) return;
        
        const path = state.lastResult.paths[state.selectedPathIndex];
        if (!path) return;
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer && typeof renderer.fitView === 'function') {
                renderer.fitView(path.nodes, 0.2);
            }
            return;
        }
        
        // Cytoscape implementation
        const cy = getCytoscape();
        if (!cy) return;
        
        const pathNodes = cy.nodes().filter(n => path.nodes.includes(n.id()));
        if (pathNodes.length > 0) {
            cy.animate({
                fit: { eles: pathNodes, padding: 80 },
                duration: 500,
            });
        }
    }
    
    function copyPath() {
        if (!state.lastResult || state.selectedPathIndex < 0) return;
        
        const path = state.lastResult.paths[state.selectedPathIndex];
        if (!path) return;
        
        navigator.clipboard.writeText(path.nodes.join(' -> ')).then(() => {
            showToast('Path copied!', 'success');
        });
    }
    
    /**
     * Isolate path - hide all nodes except those in the selected path(s)
     */
    function isolatePath() {
        // Collect all nodes from selected paths
        const pathNodeIds = new Set();
        
        if (state.selectedPaths.size > 0) {
            // Multiple paths selected
            state.selectedPaths.forEach(idx => {
                const path = state.lastResult?.paths[idx];
                if (path?.nodes) {
                    path.nodes.forEach(n => pathNodeIds.add(n));
                }
            });
        } else if (state.selectedPathIndex >= 0) {
            // Single path selected
            const path = state.lastResult?.paths[state.selectedPathIndex];
            if (path?.nodes) {
                path.nodes.forEach(n => pathNodeIds.add(n));
            }
        }
        
        if (pathNodeIds.size === 0) {
            showToast('No path selected', 'error');
            return;
        }
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer && typeof renderer.showOnlyNodes === 'function') {
                renderer.showOnlyNodes(Array.from(pathNodeIds));
                renderer.fitView(Array.from(pathNodeIds), 0.1);
                state.isIsolated = true;
                state.hiddenNodes = [];
                showToast(`Isolated ${pathNodeIds.size} path nodes`, 'success');
                updateIsolateButtons();
            }
            return;
        }
        
        // Cytoscape implementation
        const cy = getCytoscape();
        if (!cy) return;
        
        cy.batch(() => {
            // Hide nodes not in path
            cy.nodes().forEach(node => {
                if (!pathNodeIds.has(node.id())) {
                    node.style('display', 'none');
                    state.hiddenNodes.push(node.id());
                }
            });
            
            // Hide edges not connected to path nodes
            cy.edges().forEach(edge => {
                const src = edge.source().id();
                const tgt = edge.target().id();
                if (!pathNodeIds.has(src) || !pathNodeIds.has(tgt)) {
                    edge.style('display', 'none');
                }
            });
        });
        
        state.isIsolated = true;
        
        // Fit to visible nodes
        const visibleNodes = cy.nodes().filter(n => pathNodeIds.has(n.id()));
        if (visibleNodes.length > 0) {
            cy.animate({
                fit: { eles: visibleNodes, padding: 50 },
                duration: 500
            });
        }
        
        showToast(`Isolated ${pathNodeIds.size} path nodes`, 'success');
        updateIsolateButtons();
    }
    
    /**
     * Show all nodes - restore from isolated view
     */
    function showAllNodes() {
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer && typeof renderer.showAllNodes === 'function') {
                renderer.showAllNodes();
                renderer.fitView();
                state.isIsolated = false;
                state.hiddenNodes = [];
                showToast('Showing all nodes', 'success');
                updateIsolateButtons();
            }
            return;
        }
        
        // Cytoscape implementation
        const cy = getCytoscape();
        if (!cy) return;
        
        cy.batch(() => {
            // Show all nodes
            cy.nodes().style('display', 'element');
            // Show all edges
            cy.edges().style('display', 'element');
        });
        
        state.isIsolated = false;
        state.hiddenNodes = [];
        
        // Fit to all
        cy.fit(50);
        
        showToast('Showing all nodes', 'success');
        updateIsolateButtons();
    }
    
    /**
     * Update isolate/show all button states
     */
    function updateIsolateButtons() {
        const isolateBtn = document.querySelector('.isolate-btn');
        const showAllBtn = document.querySelector('.show-all-btn');
        
        if (isolateBtn) {
            isolateBtn.disabled = state.isIsolated;
            isolateBtn.style.opacity = state.isIsolated ? '0.5' : '1';
        }
        if (showAllBtn) {
            showAllBtn.disabled = !state.isIsolated;
            showAllBtn.style.opacity = state.isIsolated ? '1' : '0.5';
        }
    }
    
    function clearResults() {
        state.lastResult = null;
        state.selectedPathIndex = -1;
        state.selectedPaths.clear();
        clearHighlights();
        hidePathDetails();
        
        // If isolated, show all first
        if (state.isIsolated) {
            showAllNodes();
        }
        
        const resultsSection = document.getElementById('path-results');
        if (resultsSection) resultsSection.style.display = 'none';
        
        const listEl = document.getElementById('path-results-list');
        if (listEl) listEl.innerHTML = '';
    }
    
    // ==========================================================================
    // UTILITIES
    // ==========================================================================
    
    function showLoading(show) {
        const btn = document.getElementById('find-paths-btn');
        if (btn) {
            btn.disabled = show;
            btn.textContent = show ? 'Finding...' : 'Find Paths';
        }
    }
    
    function showToast(message, type = 'info') {
        if (typeof Toast !== 'undefined' && Toast.show) {
            Toast.show(message, type);
        } else if (typeof Toast !== 'undefined') {
            // Try other Toast methods
            if (type === 'error' && Toast.error) Toast.error(message);
            else if (type === 'success' && Toast.success) Toast.success(message);
            else if (Toast.info) Toast.info(message);
            else console.log(`[${type}] ${message}`);
        } else {
            console.log(`[PathAnalysis] [${type}] ${message}`);
        }
    }
    
    // ==========================================================================
    // PUBLIC API
    // ==========================================================================
    
    return {
        init,
        findPaths,
        selectPath,
        clearResults,
        clearHighlights,
        setSource,
        setTarget,
        swapNodes,
        onNodeClick,
        startPicking,
        cancelPicking,
        showPathDetails,
        hidePathDetails,
        showMultiplePathDetails,
        zoomToNode,
        fitToPath,
        fitToSelectedPaths,
        copyPath,
        copyAllPaths,
        isolatePath,
        showAllNodes,
        togglePathSelection,
        toggleSelectAll,
        highlightMultiplePaths,
        getState: () => ({ ...state }),
        isPickingNode: () => state.pickMode !== null,
    };
})();

// Make available globally
window.PathAnalysis = PathAnalysis;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => PathAnalysis.init());
} else {
    setTimeout(() => PathAnalysis.init(), 0);
}