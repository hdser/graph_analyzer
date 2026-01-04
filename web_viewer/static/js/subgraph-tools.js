/**
 * Subgraph Tools Module
 * 
 * Extract and visualize subgraphs: neighborhoods, ego graphs,
 * connected components, and induced subgraphs.
 * 
 * Location: web_viewer/static/js/subgraph-tools.js
 */

const SubgraphTools = (function() {
    'use strict';
    
    // State
    let state = {
        centerNode: null,
        seedNodes: [],
        pickMode: null,
        lastResult: null,
        isIsolated: false,
        highlightedElements: new Map(),
    };
    
    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
    
    function init() {
        console.log('[SubgraphTools] Initializing...');
        
        const panel = document.getElementById('panel-subgraph');
        if (!panel) {
            console.warn('[SubgraphTools] Panel not found');
            return;
        }
        
        // Mode selection
        document.getElementById('subgraph-mode')?.addEventListener('change', onModeChange);
        
        // Center node input
        const centerInput = document.getElementById('subgraph-center-input');
        if (centerInput) {
            centerInput.addEventListener('input', (e) => {
                state.centerNode = e.target.value.trim() || null;
            });
        }
        
        // Add selected seeds
        document.getElementById('add-selected-seeds-btn')?.addEventListener('click', addSelectedAsSeeds);
        
        // Action buttons
        document.getElementById('extract-subgraph-btn')?.addEventListener('click', extractSubgraph);
        document.getElementById('clear-subgraph-btn')?.addEventListener('click', clearResults);
        
        // Result action buttons
        document.getElementById('highlight-subgraph-btn')?.addEventListener('click', highlightSubgraph);
        document.getElementById('isolate-subgraph-btn')?.addEventListener('click', isolateSubgraph);
        document.getElementById('show-all-subgraph-btn')?.addEventListener('click', showAllNodes);
        document.getElementById('export-subgraph-btn')?.addEventListener('click', exportSubgraph);
        
        // Initial mode setup
        onModeChange();
        
        console.log('[SubgraphTools] Initialized');
    }
    
    // ==========================================================================
    // CYTOSCAPE ACCESS
    // ==========================================================================
    
    function getCytoscape() {
        if (typeof State !== 'undefined' && State.cy) {
            return State.cy;
        }
        if (typeof getCy === 'function') {
            return getCy();
        }
        return null;
    }
    
    // ==========================================================================
    // MODE HANDLING
    // ==========================================================================
    
    function onModeChange() {
        const mode = document.getElementById('subgraph-mode')?.value || 'neighborhood';
        
        const seedSection = document.getElementById('seed-nodes-section');
        const hopsRow = document.getElementById('subgraph-hops-row');
        const centerInput = document.getElementById('subgraph-center-input');
        
        // Show/hide seed nodes section
        if (seedSection) {
            seedSection.style.display = (mode === 'induced' || mode === 'k_hop') ? 'block' : 'none';
        }
        
        // Show/hide hops parameter
        if (hopsRow) {
            hopsRow.style.display = (mode === 'component' || mode === 'induced') ? 'none' : 'flex';
        }
        
        // Update center input placeholder
        if (centerInput) {
            if (mode === 'induced') {
                centerInput.placeholder = '(optional)';
            } else {
                centerInput.placeholder = 'Node ID or pick...';
            }
        }
    }
    
    // ==========================================================================
    // NODE PICKING
    // ==========================================================================
    
    function startPicking(mode) {
        state.pickMode = mode;
        
        const btn = document.getElementById('pick-subgraph-center-btn');
        if (btn) btn.classList.add('picking');
        
        showToast('Click a node to set as center', 'info');
        
        const cyDiv = document.getElementById('cy');
        if (cyDiv) cyDiv.style.cursor = 'crosshair';
    }
    
    function onNodeClick(nodeId) {
        if (!state.pickMode) return false;
        
        if (state.pickMode === 'center') {
            setCenter(nodeId);
        }
        
        cancelPicking();
        return true;
    }
    
    function cancelPicking() {
        state.pickMode = null;
        document.querySelectorAll('#panel-subgraph .mini-icon-btn.picking').forEach(btn => {
            btn.classList.remove('picking');
        });
        const cyDiv = document.getElementById('cy');
        if (cyDiv) cyDiv.style.cursor = '';
    }
    
    function setCenter(nodeId) {
        state.centerNode = nodeId;
        const input = document.getElementById('subgraph-center-input');
        if (input) input.value = nodeId;
    }
    
    // ==========================================================================
    // SEED NODES MANAGEMENT
    // ==========================================================================
    
    function addSelectedAsSeeds() {
        const cy = getCytoscape();
        if (!cy) return;
        
        const selected = cy.nodes(':selected');
        if (selected.length === 0) {
            showToast('No nodes selected', 'warning');
            return;
        }
        
        selected.forEach(node => {
            const id = node.id();
            if (!state.seedNodes.includes(id)) {
                state.seedNodes.push(id);
            }
        });
        
        updateSeedNodesList();
        showToast(`Added ${selected.length} node(s) as seeds`, 'success');
    }
    
    function removeSeed(nodeId) {
        state.seedNodes = state.seedNodes.filter(id => id !== nodeId);
        updateSeedNodesList();
    }
    
    function clearSeeds() {
        state.seedNodes = [];
        updateSeedNodesList();
    }
    
    function updateSeedNodesList() {
        const list = document.getElementById('seed-nodes-list');
        if (!list) return;
        
        if (state.seedNodes.length === 0) {
            list.innerHTML = '<div class="empty-list">No seed nodes added</div>';
            return;
        }
        
        list.innerHTML = state.seedNodes.map(id => `
            <div class="seed-node-item">
                <span class="seed-node-id">${id.slice(0, 10)}...${id.slice(-6)}</span>
                <button class="remove-seed-btn" onclick="SubgraphTools.removeSeed('${id}')" title="Remove">×</button>
            </div>
        `).join('');
    }
    
    // ==========================================================================
    // SUBGRAPH EXTRACTION
    // ==========================================================================
    
    function extractSubgraph() {
        const mode = document.getElementById('subgraph-mode')?.value || 'neighborhood';
        const hops = parseInt(document.getElementById('subgraph-hops')?.value) || 1;
        const directed = document.getElementById('subgraph-directed')?.checked ?? false;
        
        // Get center node from input
        state.centerNode = document.getElementById('subgraph-center-input')?.value?.trim() || null;
        
        // Validate inputs based on mode
        if (mode === 'induced') {
            if (state.seedNodes.length === 0 && !state.centerNode) {
                showToast('Add seed nodes or specify a center node', 'error');
                return;
            }
        } else if (!state.centerNode) {
            showToast('Please specify a center node', 'error');
            return;
        }
        
        // Build node list
        const nodes = [...state.seedNodes];
        if (state.centerNode && !nodes.includes(state.centerNode)) {
            nodes.unshift(state.centerNode);
        }
        
        const graphSelect = document.getElementById('graph-select');
        const graphName = graphSelect?.value || null;
        
        let url, body;
        
        // Choose endpoint based on mode
        if (mode === 'neighborhood' || mode === 'ego' || mode === 'k_hop') {
            url = '/api/algorithms/subgraph';
            if (graphName) url += `?graph_name=${encodeURIComponent(graphName)}`;
            body = {
                nodes: nodes,
                mode: mode,
                hops: hops,
                directed: directed,
            };
        } else if (mode === 'component') {
            url = `/api/algorithms/component/${encodeURIComponent(state.centerNode)}`;
            if (graphName) url += `?graph_name=${encodeURIComponent(graphName)}`;
            url += `${graphName ? '&' : '?'}directed=${directed}`;
            body = null;
        } else if (mode === 'induced') {
            url = '/api/algorithms/subgraph';
            if (graphName) url += `?graph_name=${encodeURIComponent(graphName)}`;
            body = {
                nodes: nodes,
                mode: 'induced',
                directed: directed,
            };
        }
        
        console.log('[SubgraphTools] Extracting:', mode, body || url);
        showLoading(true);
        
        // Non-blocking: yield to event loop, then fetch
        setTimeout(() => {
            const fetchOptions = body ? {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            } : {};
            
            fetch(url, fetchOptions)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.detail || 'Failed to extract subgraph');
                    });
                }
                return response.json();
            })
            .then(result => {
                console.log('[SubgraphTools] Result:', result);
                
                state.lastResult = result;
                
                // Yield before updating UI
                setTimeout(() => {
                    displayResults(result);
                    if (result.success && result.nodes?.length > 0) {
                        highlightSubgraph();
                    }
                    showLoading(false);
                }, 0);
            })
            .catch(error => {
                console.error('[SubgraphTools] Error:', error);
                showToast(error.message, 'error');
                showLoading(false);
            });
        }, 0);
    }
    
    function displayResults(result) {
        const resultsSection = document.getElementById('subgraph-results');
        if (!resultsSection) return;
        
        resultsSection.style.display = 'block';
        
        document.getElementById('subgraph-node-count').textContent = result.node_count || 0;
        document.getElementById('subgraph-edge-count').textContent = result.edge_count || 0;
        document.getElementById('subgraph-time').textContent = 
            result.computation_time_ms ? `${result.computation_time_ms.toFixed(1)}ms` : '-';
        
        if (!result.success) {
            showToast(result.message || 'Extraction failed', 'error');
        }
    }
    
    // ==========================================================================
    // VISUALIZATION
    // ==========================================================================
    
    function highlightSubgraph() {
        if (!state.lastResult?.nodes) return;
        
        clearHighlights();
        
        const cy = getCytoscape();
        if (!cy) return;
        
        const subgraphNodes = new Set(state.lastResult.nodes);
        const centerNode = state.lastResult.center || state.centerNode;
        
        // Non-blocking: yield to event loop before heavy work
        setTimeout(() => {
            cy.batch(() => {
                // Highlight subgraph nodes
                subgraphNodes.forEach(nodeId => {
                    const node = cy.getElementById(nodeId);
                    if (node.length) {
                        const isCenter = nodeId === centerNode;
                        node.style({
                            'background-color': isCenter ? '#22c55e' : '#00d4ff',
                            'border-color': '#ffffff',
                            'border-width': isCenter ? 4 : 2,
                            'width': isCenter ? 35 : 25,
                            'height': isCenter ? 35 : 25,
                            'z-index': 9999,
                        });
                        state.highlightedElements.set(nodeId, node);
                    }
                });
            });
            
            // Highlight edges in chunks to avoid blocking
            if (state.lastResult.edges && state.lastResult.edges.length > 0) {
                highlightEdgesChunked(cy, state.lastResult.edges, 0);
            } else {
                finishHighlight(cy, subgraphNodes);
            }
        }, 0);
        
        // Show details in info panel
        showSubgraphDetails();
    }
    
    function highlightEdgesChunked(cy, edges, startIndex) {
        const CHUNK_SIZE = 100;
        const endIndex = Math.min(startIndex + CHUNK_SIZE, edges.length);
        
        cy.batch(() => {
            for (let i = startIndex; i < endIndex; i++) {
                const edge = edges[i];
                let cyEdge = cy.getElementById(`${edge.source}-${edge.target}`);
                if (!cyEdge.length) {
                    cyEdge = cy.edges().filter(e => 
                        (e.source().id() === edge.source && e.target().id() === edge.target) ||
                        (e.source().id() === edge.target && e.target().id() === edge.source)
                    ).first();
                }
                
                if (cyEdge.length) {
                    cyEdge.style({
                        'line-color': '#00d4ff',
                        'width': 3,
                        'opacity': 1,
                        'z-index': 9998,
                    });
                    state.highlightedElements.set(cyEdge.id(), cyEdge);
                }
            }
        });
        
        if (endIndex < edges.length) {
            // More edges to process - yield and continue
            setTimeout(() => highlightEdgesChunked(cy, edges, endIndex), 0);
        } else {
            // Done - fit to view
            const subgraphNodes = new Set(state.lastResult.nodes);
            finishHighlight(cy, subgraphNodes);
        }
    }
    
    function finishHighlight(cy, subgraphNodes) {
        const nodes = cy.nodes().filter(n => subgraphNodes.has(n.id()));
        if (nodes.length > 0 && nodes.length < 1000) {
            cy.animate({
                fit: { eles: nodes, padding: 80 },
                duration: 500,
            });
        }
        showToast(`Highlighting ${subgraphNodes.size} nodes`, 'info');
    }
    
    function showSubgraphDetails() {
        const result = state.lastResult;
        if (!result) return;
        
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
        if (headerTitle) headerTitle.textContent = 'Subgraph Details';
        
        const modeLabels = {
            'neighborhood': 'Neighborhood',
            'ego': 'Ego Graph',
            'component': 'Component',
            'induced': 'Induced',
            'k_hop': 'K-Hop',
        };
        
        const centerNode = result.center || state.centerNode;
        
        let html = `
            <div id="subgraph-detail-panel">
                <div class="path-stats">
                    <div class="path-stat">
                        <div class="path-stat-label">Mode</div>
                        <div class="path-stat-value">${modeLabels[result.mode] || result.mode}</div>
                    </div>
                    <div class="path-stat">
                        <div class="path-stat-label">Nodes</div>
                        <div class="path-stat-value">${result.node_count || 0}</div>
                    </div>
                </div>
                <div class="path-stats">
                    <div class="path-stat">
                        <div class="path-stat-label">Edges</div>
                        <div class="path-stat-value">${result.edge_count || 0}</div>
                    </div>
                    <div class="path-stat">
                        <div class="path-stat-label">Hops</div>
                        <div class="path-stat-value">${result.hops || 1}</div>
                    </div>
                </div>
        `;
        
        if (centerNode) {
            html += `
                <div class="path-node-list-header">Center Node</div>
                <div class="path-node-list">
                    <div class="path-node-item" onclick="SubgraphTools.zoomToNode('${centerNode}')" title="Click to zoom">
                        <span class="path-node-hop source">C</span>
                        <span class="path-node-id">${centerNode}</span>
                    </div>
                </div>
            `;
        }
        
        html += `
                <div class="path-actions">
                    <button onclick="SubgraphTools.highlightSubgraph()">Highlight</button>
                    <button onclick="SubgraphTools.isolateSubgraph()" class="isolate-btn">Isolate</button>
                </div>
                <div class="path-actions">
                    <button onclick="SubgraphTools.showAllNodes()" class="show-all-btn">Show All</button>
                    <button onclick="SubgraphTools.exportSubgraph()">Export JSON</button>
                </div>
            </div>
        `;
        
        // Remove existing detail panels
        document.getElementById('subgraph-detail-panel')?.remove();
        document.getElementById('path-detail-panel')?.remove();
        document.getElementById('flow-detail-panel')?.remove();
        
        // Insert after header
        const infoHeader = infoPanel.querySelector('.info-header');
        if (infoHeader) {
            infoHeader.insertAdjacentHTML('afterend', html);
        } else {
            infoPanel.insertAdjacentHTML('afterbegin', html);
        }
        
        infoPanel.style.display = 'block';
    }
    
    function hideSubgraphDetails() {
        document.getElementById('subgraph-detail-panel')?.remove();
        
        // Restore original header title and show hidden sections
        const infoPanel = document.getElementById('info-panel');
        if (infoPanel) {
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
    }
    
    function zoomToNode(nodeId) {
        const cy = getCytoscape();
        if (!cy) return;
        
        const node = cy.getElementById(nodeId);
        if (node.length) {
            cy.animate({
                center: { eles: node },
                zoom: 2,
                duration: 300,
            });
        }
    }
    
    function isolateSubgraph() {
        if (!state.lastResult?.nodes) return;
        
        const cy = getCytoscape();
        if (!cy) return;
        
        const subgraphNodes = new Set(state.lastResult.nodes);
        
        // Non-blocking: yield to event loop
        setTimeout(() => {
            cy.batch(() => {
                // Hide nodes not in subgraph
                cy.nodes().forEach(node => {
                    if (!subgraphNodes.has(node.id())) {
                        node.style('display', 'none');
                    }
                });
                
                // Hide edges not within subgraph
                cy.edges().forEach(edge => {
                    const src = edge.source().id();
                    const tgt = edge.target().id();
                    if (!subgraphNodes.has(src) || !subgraphNodes.has(tgt)) {
                        edge.style('display', 'none');
                    }
                });
            });
            
            state.isIsolated = true;
            
            // Fit to visible
            const visible = cy.nodes().filter(n => subgraphNodes.has(n.id()));
            if (visible.length > 0) {
                cy.animate({
                    fit: { eles: visible, padding: 50 },
                    duration: 500
                });
            }
            
            showToast(`Isolated ${subgraphNodes.size} nodes`, 'success');
        }, 0);
    }
    
    function showAllNodes() {
        const cy = getCytoscape();
        if (!cy) return;
        
        // Non-blocking: yield to event loop
        setTimeout(() => {
            cy.batch(() => {
                cy.nodes().style('display', 'element');
                cy.edges().style('display', 'element');
            });
            
            state.isIsolated = false;
            cy.fit(50);
            
            showToast('Showing all nodes', 'success');
        }, 0);
    }
    
    function clearHighlights() {
        const cy = getCytoscape();
        if (!cy) return;
        
        cy.batch(() => {
            state.highlightedElements.forEach((ele, id) => {
                const element = cy.getElementById(id);
                if (element.length) {
                    if (element.isNode()) {
                        element.removeStyle('background-color border-color border-width width height z-index');
                    } else if (element.isEdge()) {
                        element.removeStyle('line-color line-style target-arrow-color source-arrow-color width opacity z-index');
                    }
                }
            });
        });
        
        state.highlightedElements.clear();
    }
    
    function clearResults() {
        state.lastResult = null;
        clearHighlights();
        
        if (state.isIsolated) {
            showAllNodes();
        }
        
        const resultsSection = document.getElementById('subgraph-results');
        if (resultsSection) resultsSection.style.display = 'none';
        
        // Also clear the info panel
        hideSubgraphDetails();
    }
    
    // ==========================================================================
    // EXPORT
    // ==========================================================================
    
    function exportSubgraph() {
        if (!state.lastResult) {
            showToast('No subgraph to export', 'warning');
            return;
        }
        
        const data = {
            mode: state.lastResult.mode,
            center: state.lastResult.center,
            nodes: state.lastResult.nodes,
            edges: state.lastResult.edges,
            node_count: state.lastResult.node_count,
            edge_count: state.lastResult.edge_count,
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `subgraph_${state.lastResult.mode}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        showToast('Subgraph exported', 'success');
    }
    
    // ==========================================================================
    // UTILITIES
    // ==========================================================================
    
    function showLoading(show) {
        const btn = document.getElementById('extract-subgraph-btn');
        if (btn) {
            btn.disabled = show;
            btn.textContent = show ? 'Extracting...' : 'Extract';
        }
    }
    
    function showToast(message, type = 'info') {
        if (typeof Toast !== 'undefined') {
            if (type === 'error' && Toast.error) Toast.error(message);
            else if (type === 'success' && Toast.success) Toast.success(message);
            else if (type === 'warning' && Toast.warning) Toast.warning(message);
            else if (Toast.info) Toast.info(message);
            else console.log(`[${type}] ${message}`);
        } else {
            console.log(`[SubgraphTools] [${type}] ${message}`);
        }
    }
    
    // ==========================================================================
    // PUBLIC API
    // ==========================================================================
    
    return {
        init,
        extractSubgraph,
        highlightSubgraph,
        isolateSubgraph,
        showAllNodes,
        clearResults,
        clearHighlights,
        setCenter,
        addSelectedAsSeeds,
        removeSeed,
        clearSeeds,
        onNodeClick,
        startPicking,
        cancelPicking,
        exportSubgraph,
        zoomToNode,
        getState: () => ({ ...state }),
        isPickingNode: () => state.pickMode !== null,
    };
})();

// Make available globally
window.SubgraphTools = SubgraphTools;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => SubgraphTools.init());
} else {
    setTimeout(() => SubgraphTools.init(), 0);
}