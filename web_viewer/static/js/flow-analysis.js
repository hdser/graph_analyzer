/**
 * Flow Analysis Module
 * 
 * Network flow analysis: maximum flow, minimum cut,
 * edge connectivity, and node connectivity.
 * 
 * Location: web_viewer/static/js/flow-analysis.js
 */

const FlowAnalysis = (function() {
    'use strict';
    
    // State
    let state = {
        sourceNode: null,
        targetNode: null,
        pickMode: null,
        lastResult: null,
        analysisType: 'max_flow',
        isIsolated: false,
        highlightedElements: new Map(),
    };
    
    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
    
    function init() {
        console.log('[FlowAnalysis] Initializing...');
        
        const panel = document.getElementById('panel-flow');
        if (!panel) {
            console.warn('[FlowAnalysis] Panel not found');
            return;
        }
        
        // Analysis type selection
        document.getElementById('flow-analysis-type')?.addEventListener('change', onAnalysisTypeChange);
        
        // Source input
        const sourceInput = document.getElementById('flow-source-input');
        if (sourceInput) {
            sourceInput.addEventListener('input', (e) => {
                state.sourceNode = e.target.value.trim() || null;
            });
        }
        
        // Target input
        const targetInput = document.getElementById('flow-target-input');
        if (targetInput) {
            targetInput.addEventListener('input', (e) => {
                state.targetNode = e.target.value.trim() || null;
            });
        }
        
        // Swap button
        document.getElementById('swap-flow-nodes-btn')?.addEventListener('click', swapNodes);
        
        // Action buttons
        document.getElementById('compute-flow-btn')?.addEventListener('click', computeFlow);
        document.getElementById('clear-flow-btn')?.addEventListener('click', clearResults);
        
        // Result action buttons
        document.getElementById('highlight-flow-btn')?.addEventListener('click', highlightFlow);
        document.getElementById('isolate-flow-btn')?.addEventListener('click', isolateFlow);
        document.getElementById('show-partition-btn')?.addEventListener('click', showPartition);
        document.getElementById('show-all-flow-btn')?.addEventListener('click', showAllNodes);
        
        // Populate capacity dropdown
        populateCapacityDropdown();
        
        // Update capacity dropdown when graph changes
        document.getElementById('graph-select')?.addEventListener('change', populateCapacityDropdown);
        
        // Initial setup
        onAnalysisTypeChange();
        
        console.log('[FlowAnalysis] Initialized');
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
    // ANALYSIS TYPE HANDLING
    // ==========================================================================
    
    function onAnalysisTypeChange() {
        const type = document.getElementById('flow-analysis-type')?.value || 'max_flow';
        state.analysisType = type;
        
        const capacityRow = document.getElementById('flow-capacity-row');
        const partitionBtn = document.getElementById('show-partition-btn');
        
        // Show capacity for max_flow and min_cut
        if (capacityRow) {
            capacityRow.style.display = (type === 'max_flow' || type === 'min_cut') ? 'flex' : 'none';
        }
        
        // Partition button only for min_cut
        if (partitionBtn) {
            partitionBtn.style.display = type === 'min_cut' ? 'inline-block' : 'none';
        }
    }
    
    function populateCapacityDropdown() {
        const select = document.getElementById('flow-capacity-attr');
        if (!select) return;
        
        // Get graph name
        const graphSelect = document.getElementById('graph-select');
        const graphName = graphSelect?.value || null;
        
        let url = '/api/algorithms/edge-attributes';
        if (graphName) url += `?graph_name=${encodeURIComponent(graphName)}`;
        
        // Fetch edge attributes from API
        fetch(url)
        .then(response => response.json())
        .then(data => {
            // Clear existing options
            select.innerHTML = '<option value="">Unit Capacity (1.0)</option>';
            
            // Add numeric attributes
            if (data.numeric_attributes) {
                data.numeric_attributes.forEach(attr => {
                    select.innerHTML += `<option value="${attr}">${attr}</option>`;
                });
            }
        })
        .catch(err => {
            console.warn('[FlowAnalysis] Could not fetch edge attributes:', err);
            // Fallback: just keep Unit Capacity option
            select.innerHTML = '<option value="">Unit Capacity (1.0)</option>';
        });
    }
    
    // ==========================================================================
    // NODE PICKING
    // ==========================================================================
    
    function startPicking(mode) {
        state.pickMode = mode;
        
        const btn = mode === 'source' 
            ? document.getElementById('pick-flow-source-btn')
            : document.getElementById('pick-flow-target-btn');
        
        if (btn) btn.classList.add('picking');
        
        showToast(`Click a node to set as ${mode}`, 'info');
        
        const cyDiv = document.getElementById('cy');
        if (cyDiv) cyDiv.style.cursor = 'crosshair';
    }
    
    function onNodeClick(nodeId) {
        if (!state.pickMode) return false;
        
        if (state.pickMode === 'source') {
            setSource(nodeId);
        } else if (state.pickMode === 'target') {
            setTarget(nodeId);
        }
        
        cancelPicking();
        return true;
    }
    
    function cancelPicking() {
        state.pickMode = null;
        document.querySelectorAll('#panel-flow .mini-icon-btn.picking').forEach(btn => {
            btn.classList.remove('picking');
        });
        const cyDiv = document.getElementById('cy');
        if (cyDiv) cyDiv.style.cursor = '';
    }
    
    function setSource(nodeId) {
        state.sourceNode = nodeId;
        const input = document.getElementById('flow-source-input');
        if (input) input.value = nodeId;
    }
    
    function setTarget(nodeId) {
        state.targetNode = nodeId;
        const input = document.getElementById('flow-target-input');
        if (input) input.value = nodeId;
    }
    
    function swapNodes() {
        const temp = state.sourceNode;
        setSource(state.targetNode || '');
        setTarget(temp || '');
    }
    
    // ==========================================================================
    // FLOW COMPUTATION
    // ==========================================================================
    
    function computeFlow() {
        // Get values from inputs
        state.sourceNode = document.getElementById('flow-source-input')?.value?.trim() || null;
        state.targetNode = document.getElementById('flow-target-input')?.value?.trim() || null;
        
        if (!state.sourceNode || !state.targetNode) {
            showToast('Please specify both source and target nodes', 'error');
            return;
        }
        
        const type = document.getElementById('flow-analysis-type')?.value || 'max_flow';
        const capacity = document.getElementById('flow-capacity-attr')?.value || '';
        
        const graphSelect = document.getElementById('graph-select');
        const graphName = graphSelect?.value || null;
        
        let url, body;
        
        switch (type) {
            case 'max_flow':
                url = '/api/algorithms/max-flow';
                body = {
                    source: state.sourceNode,
                    target: state.targetNode,
                    capacity: capacity,  // Empty string = unit capacity
                };
                break;
            case 'min_cut':
                url = '/api/algorithms/min-cut';
                body = {
                    source: state.sourceNode,
                    target: state.targetNode,
                    capacity: capacity,  // Empty string = unit capacity
                };
                break;
            case 'edge_connectivity':
                url = `/api/algorithms/edge-connectivity?source=${encodeURIComponent(state.sourceNode)}&target=${encodeURIComponent(state.targetNode)}`;
                body = null;
                break;
            case 'node_connectivity':
                url = `/api/algorithms/node-connectivity?source=${encodeURIComponent(state.sourceNode)}&target=${encodeURIComponent(state.targetNode)}`;
                body = null;
                break;
        }
        
        if (graphName) {
            url += (url.includes('?') ? '&' : '?') + `graph_name=${encodeURIComponent(graphName)}`;
        }
        
        console.log('[FlowAnalysis] Computing:', type, body || url);
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
                        throw new Error(err.detail || 'Failed to compute flow');
                    });
                }
                return response.json();
            })
            .then(result => {
                console.log('[FlowAnalysis] Result:', result);
                
                state.lastResult = result;
                state.lastResult.type = type;
                
                // Yield before updating UI
                setTimeout(() => {
                    displayResults(result, type);
                    if (result.success) {
                        highlightFlow();
                    }
                    showLoading(false);
                }, 0);
            })
            .catch(error => {
                console.error('[FlowAnalysis] Error:', error);
                showToast(error.message, 'error');
                showLoading(false);
            });
        }, 0);
    }
    
    function displayResults(result, type) {
        const resultsSection = document.getElementById('flow-results');
        if (!resultsSection) return;
        
        resultsSection.style.display = 'block';
        
        // Update header
        const typeLabels = {
            'max_flow': 'Maximum Flow',
            'min_cut': 'Minimum Cut',
            'edge_connectivity': 'Edge Connectivity',
            'node_connectivity': 'Node Connectivity',
        };
        
        document.getElementById('flow-result-type').textContent = typeLabels[type] || type;
        document.getElementById('flow-result-time').textContent = 
            result.computation_time_ms ? `${result.computation_time_ms.toFixed(1)}ms` : '';
        
        // Update value based on type
        const valueLabel = document.getElementById('flow-value-label');
        const valueEl = document.getElementById('flow-value');
        const edgeCountEl = document.getElementById('flow-edge-count');
        const edgesLabel = document.getElementById('flow-edges-label');
        const edgesSection = document.getElementById('flow-edges-section');
        const partitionSection = document.getElementById('partition-section');
        
        switch (type) {
            case 'max_flow':
                valueLabel.textContent = 'Flow Value';
                valueEl.textContent = result.flow_value?.toFixed(2) || '0';
                edgesLabel.textContent = 'Flow Edges';
                edgeCountEl.textContent = result.flow_edges?.length || 0;
                edgesSection.style.display = 'block';
                partitionSection.style.display = 'none';
                displayFlowEdges(result.flow_edges, 'flow');
                break;
                
            case 'min_cut':
                valueLabel.textContent = 'Cut Value';
                valueEl.textContent = result.cut_value?.toFixed(2) || '0';
                edgesLabel.textContent = 'Cut Edges';
                edgeCountEl.textContent = result.cut_edges?.length || 0;
                edgesSection.style.display = 'block';
                partitionSection.style.display = 'block';
                displayFlowEdges(result.cut_edges, 'cut');
                displayPartitionStats(result.partition);
                break;
                
            case 'edge_connectivity':
            case 'node_connectivity':
                valueLabel.textContent = 'Connectivity';
                valueEl.textContent = result.connectivity || '0';
                edgesSection.style.display = 'none';
                partitionSection.style.display = 'none';
                break;
        }
        
        // Show partition button for min_cut
        const partitionBtn = document.getElementById('show-partition-btn');
        if (partitionBtn) {
            partitionBtn.style.display = type === 'min_cut' ? 'inline-block' : 'none';
        }
        
        if (!result.success) {
            showToast(result.message || 'Computation failed', 'error');
        }
    }
    
    function displayFlowEdges(edges, type) {
        const list = document.getElementById('flow-edges-list');
        const title = document.getElementById('flow-edges-title');
        if (!list) return;
        
        title.textContent = type === 'flow' ? 'Flow Edges' : 'Cut Edges';
        
        if (!edges || edges.length === 0) {
            list.innerHTML = '<div class="empty-list">No edges</div>';
            return;
        }
        
        // Limit display to first 50
        const displayEdges = edges.slice(0, 50);
        
        list.innerHTML = displayEdges.map(edge => {
            const srcShort = edge.source.slice(0, 8) + '...';
            const tgtShort = edge.target.slice(0, 8) + '...';
            const flowVal = edge.flow !== undefined ? `: ${edge.flow.toFixed(2)}` : '';
            
            return `
                <div class="flow-edge-item" title="${edge.source} → ${edge.target}">
                    <span class="edge-route">${srcShort} → ${tgtShort}</span>
                    <span class="edge-flow">${flowVal}</span>
                </div>
            `;
        }).join('');
        
        if (edges.length > 50) {
            list.innerHTML += `<div class="more-edges">... and ${edges.length - 50} more</div>`;
        }
    }
    
    function displayPartitionStats(partition) {
        if (!partition) return;
        
        const [sourceSide, targetSide] = partition;
        
        document.getElementById('partition-source-count').textContent = 
            sourceSide?.length || 0;
        document.getElementById('partition-target-count').textContent = 
            targetSide?.length || 0;
    }
    
    // ==========================================================================
    // VISUALIZATION
    // ==========================================================================
    
    function highlightFlow() {
        if (!state.lastResult) return;
        
        clearHighlights();
        
        const cy = getCytoscape();
        if (!cy) return;
        
        const type = state.lastResult.type;
        
        // Non-blocking: yield to event loop before heavy work
        setTimeout(() => {
            cy.batch(() => {
                // Highlight source and target
                const sourceNode = cy.getElementById(state.sourceNode);
                const targetNode = cy.getElementById(state.targetNode);
                
                if (sourceNode.length) {
                    sourceNode.style({
                        'background-color': '#22c55e',
                        'border-color': '#ffffff',
                        'border-width': 4,
                        'width': 40,
                        'height': 40,
                        'z-index': 9999,
                    });
                    state.highlightedElements.set(state.sourceNode, sourceNode);
                }
                
                if (targetNode.length) {
                    targetNode.style({
                        'background-color': '#ef4444',
                        'border-color': '#ffffff',
                        'border-width': 4,
                        'width': 40,
                        'height': 40,
                        'z-index': 9999,
                    });
                    state.highlightedElements.set(state.targetNode, targetNode);
                }
            });
            
            // Highlight edges in chunks to avoid blocking
            const edges = type === 'max_flow' ? state.lastResult.flow_edges : 
                         type === 'min_cut' ? state.lastResult.cut_edges : [];
            
            if (edges && edges.length > 0) {
                highlightEdgesChunked(cy, edges, type, 0);
            } else {
                finishFlowHighlight(cy);
            }
            
            // Show details in info panel
            showFlowDetails();
        }, 0);
    }
    
    function highlightEdgesChunked(cy, edges, type, startIndex) {
        const CHUNK_SIZE = 50;
        const endIndex = Math.min(startIndex + CHUNK_SIZE, edges.length);
        
        // Find max flow for normalization (only once)
        const maxFlow = type === 'max_flow' ? 
            Math.max(...edges.map(e => e.flow || 1), 1) : 1;
        
        cy.batch(() => {
            for (let i = startIndex; i < endIndex; i++) {
                const edge = edges[i];
                
                // Try to find existing edge using selectors (like path analysis)
                let cyEdge = cy.edges(`[source="${edge.source}"][target="${edge.target}"]`);
                if (!cyEdge.length) {
                    cyEdge = cy.edges(`[source="${edge.target}"][target="${edge.source}"]`);
                }
                
                if (cyEdge.length) {
                    // Style existing edge
                    if (type === 'max_flow') {
                        const norm = (edge.flow || 1) / maxFlow;
                        const width = 2 + norm * 6;
                        cyEdge.style({
                            'line-color': '#00d4ff',
                            'target-arrow-color': '#00d4ff',
                            'source-arrow-color': '#00d4ff',
                            'width': width,
                            'opacity': 1,
                            'z-index': 9998,
                        });
                    } else {
                        cyEdge.style({
                            'line-color': '#ef4444',
                            'line-style': 'dashed',
                            'width': 5,
                            'opacity': 1,
                            'z-index': 9998,
                        });
                    }
                    state.highlightedElements.set(cyEdge.id(), cyEdge);
                } else {
                    // Create temporary edge (like path analysis does)
                    const tempEdgeId = `flow-edge-${i}`;
                    try {
                        cy.add({
                            group: 'edges',
                            data: {
                                id: tempEdgeId,
                                source: edge.source,
                                target: edge.target,
                                _flowTemp: true
                            }
                        });
                        
                        const newEdge = cy.getElementById(tempEdgeId);
                        if (newEdge.length) {
                            if (type === 'max_flow') {
                                const norm = (edge.flow || 1) / maxFlow;
                                const width = 2 + norm * 6;
                                newEdge.style({
                                    'line-color': '#00d4ff',
                                    'target-arrow-color': '#00d4ff',
                                    'width': width,
                                    'curve-style': 'bezier',
                                    'line-style': 'solid',
                                    'opacity': 1,
                                    'z-index': 9998,
                                });
                            } else {
                                newEdge.style({
                                    'line-color': '#ef4444',
                                    'line-style': 'dashed',
                                    'width': 5,
                                    'curve-style': 'bezier',
                                    'opacity': 1,
                                    'z-index': 9998,
                                });
                            }
                            state.highlightedElements.set(tempEdgeId, { _isTemp: true });
                        }
                    } catch (e) {
                        console.warn(`[FlowAnalysis] Could not create edge: ${edge.source} -> ${edge.target}`, e);
                    }
                }
                
                // Also highlight connected nodes (except source/target)
                [edge.source, edge.target].forEach(nodeId => {
                    if (nodeId !== state.sourceNode && nodeId !== state.targetNode && 
                        !state.highlightedElements.has(nodeId)) {
                        const node = cy.getElementById(nodeId);
                        if (node.length) {
                            node.style({
                                'background-color': '#00d4ff',
                                'border-color': '#ffffff',
                                'border-width': 2,
                                'width': 25,
                                'height': 25,
                                'z-index': 9997,
                            });
                            state.highlightedElements.set(nodeId, node);
                        }
                    }
                });
            }
        });
        
        if (endIndex < edges.length) {
            // More edges to process - yield and continue
            setTimeout(() => highlightEdgesChunked(cy, edges, type, endIndex), 0);
        } else {
            // Done - fit to view
            finishFlowHighlight(cy);
        }
    }
    
    function finishFlowHighlight(cy) {
        if (state.highlightedElements.size > 0) {
            const elements = cy.collection();
            state.highlightedElements.forEach(ele => elements.merge(ele));
            
            if (elements.length > 0 && elements.length < 500) {
                cy.animate({
                    fit: { eles: elements, padding: 80 },
                    duration: 500,
                });
            }
        }
    }
    
    function showFlowDetails() {
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
        if (headerTitle) headerTitle.textContent = 'Flow Analysis';
        
        const type = result.type;
        const typeLabels = {
            'max_flow': 'Maximum Flow',
            'min_cut': 'Minimum Cut',
            'edge_connectivity': 'Edge Connectivity',
            'node_connectivity': 'Node Connectivity',
        };
        
        const edges = type === 'max_flow' ? result.flow_edges : 
                     type === 'min_cut' ? result.cut_edges : [];
        
        const flowValue = type === 'max_flow' ? result.flow_value : 
                         type === 'min_cut' ? result.cut_value :
                         result.connectivity;
        
        let html = `
            <div id="flow-detail-panel">
                <div class="path-stats">
                    <div class="path-stat">
                        <div class="path-stat-label">${typeLabels[type] || type}</div>
                        <div class="path-stat-value">${typeof flowValue === 'number' ? flowValue.toFixed(2) : flowValue || '0'}</div>
                    </div>
                    <div class="path-stat">
                        <div class="path-stat-label">Edges</div>
                        <div class="path-stat-value">${edges?.length || 0}</div>
                    </div>
                </div>
                
                <div class="path-node-list-header">Flow Path</div>
                <div class="path-node-list">
                    <div class="path-node-item" onclick="FlowAnalysis.zoomToNode('${state.sourceNode}')" title="Click to zoom">
                        <span class="path-node-hop source">S</span>
                        <span class="path-node-id">${state.sourceNode}</span>
                    </div>
        `;
        
        // Flow edges (show like path)
        if (edges && edges.length > 0) {
            const displayEdges = edges.slice(0, 50);
            displayEdges.forEach((edge, i) => {
                const flowVal = edge.flow !== undefined ? ` (${edge.flow.toFixed(2)})` : '';
                html += `
                    <div class="path-edge-info">↓ ${type === 'max_flow' ? 'flow' : 'cut'}${flowVal}</div>
                    <div class="path-node-item" onclick="FlowAnalysis.zoomToEdge('${edge.source}', '${edge.target}')" title="Click to zoom">
                        <span class="path-node-hop">${i + 1}</span>
                        <span class="path-node-id">${edge.source.slice(0, 10)}...→ ${edge.target.slice(-10)}</span>
                    </div>
                `;
            });
            
            if (edges.length > 50) {
                html += `<div class="path-edge-info">... and ${edges.length - 50} more edges</div>`;
            }
        }
        
        html += `
                    <div class="path-edge-info">↓</div>
                    <div class="path-node-item" onclick="FlowAnalysis.zoomToNode('${state.targetNode}')" title="Click to zoom">
                        <span class="path-node-hop target">T</span>
                        <span class="path-node-id">${state.targetNode}</span>
                    </div>
                </div>
        `;
        
        // Partition info for min_cut
        if (type === 'min_cut' && result.partition) {
            const [sourceSide, targetSide] = result.partition;
            html += `
                <div class="path-stats" style="margin-top: 12px;">
                    <div class="path-stat" style="background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4);">
                        <div class="path-stat-label" style="color: #22c55e;">Source Side</div>
                        <div class="path-stat-value">${sourceSide.length}</div>
                    </div>
                    <div class="path-stat" style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4);">
                        <div class="path-stat-label" style="color: #ef4444;">Target Side</div>
                        <div class="path-stat-value">${targetSide.length}</div>
                    </div>
                </div>
            `;
        }
        
        html += `
                <div class="path-actions">
                    <button onclick="FlowAnalysis.fitToFlow()">Fit to Flow</button>
                    <button onclick="FlowAnalysis.isolateFlow()" class="isolate-btn">Isolate</button>
                </div>
                <div class="path-actions">
                    <button onclick="FlowAnalysis.showAllNodes()" class="show-all-btn">Show All</button>
                    ${type === 'min_cut' ? '<button onclick="FlowAnalysis.showPartition()">Show Partition</button>' : ''}
                </div>
            </div>
        `;
        
        // Remove existing detail panels
        document.getElementById('flow-detail-panel')?.remove();
        document.getElementById('path-detail-panel')?.remove();
        document.getElementById('subgraph-detail-panel')?.remove();
        
        // Insert after header
        const infoHeader = infoPanel.querySelector('.info-header');
        if (infoHeader) {
            infoHeader.insertAdjacentHTML('afterend', html);
        } else {
            infoPanel.insertAdjacentHTML('afterbegin', html);
        }
        
        infoPanel.style.display = 'block';
    }
    
    function hideFlowDetails() {
        document.getElementById('flow-detail-panel')?.remove();
        
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
        if (!cy || !nodeId) return;
        
        const node = cy.getElementById(nodeId);
        if (node.length) {
            cy.animate({
                center: { eles: node },
                zoom: 2,
                duration: 300,
            });
        }
    }
    
    function zoomToEdge(sourceId, targetId) {
        const cy = getCytoscape();
        if (!cy) return;
        
        const srcNode = cy.getElementById(sourceId);
        const tgtNode = cy.getElementById(targetId);
        
        if (srcNode.length && tgtNode.length) {
            cy.animate({
                fit: { eles: srcNode.union(tgtNode), padding: 100 },
                duration: 300,
            });
        }
    }
    
    function fitToFlow() {
        const cy = getCytoscape();
        if (!cy || state.highlightedElements.size === 0) return;
        
        const elements = cy.collection();
        state.highlightedElements.forEach(ele => elements.merge(ele));
        
        if (elements.length > 0) {
            cy.animate({
                fit: { eles: elements, padding: 80 },
                duration: 500,
            });
        }
    }
    
    function highlightFlowEdges(cy, flowEdges) {
        // This is now handled by highlightEdgesChunked
    }
    
    function highlightCutEdges(cy, cutEdges) {
        // This is now handled by highlightEdgesChunked
    }
    
    function showPartition() {
        if (!state.lastResult?.partition) {
            showToast('No partition data available', 'warning');
            return;
        }
        
        const cy = getCytoscape();
        if (!cy) return;
        
        const [sourceSide, targetSide] = state.lastResult.partition;
        const sourceSet = new Set(sourceSide);
        const targetSet = new Set(targetSide);
        
        // Non-blocking: yield to event loop
        setTimeout(() => {
            cy.batch(() => {
                // Color source side green, target side red
                cy.nodes().forEach(node => {
                    const id = node.id();
                    if (sourceSet.has(id)) {
                        node.style({
                            'background-color': 'rgba(34, 197, 94, 0.6)',
                            'border-color': '#22c55e',
                            'border-width': 2,
                        });
                    } else if (targetSet.has(id)) {
                        node.style({
                            'background-color': 'rgba(239, 68, 68, 0.6)',
                            'border-color': '#ef4444',
                            'border-width': 2,
                        });
                    }
                });
            });
            
            showToast(`Source side: ${sourceSide.length}, Target side: ${targetSide.length}`, 'info');
        }, 0);
    }
    
    function isolateFlow() {
        if (!state.lastResult) return;
        
        const cy = getCytoscape();
        if (!cy) return;
        
        // Collect all nodes involved
        const involvedNodes = new Set([state.sourceNode, state.targetNode]);
        
        const edges = state.lastResult.flow_edges || state.lastResult.cut_edges || [];
        edges.forEach(edge => {
            involvedNodes.add(edge.source);
            involvedNodes.add(edge.target);
        });
        
        // Non-blocking: yield to event loop
        setTimeout(() => {
            cy.batch(() => {
                cy.nodes().forEach(node => {
                    if (!involvedNodes.has(node.id())) {
                        node.style('display', 'none');
                    }
                });
                
                cy.edges().forEach(edge => {
                    const src = edge.source().id();
                    const tgt = edge.target().id();
                    if (!involvedNodes.has(src) || !involvedNodes.has(tgt)) {
                        edge.style('display', 'none');
                    }
                });
            });
            
            state.isIsolated = true;
            
            const visible = cy.nodes().filter(n => involvedNodes.has(n.id()));
            if (visible.length > 0) {
                cy.animate({
                    fit: { eles: visible, padding: 50 },
                    duration: 500
                });
            }
            
            showToast(`Isolated ${involvedNodes.size} nodes`, 'success');
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
            
            showToast('Showing all nodes', 'info');
        }, 0);
    }
    
    function clearHighlights() {
        const cy = getCytoscape();
        if (!cy) return;
        
        cy.batch(() => {
            // Remove temporary flow edges
            cy.edges('[_flowTemp]').remove();
            
            // Reset styled elements
            state.highlightedElements.forEach((value, id) => {
                if (value._isTemp) return; // Already removed above
                
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
        
        const resultsSection = document.getElementById('flow-results');
        if (resultsSection) resultsSection.style.display = 'none';
        
        // Also clear the info panel
        hideFlowDetails();
    }
    
    // ==========================================================================
    // UTILITIES
    // ==========================================================================
    
    function showLoading(show) {
        const btn = document.getElementById('compute-flow-btn');
        if (btn) {
            btn.disabled = show;
            btn.textContent = show ? 'Computing...' : 'Compute';
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
            console.log(`[FlowAnalysis] [${type}] ${message}`);
        }
    }
    
    // ==========================================================================
    // PUBLIC API
    // ==========================================================================
    
    return {
        init,
        computeFlow,
        highlightFlow,
        isolateFlow,
        showPartition,
        showAllNodes,
        clearResults,
        clearHighlights,
        setSource,
        setTarget,
        swapNodes,
        onNodeClick,
        startPicking,
        cancelPicking,
        populateCapacityDropdown,
        zoomToNode,
        zoomToEdge,
        fitToFlow,
        getState: () => ({ ...state }),
        isPickingNode: () => state.pickMode !== null,
    };
})();

// Make available globally
window.FlowAnalysis = FlowAnalysis;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => FlowAnalysis.init());
} else {
    setTimeout(() => FlowAnalysis.init(), 0);
}