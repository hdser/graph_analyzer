/**
 * Capacity Flow Module
 * 
 * Circles protocol capacity graph and max flow computation.
 * Handles capacity graph visualization, max flow analysis, and path highlighting.
 * 
 * Location: web_viewer/static/js/capacity-flow.js
 */

const CapacityFlow = (function() {
    'use strict';
    
    // ==========================================================================
    // STATE
    // ==========================================================================
    
    let state = {
        // Flow computation
        sourceNode: null,
        targetNode: null,
        lastResult: null,
        selectedPathIndex: -1,
        
        // Graph state
        graphBuilt: false,
        graphStats: null,
        capacityGraphLoaded: false,
        edgesLoaded: 0,
        totalEdges: 0,
        trustCount: 0,
        balanceCount: 0,
        
        // Visualization state
        originalStyles: new Map(),
        isIsolated: false,
        hiddenNodes: [],
        
        // Graph switching
        previousGraphId: null,
        isCapacityView: false,
        capacityGraphData: null,
        
        // Algorithms
        availableAlgorithms: [],
    };
    
    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
    
    function init() {
        console.log('[CapacityFlow] Initializing...');
        
        // Tab switching
        document.querySelectorAll('.flow-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                switchTab(this.dataset.tab);
            });
        });
        
        // Build button
        const buildBtn = document.getElementById('cf-build-btn');
        if (buildBtn) {
            buildBtn.addEventListener('click', buildCapacityGraph);
        }
        
        // View Graph button
        const viewBtn = document.getElementById('cf-view-graph-btn');
        if (viewBtn) {
            viewBtn.addEventListener('click', handleViewGraphClick);
        }
        
        // Load more edges button
        const loadEdgesBtn = document.getElementById('cf-load-edges-btn');
        if (loadEdgesBtn) {
            loadEdgesBtn.addEventListener('click', loadMoreEdges);
        }
        
        // Compute button
        const computeBtn = document.getElementById('cf-compute-btn');
        if (computeBtn) {
            computeBtn.addEventListener('click', computeMaxFlow);
        }
        
        // Clear button
        const clearBtn = document.getElementById('cf-clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', clearResults);
        }
        
        // Visualization buttons in results section
        const highlightBtn = document.getElementById('cf-highlight-btn');
        if (highlightBtn) {
            highlightBtn.addEventListener('click', highlightAllPaths);
        }
        
        const isolateBtn = document.getElementById('cf-isolate-btn');
        if (isolateBtn) {
            isolateBtn.addEventListener('click', isolateAllPaths);
        }
        
        const showAllBtn = document.getElementById('cf-show-all-btn');
        if (showAllBtn) {
            showAllBtn.addEventListener('click', showAllNodes);
        }
        
        // Load algorithms
        loadAlgorithms();
        
        // Check status
        checkStatus();
        
        console.log('[CapacityFlow] Initialized');
    }
    
    function switchTab(tabId) {
        document.querySelectorAll('.flow-tab').forEach(function(t) {
            t.classList.toggle('active', t.dataset.tab === tabId);
        });
        
        document.querySelectorAll('.flow-tab-content').forEach(function(c) {
            c.classList.toggle('active', c.id === tabId + '-tab');
        });
    }
    
    // ==========================================================================
    // RENDERER ACCESS
    // ==========================================================================
    
    /**
     * Get the current renderer (supports both Cytoscape and Cosmos)
     */
    function getRenderer() {
        if (typeof State !== 'undefined' && State.renderer) {
            return State.renderer;
        }
        console.warn('[CapacityFlow] Could not find renderer');
        return null;
    }
    
    /**
     * Get Cytoscape instance if available (for Cytoscape-specific operations)
     */
    function getCytoscape() {
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
     * Check if we're using Cytoscape renderer
     */
    function isCytoscapeRenderer() {
        return typeof State !== 'undefined' && State.rendererType === 'cytoscape';
    }
    
    /**
     * Check if we're using Cosmos renderer
     */
    function isCosmosRenderer() {
        return typeof State !== 'undefined' && State.rendererType === 'cosmos';
    }
    
    // ==========================================================================
    // API CALLS
    // ==========================================================================
    
    async function checkStatus() {
        try {
            const response = await fetch('/api/capacity-flow/status');
            const data = await response.json();
            
            if (!data.available) {
                showError('Capacity flow engine not available');
            }
        } catch (err) {
            console.error('[CapacityFlow] Status check failed:', err);
        }
    }
    
    async function loadAlgorithms() {
        try {
            const response = await fetch('/api/capacity-flow/algorithms');
            const data = await response.json();
            
            state.availableAlgorithms = data.algorithms || [];
            
            const algoSelect = document.getElementById('cf-algorithm-select');
            if (algoSelect && state.availableAlgorithms.length > 0) {
                algoSelect.innerHTML = '';
                state.availableAlgorithms.forEach(function(algo) {
                    const opt = document.createElement('option');
                    opt.value = algo.id;
                    // Use unicode checkmark instead of [ok] for better display
                    opt.textContent = algo.label;
                    opt.title = algo.description || '';
                    opt.dataset.backend = algo.backend;
                    opt.dataset.algorithm = algo.algorithm;
                    opt.dataset.supportsCutoff = algo.supports_cutoff;
                    algoSelect.appendChild(opt);
                });
                
                algoSelect.addEventListener('change', updateCutoffNote);
                updateCutoffNote();
            }
        } catch (err) {
            console.error('[CapacityFlow] Failed to load algorithms:', err);
        }
    }
    
    function updateCutoffNote() {
        const algoSelect = document.getElementById('cf-algorithm-select');
        const cutoffNote = document.getElementById('cf-cutoff-note');
        
        if (!algoSelect || !cutoffNote) return;
        
        const selected = algoSelect.selectedOptions[0];
        const supportsCutoff = selected && selected.dataset.supportsCutoff === 'true';
        
        cutoffNote.textContent = supportsCutoff ? '' : '(ignored)';
        cutoffNote.style.display = supportsCutoff ? 'none' : 'inline';
    }
    
    // ==========================================================================
    // BUILD CAPACITY GRAPH
    // ==========================================================================
    
    async function buildCapacityGraph() {
        const btn = document.getElementById('cf-build-btn');
        const status = document.getElementById('cf-status');
        
        try {
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Building...';
            }
            if (status) {
                status.textContent = 'Building';
                status.className = 'cf-badge building';
            }
            
            const response = await fetch('/api/capacity-flow/build', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    force_rebuild: true,
                    include_groups: true
                })
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Build failed');
            }
            
            const data = await response.json();
            console.log('[CapacityFlow] Build response:', data);
            
            state.graphBuilt = true;
            state.graphStats = data.stats;
            state.totalEdges = data.stats ? (data.stats.num_edges || 0) : 0;
            state.trustCount = data.trust_count || 0;
            state.balanceCount = data.balance_count || 0;
            
            if (status) {
                status.textContent = 'Ready';
                status.className = 'cf-badge built';
            }
            
            displayStats(data.stats, data);
            
            // Enable view graph button
            const viewBtn = document.getElementById('cf-view-graph-btn');
            if (viewBtn) {
                viewBtn.disabled = false;
            }
            
            showToast('Capacity graph built: ' + formatNumber(state.trustCount) + ' trusts, ' + formatNumber(state.balanceCount) + ' balances', 'success');
            
        } catch (err) {
            console.error('[CapacityFlow] Build failed:', err);
            if (status) {
                status.textContent = 'Failed';
                status.className = 'cf-badge not-built';
            }
            showError(err.message);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = 'Build Graph';
            }
        }
    }
    
    // ==========================================================================
    // VIEW / SWITCH GRAPH
    // ==========================================================================
    
    function handleViewGraphClick() {
        if (state.isCapacityView) {
            switchToTrustGraph();
        } else {
            viewCapacityGraph();
        }
    }
    
    async function viewCapacityGraph() {
        if (!state.graphBuilt) {
            showToast('Build the capacity graph first', 'warning');
            return;
        }
        
        const btn = document.getElementById('cf-view-graph-btn');
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Loading...';
        }
        
        try {
            // Save current graph ID for switching back
            if (typeof State !== 'undefined' && State.currentGraph) {
                state.previousGraphId = State.currentGraph;
            }
            
            console.log('[CapacityFlow] Fetching capacity graph nodes...');
            const response = await fetch('/api/capacity-flow/graph/nodes?use_trust_layout=true');
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || err.error || 'Failed to get nodes');
            }
            
            const data = await response.json();
            console.log('[CapacityFlow] Loaded', data.node_count, 'nodes,', data.positions_count, 'with positions');
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to get graph data');
            }
            
            // Store data for later
            state.capacityGraphData = data;
            
            // Create appropriate renderer for capacity graph size
            // Capacity graphs are typically large (20k+ nodes) so prefer CosmosGL
            const container = document.getElementById('cy');
            if (!container) {
                throw new Error('Graph container not found');
            }
            
            const nodeCount = data.node_count || data.nodes?.length || 0;
            console.log('[CapacityFlow] Creating renderer for', nodeCount, 'nodes');
            
            // Use RendererFactory to create appropriate renderer
            // Force cosmos for large graphs, or use auto selection
            let renderer;
            if (typeof RendererFactory !== 'undefined') {
                renderer = RendererFactory.create(container, {
                    expectedNodeCount: nodeCount,
                    // For capacity graphs, prefer cosmos since they're usually large
                    rendererPreference: nodeCount > 5000 ? 'cosmos' : State.rendererPreference
                });
                
                // Update state with new renderer
                State.setRenderer(renderer);
                
                console.log('[CapacityFlow] Created renderer:', renderer.getType());
            } else {
                // Fallback to existing renderer
                renderer = getRenderer();
                if (!renderer) {
                    throw new Error('Renderer not available');
                }
            }
            
            // Load nodes into the graph
            loadCapacityNodesIntoRenderer(renderer, data);
            
            // Update renderer indicator
            if (typeof GraphLoader !== 'undefined' && GraphLoader.updateRendererIndicator) {
                GraphLoader.updateRendererIndicator();
            }
            
            state.capacityGraphLoaded = true;
            state.isCapacityView = true;
            state.edgesLoaded = 0;
            state.totalEdges = data.edge_count || 0;
            
            updateEdgeLoadingUI();
            updateGraphSwitchUI();
            
            showToast('Loaded ' + formatNumber(data.node_count) + ' capacity nodes [' + renderer.getType() + ']', 'success');
            
        } catch (err) {
            console.error('[CapacityFlow] Failed to view graph:', err);
            showError(err.message);
        } finally {
            if (btn) {
                btn.disabled = false;
                updateGraphSwitchUI();
            }
        }
    }
    
    async function switchToTrustGraph() {
        if (!state.previousGraphId) {
            showToast('No previous graph to switch to', 'warning');
            return;
        }
        
        try {
            // Use GraphLoader to display the original graph
            if (typeof GraphLoader !== 'undefined' && GraphLoader.displayGraph) {
                await GraphLoader.displayGraph(state.previousGraphId);
                state.isCapacityView = false;
                updateGraphSwitchUI();
                showToast('Switched back to ' + state.previousGraphId, 'success');
            } else {
                showToast('GraphLoader not available', 'error');
            }
        } catch (err) {
            console.error('[CapacityFlow] Failed to switch graph:', err);
            showError(err.message);
        }
    }
    
    function updateGraphSwitchUI() {
        const btn = document.getElementById('cf-view-graph-btn');
        if (btn) {
            if (state.isCapacityView) {
                btn.textContent = 'Back to Trust';
            } else {
                btn.textContent = 'View Graph';
            }
        }
    }
    
    // ==========================================================================
    // LOAD EDGES
    // ==========================================================================
    
    async function loadMoreEdges() {
        if (!state.capacityGraphLoaded) {
            showToast('Load the graph first', 'warning');
            return;
        }
        
        const btn = document.getElementById('cf-load-edges-btn');
        var batchSize = 10000;
        
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Loading...';
        }
        
        try {
            const response = await fetch('/api/capacity-flow/graph/edges?offset=' + state.edgesLoaded + '&limit=' + batchSize);
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to load edges');
            }
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load edges');
            }
            
            addEdgesToCytoscape(data.edges);
            
            state.edgesLoaded += data.returned;
            
            updateEdgeLoadingUI();
            
        } catch (err) {
            console.error('[CapacityFlow] Failed to load edges:', err);
            showError(err.message);
        } finally {
            if (btn) {
                btn.disabled = state.edgesLoaded >= state.totalEdges;
                btn.textContent = state.edgesLoaded >= state.totalEdges ? 'All Loaded' : 'Load Edges';
            }
        }
    }
    
    function updateEdgeLoadingUI() {
        const container = document.getElementById('cf-edge-loading');
        const progress = document.getElementById('cf-edge-progress');
        const btn = document.getElementById('cf-load-edges-btn');
        
        if (container) {
            container.style.display = state.capacityGraphLoaded ? 'flex' : 'none';
        }
        
        if (progress) {
            progress.textContent = formatNumber(state.edgesLoaded) + ' / ' + formatNumber(state.totalEdges) + ' edges';
        }
        
        if (btn) {
            btn.disabled = state.edgesLoaded >= state.totalEdges;
            btn.textContent = state.edgesLoaded >= state.totalEdges ? 'All Loaded' : 'Load Edges';
        }
    }
    
    // ==========================================================================
    // CAPACITY GRAPH VISUALIZATION
    // ==========================================================================
    
    /**
     * Load capacity graph nodes into the renderer (works with both Cytoscape and Cosmos)
     */
    function loadCapacityNodesIntoRenderer(renderer, data) {
        console.log('[CapacityFlow] Loading', data.nodes ? data.nodes.length : 0, 'nodes into renderer');
        console.log('[CapacityFlow] Renderer type:', renderer.getType ? renderer.getType() : 'unknown');
        
        // Debug: show sample of raw data structure
        if (data.nodes && data.nodes.length > 0) {
            console.log('[CapacityFlow] Sample raw node structure:', JSON.stringify(data.nodes[0]).substring(0, 500));
        }
        
        // Color scheme for node types
        const nodeColors = {
            'avatar': '#4A90E2',
            'token_pool': '#9B59B6',
            'group': '#2ECC71',
            'virtual_sink': '#E74C3C'
        };
        
        // Check if we're using Cytoscape (needs special handling)
        if (isCytoscapeRenderer() && getCytoscape()) {
            console.log('[CapacityFlow] Using Cytoscape path');
            // Use the existing Cytoscape-specific function
            loadCapacityNodesIntoCytoscape(getCytoscape(), data);
            return;
        }
        
        console.log('[CapacityFlow] Using CosmosGL path');
        
        // For Cosmos or other renderers, use the renderer abstraction
        // Clear existing data
        renderer.clear();
        
        // Transform nodes for renderer - Cytoscape format has { data: { id, ... }, position: { x, y } }
        const nodes = data.nodes.map((node, index) => {
            // Extract node data - handle both Cytoscape format and flat format
            let nodeData;
            let nodeId;
            
            if (node.data && typeof node.data === 'object') {
                // Cytoscape format: { data: { id, type, ... }, position: { x, y } }
                nodeData = Object.assign({}, node.data);
                nodeId = node.data.id;
            } else {
                // Flat format: { id, type, x, y, ... }
                nodeData = Object.assign({}, node);
                nodeId = node.id;
            }
            
            // Ensure id is set - try multiple sources
            if (!nodeId) {
                nodeId = nodeData.id || nodeData.address || node.id || `node_${index}`;
            }
            nodeData.id = nodeId;
            
            // Add position if available
            if (node.position && node.position.x !== undefined && node.position.y !== undefined) {
                nodeData.x = node.position.x;
                nodeData.y = node.position.y;
            } else if (node.x !== undefined && node.y !== undefined) {
                nodeData.x = node.x;
                nodeData.y = node.y;
            }
            
            // Add color based on type
            nodeData._color = nodeColors[nodeData.type] || '#666666';
            
            return nodeData;
        });
        
        console.log('[CapacityFlow] Transformed', nodes.length, 'nodes for CosmosGL');
        
        // Debug: show sample transformed nodes
        if (nodes.length > 0) {
            console.log('[CapacityFlow] Sample transformed node IDs:', nodes.slice(0, 5).map(n => n.id));
        }
        
        // Load nodes into renderer (empty edges for now)
        renderer.setData(nodes, []);
        
        // Verify nodes were loaded correctly
        console.log('[CapacityFlow] Renderer node count:', renderer.nodeIds?.length);
        if (renderer.nodeIds?.length > 0) {
            console.log('[CapacityFlow] Renderer first 5 node IDs:', renderer.nodeIds.slice(0, 5));
        }
        
        // Apply node colors based on type
        applyCapacityNodeColors(renderer, nodes, nodeColors);
        
        // Setup event handlers for CosmosGL
        setupCosmosEventHandlers(renderer);
        
        // Fit view
        renderer.fitView();
        
        // Pause simulation after initial layout
        if (typeof renderer.pauseSimulation === 'function') {
            setTimeout(() => {
                renderer.pauseSimulation();
            }, 1000);
        }
        
        // Update header counts
        const nodeCountEl = document.getElementById('node-count');
        const edgeCountEl = document.getElementById('edge-count');
        if (nodeCountEl) nodeCountEl.textContent = nodes.length + ' nodes';
        if (edgeCountEl) edgeCountEl.textContent = '0 edges';
        
        console.log('[CapacityFlow] CosmosGL graph loaded successfully');
    }
    
    /**
     * Setup event handlers for CosmosGL renderer
     */
    function setupCosmosEventHandlers(renderer) {
        if (!renderer || !renderer.graph) return;
        
        console.log('[CapacityFlow] Setting up CosmosGL event handlers');
        
        // Node click handler
        renderer.graph.setConfig({
            onClick: (node, index, position, event) => {
                if (node) {
                    const nodeId = renderer.nodeIds[index];
                    const nodeData = renderer.nodeDataMap.get(nodeId);
                    
                    console.log('[CapacityFlow] Node clicked:', nodeId);
                    
                    // Use InfoPanel if available
                    if (typeof InfoPanel !== 'undefined' && InfoPanel.showNodeData) {
                        InfoPanel.showNodeData(nodeData);
                    }
                    
                    // Update selection
                    renderer.setSelectedNode(nodeId);
                }
            }
        });
    }
    
    /**
     * Apply capacity node colors based on node type
     */
    function applyCapacityNodeColors(renderer, nodes, colorMap) {
        if (!renderer || !renderer.graph) return;
        
        // Build color array for Cosmos
        const colors = new Float32Array(nodes.length * 4);
        
        nodes.forEach((node, index) => {
            const colorHex = colorMap[node.type] || '#666666';
            const rgba = RendererSettings.hexToRgba(colorHex);
            
            colors[index * 4] = rgba[0];
            colors[index * 4 + 1] = rgba[1];
            colors[index * 4 + 2] = rgba[2];
            colors[index * 4 + 3] = rgba[3];
        });
        
        // Apply colors to Cosmos
        if (typeof renderer.graph.setPointColors === 'function') {
            renderer.graph.setPointColors(colors);
            renderer.graph.render();
        }
    }
    
    function loadCapacityNodesIntoCytoscape(cy, data) {
        console.log('[CapacityFlow] Loading', data.nodes ? data.nodes.length : 0, 'nodes into Cytoscape');
        
        // Clear existing elements
        cy.elements().remove();
        
        // Color scheme for node types
        var nodeColors = {
            'avatar': '#4A90E2',
            'token_pool': '#9B59B6',
            'group': '#2ECC71',
            'virtual_sink': '#E74C3C'
        };
        
        // Add nodes
        var cyNodes = [];
        for (var i = 0; i < data.nodes.length; i++) {
            var node = data.nodes[i];
            var nodeData = {
                data: Object.assign({}, node.data)
            };
            
            if (node.position && node.position.x !== undefined && node.position.y !== undefined) {
                nodeData.position = { x: node.position.x, y: node.position.y };
            }
            
            cyNodes.push(nodeData);
        }
        
        cy.add(cyNodes);
        
        // Apply style - no labels for clean display
        cy.style()
            .selector('node')
            .style({
                'background-color': function(ele) { return nodeColors[ele.data('type')] || '#666'; },
                'label': '',
                'width': 12,
                'height': 12
            })
            .selector('node[type="token_pool"]')
            .style({
                'shape': 'diamond',
                'width': 8,
                'height': 8,
                'opacity': 0.7
            })
            .selector('node[type="group"]')
            .style({
                'shape': 'hexagon',
                'width': 18,
                'height': 18
            })
            .selector('node:selected')
            .style({
                'border-width': 3,
                'border-color': '#fff'
            })
            .update();
        
        // Run layout if needed
        if (!data.has_positions || data.positions_count < data.node_count / 2) {
            console.log('[CapacityFlow] Running layout...');
            cy.layout({
                name: 'cose',
                animate: false,
                nodeRepulsion: function(node) {
                    return node.data('type') === 'token_pool' ? 800000 : 400000;
                },
                nodeOverlap: 30,
                idealEdgeLength: 80,
                randomize: false
            }).run();
        } else {
            // Nudge token pools away from avatars
            nudgeTokenPools(cy);
        }
        
        cy.fit(50);
        
        // Update header counts
        var nodeCountEl = document.getElementById('node-count');
        var edgeCountEl = document.getElementById('edge-count');
        if (nodeCountEl) nodeCountEl.textContent = cy.nodes().length + ' nodes';
        if (edgeCountEl) edgeCountEl.textContent = '0 edges';
        
        // Setup click handler to show node info using InfoPanel
        setupNodeClickHandler(cy);
    }
    
    function setupNodeClickHandler(cy) {
        // Remove any existing tap handlers
        cy.off('tap', 'node');
        
        // Add handler that uses the standard InfoPanel
        cy.on('tap', 'node', function(evt) {
            var node = evt.target;
            
            // Use the standard InfoPanel if available
            if (typeof InfoPanel !== 'undefined' && InfoPanel.showNode) {
                InfoPanel.showNode(node);
            } else {
                console.log('[CapacityFlow] Node clicked:', node.data());
            }
        });
    }
    
    function nudgeTokenPools(cy) {
        var tokenPools = cy.nodes('[type="token_pool"]');
        tokenPools.forEach(function(tp) {
            var pos = tp.position();
            var offset = 25;
            tp.position({
                x: pos.x + (Math.random() - 0.5) * offset,
                y: pos.y + (Math.random() - 0.5) * offset
            });
        });
    }
    
    function addEdgesToCytoscape(edges) {
        const edgeColors = {
            'balance': '#3498DB',
            'trust': '#9B59B6',
            'mint': '#2ECC71'
        };
        
        // Check if we're using Cytoscape
        if (isCytoscapeRenderer() && getCytoscape()) {
            const cy = getCytoscape();
            
            const cyEdges = edges.map(edge => ({
                data: Object.assign({}, edge.data)
            }));
            
            cy.add(cyEdges);
            
            cy.style()
                .selector('edge')
                .style({
                    'line-color': function(ele) { return edgeColors[ele.data('type')] || '#666'; },
                    'width': 1,
                    'opacity': 0.4,
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': function(ele) { return edgeColors[ele.data('type')] || '#666'; },
                    'arrow-scale': 0.4
                })
                .update();
            
            // Update edge count
            const edgeCountEl = document.getElementById('edge-count');
            if (edgeCountEl) edgeCountEl.textContent = cy.edges().length + ' edges';
        } else {
            // For Cosmos or other renderers
            const renderer = getRenderer();
            if (!renderer) return;
            
            // Transform edges for renderer
            const rendererEdges = edges.map(edge => Object.assign({}, edge.data));
            
            // Add edges to renderer
            renderer.addEdges(rendererEdges);
            
            // Apply edge style
            renderer.setEdgeStyle({
                color: '#3498DB',
                opacity: 0.4,
                width: 1
            });
            
            // Update edge count
            const edgeCountEl = document.getElementById('edge-count');
            const totalEdges = renderer.edgeDataMap ? renderer.edgeDataMap.size : edges.length;
            if (edgeCountEl) edgeCountEl.textContent = totalEdges + ' edges';
        }
    }
    
    // ==========================================================================
    // MAX FLOW COMPUTATION
    // ==========================================================================
    
    async function computeMaxFlow() {
        var sourceInput = document.getElementById('flow-source-input');
        var sinkInput = document.getElementById('flow-target-input');
        
        var source = sourceInput ? sourceInput.value.trim() : '';
        var sink = sinkInput ? sinkInput.value.trim() : '';
        
        if (!source || !sink) {
            showToast('Enter Source and Sink addresses', 'error');
            return;
        }
        
        var algoSelect = document.getElementById('cf-algorithm-select');
        var selected = algoSelect ? algoSelect.selectedOptions[0] : null;
        var backend = selected ? (selected.dataset.backend || 'networkx') : 'networkx';
        var algorithm = selected ? (selected.dataset.algorithm || 'edmonds_karp') : 'edmonds_karp';
        
        var cutoffInput = document.getElementById('cf-cutoff-input');
        var cutoffValue = cutoffInput ? cutoffInput.value : '';
        var cutoff = cutoffValue ? parseInt(cutoffValue) : null;
        
        var decomposeCheckbox = document.getElementById('cf-decompose-paths');
        var simplifyCheckbox = document.getElementById('cf-simplify-paths');
        var decompose = decomposeCheckbox ? decomposeCheckbox.checked : true;
        var simplify = simplifyCheckbox ? simplifyCheckbox.checked : true;
        
        showLoading(true);
        hideError();
        
        try {
            var response = await fetch('/api/capacity-flow/max-flow', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source: source,
                    sink: sink,
                    backend: backend,
                    algorithm: algorithm,
                    cutoff: cutoff,
                    decompose_paths: decompose,
                    simplify_paths: simplify,
                    max_paths: 100
                })
            });
            
            if (!response.ok) {
                var err = await response.json();
                throw new Error(err.detail || 'Computation failed');
            }
            
            var data = await response.json();
            console.log('[CapacityFlow] Max flow result:', data);
            
            state.lastResult = data;
            state.sourceNode = source.toLowerCase();
            state.targetNode = sink.toLowerCase();
            state.selectedPathIndex = -1;
            
            // Reset isolation state
            state.isIsolated = false;
            state.hiddenNodes = [];
            
            displayResult(data);
            showToast('Max flow: ' + formatNumber(data.max_flow), 'success');
            
        } catch (err) {
            console.error('[CapacityFlow] Computation failed:', err);
            showError(err.message);
        } finally {
            showLoading(false);
        }
    }
    
    // ==========================================================================
    // DISPLAY FUNCTIONS
    // ==========================================================================
    
    function displayStats(stats, buildData) {
        var container = document.getElementById('cf-graph-stats');
        if (!container) return;
        
        var avatars = stats ? (stats.num_avatars || 0) : 0;
        var tokenPools = stats ? (stats.num_token_pools || 0) : 0;
        var groups = stats ? (stats.num_groups || 0) : 0;
        var edges = stats ? (stats.num_edges || 0) : 0;
        var trusts = buildData ? (buildData.trust_count || 0) : state.trustCount || 0;
        var balances = buildData ? (buildData.balance_count || 0) : state.balanceCount || 0;
        
        container.innerHTML = 
            '<div class="stat-row"><span>Avatars</span><span class="stat-value">' + formatNumber(avatars) + '</span></div>' +
            '<div class="stat-row"><span>Token Pools</span><span class="stat-value">' + formatNumber(tokenPools) + '</span></div>' +
            '<div class="stat-row"><span>Groups</span><span class="stat-value">' + formatNumber(groups) + '</span></div>' +
            '<div class="stat-row"><span>Edges</span><span class="stat-value">' + formatNumber(edges) + '</span></div>' +
            '<div class="stat-row"><span>Trusts</span><span class="stat-value">' + formatNumber(trusts) + '</span></div>' +
            '<div class="stat-row"><span>Balances</span><span class="stat-value">' + formatNumber(balances) + '</span></div>';
        container.style.display = 'block';
    }
    
    function displayResult(result) {
        var container = document.getElementById('cf-results');
        if (!container) return;
        
        var valueEl = document.getElementById('cf-max-flow-value');
        if (valueEl) valueEl.textContent = formatNumber(result.max_flow);
        
        var pathCount = document.getElementById('cf-path-count');
        var compTime = document.getElementById('cf-computation-time');
        var backendUsed = document.getElementById('cf-backend-used');
        
        if (pathCount) pathCount.textContent = (result.path_count || 0) + ' paths';
        if (compTime) compTime.textContent = (result.computation_time_ms || 0).toFixed(0) + 'ms';
        if (backendUsed) backendUsed.textContent = result.backend || '';
        
        // Display token flows (computed from paths as sink inflows)
        displayTokenFlows(result.token_flows || {}, result.max_flow);
        
        // Display paths
        displayPaths(result.paths || []);
        
        container.style.display = 'block';
    }
    
    function displayTokenFlows(tokenFlows, maxFlow) {
        var container = document.getElementById('cf-token-flows');
        var list = document.getElementById('cf-token-flows-list');
        if (!container || !list) return;
        
        // Compute sink inflows from paths (should sum to max_flow)
        var sinkInflows = computeSinkInflowsByToken();
        
        var flowData = Object.keys(sinkInflows).length > 0 ? sinkInflows : tokenFlows;
        
        if (Object.keys(flowData).length === 0) {
            container.style.display = 'none';
            return;
        }
        
        var entries = Object.entries(flowData).sort(function(a, b) { return b[1] - a[1]; }).slice(0, 10);
        var total = 0;
        for (var i = 0; i < entries.length; i++) {
            total += entries[i][1];
        }
        
        var html = '';
        for (var i = 0; i < entries.length; i++) {
            var token = entries[i][0];
            var flow = entries[i][1];
            html += '<div class="token-flow-item">' +
                '<span class="token-addr">' + shortenAddress(token) + '</span>' +
                '<span class="token-flow">' + formatNumber(flow) + '</span>' +
                '</div>';
        }
        
        // Show total which should equal max_flow
        html += '<div class="token-flow-item token-flow-total">' +
            '<span class="token-addr">Total</span>' +
            '<span class="token-flow">' + formatNumber(total) + '</span>' +
            '</div>';
        
        list.innerHTML = html;
        container.style.display = 'block';
    }
    
    function computeSinkInflowsByToken() {
        var sinkInflows = {};
        
        if (!state.lastResult || !state.lastResult.paths) return sinkInflows;
        
        var paths = state.lastResult.paths;
        for (var p = 0; p < paths.length; p++) {
            var path = paths[p];
            var nodes = path.nodes || [];
            var flow = path.flow || 0;
            
            if (nodes.length < 2) continue;
            
            // The second-to-last node tells us which token is being delivered
            var secondToLast = nodes[nodes.length - 2].toLowerCase();
            
            var tokenId = secondToLast;
            if (tokenId.indexOf('t_') === 0) {
                tokenId = tokenId.substring(2);
            } else if (tokenId.indexOf('a_') === 0) {
                tokenId = tokenId.substring(2);
            }
            
            if (!sinkInflows[tokenId]) {
                sinkInflows[tokenId] = 0;
            }
            sinkInflows[tokenId] += flow;
        }
        
        return sinkInflows;
    }
    
    function displayPaths(paths) {
        var list = document.getElementById('cf-paths-list');
        if (!list) return;
        
        if (paths.length === 0) {
            list.innerHTML = '<div class="no-data">No paths found</div>';
            return;
        }
        
        var html = '';
        var maxDisplay = Math.min(paths.length, 30);
        for (var i = 0; i < maxDisplay; i++) {
            var path = paths[i];
            var nodes = path.nodes || [];
            html += '<div class="path-item" data-index="' + i + '">' +
                '<span class="path-nodes">' + formatPathNodes(nodes) + '</span>' +
                '<span class="path-flow">' + formatNumber(path.flow) + '</span>' +
                '</div>';
        }
        
        if (paths.length > 30) {
            html += '<div class="no-data">+' + (paths.length - 30) + ' more</div>';
        }
        
        list.innerHTML = html;
        
        // Click handler for path selection
        var pathItems = list.querySelectorAll('.path-item');
        for (var i = 0; i < pathItems.length; i++) {
            pathItems[i].addEventListener('click', function() {
                var idx = parseInt(this.dataset.index);
                selectPath(idx);
            });
        }
    }
    
    function formatPathNodes(nodes) {
        if (!nodes || nodes.length === 0) return '-';
        
        function fmt(id) {
            var clean = id;
            if (clean.indexOf('a_') === 0) clean = clean.substring(2);
            if (clean.indexOf('t_') === 0) clean = clean.substring(2);
            return clean.length > 12 ? clean.slice(0, 6) + '...' + clean.slice(-4) : clean;
        }
        
        if (nodes.length > 5) {
            var first = [fmt(nodes[0]), fmt(nodes[1])];
            var last = [fmt(nodes[nodes.length - 2]), fmt(nodes[nodes.length - 1])];
            return first.join(' -> ') + ' -> ... -> ' + last.join(' -> ');
        }
        
        var result = [];
        for (var i = 0; i < nodes.length; i++) {
            result.push(fmt(nodes[i]));
        }
        return result.join(' -> ');
    }
    
    // ==========================================================================
    // PATH SELECTION & HIGHLIGHTING
    // ==========================================================================
    
    function selectPath(index) {
        if (!state.lastResult || !state.lastResult.paths) return;
        
        var path = state.lastResult.paths[index];
        if (!path) return;
        
        state.selectedPathIndex = index;
        
        // Update UI selection
        var pathItems = document.querySelectorAll('#cf-paths-list .path-item');
        for (var i = 0; i < pathItems.length; i++) {
            pathItems[i].classList.toggle('selected', parseInt(pathItems[i].dataset.index) === index);
        }
        
        // If isolated, first show all then highlight selected path
        if (state.isIsolated) {
            restoreAllNodesQuietly();
        }
        
        // Highlight this path
        highlightPath(path);
        
        // Show path details in right sidebar
        showPathDetails(path, index);
    }
    
    function cleanNodeIdForDisplay(nodeId) {
        var id = (nodeId || '').toLowerCase();
        if (id.indexOf('a_') === 0) id = id.substring(2);
        if (id.indexOf('t_') === 0) id = id.substring(2);
        return id;
    }
    
    function findCyNode(cy, nodeId) {
        if (!cy || !nodeId) return null;
        
        var id = nodeId.toLowerCase();
        
        // Try exact match first
        var node = cy.getElementById(id);
        if (node && node.length) return node;
        
        // Try with a_ prefix (avatar)
        node = cy.getElementById('a_' + id);
        if (node && node.length) return node;
        
        // Try with t_ prefix (token pool)
        node = cy.getElementById('t_' + id);
        if (node && node.length) return node;
        
        // Try without prefix if it has one
        if (id.indexOf('a_') === 0) {
            node = cy.getElementById(id.substring(2));
            if (node && node.length) return node;
        }
        if (id.indexOf('t_') === 0) {
            node = cy.getElementById(id.substring(2));
            if (node && node.length) return node;
        }
        
        // Try with address data attribute
        node = cy.nodes().filter(function(n) {
            var addr = n.data('address');
            return addr && addr.toLowerCase() === id.replace(/^[at]_/, '');
        });
        
        return (node && node.length) ? node : null;
    }
    
    function highlightPath(path) {
        if (!path || !path.nodes) return;
        
        clearHighlights();
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            highlightPathCosmos(path);
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        var nodes = path.nodes;
        var foundNodeIds = [];
        
        cy.batch(function() {
            // Highlight path nodes
            for (var i = 0; i < nodes.length; i++) {
                var nodeId = nodes[i];
                var node = findCyNode(cy, nodeId);
                
                if (node && node.length) {
                    var color, size;
                    if (i === 0) {
                        color = '#22c55e';
                        size = 40;
                    } else if (i === nodes.length - 1) {
                        color = '#ef4444';
                        size = 40;
                    } else {
                        color = '#00d4ff';
                        size = 30;
                    }
                    
                    node.style({
                        'background-color': color,
                        'border-color': '#ffffff',
                        'border-width': 4,
                        'width': size,
                        'height': size,
                        'z-index': 9999
                    });
                    
                    state.originalStyles.set(node.id(), { type: 'node' });
                    foundNodeIds.push(node.id());
                }
            }
            
            // Draw path edges
            for (var i = 0; i < nodes.length - 1; i++) {
                var sourceNode = findCyNode(cy, nodes[i]);
                var targetNode = findCyNode(cy, nodes[i + 1]);
                
                if (!sourceNode || !sourceNode.length || !targetNode || !targetNode.length) continue;
                
                var sourceId = sourceNode.id();
                var targetId = targetNode.id();
                
                var edge = cy.edges('[source="' + sourceId + '"][target="' + targetId + '"]');
                if (!edge.length) {
                    edge = cy.edges('[source="' + targetId + '"][target="' + sourceId + '"]');
                }
                
                if (edge && edge.length) {
                    edge.style({
                        'line-color': '#00d4ff',
                        'target-arrow-color': '#00d4ff',
                        'width': 5,
                        'opacity': 1,
                        'z-index': 9998
                    });
                    state.originalStyles.set(edge.id(), { type: 'edge' });
                } else {
                    // Create temporary edge
                    var tempEdgeId = 'cf-path-edge-' + i;
                    try {
                        cy.add({
                            group: 'edges',
                            data: { id: tempEdgeId, source: sourceId, target: targetId, _pathTemp: true }
                        });
                        cy.getElementById(tempEdgeId).style({
                            'line-color': '#00d4ff',
                            'target-arrow-color': '#00d4ff',
                            'width': 5,
                            'curve-style': 'bezier',
                            'opacity': 1,
                            'z-index': 9998
                        });
                        state.originalStyles.set(tempEdgeId, { type: 'temp' });
                    } catch (e) {
                        console.warn('[CapacityFlow] Could not create temp edge:', e);
                    }
                }
            }
        });
        
        // Fit view to found path nodes
        if (foundNodeIds.length > 0) {
            var pathNodes = cy.nodes().filter(function(n) {
                return foundNodeIds.indexOf(n.id()) >= 0;
            });
            if (pathNodes.length > 0) {
                cy.animate({
                    fit: { eles: pathNodes, padding: 100 },
                    duration: 500
                });
            }
        }
    }
    
    /**
     * Highlight a single path using CosmosGL renderer
     */
    function highlightPathCosmos(path) {
        const renderer = getRenderer();
        if (!renderer) {
            console.warn('[CapacityFlow] No renderer available for CosmosGL highlighting');
            return;
        }
        
        console.log('[CapacityFlow] Highlighting path with CosmosGL:', path.nodes.length, 'nodes');
        console.log('[CapacityFlow] Path nodes:', path.nodes);
        console.log('[CapacityFlow] Renderer has', renderer.nodeIds?.length || 0, 'nodes');
        
        const nodes = path.nodes;
        const nodeColorMap = new Map();
        const edgePairs = [];
        
        // Build node color map
        for (let i = 0; i < nodes.length; i++) {
            const nodeId = nodes[i];
            // Find actual node ID in renderer (handle prefixes)
            const actualNodeId = findCosmosNodeId(renderer, nodeId);
            
            console.log('[CapacityFlow] Node', i, ':', nodeId, '->', actualNodeId, 
                       '(found:', renderer.nodeIndices?.has(actualNodeId), ')');
            
            if (actualNodeId && renderer.nodeIndices?.has(actualNodeId)) {
                let type;
                if (i === 0) {
                    type = 'source';
                } else if (i === nodes.length - 1) {
                    type = 'target';
                } else {
                    type = 'intermediate';
                }
                
                const color = type === 'source' ? '#22c55e' : type === 'target' ? '#ef4444' : '#00d4ff';
                nodeColorMap.set(actualNodeId, { color: color, type: type });
            } else {
                console.warn('[CapacityFlow] Node not found in renderer:', nodeId);
            }
        }
        
        // Build edge pairs
        for (let i = 0; i < nodes.length - 1; i++) {
            const sourceId = findCosmosNodeId(renderer, nodes[i]);
            const targetId = findCosmosNodeId(renderer, nodes[i + 1]);
            
            if (sourceId && targetId && renderer.nodeIndices?.has(sourceId) && renderer.nodeIndices?.has(targetId)) {
                edgePairs.push({ source: sourceId, target: targetId });
            }
        }
        
        console.log('[CapacityFlow] Built nodeColorMap with', nodeColorMap.size, 'nodes');
        console.log('[CapacityFlow] Built edgePairs with', edgePairs.length, 'edges');
        
        // Apply highlighting
        if (nodeColorMap.size > 0 && typeof renderer.highlightPathNodes === 'function') {
            renderer.highlightPathNodes(nodeColorMap);
        }
        
        if (edgePairs.length > 0 && typeof renderer.highlightPathEdges === 'function') {
            renderer.highlightPathEdges(edgePairs, '#00d4ff', 1.0);
        }
        
        // Fit view to path nodes
        if (nodeColorMap.size > 0 && typeof renderer.fitView === 'function') {
            renderer.fitView(Array.from(nodeColorMap.keys()), 0.2);
        }
        
        // Store for cleanup
        state.originalStyles.set('_cosmosPath', { nodes: Array.from(nodeColorMap.keys()), edges: edgePairs });
    }
    
    /**
     * Find node ID in CosmosGL renderer (handles prefixes like a_, t_)
     */
    function findCosmosNodeId(renderer, nodeId) {
        if (!renderer || !renderer.nodeIndices) {
            console.warn('[CapacityFlow] findCosmosNodeId: No renderer or nodeIndices');
            return nodeId;
        }
        
        // Try direct match
        if (renderer.nodeIndices.has(nodeId)) {
            return nodeId;
        }
        
        // Try with prefixes
        const prefixes = ['', 'a_', 't_'];
        const cleanId = nodeId.replace(/^[at]_/, '');
        
        for (const prefix of prefixes) {
            const testId = prefix + cleanId;
            if (renderer.nodeIndices.has(testId)) {
                return testId;
            }
        }
        
        // Try without prefix if it has one
        if (renderer.nodeIndices.has(cleanId)) {
            return cleanId;
        }
        
        // Debug: Print some sample node IDs from renderer for comparison
        if (renderer.nodeIds && renderer.nodeIds.length > 0) {
            console.log('[CapacityFlow] Looking for:', nodeId, '(cleaned:', cleanId, ')');
            console.log('[CapacityFlow] Sample renderer node IDs:', renderer.nodeIds.slice(0, 5));
        }
        
        return null; // Return null if not found to indicate failure
    }
    
    function highlightAllPaths() {
        if (!state.lastResult || !state.lastResult.paths || state.lastResult.paths.length === 0) {
            showToast('No paths to highlight. Compute flow first.', 'warning');
            return;
        }
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            highlightAllPathsCosmos();
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        // Reset isolation first
        if (state.isIsolated) {
            restoreAllNodesQuietly();
        }
        
        clearHighlights();
        
        var nodeIdToType = new Map();
        var edgePairs = [];
        
        var paths = state.lastResult.paths;
        for (var p = 0; p < paths.length; p++) {
            var path = paths[p];
            var nodes = path.nodes || [];
            
            for (var i = 0; i < nodes.length; i++) {
                var cyNode = findCyNode(cy, nodes[i]);
                if (!cyNode || !cyNode.length) continue;
                
                var cyNodeId = cyNode.id();
                
                if (i === 0) {
                    if (!nodeIdToType.has(cyNodeId)) nodeIdToType.set(cyNodeId, 'source');
                } else if (i === nodes.length - 1) {
                    if (!nodeIdToType.has(cyNodeId) || nodeIdToType.get(cyNodeId) === 'intermediate') {
                        nodeIdToType.set(cyNodeId, 'target');
                    }
                } else {
                    if (!nodeIdToType.has(cyNodeId)) nodeIdToType.set(cyNodeId, 'intermediate');
                }
            }
            
            for (var i = 0; i < nodes.length - 1; i++) {
                var s = findCyNode(cy, nodes[i]);
                var t = findCyNode(cy, nodes[i + 1]);
                if (s && s.length && t && t.length) {
                    edgePairs.push({ source: s.id(), target: t.id() });
                }
            }
        }
        
        if (nodeIdToType.size === 0) {
            showToast('No matching nodes found in graph', 'warning');
            return;
        }
        
        cy.batch(function() {
            nodeIdToType.forEach(function(type, nodeId) {
                var node = cy.getElementById(nodeId);
                if (node && node.length) {
                    var color = type === 'source' ? '#22c55e' : type === 'target' ? '#ef4444' : '#00d4ff';
                    var size = type === 'intermediate' ? 25 : 35;
                    
                    node.style({
                        'background-color': color,
                        'border-color': '#ffffff',
                        'border-width': 3,
                        'width': size,
                        'height': size,
                        'z-index': 9999
                    });
                    state.originalStyles.set(nodeId, { type: 'node' });
                }
            });
            
            var edgeSet = new Set();
            var tempIdx = 0;
            for (var i = 0; i < edgePairs.length; i++) {
                var source = edgePairs[i].source;
                var target = edgePairs[i].target;
                var key = source + '|' + target;
                var reverseKey = target + '|' + source;
                if (edgeSet.has(key) || edgeSet.has(reverseKey)) continue;
                edgeSet.add(key);
                
                var edge = cy.edges('[source="' + source + '"][target="' + target + '"]');
                if (!edge.length) {
                    edge = cy.edges('[source="' + target + '"][target="' + source + '"]');
                }
                
                if (edge && edge.length) {
                    edge.style({
                        'line-color': '#00d4ff',
                        'target-arrow-color': '#00d4ff',
                        'width': 4,
                        'opacity': 1,
                        'z-index': 9998
                    });
                    state.originalStyles.set(edge.id(), { type: 'edge' });
                } else {
                    var tempId = 'cf-multi-edge-' + tempIdx++;
                    try {
                        cy.add({
                            group: 'edges',
                            data: { id: tempId, source: source, target: target, _pathTemp: true }
                        });
                        cy.getElementById(tempId).style({
                            'line-color': '#00d4ff',
                            'width': 4,
                            'opacity': 1,
                            'z-index': 9998
                        });
                        state.originalStyles.set(tempId, { type: 'temp' });
                    } catch (e) {}
                }
            }
        });
        
        // Fit to all path nodes
        var allPathNodeIds = [];
        nodeIdToType.forEach(function(type, nodeId) {
            allPathNodeIds.push(nodeId);
        });
        
        var pathNodes = cy.nodes().filter(function(n) {
            return allPathNodeIds.indexOf(n.id()) >= 0;
        });
        if (pathNodes.length > 0) {
            cy.animate({
                fit: { eles: pathNodes, padding: 80 },
                duration: 500
            });
        }
        
        showToast('Highlighting ' + paths.length + ' paths (' + nodeIdToType.size + ' nodes)', 'info');
    }
    
    /**
     * Highlight all paths using CosmosGL renderer
     */
    function highlightAllPathsCosmos() {
        const renderer = getRenderer();
        if (!renderer) {
            console.warn('[CapacityFlow] No renderer available for CosmosGL highlighting');
            return;
        }
        
        console.log('[CapacityFlow] Highlighting all paths with CosmosGL');
        
        // Reset isolation first
        if (state.isIsolated) {
            restoreAllNodesQuietly();
        }
        
        clearHighlights();
        
        console.log('[CapacityFlow] Highlighting all paths in CosmosGL');
        
        const nodeColorMap = new Map();
        const edgePairs = [];
        const paths = state.lastResult.paths;
        
        // Collect all nodes and edges from all paths
        for (let p = 0; p < paths.length; p++) {
            const path = paths[p];
            const nodes = path.nodes || [];
            
            for (let i = 0; i < nodes.length; i++) {
                const nodeId = nodes[i];
                const actualNodeId = findCosmosNodeId(renderer, nodeId);
                
                if (actualNodeId && renderer.nodeIndices?.has(actualNodeId) && !nodeColorMap.has(actualNodeId)) {
                    let type;
                    if (i === 0) {
                        type = 'source';
                    } else if (i === nodes.length - 1) {
                        type = 'target';
                    } else {
                        type = 'intermediate';
                    }
                    
                    // Don't downgrade source/target to intermediate
                    const existing = nodeColorMap.get(actualNodeId);
                    if (!existing || (type !== 'intermediate' && existing.type === 'intermediate')) {
                        const color = type === 'source' ? '#22c55e' : type === 'target' ? '#ef4444' : '#00d4ff';
                        nodeColorMap.set(actualNodeId, { color: color, type: type });
                    }
                }
            }
            
            // Build edge pairs
            for (let i = 0; i < nodes.length - 1; i++) {
                const sourceId = findCosmosNodeId(renderer, nodes[i]);
                const targetId = findCosmosNodeId(renderer, nodes[i + 1]);
                
                if (sourceId && targetId && renderer.nodeIndices?.has(sourceId) && renderer.nodeIndices?.has(targetId)) {
                    edgePairs.push({ source: sourceId, target: targetId });
                }
            }
        }
        
        console.log('[CapacityFlow] Built nodeColorMap with', nodeColorMap.size, 'nodes');
        console.log('[CapacityFlow] Built edgePairs with', edgePairs.length, 'edges');
        
        if (nodeColorMap.size === 0) {
            showToast('No matching nodes found in graph', 'warning');
            return;
        }
        
        // Apply highlighting
        if (typeof renderer.highlightPathNodes === 'function') {
            renderer.highlightPathNodes(nodeColorMap);
        }
        
        if (edgePairs.length > 0 && typeof renderer.highlightPathEdges === 'function') {
            renderer.highlightPathEdges(edgePairs, '#00d4ff', 1.0);
        }
        
        // Fit view
        if (typeof renderer.fitView === 'function') {
            renderer.fitView(Array.from(nodeColorMap.keys()), 0.2);
        }
        
        // Store for cleanup
        state.originalStyles.set('_cosmosAllPaths', { nodes: Array.from(nodeColorMap.keys()), edges: edgePairs });
        
        showToast('Highlighting ' + paths.length + ' paths (' + nodeColorMap.size + ' nodes)', 'info');
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
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) {
            state.originalStyles.clear();
            return;
        }
        
        cy.batch(function() {
            // Remove temporary path edges
            cy.edges('[_pathTemp]').remove();
            
            // Reset styled elements
            state.originalStyles.forEach(function(info, key) {
                if (info.type === 'temp') return;
                if (key === '_cosmosPath' || key === '_cosmosAllPaths') return; // Cosmos markers
                
                var ele = cy.getElementById(key);
                if (ele && ele.length) {
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
    // PATH DETAILS PANEL (Right sidebar)
    // ==========================================================================
    
    function showPathDetails(path, index) {
        var infoPanel = document.getElementById('info-panel');
        if (!infoPanel) return;
        
        // Hide other sections
        var nodeInfo = document.getElementById('node-info');
        var edgeInfo = document.getElementById('edge-info');
        var multiInfo = document.getElementById('multi-info');
        if (nodeInfo) nodeInfo.style.display = 'none';
        if (edgeInfo) edgeInfo.style.display = 'none';
        if (multiInfo) multiInfo.style.display = 'none';
        
        // Update header
        var headerTitle = infoPanel.querySelector('.info-header h3');
        if (headerTitle) headerTitle.textContent = 'Flow Path Details';
        
        var nodes = path.nodes || [];
        var hops = nodes.length - 1;
        
        var html = '<div id="flow-detail-panel">' +
            '<div class="flow-stats">' +
            '<div class="flow-stat"><div class="flow-stat-label">Flow</div><div class="flow-stat-value highlight">' + formatNumber(path.flow) + '</div></div>' +
            '<div class="flow-stat"><div class="flow-stat-label">Hops</div><div class="flow-stat-value">' + hops + '</div></div>' +
            '</div>' +
            '<div class="path-node-list-header">Path Nodes (' + nodes.length + ')</div>' +
            '<div class="path-node-list">';
        
        for (var i = 0; i < nodes.length; i++) {
            var nodeId = nodes[i];
            var isSource = i === 0;
            var isTarget = i === nodes.length - 1;
            var isTokenPool = nodeId.toLowerCase().indexOf('t_') === 0;
            
            var hopClass = isSource ? 'source' : isTarget ? 'target' : isTokenPool ? 'token' : '';
            var hopLabel = isSource ? 'S' : isTarget ? 'T' : isTokenPool ? '*' : i;
            var itemClass = isTokenPool ? 'token-pool' : '';
            var displayId = cleanNodeIdForDisplay(nodeId);
            
            html += '<div class="path-node-item ' + itemClass + '" data-node-id="' + nodeId + '">' +
                '<span class="path-node-hop ' + hopClass + '">' + hopLabel + '</span>' +
                '<span class="path-node-id">' + displayId + '</span>' +
                '</div>';
            
            if (i < nodes.length - 1) {
                html += '<div class="path-edge-info">| hop ' + (i + 1) + '</div>';
            }
        }
        
        html += '</div>' +
            '<div class="path-actions">' +
            '<button class="fit-btn" id="fd-fit-btn">Fit to Path</button>' +
            '<button id="fd-copy-btn">Copy Path</button>' +
            '</div>' +
            '<div class="path-actions">' +
            '<button class="isolate-btn" id="fd-isolate-btn">Isolate Path</button>' +
            '<button class="show-all-btn" id="fd-show-all-btn">Show All</button>' +
            '</div>' +
            '</div>';
        
        // Remove existing detail panels
        var existingFlowPanel = document.getElementById('flow-detail-panel');
        if (existingFlowPanel) existingFlowPanel.remove();
        var existingPathPanel = document.getElementById('path-detail-panel');
        if (existingPathPanel) existingPathPanel.remove();
        
        // Insert after header
        var infoHeader = infoPanel.querySelector('.info-header');
        if (infoHeader) {
            infoHeader.insertAdjacentHTML('afterend', html);
        } else {
            infoPanel.insertAdjacentHTML('afterbegin', html);
        }
        
        // Add event listeners
        var fitBtn = document.getElementById('fd-fit-btn');
        if (fitBtn) fitBtn.addEventListener('click', fitToPath);
        
        var copyBtn = document.getElementById('fd-copy-btn');
        if (copyBtn) copyBtn.addEventListener('click', copyPath);
        
        var isolateBtn = document.getElementById('fd-isolate-btn');
        if (isolateBtn) isolateBtn.addEventListener('click', isolatePath);
        
        var showAllBtn = document.getElementById('fd-show-all-btn');
        if (showAllBtn) showAllBtn.addEventListener('click', showAllNodes);
        
        // Node click to zoom
        var nodeItems = document.querySelectorAll('#flow-detail-panel .path-node-item');
        for (var i = 0; i < nodeItems.length; i++) {
            nodeItems[i].addEventListener('click', function() {
                zoomToNode(this.dataset.nodeId);
            });
        }
        
        infoPanel.style.display = 'flex';
        updateIsolateButtons();
    }
    
    function hidePathDetails() {
        var flowPanel = document.getElementById('flow-detail-panel');
        if (flowPanel) flowPanel.remove();
        
        var infoPanel = document.getElementById('info-panel');
        if (infoPanel) {
            var headerTitle = infoPanel.querySelector('.info-header h3');
            if (headerTitle) headerTitle.textContent = 'Information';
        }
    }
    
    function zoomToNode(nodeId) {
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer) {
                const actualNodeId = findCosmosNodeId(renderer, nodeId);
                if (actualNodeId && typeof renderer.zoomToNode === 'function') {
                    renderer.zoomToNode(actualNodeId, 2, 400);
                }
            }
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        var node = findCyNode(cy, nodeId);
        if (node && node.length) {
            cy.animate({
                center: { eles: node },
                zoom: 2,
                duration: 400
            });
        }
    }
    
    function fitToPath() {
        if (!state.lastResult || state.selectedPathIndex < 0) return;
        
        var path = state.lastResult.paths[state.selectedPathIndex];
        if (!path) return;
        
        var nodes = path.nodes || [];
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer) {
                const nodeIds = [];
                for (let i = 0; i < nodes.length; i++) {
                    const actualNodeId = findCosmosNodeId(renderer, nodes[i]);
                    if (actualNodeId) {
                        nodeIds.push(actualNodeId);
                    }
                }
                
                if (nodeIds.length > 0 && typeof renderer.fitView === 'function') {
                    renderer.fitView(nodeIds, 0.2);
                }
            }
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        var foundIds = [];
        
        for (var i = 0; i < nodes.length; i++) {
            var node = findCyNode(cy, nodes[i]);
            if (node && node.length) foundIds.push(node.id());
        }
        
        if (foundIds.length > 0) {
            var pathNodes = cy.nodes().filter(function(n) {
                return foundIds.indexOf(n.id()) >= 0;
            });
            if (pathNodes.length > 0) {
                cy.animate({
                    fit: { eles: pathNodes, padding: 80 },
                    duration: 500
                });
            }
        }
    }
    
    function copyPath() {
        if (!state.lastResult || state.selectedPathIndex < 0) return;
        
        var path = state.lastResult.paths[state.selectedPathIndex];
        if (!path) return;
        
        var nodes = path.nodes || [];
        var cleanNodes = [];
        for (var i = 0; i < nodes.length; i++) {
            cleanNodes.push(cleanNodeIdForDisplay(nodes[i]));
        }
        
        navigator.clipboard.writeText(cleanNodes.join(' -> ')).then(function() {
            showToast('Path copied!', 'success');
        });
    }
    
    // ==========================================================================
    // ISOLATE / SHOW ALL
    // ==========================================================================
    
    function isolatePath() {
        if (state.selectedPathIndex < 0 || !state.lastResult || !state.lastResult.paths) {
            showToast('No path selected', 'error');
            return;
        }
        
        var path = state.lastResult.paths[state.selectedPathIndex];
        var nodes = path.nodes || [];
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (!renderer) return;
            
            console.log('[CapacityFlow] Isolating path in CosmosGL, nodes:', nodes);
            
            const pathNodeIds = [];
            for (let i = 0; i < nodes.length; i++) {
                const actualNodeId = findCosmosNodeId(renderer, nodes[i]);
                if (actualNodeId && renderer.nodeIndices?.has(actualNodeId)) {
                    pathNodeIds.push(actualNodeId);
                } else {
                    console.warn('[CapacityFlow] Node not found for isolation:', nodes[i]);
                }
            }
            
            console.log('[CapacityFlow] Found', pathNodeIds.length, 'nodes for isolation');
            
            if (pathNodeIds.length === 0) {
                showToast('No nodes found for path', 'error');
                return;
            }
            
            if (typeof renderer.showOnlyNodes === 'function') {
                renderer.showOnlyNodes(pathNodeIds);
            }
            
            if (typeof renderer.fitView === 'function') {
                renderer.fitView(pathNodeIds, 0.1);
            }
            
            state.isIsolated = true;
            showToast('Isolated ' + pathNodeIds.length + ' path nodes', 'success');
            updateIsolateButtons();
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        var pathNodeIds = new Set();
        
        for (var i = 0; i < nodes.length; i++) {
            var cyNode = findCyNode(cy, nodes[i]);
            if (cyNode && cyNode.length) {
                pathNodeIds.add(cyNode.id());
            }
        }
        
        if (pathNodeIds.size === 0) {
            showToast('No nodes found for path', 'error');
            return;
        }
        
        // First restore all if already isolated
        if (state.isIsolated) {
            restoreAllNodesQuietly();
        }
        
        state.hiddenNodes = [];
        
        cy.batch(function() {
            cy.nodes().forEach(function(node) {
                if (!pathNodeIds.has(node.id())) {
                    node.style('display', 'none');
                    state.hiddenNodes.push(node.id());
                }
            });
            
            cy.edges().forEach(function(edge) {
                if (!pathNodeIds.has(edge.source().id()) || !pathNodeIds.has(edge.target().id())) {
                    edge.style('display', 'none');
                }
            });
        });
        
        state.isIsolated = true;
        
        var visibleNodes = cy.nodes().filter(function(n) { return pathNodeIds.has(n.id()); });
        if (visibleNodes.length > 0) {
            cy.animate({
                fit: { eles: visibleNodes, padding: 50 },
                duration: 500
            });
        }
        
        showToast('Isolated ' + pathNodeIds.size + ' path nodes', 'success');
        updateIsolateButtons();
    }
    
    function isolateAllPaths() {
        if (!state.lastResult || !state.lastResult.paths || state.lastResult.paths.length === 0) {
            showToast('No paths to isolate', 'warning');
            return;
        }
        
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (!renderer) return;
            
            // First restore all if already isolated
            if (state.isIsolated) {
                restoreAllNodesQuietly();
            }
            
            console.log('[CapacityFlow] Isolating all paths in CosmosGL');
            
            const pathNodeIds = [];
            const paths = state.lastResult.paths;
            
            for (let p = 0; p < paths.length; p++) {
                const nodes = paths[p].nodes || [];
                for (let i = 0; i < nodes.length; i++) {
                    const actualNodeId = findCosmosNodeId(renderer, nodes[i]);
                    if (actualNodeId && renderer.nodeIndices?.has(actualNodeId) && pathNodeIds.indexOf(actualNodeId) === -1) {
                        pathNodeIds.push(actualNodeId);
                    }
                }
            }
            
            console.log('[CapacityFlow] Found', pathNodeIds.length, 'unique nodes for isolation');
            
            if (pathNodeIds.length === 0) {
                showToast('No matching nodes found', 'warning');
                return;
            }
            
            if (typeof renderer.showOnlyNodes === 'function') {
                renderer.showOnlyNodes(pathNodeIds);
            }
            
            if (typeof renderer.fitView === 'function') {
                renderer.fitView(pathNodeIds, 0.1);
            }
            
            state.isIsolated = true;
            showToast('Isolated ' + pathNodeIds.length + ' nodes from all paths', 'success');
            updateIsolateButtons();
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        // First restore all if already isolated
        if (state.isIsolated) {
            restoreAllNodesQuietly();
        }
        
        var pathNodeIds = new Set();
        
        var paths = state.lastResult.paths;
        for (var p = 0; p < paths.length; p++) {
            var nodes = paths[p].nodes || [];
            for (var i = 0; i < nodes.length; i++) {
                var cyNode = findCyNode(cy, nodes[i]);
                if (cyNode && cyNode.length) {
                    pathNodeIds.add(cyNode.id());
                }
            }
        }
        
        if (pathNodeIds.size === 0) {
            showToast('No matching nodes found', 'warning');
            return;
        }
        
        state.hiddenNodes = [];
        
        cy.batch(function() {
            cy.nodes().forEach(function(node) {
                if (!pathNodeIds.has(node.id())) {
                    node.style('display', 'none');
                    state.hiddenNodes.push(node.id());
                }
            });
            
            cy.edges().forEach(function(edge) {
                if (!pathNodeIds.has(edge.source().id()) || !pathNodeIds.has(edge.target().id())) {
                    edge.style('display', 'none');
                }
            });
        });
        
        state.isIsolated = true;
        
        var visibleNodes = cy.nodes().filter(function(n) { return pathNodeIds.has(n.id()); });
        if (visibleNodes.length > 0) {
            cy.animate({
                fit: { eles: visibleNodes, padding: 50 },
                duration: 500
            });
        }
        
        showToast('Isolated ' + pathNodeIds.size + ' nodes from all paths', 'success');
        updateIsolateButtons();
    }
    
    function restoreAllNodesQuietly() {
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer && typeof renderer.showAllNodes === 'function') {
                renderer.showAllNodes();
            }
            state.isIsolated = false;
            state.hiddenNodes = [];
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        console.log('[CapacityFlow] Restoring all nodes quietly...');
        
        // Use direct style setting instead of removeStyle which causes errors
        // Setting display to empty string or 'element' restores visibility
        try {
            var allNodes = cy.nodes();
            var allEdges = cy.edges();
            
            // Restore hidden nodes by ID
            if (state.hiddenNodes && state.hiddenNodes.length > 0) {
                for (var i = 0; i < state.hiddenNodes.length; i++) {
                    var nodeId = state.hiddenNodes[i];
                    var node = cy.getElementById(nodeId);
                    if (node && node.length) {
                        node.style('display', 'element');
                    }
                }
            } else {
                // Fallback: show all nodes
                allNodes.style('display', 'element');
            }
            
            // Show all edges
            allEdges.style('display', 'element');
            
        } catch (err) {
            console.error('[CapacityFlow] Error restoring nodes:', err);
            // Last resort fallback
            try {
                cy.nodes().style('display', 'element');
                cy.edges().style('display', 'element');
            } catch (e) {}
        }
        
        state.isIsolated = false;
        state.hiddenNodes = [];
    }
    
    function showAllNodes() {
        // Check if using CosmosGL renderer
        if (isCosmosRenderer()) {
            const renderer = getRenderer();
            if (renderer) {
                if (typeof renderer.showAllNodes === 'function') {
                    renderer.showAllNodes();
                }
                if (typeof renderer.fitView === 'function') {
                    renderer.fitView();
                }
            }
            state.isIsolated = false;
            state.hiddenNodes = [];
            showToast('Showing all nodes', 'success');
            updateIsolateButtons();
            return;
        }
        
        // Cytoscape implementation
        var cy = getCytoscape();
        if (!cy) return;
        
        console.log('[CapacityFlow] Showing all nodes...');
        
        try {
            // Direct style setting on the collection
            cy.nodes().style('display', 'element');
            cy.edges().style('display', 'element');
        } catch (err) {
            console.error('[CapacityFlow] Error in showAllNodes:', err);
        }
        
        state.isIsolated = false;
        state.hiddenNodes = [];
        
        setTimeout(function() {
            try {
                cy.fit(50);
            } catch (e) {}
        }, 50);
        
        showToast('Showing all nodes', 'success');
        updateIsolateButtons();
    }
    
    function updateIsolateButtons() {
        // Update buttons in the detail panel
        var isolateBtn = document.getElementById('fd-isolate-btn');
        var showAllBtn = document.getElementById('fd-show-all-btn');
        
        if (isolateBtn) {
            isolateBtn.disabled = state.isIsolated && state.selectedPathIndex >= 0;
        }
        if (showAllBtn) {
            showAllBtn.disabled = !state.isIsolated;
        }
        
        // Update buttons in the results section
        var cfIsolateBtn = document.getElementById('cf-isolate-btn');
        var cfShowAllBtn = document.getElementById('cf-show-all-btn');
        
        if (cfIsolateBtn) {
            cfIsolateBtn.disabled = state.isIsolated;
        }
        if (cfShowAllBtn) {
            cfShowAllBtn.disabled = !state.isIsolated;
        }
    }
    
    // ==========================================================================
    // CLEAR & UTILITIES
    // ==========================================================================
    
    function clearResults() {
        state.lastResult = null;
        state.selectedPathIndex = -1;
        
        clearHighlights();
        hidePathDetails();
        
        // Hide results section
        var container = document.getElementById('cf-results');
        if (container) container.style.display = 'none';
        
        // Clear paths list
        var list = document.getElementById('cf-paths-list');
        if (list) list.innerHTML = '';
        
        // Clear token flows
        var tokenFlows = document.getElementById('cf-token-flows');
        if (tokenFlows) tokenFlows.style.display = 'none';
        
        // Show all nodes if isolated
        if (state.isIsolated) {
            showAllNodes();
        }
        
        showToast('Results cleared', 'info');
    }
    
    function showLoading(show) {
        var btn = document.getElementById('cf-compute-btn');
        var loading = document.getElementById('cf-loading');
        
        if (btn) {
            btn.disabled = show;
            btn.textContent = show ? 'Computing...' : 'Compute Flow';
        }
        
        if (loading) {
            loading.style.display = show ? 'flex' : 'none';
        }
    }
    
    function showError(message) {
        var errorEl = document.getElementById('cf-error');
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.style.display = 'block';
        }
        showToast(message, 'error');
    }
    
    function hideError() {
        var errorEl = document.getElementById('cf-error');
        if (errorEl) errorEl.style.display = 'none';
    }
    
    function showToast(message, type) {
        type = type || 'info';
        if (typeof window.showToast === 'function') {
            window.showToast(message, type);
        } else {
            console.log('[' + type.toUpperCase() + '] ' + message);
        }
    }
    
    function formatNumber(num) {
        if (num === null || num === undefined) return '0';
        if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
        return Math.round(num).toString();
    }
    
    function shortenAddress(addr) {
        if (!addr) return '';
        if (addr.length <= 12) return addr;
        return addr.slice(0, 6) + '...' + addr.slice(-4);
    }
    
    // ==========================================================================
    // PUBLIC API
    // ==========================================================================
    
    return {
        init: init,
        buildCapacityGraph: buildCapacityGraph,
        viewCapacityGraph: viewCapacityGraph,
        switchToTrustGraph: switchToTrustGraph,
        computeMaxFlow: computeMaxFlow,
        clearResults: clearResults,
        highlightAllPaths: highlightAllPaths,
        selectPath: selectPath,
        isolatePath: isolatePath,
        isolateAllPaths: isolateAllPaths,
        showAllNodes: showAllNodes,
        fitToPath: fitToPath,
        copyPath: copyPath,
        zoomToNode: zoomToNode,
        getState: function() { return state; }
    };
})();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    CapacityFlow.init();
});