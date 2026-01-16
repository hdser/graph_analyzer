/**
 * Cosmos Adapter
 * 
 * Wraps @cosmograph/cosmos to implement GraphRendererInterface.
 * Provides GPU-accelerated graph visualization for large graphs.
 * 
 * Features:
 * - Position preservation on edge add/remove
 * - Edge visibility controls (show/hide without layout change)
 * - Enhanced simulation parameters
 * - Layout snapshots for save/restore
 * - Simulation progress monitoring
 * 
 * Note: This requires @cosmograph/cosmos to be loaded via CDN or npm.
 */

class CosmosAdapter extends GraphRendererInterface {
    /**
     * @param {HTMLElement} container - DOM container
     * @param {Object} options - Renderer options
     */
    constructor(container, options = {}) {
        super(container, options);
        
        this.graph = null;
        this.nodeIndices = new Map();  // id -> index
        this.nodeIds = [];             // index -> id
        this.edgeIndices = new Map();  // "source-target" -> index
        
        // State tracking
        this.selectedIndices = new Set();
        this.highlightedIndices = new Set();
        this._currentColorMetric = null;
        this._currentColorScale = null;
        this._hoveredIndex = null;
        this._performanceMode = true;
        
        // Edge tracking for queries
        this.incomingEdges = new Map();  // nodeId -> [sourceIds]
        this.outgoingEdges = new Map();  // nodeId -> [targetIds]
        
        // Position cache
        this.positions = null;
        
        // Path/Flow highlighting state
        this._pathNodeColors = new Map();  // nodeId -> [r,g,b,a]
        this._pathEdgeColors = new Map();  // edgeKey -> [r,g,b,a]
        this._isPathHighlightActive = false;
        
        // ========== NEW: Edge visibility & position preservation ==========
        this._edgesVisible = true;           // Whether edges are currently visible
        this._storedEdgeData = [];           // Store edge data when hidden
        this._edgeLinkData = null;           // Cached link data array
        
        // ========== NEW: Layout snapshots ==========
        this._layoutSnapshots = new Map();   // name -> Float32Array of positions
        
        // ========== NEW: Simulation state monitoring ==========
        this._simulationRunning = false;
        this._simulationProgress = 0;
        this._simulationAlpha = 0;
        this._onSimulationTickCallback = null;
        this._onSimulationEndCallback = null;
        
        // ========== NEW: Position preservation flags ==========
        this._preservePositionsOnEdgeChange = true;
        this._autoFitAfterEdgeChange = false;
        
        this.initialize();
    }
    
    /**
     * Initialize cosmos.gl graph
     * Uses local bundle: window.cosmosgl.Graph
     */
    initialize() {
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const styleConfig = RendererSettings.getStyleConfig();
        
        // Check if cosmos.gl is available from our local bundle
        if (typeof window.cosmosgl === 'undefined' || typeof window.cosmosgl.Graph !== 'function') {
            console.error('[CosmosAdapter] cosmos.gl not loaded');
            throw new Error('cosmos.gl library not loaded. Ensure cosmos-gl-browser.min.js is included.');
        }
        
        const CosmosGraph = window.cosmosgl.Graph;
        console.log('[CosmosAdapter] Initializing with cosmos.gl Graph');
        
        // Store reference for later use
        this.CosmosGraph = CosmosGraph;
        
        const config = {
            // Space and simulation
            spaceSize: cosmosConfig.spaceSize,
            simulationFriction: cosmosConfig.simulation.friction,
            simulationGravity: cosmosConfig.simulation.gravity,
            simulationRepulsion: cosmosConfig.simulation.repulsion,
            simulationLinkDistance: cosmosConfig.simulation.linkDistance,
            simulationLinkSpring: cosmosConfig.simulation.linkSpring,
            simulationDecay: cosmosConfig.simulation.decay || 5000,
            simulationCenter: cosmosConfig.simulation.center || 0,
            simulationRepulsionTheta: cosmosConfig.simulation.repulsionTheta || 1.15,
            simulationCluster: cosmosConfig.simulation.cluster || 0.1,
            
            // Visual appearance
            backgroundColor: cosmosConfig.backgroundColor,
            pointSize: cosmosConfig.pointSize,
            linkWidth: cosmosConfig.linkWidth,
            curvedLinks: cosmosConfig.curvedLinks,
            
            // Default colors
            pointColor: RendererSettings.hexToRgba(styleConfig.defaultNodeColor),
            linkColor: RendererSettings.hexToRgba(styleConfig.defaultEdgeColor, styleConfig.defaultEdgeOpacity),
            
            // View control
            fitViewOnInit: false,
            fitViewPadding: 0.1,
            rescalePositions: false,
            
            // Interaction
            enableDrag: cosmosConfig.enableDrag,
            enableZoom: true,
            enableRightClickRepulsion: cosmosConfig.enableRightClickRepulsion || false,
            simulationRepulsionFromMouse: cosmosConfig.simulation.repulsionFromMouse || 2.0,
            
            // Events (v2.0 uses index-based events)
            onClick: (pointIndex) => this.handleClick(pointIndex),
            onMouseMove: (pointIndex) => this.handleHover(pointIndex),
            onZoom: (zoom) => this.emit('viewportChange', { zoom }),
            
            // Simulation callbacks
            onSimulationStart: () => this._handleSimulationStart(),
            onSimulationTick: (alpha, hoverInfo) => this._handleSimulationTick(alpha, hoverInfo),
            onSimulationEnd: () => this._handleSimulationEnd(),
            onSimulationPause: () => this._handleSimulationPause(),
            onSimulationUnpause: () => this._handleSimulationUnpause()
        };
        
        this.graph = new this.CosmosGraph(this.container, config);
        
        // Setup WebGL context loss handling
        this.setupContextLossHandling();
        
        // Add container click handler as fallback for background clicks
        this.setupContainerClickHandler();
    }
    
    // ============================================================================
    // SIMULATION EVENT HANDLERS (NEW)
    // ============================================================================
    
    _handleSimulationStart() {
        this._simulationRunning = true;
        this._simulationProgress = 0;
        console.log('[CosmosAdapter] Simulation started');
        this.emit('simulationStart', {});
    }
    
    _handleSimulationTick(alpha, hoverInfo) {
        this._simulationAlpha = alpha;
        this._simulationProgress = this.graph.progress || 0;
        
        // Update positions cache from graph
        if (this.graph.getPointPositions) {
            this.positions = this.graph.getPointPositions();
        }
        
        // Call external tick callback if registered
        if (this._onSimulationTickCallback) {
            this._onSimulationTickCallback({
                alpha: alpha,
                progress: this._simulationProgress,
                hoverInfo: hoverInfo
            });
        }
        
        this.emit('simulationTick', { 
            alpha: alpha, 
            progress: this._simulationProgress 
        });
    }
    
    _handleSimulationEnd() {
        this._simulationRunning = false;
        this._simulationProgress = 1;
        console.log('[CosmosAdapter] Simulation ended');
        
        // Update positions cache
        if (this.graph.getPointPositions) {
            this.positions = this.graph.getPointPositions();
        }
        
        // Call external end callback if registered
        if (this._onSimulationEndCallback) {
            this._onSimulationEndCallback();
        }
        
        this.emit('simulationEnd', {});
    }
    
    _handleSimulationPause() {
        this._simulationRunning = false;
        console.log('[CosmosAdapter] Simulation paused');
        
        // Update positions cache
        if (this.graph.getPointPositions) {
            this.positions = this.graph.getPointPositions();
        }
        
        this.emit('simulationPause', {});
    }
    
    _handleSimulationUnpause() {
        this._simulationRunning = true;
        console.log('[CosmosAdapter] Simulation unpaused');
        this.emit('simulationUnpause', {});
    }
    
    /**
     * Setup container click handler for background clicks
     */
    setupContainerClickHandler() {
        this._nodeClickedRecently = false;
        this._lastClickTime = 0;
        
        this.container.addEventListener('click', (e) => {
            setTimeout(() => {
                const timeSinceLastClick = Date.now() - (this._lastClickTime || 0);
                if (timeSinceLastClick > 150 && this.selectedIndices.size > 0) {
                    console.log('[CosmosAdapter] Container click - clearing selection (fallback)');
                    this.clearSelection();
                    this.emit('backgroundClick', {});
                }
            }, 100);
        }, true);
    }
    
    /**
     * Setup WebGL context loss/restore handling
     */
    setupContextLossHandling() {
        this.container.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            console.warn('[CosmosAdapter] WebGL context lost');
            this.emit('contextLost', {});
        });
        
        this.container.addEventListener('webglcontextrestored', () => {
            console.log('[CosmosAdapter] WebGL context restored');
            this.reinitialize();
            this.emit('contextRestored', {});
        });
    }
    
    /**
     * Reinitialize after context loss
     */
    reinitialize() {
        const nodes = Array.from(this.nodeDataMap.values());
        const edges = Array.from(this.edgeDataMap.values());
        
        if (this.graph) {
            this.graph.destroy();
        }
        this.initialize();
        
        if (nodes.length > 0) {
            this.setData(nodes, edges);
        }
    }
    
    // ============================================================================
    // DATA METHODS
    // ============================================================================
    
    setData(nodes, edges) {
        // Clear existing mappings
        this.nodeIndices.clear();
        this.nodeIds = [];
        this.nodeDataMap.clear();
        this.edgeIndices.clear();
        this.edgeDataMap.clear();
        this.selectedIndices.clear();
        this.highlightedIndices.clear();
        this.incomingEdges.clear();
        this.outgoingEdges.clear();
        
        // Clear path highlights
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        this._isPathHighlightActive = false;
        
        // Reset edge visibility state
        this._storedEdgeData = [];
        this._edgeLinkData = null;
        
        // Build node index mapping
        nodes.forEach((node, index) => {
            let nodeId = node.id;
            if (nodeId === undefined || nodeId === null || nodeId === '') {
                nodeId = node.data?.id || node.address || node.label || `node_${index}`;
                console.warn(`[CosmosAdapter] Node at index ${index} missing ID, using:`, nodeId);
            }
            
            this.nodeIndices.set(nodeId, index);
            this.nodeIds[index] = nodeId;
            this.nodeDataMap.set(nodeId, node);
            
            this.incomingEdges.set(nodeId, []);
            this.outgoingEdges.set(nodeId, []);
        });
        
        console.log('[CosmosAdapter] setData: Loaded', this.nodeIds.length, 'nodes');
        if (this.nodeIds.length > 0) {
            console.log('[CosmosAdapter] setData: First 5 node IDs:', this.nodeIds.slice(0, 5));
        }
        
        // Build position array
        this.positions = new Float32Array(nodes.length * 2);
        nodes.forEach((node, i) => {
            this.positions[i * 2] = node.x !== undefined ? node.x : Math.random() * 1000;
            this.positions[i * 2 + 1] = node.y !== undefined ? node.y : Math.random() * 1000;
        });
        
        // Build links array (indices, not IDs)
        const linkData = new Float32Array(edges.length * 2);
        edges.forEach((edge, i) => {
            const sourceIndex = this.nodeIndices.get(edge.source);
            const targetIndex = this.nodeIndices.get(edge.target);
            
            if (sourceIndex !== undefined && targetIndex !== undefined) {
                linkData[i * 2] = sourceIndex;
                linkData[i * 2 + 1] = targetIndex;
                
                const edgeId = edge.id || `${edge.source}-${edge.target}`;
                this.edgeIndices.set(edgeId, i);
                this.edgeDataMap.set(edgeId, edge);
                
                this.incomingEdges.get(edge.target)?.push(edge.source);
                this.outgoingEdges.get(edge.source)?.push(edge.target);
            }
        });
        
        // Store link data for edge visibility toggle
        this._edgeLinkData = linkData;
        
        // Set data to cosmos
        this.graph.setPointPositions(this.positions);
        
        // Only set links if edges should be visible
        if (this._edgesVisible && linkData.length > 0) {
            this.graph.setLinks(linkData);
        } else {
            this.graph.setLinks(new Float32Array(0));
        }
        
        // Apply default colors
        this.applyDefaultColors();
        
        this.graph.render();
        
        // Fit view after data is set
        setTimeout(() => this.graph.fitView(), 100);
    }
    
    updatePositions(positions) {
        const posArray = new Float32Array(this.nodeIds.length * 2);
        const posMap = positions instanceof Map ? positions : new Map(Object.entries(positions));
        
        this.nodeIds.forEach((id, index) => {
            const pos = posMap.get(id);
            if (pos) {
                posArray[index * 2] = pos.x;
                posArray[index * 2 + 1] = pos.y;
                
                const nodeData = this.nodeDataMap.get(id);
                if (nodeData) {
                    nodeData.x = pos.x;
                    nodeData.y = pos.y;
                }
            }
        });
        
        this.positions = posArray;
        this.graph.setPointPositions(posArray, true); // dontRescale = true
        this.graph.render();
    }
    
    addNodes(nodes) {
        const existingNodes = Array.from(this.nodeDataMap.values());
        const existingEdges = Array.from(this.edgeDataMap.values());
        this.setData([...existingNodes, ...nodes], existingEdges);
    }
    
    // ============================================================================
    // EDGE MANAGEMENT WITH POSITION PRESERVATION (NEW/IMPROVED)
    // ============================================================================
    
    /**
     * Capture current node positions from cosmos.gl
     * @returns {Float32Array} Current positions array
     */
    capturePositions() {
        if (this.graph.getPointPositions) {
            this.positions = this.graph.getPointPositions();
        }
        return this.positions ? new Float32Array(this.positions) : null;
    }
    
    /**
     * Restore positions without rescaling
     * @param {Float32Array} positions - Positions to restore
     */
    restorePositions(positions) {
        if (!positions || positions.length === 0) return;
        
        this.positions = positions;
        this.graph.setPointPositions(positions, true); // dontRescale = true
        this.graph.render();
    }
    
    /**
     * Add edges while preserving current node positions
     * This is the key fix for the layout loss issue
     */
    addEdges(edges) {
        // Step 1: Capture current positions BEFORE any changes
        const savedPositions = this.capturePositions();
        const wasSimulationRunning = this._simulationRunning;
        
        // Step 2: Pause simulation to prevent layout changes
        this.graph.pause();
        
        // Step 3: Add edges to tracking
        edges.forEach(edge => {
            const edgeId = edge.id || `${edge.source}-${edge.target}`;
            this.edgeDataMap.set(edgeId, edge);
            this.incomingEdges.get(edge.target)?.push(edge.source);
            this.outgoingEdges.get(edge.source)?.push(edge.target);
        });
        
        // Step 4: Rebuild links array
        const allEdges = Array.from(this.edgeDataMap.values());
        const linkData = new Float32Array(allEdges.length * 2);
        
        allEdges.forEach((edge, i) => {
            const sourceIndex = this.nodeIndices.get(edge.source);
            const targetIndex = this.nodeIndices.get(edge.target);
            
            if (sourceIndex !== undefined && targetIndex !== undefined) {
                linkData[i * 2] = sourceIndex;
                linkData[i * 2 + 1] = targetIndex;
            }
        });
        
        // Store for edge visibility toggle
        this._edgeLinkData = linkData;
        
        // Step 5: Set links (only if edges should be visible)
        if (this._edgesVisible) {
            this.graph.setLinks(linkData);
        }
        
        // Step 6: CRITICAL - Restore positions to prevent layout jump
        if (this._preservePositionsOnEdgeChange && savedPositions) {
            this.graph.setPointPositions(savedPositions, true); // dontRescale = true
            this.positions = savedPositions;
        }
        
        // Step 7: Render
        this.graph.render();
        
        // Step 8: Optionally fit view or resume simulation
        if (this._autoFitAfterEdgeChange) {
            this.graph.fitView();
        }
        
        // Keep simulation paused - user can restart if they want layout to adapt
        console.log('[CosmosAdapter] Added', edges.length, 'edges with position preservation');
        
        this.emit('edgesAdded', { count: edges.length });
    }
    
    /**
     * Remove edges while preserving current node positions
     */
    removeEdges(edgeIds) {
        // Capture positions before changes
        const savedPositions = this.capturePositions();
        
        // Pause simulation
        this.graph.pause();
        
        // Remove edges from tracking
        edgeIds.forEach(id => {
            const edge = this.edgeDataMap.get(id);
            if (edge) {
                // Remove from incoming/outgoing tracking
                const incoming = this.incomingEdges.get(edge.target);
                if (incoming) {
                    const idx = incoming.indexOf(edge.source);
                    if (idx > -1) incoming.splice(idx, 1);
                }
                const outgoing = this.outgoingEdges.get(edge.source);
                if (outgoing) {
                    const idx = outgoing.indexOf(edge.target);
                    if (idx > -1) outgoing.splice(idx, 1);
                }
                
                this.edgeDataMap.delete(id);
                this.edgeIndices.delete(id);
            }
        });
        
        // Rebuild links array
        const allEdges = Array.from(this.edgeDataMap.values());
        const linkData = new Float32Array(allEdges.length * 2);
        
        allEdges.forEach((edge, i) => {
            const sourceIndex = this.nodeIndices.get(edge.source);
            const targetIndex = this.nodeIndices.get(edge.target);
            
            if (sourceIndex !== undefined && targetIndex !== undefined) {
                linkData[i * 2] = sourceIndex;
                linkData[i * 2 + 1] = targetIndex;
                this.edgeIndices.set(edge.id || `${edge.source}-${edge.target}`, i);
            }
        });
        
        this._edgeLinkData = linkData;
        
        // Set links
        if (this._edgesVisible) {
            this.graph.setLinks(linkData);
        }
        
        // Restore positions
        if (this._preservePositionsOnEdgeChange && savedPositions) {
            this.graph.setPointPositions(savedPositions, true);
            this.positions = savedPositions;
        }
        
        this.graph.render();
        
        console.log('[CosmosAdapter] Removed', edgeIds.length, 'edges with position preservation');
        
        this.emit('edgesRemoved', { count: edgeIds.length });
    }
    
    /**
     * Clear all edges while preserving node positions
     */
    clearEdges() {
        const savedPositions = this.capturePositions();
        const edgeCount = this.edgeDataMap.size;
        
        this.graph.pause();
        
        // Clear all edge data
        this.edgeDataMap.clear();
        this.edgeIndices.clear();
        this._edgeLinkData = new Float32Array(0);
        
        // Clear incoming/outgoing references
        this.nodeIds.forEach(id => {
            this.incomingEdges.set(id, []);
            this.outgoingEdges.set(id, []);
        });
        
        // Clear path edge colors
        this._pathEdgeColors.clear();
        
        // Set empty links
        this.graph.setLinks(new Float32Array(0));
        
        // Restore positions
        if (this._preservePositionsOnEdgeChange && savedPositions) {
            this.graph.setPointPositions(savedPositions, true);
            this.positions = savedPositions;
        }
        
        this.graph.render();
        
        console.log('[CosmosAdapter] Cleared', edgeCount, 'edges with position preservation');
        
        this.emit('edgesCleared', { count: edgeCount });
    }
    
    removeElements(nodeIds = [], edgeIds = []) {
        // If only removing edges, use position-preserving method
        if (nodeIds.length === 0 && edgeIds.length > 0) {
            this.removeEdges(edgeIds);
            return;
        }
        
        // For node removal, we need full rebuild
        nodeIds.forEach(id => this.nodeDataMap.delete(id));
        edgeIds.forEach(id => this.edgeDataMap.delete(id));
        
        // Filter edges that reference removed nodes
        this.edgeDataMap.forEach((edge, id) => {
            if (nodeIds.includes(edge.source) || nodeIds.includes(edge.target)) {
                this.edgeDataMap.delete(id);
            }
        });
        
        const nodes = Array.from(this.nodeDataMap.values());
        const edges = Array.from(this.edgeDataMap.values());
        this.setData(nodes, edges);
    }
    
    clear() {
        this.nodeIndices.clear();
        this.nodeIds = [];
        this.nodeDataMap.clear();
        this.edgeIndices.clear();
        this.edgeDataMap.clear();
        this.selectedIndices.clear();
        this.highlightedIndices.clear();
        this.incomingEdges.clear();
        this.outgoingEdges.clear();
        
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        this._isPathHighlightActive = false;
        
        this._storedEdgeData = [];
        this._edgeLinkData = null;
        this._layoutSnapshots.clear();
        
        this.graph.setPointPositions(new Float32Array(0));
        this.graph.setLinks(new Float32Array(0));
        this.graph.render();
    }
    
    // ============================================================================
    // EDGE VISIBILITY CONTROLS (NEW)
    // ============================================================================
    
    /**
     * Show edges (makes them visible without affecting layout)
     */
    showEdges() {
        if (this._edgesVisible) return;
        
        this._edgesVisible = true;
        
        // Capture positions before changes
        const savedPositions = this.capturePositions();
        
        // Pause simulation
        this.graph.pause();
        
        // Restore link data if we have it
        if (this._edgeLinkData && this._edgeLinkData.length > 0) {
            this.graph.setLinks(this._edgeLinkData);
        }
        
        // Restore positions
        if (savedPositions) {
            this.graph.setPointPositions(savedPositions, true);
            this.positions = savedPositions;
        }
        
        this.graph.render();
        
        console.log('[CosmosAdapter] Edges shown (layout preserved)');
        this.emit('edgesVisibilityChanged', { visible: true });
    }
    
    /**
     * Hide edges (makes them invisible without affecting layout)
     */
    hideEdges() {
        if (!this._edgesVisible) return;
        
        this._edgesVisible = false;
        
        // Capture positions before changes
        const savedPositions = this.capturePositions();
        
        // Pause simulation
        this.graph.pause();
        
        // Set empty links (visually hides edges)
        this.graph.setLinks(new Float32Array(0));
        
        // Restore positions to prevent any layout changes
        if (savedPositions) {
            this.graph.setPointPositions(savedPositions, true);
            this.positions = savedPositions;
        }
        
        this.graph.render();
        
        console.log('[CosmosAdapter] Edges hidden (layout preserved)');
        this.emit('edgesVisibilityChanged', { visible: false });
    }
    
    /**
     * Toggle edge visibility
     * @returns {boolean} New visibility state
     */
    toggleEdges() {
        if (this._edgesVisible) {
            this.hideEdges();
        } else {
            this.showEdges();
        }
        return this._edgesVisible;
    }
    
    /**
     * Check if edges are currently visible
     * @returns {boolean}
     */
    areEdgesVisible() {
        return this._edgesVisible;
    }
    
    /**
     * Get edge count (including hidden edges)
     * @returns {number}
     */
    getTotalEdgeCount() {
        return this.edgeDataMap.size;
    }
    
    // ============================================================================
    // LAYOUT SNAPSHOTS (NEW)
    // ============================================================================
    
    /**
     * Create a named snapshot of current positions
     * @param {string} name - Snapshot name
     * @returns {boolean} Success
     */
    createSnapshot(name = 'default') {
        const positions = this.capturePositions();
        if (!positions) return false;
        
        this._layoutSnapshots.set(name, new Float32Array(positions));
        console.log('[CosmosAdapter] Created snapshot:', name);
        return true;
    }
    
    /**
     * Restore a named snapshot
     * @param {string} name - Snapshot name
     * @returns {boolean} Success
     */
    restoreSnapshot(name = 'default') {
        const positions = this._layoutSnapshots.get(name);
        if (!positions) {
            console.warn('[CosmosAdapter] Snapshot not found:', name);
            return false;
        }
        
        this.graph.pause();
        this.restorePositions(new Float32Array(positions));
        
        console.log('[CosmosAdapter] Restored snapshot:', name);
        return true;
    }
    
    /**
     * Delete a snapshot
     * @param {string} name - Snapshot name
     * @returns {boolean} Success
     */
    deleteSnapshot(name) {
        return this._layoutSnapshots.delete(name);
    }
    
    /**
     * Get list of snapshot names
     * @returns {string[]}
     */
    getSnapshotNames() {
        return Array.from(this._layoutSnapshots.keys());
    }
    
    /**
     * Export current positions as JSON object
     * @returns {Object} { nodeId: {x, y}, ... }
     */
    exportPositions() {
        const positions = this.capturePositions();
        if (!positions) return {};
        
        const result = {};
        this.nodeIds.forEach((id, index) => {
            result[id] = {
                x: positions[index * 2],
                y: positions[index * 2 + 1]
            };
        });
        
        return result;
    }
    
    /**
     * Import positions from JSON object
     * @param {Object} positions - { nodeId: {x, y}, ... }
     * @param {boolean} fit - Whether to fit view after import
     */
    importPositions(positions, fit = true) {
        this.graph.pause();
        this.updatePositions(positions);
        
        if (fit) {
            this.graph.fitView();
        }
        
        console.log('[CosmosAdapter] Imported positions for', Object.keys(positions).length, 'nodes');
    }
    
    // ============================================================================
    // VISUAL STYLING
    // ============================================================================
    
    applyDefaultColors() {
        const styleConfig = RendererSettings.getStyleConfig();
        const defaultColor = RendererSettings.hexToRgba(styleConfig.defaultNodeColor);
        
        const colors = new Float32Array(this.nodeIds.length * 4);
        for (let i = 0; i < this.nodeIds.length; i++) {
            colors[i * 4] = defaultColor[0];
            colors[i * 4 + 1] = defaultColor[1];
            colors[i * 4 + 2] = defaultColor[2];
            colors[i * 4 + 3] = defaultColor[3];
        }
        
        this.graph.setPointColors(colors);
    }
    
    applyNodeColors(metricName, colorScale) {
        this._currentColorMetric = metricName;
        this._currentColorScale = colorScale;
        
        const gradient = ColorGradients.get(colorScale.gradient || 'spectral');
        
        let min = colorScale.min;
        let max = colorScale.max;
        
        if (min === undefined || max === undefined) {
            const values = this.nodeIds
                .map(id => this.nodeDataMap.get(id)?.[metricName])
                .filter(v => typeof v === 'number' && !isNaN(v));
            
            if (values.length > 0) {
                min = min !== undefined ? min : Math.min(...values);
                max = max !== undefined ? max : Math.max(...values);
            }
        }
        
        const colors = new Float32Array(this.nodeIds.length * 4);
        const styleConfig = RendererSettings.getStyleConfig();
        const defaultColor = RendererSettings.hexToRgba(styleConfig.defaultNodeColor);
        
        this.nodeIds.forEach((id, index) => {
            const node = this.nodeDataMap.get(id);
            const value = node?.[metricName];
            
            if (this._isPathHighlightActive && this._pathNodeColors.has(id)) {
                const pathColor = this._pathNodeColors.get(id);
                colors[index * 4] = pathColor[0];
                colors[index * 4 + 1] = pathColor[1];
                colors[index * 4 + 2] = pathColor[2];
                colors[index * 4 + 3] = pathColor[3];
            } else if (this.selectedIndices.has(index)) {
                const selColor = RendererSettings.getSelectionColorRgba();
                colors[index * 4] = selColor[0];
                colors[index * 4 + 1] = selColor[1];
                colors[index * 4 + 2] = selColor[2];
                colors[index * 4 + 3] = selColor[3];
            } else if (this.highlightedIndices.has(index)) {
                const hlColor = RendererSettings.getHighlightColorRgba();
                colors[index * 4] = hlColor[0];
                colors[index * 4 + 1] = hlColor[1];
                colors[index * 4 + 2] = hlColor[2];
                colors[index * 4 + 3] = hlColor[3];
            } else if (typeof value === 'number' && !isNaN(value) && max > min) {
                const norm = Math.max(0, Math.min(1, (value - min) / (max - min)));
                const rgb = ColorGradients.interpolateRgba(gradient, norm);
                colors[index * 4] = rgb[0];
                colors[index * 4 + 1] = rgb[1];
                colors[index * 4 + 2] = rgb[2];
                colors[index * 4 + 3] = rgb[3];
            } else {
                colors[index * 4] = defaultColor[0];
                colors[index * 4 + 1] = defaultColor[1];
                colors[index * 4 + 2] = defaultColor[2];
                colors[index * 4 + 3] = defaultColor[3];
            }
        });
        
        this.graph.setPointColors(colors);
        this.graph.render();
    }
    
    applyNodeSizes(metricName, sizeScale) {
        const { min: sizeMin, max: sizeMax } = sizeScale;
        const sizes = new Float32Array(this.nodeIds.length);
        
        const values = this.nodeIds
            .map(id => this.nodeDataMap.get(id)?.[metricName])
            .filter(v => typeof v === 'number' && !isNaN(v));
        
        if (values.length === 0) {
            sizes.fill((sizeMin + sizeMax) / 2);
            this.graph.setPointSizes(sizes);
            this.graph.render();
            return;
        }
        
        const valueMin = Math.min(...values);
        const valueMax = Math.max(...values);
        
        this.nodeIds.forEach((id, index) => {
            const node = this.nodeDataMap.get(id);
            const value = node?.[metricName];
            
            if (typeof value === 'number' && !isNaN(value) && valueMax > valueMin) {
                const norm = (value - valueMin) / (valueMax - valueMin);
                sizes[index] = sizeMin + norm * (sizeMax - sizeMin);
            } else {
                sizes[index] = (sizeMin + sizeMax) / 2;
            }
        });
        
        this.graph.setPointSizes(sizes);
        this.graph.render();
    }
    
    setEdgeStyle(style) {
        if (!this._edgeStyle) {
            this._edgeStyle = { color: '#fcfafa', opacity: 0.2, width: 1 };
        }
        
        if (style.color !== undefined) this._edgeStyle.color = style.color;
        if (style.opacity !== undefined) this._edgeStyle.opacity = style.opacity;
        if (style.width !== undefined) this._edgeStyle.width = style.width;
        
        try {
            const edgeCount = this.edgeDataMap.size;
            if (edgeCount > 0) {
                const opacity = this._edgeStyle.opacity !== undefined ? this._edgeStyle.opacity : 0.2;
                const rgba = RendererSettings.hexToRgba(this._edgeStyle.color, opacity);
                
                const colors = new Float32Array(edgeCount * 4);
                for (let i = 0; i < edgeCount; i++) {
                    colors[i * 4] = rgba[0];
                    colors[i * 4 + 1] = rgba[1];
                    colors[i * 4 + 2] = rgba[2];
                    colors[i * 4 + 3] = rgba[3];
                }
                
                if (typeof this.graph.setLinkColors === 'function') {
                    this.graph.setLinkColors(colors);
                } else if (typeof this.graph.setConfig === 'function') {
                    this.graph.setConfig({ linkColor: rgba });
                }
            }
            
            if (this._edgeStyle.width !== undefined) {
                const width = Math.max(0.1, Math.min(10, this._edgeStyle.width));
                
                if (typeof this.graph.setLinkWidth === 'function') {
                    this.graph.setLinkWidth(width);
                } else if (typeof this.graph.setConfig === 'function') {
                    this.graph.setConfig({ linkWidth: width });
                }
            }
            
            this.graph.render();
        } catch (err) {
            console.warn('[CosmosAdapter] Error setting edge style:', err);
        }
    }
    
    applyDefaultEdgeColors() {
        const styleConfig = RendererSettings.getStyleConfig();
        this.setEdgeStyle({
            color: styleConfig.defaultEdgeColor,
            opacity: styleConfig.defaultEdgeOpacity,
            width: 1
        });
    }
    
    resetStyle() {
        this._currentColorMetric = null;
        this._currentColorScale = null;
        this.clearPathHighlights();
        this.applyDefaultColors();
        this.applyDefaultEdgeColors();
        
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const sizes = new Float32Array(this.nodeIds.length);
        sizes.fill(cosmosConfig.pointSize);
        this.graph.setPointSizes(sizes);
        
        this.graph.render();
    }
    
    setPerformanceMode(enabled) {
        this._performanceMode = enabled;
    }
    
    // ============================================================================
    // NODE VISIBILITY (via alpha channel)
    // ============================================================================
    
    _hiddenNodes = new Set();
    
    showOnlyNodes(nodeIdsToShow) {
        const showSet = new Set(nodeIdsToShow);
        this._hiddenNodes.clear();
        
        this.nodeIds.forEach(id => {
            if (!showSet.has(id)) {
                this._hiddenNodes.add(id);
            }
        });
        
        this.updateNodeVisibility();
        console.log('[CosmosAdapter] Showing only', nodeIdsToShow.length, 'nodes, hiding', this._hiddenNodes.size);
    }
    
    hideNodes(nodeIdsToHide) {
        nodeIdsToHide.forEach(id => this._hiddenNodes.add(id));
        this.updateNodeVisibility();
        console.log('[CosmosAdapter] Hidden', nodeIdsToHide.length, 'nodes');
    }
    
    showAllNodes() {
        this._hiddenNodes.clear();
        this.updateNodeVisibility();
        console.log('[CosmosAdapter] All nodes visible');
    }
    
    isNodeHidden(nodeId) {
        return this._hiddenNodes.has(nodeId);
    }
    
    getHiddenNodeIds() {
        return Array.from(this._hiddenNodes);
    }
    
    updateNodeVisibility() {
        const colors = new Float32Array(this.nodeIds.length * 4);
        const styleConfig = RendererSettings.getStyleConfig();
        const defaultColor = RendererSettings.hexToRgba(styleConfig.defaultNodeColor);
        
        let gradient = null;
        let min = 0, max = 1;
        
        if (this._currentColorMetric && this._currentColorScale) {
            gradient = ColorGradients.get(this._currentColorScale.gradient || 'spectral');
            
            const values = this.nodeIds
                .map(id => this.nodeDataMap.get(id)?.[this._currentColorMetric])
                .filter(v => typeof v === 'number' && !isNaN(v));
            
            if (values.length > 0) {
                min = this._currentColorScale.min !== undefined ? this._currentColorScale.min : Math.min(...values);
                max = this._currentColorScale.max !== undefined ? this._currentColorScale.max : Math.max(...values);
            }
        }
        
        this.nodeIds.forEach((id, index) => {
            const isHidden = this._hiddenNodes.has(id);
            let color;
            
            if (this._isPathHighlightActive && this._pathNodeColors.has(id)) {
                color = this._pathNodeColors.get(id);
            } else if (this.selectedIndices.has(index)) {
                color = RendererSettings.getSelectionColorRgba();
            } else if (this.highlightedIndices.has(index)) {
                color = RendererSettings.getHighlightColorRgba();
            } else if (gradient && this._currentColorMetric) {
                const node = this.nodeDataMap.get(id);
                const value = node?.[this._currentColorMetric];
                
                if (typeof value === 'number' && !isNaN(value) && max > min) {
                    const norm = Math.max(0, Math.min(1, (value - min) / (max - min)));
                    color = ColorGradients.interpolateRgba(gradient, norm);
                } else {
                    color = defaultColor;
                }
            } else {
                color = defaultColor;
            }
            
            colors[index * 4] = color[0];
            colors[index * 4 + 1] = color[1];
            colors[index * 4 + 2] = color[2];
            colors[index * 4 + 3] = isHidden ? 0 : (color[3] !== undefined ? color[3] : 1);
        });
        
        this.graph.setPointColors(colors);
        this.updateEdgeVisibility();
        this.graph.render();
    }
    
    updateEdgeVisibility() {
        const edgeCount = this.edgeDataMap.size;
        if (edgeCount === 0) return;
        
        const edgeStyle = this._edgeStyle || { color: '#fcfafa', opacity: 0.2 };
        const baseRgba = RendererSettings.hexToRgba(edgeStyle.color, edgeStyle.opacity);
        
        const colors = new Float32Array(edgeCount * 4);
        let edgeIndex = 0;
        
        this.edgeDataMap.forEach((edge, edgeId) => {
            const sourceHidden = this._hiddenNodes.has(edge.source);
            const targetHidden = this._hiddenNodes.has(edge.target);
            const isHidden = sourceHidden || targetHidden;
            
            const edgeKey = `${edge.source}-${edge.target}`;
            const reverseKey = `${edge.target}-${edge.source}`;
            const pathColor = this._pathEdgeColors.get(edgeKey) || this._pathEdgeColors.get(reverseKey);
            
            if (this._isPathHighlightActive && pathColor && !isHidden) {
                colors[edgeIndex * 4] = pathColor[0];
                colors[edgeIndex * 4 + 1] = pathColor[1];
                colors[edgeIndex * 4 + 2] = pathColor[2];
                colors[edgeIndex * 4 + 3] = pathColor[3];
            } else {
                colors[edgeIndex * 4] = baseRgba[0];
                colors[edgeIndex * 4 + 1] = baseRgba[1];
                colors[edgeIndex * 4 + 2] = baseRgba[2];
                colors[edgeIndex * 4 + 3] = isHidden ? 0 : baseRgba[3];
            }
            
            edgeIndex++;
        });
        
        if (typeof this.graph.setLinkColors === 'function') {
            this.graph.setLinkColors(colors);
        }
    }
    
    // ============================================================================
    // PATH / FLOW HIGHLIGHTING
    // ============================================================================
    
    highlightPathNodes(nodeColorMap) {
        console.log('[CosmosAdapter] Highlighting path nodes:', nodeColorMap.size);
        
        this._pathNodeColors.clear();
        this._isPathHighlightActive = true;
        
        nodeColorMap.forEach((config, nodeId) => {
            const rgba = RendererSettings.hexToRgba(config.color, 1.0);
            this._pathNodeColors.set(nodeId, rgba);
        });
        
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const defaultSize = cosmosConfig.pointSize || 4;
        const sizes = new Float32Array(this.nodeIds.length);
        
        this.nodeIds.forEach((id, index) => {
            const config = nodeColorMap.get(id);
            if (config) {
                if (config.type === 'source' || config.type === 'target') {
                    sizes[index] = defaultSize * 3;
                } else {
                    sizes[index] = defaultSize * 2;
                }
            } else {
                sizes[index] = defaultSize;
            }
        });
        
        this.graph.setPointSizes(sizes);
        this.updateNodeVisibility();
    }
    
    highlightPathEdges(edgePairs, color = '#00d4ff', opacity = 1.0) {
        console.log('[CosmosAdapter] Highlighting path edges:', edgePairs.length);
        
        this._pathEdgeColors.clear();
        this._isPathHighlightActive = true;
        
        const rgba = RendererSettings.hexToRgba(color, opacity);
        
        edgePairs.forEach(pair => {
            const edgeKey = `${pair.source}-${pair.target}`;
            this._pathEdgeColors.set(edgeKey, rgba);
        });
        
        this.updateEdgeVisibility();
        this.graph.render();
    }
    
    clearPathHighlights() {
        console.log('[CosmosAdapter] Clearing path highlights');
        
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        this._isPathHighlightActive = false;
        
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const sizes = new Float32Array(this.nodeIds.length);
        sizes.fill(cosmosConfig.pointSize || 4);
        this.graph.setPointSizes(sizes);
        
        this.updateNodeVisibility();
    }
    
    isPathHighlightActive() {
        return this._isPathHighlightActive;
    }
    
    // ============================================================================
    // SELECTION & HIGHLIGHTING
    // ============================================================================
    
    selectNodes(nodeIds, additive = false) {
        if (!additive) {
            this.selectedIndices.clear();
        }
        
        nodeIds.forEach(id => {
            const index = this.nodeIndices.get(id);
            if (index !== undefined) {
                this.selectedIndices.add(index);
            }
        });
        
        this.updateSelectionVisuals();
        this.emit('selectionChange', { 
            nodes: this.getSelectedNodes(),
            edges: this.getSelectedEdges()
        });
    }
    
    selectEdges(edgeIds, additive = false) {
        console.warn('[CosmosAdapter] Edge selection not visually supported');
    }
    
    getSelectedNodes() {
        return Array.from(this.selectedIndices).map(idx => this.nodeIds[idx]);
    }
    
    getSelectedEdges() {
        return [];
    }
    
    clearSelection() {
        this.selectedIndices.clear();
        this.updateSelectionVisuals();
        this.emit('selectionChange', { nodes: [], edges: [] });
    }
    
    highlightNodes(nodeIds, className = 'highlighted') {
        nodeIds.forEach(id => {
            const index = this.nodeIndices.get(id);
            if (index !== undefined) {
                this.highlightedIndices.add(index);
            }
        });
        this.updateSelectionVisuals();
    }
    
    highlightNeighbors(nodeId, direction = 'both') {
        this.highlightedIndices.clear();
        
        let neighbors = [];
        switch (direction) {
            case 'in':
                neighbors = this.incomingEdges.get(nodeId) || [];
                break;
            case 'out':
                neighbors = this.outgoingEdges.get(nodeId) || [];
                break;
            default:
                neighbors = [
                    ...(this.incomingEdges.get(nodeId) || []),
                    ...(this.outgoingEdges.get(nodeId) || [])
                ];
        }
        
        neighbors.forEach(id => {
            const index = this.nodeIndices.get(id);
            if (index !== undefined) {
                this.highlightedIndices.add(index);
            }
        });
        
        this.updateSelectionVisuals();
    }
    
    clearHighlights() {
        this.highlightedIndices.clear();
        this.updateSelectionVisuals();
    }
    
    addClass(elementIds, className, type = 'nodes') {
        if (type === 'nodes' && className === 'highlighted') {
            this.highlightNodes(elementIds);
        }
    }
    
    removeClass(elementIds, className, type = 'nodes') {
        if (type === 'nodes' && className === 'highlighted') {
            elementIds.forEach(id => {
                const index = this.nodeIndices.get(id);
                if (index !== undefined) {
                    this.highlightedIndices.delete(index);
                }
            });
            this.updateSelectionVisuals();
        }
    }
    
    updateSelectionVisuals() {
        if (this._isPathHighlightActive) {
            this.updateNodeVisibility();
            return;
        }
        
        if (this._currentColorMetric && this._currentColorScale) {
            this.applyNodeColors(this._currentColorMetric, this._currentColorScale);
            return;
        }
        
        const styleConfig = RendererSettings.getStyleConfig();
        const defaultColor = RendererSettings.hexToRgba(styleConfig.defaultNodeColor);
        const selectionColor = RendererSettings.getSelectionColorRgba();
        const highlightColor = RendererSettings.getHighlightColorRgba();
        
        const colors = new Float32Array(this.nodeIds.length * 4);
        
        this.nodeIds.forEach((id, index) => {
            let color;
            if (this.selectedIndices.has(index)) {
                color = selectionColor;
            } else if (this.highlightedIndices.has(index)) {
                color = highlightColor;
            } else {
                color = defaultColor;
            }
            
            colors[index * 4] = color[0];
            colors[index * 4 + 1] = color[1];
            colors[index * 4 + 2] = color[2];
            colors[index * 4 + 3] = color[3];
        });
        
        this.graph.setPointColors(colors);
        this.graph.render();
    }
    
    // ============================================================================
    // VIEWPORT CONTROL
    // ============================================================================
    
    fitView(nodeIds = null, padding = 0.1) {
        if (!nodeIds || nodeIds.length === 0) {
            this.graph.fitView();
            return;
        }
        
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let validCount = 0;
        
        nodeIds.forEach(id => {
            const index = this.nodeIndices.get(id);
            if (index !== undefined && this.positions) {
                const x = this.positions[index * 2];
                const y = this.positions[index * 2 + 1];
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
                validCount++;
            }
        });
        
        if (validCount === 0) {
            this.graph.fitView();
            return;
        }
        
        if (validCount === 1 || (maxX - minX < 1 && maxY - minY < 1)) {
            const nodeId = nodeIds[0];
            const index = this.nodeIndices.get(nodeId);
            if (index !== undefined) {
                this.graph.zoomToPointByIndex(index, 500, 2);
            }
            return;
        }
        
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const rangeX = (maxX - minX) * (1 + padding);
        const rangeY = (maxY - minY) * (1 + padding);
        
        if (typeof this.graph.setZoomLevel === 'function') {
            const containerWidth = this.container.clientWidth;
            const containerHeight = this.container.clientHeight;
            const scaleX = containerWidth / rangeX;
            const scaleY = containerHeight / rangeY;
            const zoom = Math.min(scaleX, scaleY, 10);
            
            this.graph.setZoomLevel(zoom);
        }
        
        this.graph.fitView();
    }
    
    center() {
        this.graph.fitView();
    }
    
    zoomToNode(nodeId, zoomLevel = 3, duration = 500) {
        const index = this.nodeIndices.get(nodeId);
        if (index !== undefined) {
            this.graph.zoomToPointByIndex(index, duration, zoomLevel);
        }
    }
    
    getViewport() {
        return {
            zoom: this.graph.getZoom?.() || 1,
            pan: { x: 0, y: 0 }
        };
    }
    
    setViewport(viewport) {
        if (viewport.zoom) {
            this.graph.setZoom?.(viewport.zoom);
        }
    }
    
    setZoom(level) {
        this.graph.setZoom?.(level);
    }
    
    // ============================================================================
    // GRAPH QUERIES
    // ============================================================================
    
    getIncomingNeighbors(nodeId) {
        return this.incomingEdges.get(nodeId) || [];
    }
    
    getOutgoingNeighbors(nodeId) {
        return this.outgoingEdges.get(nodeId) || [];
    }
    
    getConnectedEdges(nodeId, direction = 'both') {
        const edges = [];
        
        if (direction === 'in' || direction === 'both') {
            (this.incomingEdges.get(nodeId) || []).forEach(sourceId => {
                const edgeId = `${sourceId}-${nodeId}`;
                const edgeData = this.edgeDataMap.get(edgeId);
                if (edgeData) {
                    edges.push({
                        id: edgeId,
                        source: sourceId,
                        target: nodeId,
                        data: edgeData
                    });
                }
            });
        }
        
        if (direction === 'out' || direction === 'both') {
            (this.outgoingEdges.get(nodeId) || []).forEach(targetId => {
                const edgeId = `${nodeId}-${targetId}`;
                const edgeData = this.edgeDataMap.get(edgeId);
                if (edgeData) {
                    edges.push({
                        id: edgeId,
                        source: nodeId,
                        target: targetId,
                        data: edgeData
                    });
                }
            });
        }
        
        return edges;
    }
    
    /**
     * Get all edge IDs
     * @returns {string[]}
     */
    getAllEdgeIds() {
        return Array.from(this.edgeDataMap.keys());
    }
    
    // ============================================================================
    // EVENT HANDLERS
    // ============================================================================
    
    handleClick(index) {
        this._lastClickTime = Date.now();
        
        const isBackgroundClick = index === undefined || index === null || index === -1 || 
                                   Number.isNaN(index) || index < 0 || index >= this.nodeIds.length;
        
        if (isBackgroundClick) {
            console.log('[CosmosAdapter] Background click detected, clearing selection');
            if (this.selectedIndices.size > 0) {
                this.clearSelection();
            }
            this.emit('backgroundClick', {});
            return;
        }
        
        this._nodeClickedRecently = true;
        setTimeout(() => { this._nodeClickedRecently = false; }, 100);
        
        const nodeId = this.nodeIds[index];
        const nodeData = this.nodeDataMap.get(nodeId);
        
        console.log('[CosmosAdapter] Node clicked:', nodeId, 'index:', index);
        
        if (this.selectedIndices.has(index)) {
            this.selectedIndices.delete(index);
        } else {
            this.selectedIndices.add(index);
        }
        
        this.updateSelectionVisuals();
        
        this.emit('nodeClick', {
            id: nodeId,
            index,
            data: nodeData,
            position: this.getNodePosition(nodeId)
        });
        
        this.emit('selectionChange', {
            nodes: this.getSelectedNodes(),
            edges: []
        });
    }
    
    handleHover(index) {
        if (index === this._hoveredIndex) return;
        this._hoveredIndex = index;
        
        if (index === undefined || index < 0 || index >= this.nodeIds.length) {
            this.emit('nodeHover', { id: null, data: null });
            return;
        }
        
        const nodeId = this.nodeIds[index];
        const nodeData = this.nodeDataMap.get(nodeId);
        
        this.emit('nodeHover', {
            id: nodeId,
            index,
            data: nodeData
        });
    }
    
    // ============================================================================
    // UTILITY
    // ============================================================================
    
    getType() {
        return 'cosmos';
    }
    
    getStats() {
        return {
            type: 'cosmos',
            nodeCount: this.nodeIds.length,
            edgeCount: this.edgeDataMap.size,
            visibleEdgeCount: this._edgesVisible ? this.edgeDataMap.size : 0,
            selectedNodes: this.selectedIndices.size,
            selectedEdges: 0,
            simulationRunning: this._simulationRunning,
            simulationProgress: this._simulationProgress,
            edgesVisible: this._edgesVisible
        };
    }
    
    getNodePosition(nodeId) {
        const index = this.nodeIndices.get(nodeId);
        if (index === undefined || !this.positions) return null;
        
        return {
            x: this.positions[index * 2],
            y: this.positions[index * 2 + 1]
        };
    }
    
    async exportPNG() {
        const canvas = this.container.querySelector('canvas');
        if (!canvas) return null;
        
        return new Promise((resolve) => {
            canvas.toBlob(resolve, 'image/png');
        });
    }
    
    // ============================================================================
    // BATCH OPERATIONS
    // ============================================================================
    
    startBatch() {
        // cosmos.gl handles batching internally
    }
    
    endBatch() {
        this.graph.render();
    }
    
    // ============================================================================
    // LIFECYCLE
    // ============================================================================
    
    render() {
        this.graph.render();
    }
    
    resize() {
        this.graph.render();
    }
    
    dispose() {
        super.dispose();
        if (this.graph) {
            this.graph.destroy();
            this.graph = null;
        }
        this.nodeIndices.clear();
        this.nodeIds = [];
        this.edgeIndices.clear();
        this.incomingEdges.clear();
        this.outgoingEdges.clear();
        this.positions = null;
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        this._layoutSnapshots.clear();
        this._storedEdgeData = [];
        this._edgeLinkData = null;
    }
    
    // ============================================================================
    // COSMOS-SPECIFIC METHODS
    // ============================================================================
    
    getGraph() {
        return this.graph;
    }
    
    /**
     * Start force simulation
     * @param {number} alpha - Initial alpha value (0-1), default 1
     */
    startSimulation(alpha = 1) {
        this.graph.start(alpha);
    }
    
    /**
     * Pause force simulation (preserves state)
     */
    pauseSimulation() {
        this.graph.pause();
    }
    
    /**
     * Unpause/resume force simulation
     */
    unpauseSimulation() {
        if (typeof this.graph.unpause === 'function') {
            this.graph.unpause();
        } else {
            this.graph.start(this._simulationAlpha || 0.3);
        }
    }
    
    /**
     * Stop force simulation (resets state)
     */
    stopSimulation() {
        if (typeof this.graph.stop === 'function') {
            this.graph.stop();
        } else {
            this.graph.pause();
        }
    }
    
    /**
     * Run a single simulation step manually
     */
    stepSimulation() {
        if (typeof this.graph.step === 'function') {
            this.graph.step();
            // Update positions
            if (this.graph.getPointPositions) {
                this.positions = this.graph.getPointPositions();
            }
        }
    }
    
    /**
     * @deprecated Use unpauseSimulation instead
     */
    restartSimulation() {
        console.warn('[CosmosAdapter] restartSimulation is deprecated, use unpauseSimulation');
        this.unpauseSimulation();
    }
    
    /**
     * Get simulation progress (0-1)
     * @returns {number}
     */
    getSimulationProgress() {
        return this.graph.progress || this._simulationProgress;
    }
    
    /**
     * Check if simulation is currently running
     * @returns {boolean}
     */
    isSimulationRunning() {
        return this.graph.isSimulationRunning || this._simulationRunning;
    }
    
    /**
     * Register callback for simulation tick events
     * @param {Function} callback - Called with {alpha, progress, hoverInfo}
     */
    onSimulationTick(callback) {
        this._onSimulationTickCallback = callback;
    }
    
    /**
     * Register callback for simulation end event
     * @param {Function} callback
     */
    onSimulationEnd(callback) {
        this._onSimulationEndCallback = callback;
    }
    
    /**
     * Set simulation parameters dynamically
     * @param {Object} params - Simulation parameters (all optional)
     */
    setSimulationParams(params) {
        if (!this.graph || typeof this.graph.setConfig !== 'function') {
            console.warn('[CosmosAdapter] setConfig not available');
            return false;
        }
        
        const config = {};
        
        // Core parameters
        if (params.repulsion !== undefined) {
            config.simulationRepulsion = params.repulsion;
        }
        if (params.gravity !== undefined) {
            config.simulationGravity = params.gravity;
        }
        if (params.linkDistance !== undefined) {
            config.simulationLinkDistance = params.linkDistance;
        }
        if (params.friction !== undefined) {
            config.simulationFriction = params.friction;
        }
        
        // Extended parameters
        if (params.decay !== undefined) {
            config.simulationDecay = params.decay;
        }
        if (params.center !== undefined) {
            config.simulationCenter = params.center;
        }
        if (params.linkSpring !== undefined) {
            config.simulationLinkSpring = params.linkSpring;
        }
        if (params.repulsionTheta !== undefined) {
            config.simulationRepulsionTheta = params.repulsionTheta;
        }
        if (params.cluster !== undefined) {
            config.simulationCluster = params.cluster;
        }
        if (params.repulsionFromMouse !== undefined) {
            config.simulationRepulsionFromMouse = params.repulsionFromMouse;
        }
        if (params.enableRightClickRepulsion !== undefined) {
            config.enableRightClickRepulsion = params.enableRightClickRepulsion;
        }
        
        try {
            this.graph.setConfig(config);
            console.log('[CosmosAdapter] Simulation params updated:', config);
            return true;
        } catch (e) {
            console.error('[CosmosAdapter] Failed to set simulation params:', e);
            return false;
        }
    }
    
    /**
     * Apply a simulation preset
     * @param {string} presetName - Name of preset to apply
     * @returns {boolean} Success
     */
    applySimulationPreset(presetName) {
        const presets = RendererSettings.getSimulationPresets?.() || {};
        const preset = presets[presetName];
        
        if (!preset) {
            console.warn('[CosmosAdapter] Preset not found:', presetName);
            return false;
        }
        
        console.log('[CosmosAdapter] Applying preset:', presetName);
        return this.setSimulationParams(preset);
    }
    
    /**
     * Set whether to preserve positions on edge changes
     * @param {boolean} preserve
     */
    setPreservePositionsOnEdgeChange(preserve) {
        this._preservePositionsOnEdgeChange = preserve;
    }
    
    /**
     * Set whether to auto-fit view after edge changes
     * @param {boolean} autoFit
     */
    setAutoFitAfterEdgeChange(autoFit) {
        this._autoFitAfterEdgeChange = autoFit;
    }
    
    getNodeIndex(nodeId) {
        return this.nodeIndices.get(nodeId);
    }
    
    getNodeIdByIndex(index) {
        return this.nodeIds[index];
    }
}

// Make available globally
window.CosmosAdapter = CosmosAdapter;