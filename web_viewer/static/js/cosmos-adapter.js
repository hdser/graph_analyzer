/**
 * Cosmos Adapter
 * 
 * Wraps @cosmograph/cosmos to implement GraphRendererInterface.
 * Provides GPU-accelerated graph visualization for large graphs.
 * 
 * Features:
 * - Position preservation on edge add/remove
 * - Edge visibility controls (show/hide without layout change)
 * - Enhanced simulation parameters with real-time updates
 * - Layout snapshots for save/restore
 * - Simulation progress monitoring
 * - Preset system for quick configuration
 * - Visual parameter controls
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
        
        // ========== Edge visibility & position preservation ==========
        this._edgesVisible = true;           // Whether edges are currently visible
        this._storedEdgeData = [];           // Store edge data when hidden
        this._edgeLinkData = null;           // Cached link data array
        
        // ========== Color state ==========
        this._baseNodeColors = null;         // The current "base" node colors (before selection)
        
        // ========== Layout snapshots ==========
        this._layoutSnapshots = new Map();   // name -> Float32Array of positions
        
        // ========== Simulation state monitoring ==========
        this._simulationRunning = false;
        this._simulationProgress = 0;
        this._simulationAlpha = 0;
        this._onSimulationTickCallback = null;
        this._onSimulationEndCallback = null;
        this._onSimulationStartCallback = null;
        this._onSimulationPauseCallback = null;
        
        // ========== Position preservation flags ==========
        this._preservePositionsOnEdgeChange = true;
        this._autoFitAfterEdgeChange = false;
        
        // ========== Current simulation parameters (for tracking) ==========
        this._currentSimParams = {
            repulsion: 1.0,
            gravity: 0.25,
            center: 0,
            repulsionTheta: 1.15,
            cluster: 0.1,
            linkDistance: 10,
            linkSpring: 1.0,
            friction: 0.85,
            decay: 5000,
            repulsionFromMouse: 2.0,
            enableRightClickRepulsion: false
        };
        
        // ========== Current visual parameters ==========
        this._currentVisualParams = {
            pointSizeScale: 1.0,
            linkWidthScale: 1.0,
            curvedLinks: true,
            curvedLinkWeight: 0.8,
            scalePointsOnZoom: false,
            scaleLinksOnZoom: false,
            showFPSMonitor: false,
            linkOpacity: 1.0,
            pointOpacity: 1.0
        };
        
        // ========== Real-time update mode ==========
        this._realTimeUpdates = true;
        this._updateDebounceTimer = null;
        this._updateDebounceDelay = 50; // ms
        
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
        
        // Initialize current params from config
        if (cosmosConfig.simulation) {
            Object.assign(this._currentSimParams, {
                friction: cosmosConfig.simulation.friction ?? 0.85,
                gravity: cosmosConfig.simulation.gravity ?? 0.25,
                repulsion: cosmosConfig.simulation.repulsion ?? 1.0,
                linkDistance: cosmosConfig.simulation.linkDistance ?? 10,
                linkSpring: cosmosConfig.simulation.linkSpring ?? 1.0,
                decay: cosmosConfig.simulation.decay ?? 100000,  // Very high for continuous simulation
                center: cosmosConfig.simulation.center ?? 0,
                repulsionTheta: cosmosConfig.simulation.repulsionTheta ?? 1.15,
                cluster: cosmosConfig.simulation.cluster ?? 0.1,
                repulsionFromMouse: cosmosConfig.simulation.repulsionFromMouse ?? 2.0
            });
        }
        
        const config = {
            // Space and simulation
            spaceSize: cosmosConfig.spaceSize,
            simulationFriction: this._currentSimParams.friction,
            simulationGravity: this._currentSimParams.gravity,
            simulationRepulsion: this._currentSimParams.repulsion,
            simulationLinkDistance: this._currentSimParams.linkDistance,
            simulationLinkSpring: this._currentSimParams.linkSpring,
            simulationDecay: this._currentSimParams.decay,
            simulationCenter: this._currentSimParams.center,
            simulationRepulsionTheta: this._currentSimParams.repulsionTheta,
            simulationCluster: this._currentSimParams.cluster,
            simulationRepulsionFromMouse: this._currentSimParams.repulsionFromMouse,
            enableRightClickRepulsion: this._currentSimParams.enableRightClickRepulsion,
            
            // Visual appearance
            backgroundColor: cosmosConfig.backgroundColor,
            pointDefaultSize: cosmosConfig.pointSize || 6,
            linkDefaultWidth: cosmosConfig.linkWidth || 1,
            curvedLinks: cosmosConfig.curvedLinks ?? true,
            pointSizeScale: this._currentVisualParams.pointSizeScale,
            linkWidthScale: this._currentVisualParams.linkWidthScale,
            
            // Default colors
            pointDefaultColor: RendererSettings.hexToRgba(styleConfig.defaultNodeColor),
            linkDefaultColor: RendererSettings.hexToRgba(styleConfig.defaultEdgeColor, styleConfig.defaultEdgeOpacity),
            
            // View control
            fitViewOnInit: false,
            fitViewPadding: 0.1,
            rescalePositions: false,
            
            // Interaction
            enableDrag: cosmosConfig.enableDrag,
            enableZoom: true,
            
            // Performance
            showFPSMonitor: this._currentVisualParams.showFPSMonitor,
            
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
    // SIMULATION EVENT HANDLERS
    // ============================================================================
    
    _handleSimulationStart() {
        this._simulationRunning = true;
        this._simulationProgress = 0;
        console.log('[CosmosAdapter] Simulation started');
        
        // Auto-stop simulation after 10 seconds on initial load
        if (!this._simulationAutoStopDone) {
            this._simulationAutoStopDone = true;
            this._simulationAutoStopTimer = setTimeout(() => {
                if (this._simulationRunning && this.graph) {
                    console.log('[CosmosAdapter] Auto-stopping simulation after 10 seconds');
                    this.pauseSimulation();
                }
            }, 10000); // 10 seconds
        }
        
        if (this._onSimulationStartCallback) {
            this._onSimulationStartCallback();
        }
        
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
                hoverInfo: hoverInfo,
                isRunning: this._simulationRunning
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
        
        if (this._onSimulationPauseCallback) {
            this._onSimulationPauseCallback();
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
        
        // Clear base colors so they get recreated fresh
        this._baseNodeColors = null;
        
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
        
        // Store edge data for isolation/visibility filtering
        this._storedEdgeData = edges.map(edge => ({
            source: edge.source,
            target: edge.target,
            id: edge.id || `${edge.source}-${edge.target}`,
            data: edge
        }));
        
        // Set data to cosmos
        this.graph.setPointPositions(this.positions);
        
        // Only set links if edges should be visible
        if (this._edgesVisible && linkData.length > 0) {
            this.graph.setLinks(linkData);
        } else {
            this.graph.setLinks(new Float32Array(0));
        }
        
        // Apply default colors for nodes and edges
        this.applyDefaultColors();
        this.applyDefaultEdgeColors();
        
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
    // EDGE MANAGEMENT WITH POSITION PRESERVATION
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
            }
        });
        
        this._edgeLinkData = linkData;
        
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
    
    // ============================================================================
    // EDGE VISIBILITY CONTROLS
    // ============================================================================
    
    /**
     * Hide all edges without affecting layout
     */
    hideEdges() {
        if (!this._edgesVisible) return;
        
        this._edgesVisible = false;
        this.graph.setLinks(new Float32Array(0));
        this.graph.render();
        
        console.log('[CosmosAdapter] Edges hidden');
        this.emit('edgesVisibilityChanged', { visible: false });
    }
    
    /**
     * Show all edges (restore from hidden state)
     */
    showEdges() {
        if (this._edgesVisible) return;
        
        this._edgesVisible = true;
        if (this._edgeLinkData && this._edgeLinkData.length > 0) {
            this.graph.setLinks(this._edgeLinkData);
        }
        this.graph.render();
        
        console.log('[CosmosAdapter] Edges shown');
        this.emit('edgesVisibilityChanged', { visible: true });
    }
    
    /**
     * Toggle edge visibility
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
     */
    areEdgesVisible() {
        return this._edgesVisible;
    }
    
    // ============================================================================
    // SIMULATION CONTROL METHODS
    // ============================================================================
    
    /**
     * Start force simulation
     * @param {number} alpha - Initial alpha value (0-1), default 1
     */
    /**
     * Start force simulation
     * @param {number} alpha - Initial alpha value (0-1), default 1
     */
    startSimulation(alpha = 1) {
        console.log('[CosmosAdapter] startSimulation called with alpha:', alpha);
        console.log('[CosmosAdapter] graph object:', !!this.graph, typeof this.graph?.start);
        if (this.graph && typeof this.graph.start === 'function') {
            this.graph.start(alpha);
            console.log('[CosmosAdapter] Simulation started');
        } else {
            console.error('[CosmosAdapter] Cannot start simulation - graph or start method not available');
        }
    }
    
    /**
     * Pause force simulation (preserves state)
     */
    /**
     * Pause force simulation (preserves state)
     */
    pauseSimulation() {
        console.log('[CosmosAdapter] pauseSimulation called');
        if (this.graph && typeof this.graph.pause === 'function') {
            this.graph.pause();
            console.log('[CosmosAdapter] Simulation paused');
        } else {
            console.error('[CosmosAdapter] Cannot pause simulation - graph or pause method not available');
        }
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
    /**
     * Run a single simulation step manually
     */
    stepSimulation() {
        console.log('[CosmosAdapter] stepSimulation called');
        if (this.graph && typeof this.graph.step === 'function') {
            this.graph.step();
            // Update positions
            if (this.graph.getPointPositions) {
                this.positions = this.graph.getPointPositions();
            }
            this.graph.render();
            console.log('[CosmosAdapter] Step executed');
        } else {
            console.error('[CosmosAdapter] Cannot step simulation - graph or step method not available');
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
     * Get current simulation alpha
     * @returns {number}
     */
    getSimulationAlpha() {
        return this._simulationAlpha;
    }
    
    /**
     * Check if simulation is currently running
     * @returns {boolean}
     */
    isSimulationRunning() {
        return this.graph.isSimulationRunning || this._simulationRunning;
    }
    
    /**
     * Register callback for simulation start event
     * @param {Function} callback
     */
    onSimulationStart(callback) {
        this._onSimulationStartCallback = callback;
    }
    
    /**
     * Register callback for simulation tick events
     * @param {Function} callback - Called with {alpha, progress, hoverInfo, isRunning}
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
     * Register callback for simulation pause event
     * @param {Function} callback
     */
    onSimulationPause(callback) {
        this._onSimulationPauseCallback = callback;
    }
    
    // ============================================================================
    // SIMULATION PARAMETER METHODS (ENHANCED)
    // ============================================================================
    
    /**
     * Set simulation parameters dynamically with immediate effect
     * @param {Object} params - Simulation parameters (all optional)
     * @param {Object} options - Options for how to apply
     * @param {boolean} options.restart - Whether to restart simulation (default: true)
     * @param {number} options.alpha - Alpha value for restart (default: 0.3)
     * @returns {boolean} Success
     */
    setSimulationParams(params, options = {}) {
        const { restart = true, alpha = 0.3 } = options;
        
        console.log('[CosmosAdapter] setSimulationParams called with:', params, options);
        
        if (!this.graph || typeof this.graph.setConfig !== 'function') {
            console.error('[CosmosAdapter] setConfig not available on graph');
            return false;
        }
        
        const config = {};
        
        // Core force parameters
        if (params.repulsion !== undefined) {
            config.simulationRepulsion = params.repulsion;
            this._currentSimParams.repulsion = params.repulsion;
        }
        if (params.gravity !== undefined) {
            config.simulationGravity = params.gravity;
            this._currentSimParams.gravity = params.gravity;
        }
        if (params.center !== undefined) {
            config.simulationCenter = params.center;
            this._currentSimParams.center = params.center;
        }
        if (params.repulsionTheta !== undefined) {
            config.simulationRepulsionTheta = params.repulsionTheta;
            this._currentSimParams.repulsionTheta = params.repulsionTheta;
        }
        if (params.cluster !== undefined) {
            config.simulationCluster = params.cluster;
            this._currentSimParams.cluster = params.cluster;
        }
        
        // Link parameters
        if (params.linkDistance !== undefined) {
            config.simulationLinkDistance = params.linkDistance;
            this._currentSimParams.linkDistance = params.linkDistance;
        }
        if (params.linkSpring !== undefined) {
            config.simulationLinkSpring = params.linkSpring;
            this._currentSimParams.linkSpring = params.linkSpring;
        }
        
        // Behavior parameters
        if (params.friction !== undefined) {
            config.simulationFriction = params.friction;
            this._currentSimParams.friction = params.friction;
        }
        if (params.decay !== undefined) {
            config.simulationDecay = params.decay;
            this._currentSimParams.decay = params.decay;
        }
        if (params.repulsionFromMouse !== undefined) {
            config.simulationRepulsionFromMouse = params.repulsionFromMouse;
            this._currentSimParams.repulsionFromMouse = params.repulsionFromMouse;
        }
        if (params.enableRightClickRepulsion !== undefined) {
            config.enableRightClickRepulsion = params.enableRightClickRepulsion;
            this._currentSimParams.enableRightClickRepulsion = params.enableRightClickRepulsion;
        }
        
        console.log('[CosmosAdapter] Applying config to graph:', config);
        
        try {
            // Apply configuration
            this.graph.setConfig(config);
            console.log('[CosmosAdapter] setConfig successful');
            
            // CRITICAL: Restart simulation with energy for changes to take effect
            if (restart && Object.keys(config).length > 0) {
                this.graph.start(alpha);
                console.log('[CosmosAdapter] Simulation restarted with alpha:', alpha);
            }
            
            this.emit('simulationParamsChanged', { params: this._currentSimParams });
            return true;
        } catch (e) {
            console.error('[CosmosAdapter] Failed to set simulation params:', e);
            return false;
        }
    }
    
    /**
     * Set a single simulation parameter (for real-time slider updates)
     * @param {string} name - Parameter name
     * @param {number|boolean} value - Parameter value
     * @param {boolean} restart - Whether to restart simulation
     * @returns {boolean} Success
     */
    setSimulationParam(name, value, restart = true) {
        return this.setSimulationParams({ [name]: value }, { restart, alpha: 0.2 });
    }
    
    /**
     * Set simulation parameter with debouncing (for slider real-time updates)
     * @param {string} name - Parameter name
     * @param {number|boolean} value - Parameter value
     */
    setSimulationParamDebounced(name, value) {
        // Update internal tracking immediately
        if (this._currentSimParams.hasOwnProperty(name)) {
            this._currentSimParams[name] = value;
        }
        
        // Debounce the actual update
        if (this._updateDebounceTimer) {
            clearTimeout(this._updateDebounceTimer);
        }
        
        this._updateDebounceTimer = setTimeout(() => {
            this.setSimulationParam(name, value, true);
        }, this._updateDebounceDelay);
    }
    
    /**
     * Get current simulation parameters
     * @returns {Object}
     */
    getSimulationParams() {
        return { ...this._currentSimParams };
    }
    
    /**
     * Apply a simulation preset by name
     * @param {string} presetName - Name of preset to apply
     * @param {number} alpha - Alpha value for restart
     * @returns {boolean} Success
     */
    applySimulationPreset(presetName, alpha = 0.5) {
        const presets = RendererSettings.getSimulationPresets?.() || {};
        const preset = presets[presetName];
        
        if (!preset) {
            console.warn('[CosmosAdapter] Preset not found:', presetName);
            return false;
        }
        
        console.log('[CosmosAdapter] Applying preset:', presetName, preset);
        return this.setSimulationParams(preset, { restart: true, alpha });
    }
    
    /**
     * Reset simulation parameters to defaults
     * @returns {boolean} Success
     */
    resetSimulationParams() {
        const defaults = RendererSettings.getSimulationPreset?.('default') || {
            repulsion: 1.0,
            gravity: 0.25,
            center: 0,
            repulsionTheta: 1.15,
            cluster: 0.1,
            linkDistance: 10,
            linkSpring: 1.0,
            friction: 0.85,
            decay: 5000,
            repulsionFromMouse: 2.0,
            enableRightClickRepulsion: false
        };
        
        return this.setSimulationParams(defaults, { restart: true, alpha: 0.5 });
    }
    
    // ============================================================================
    // VISUAL PARAMETER METHODS
    // ============================================================================
    
    /**
     * Set visual parameters dynamically
     * @param {Object} params - Visual parameters
     * @returns {boolean} Success
     */
    setVisualParams(params) {
        if (!this.graph || typeof this.graph.setConfig !== 'function') {
            console.warn('[CosmosAdapter] setConfig not available');
            return false;
        }
        
        const config = {};
        
        if (params.pointSizeScale !== undefined) {
            config.pointSizeScale = params.pointSizeScale;
            this._currentVisualParams.pointSizeScale = params.pointSizeScale;
        }
        if (params.linkWidthScale !== undefined) {
            config.linkWidthScale = params.linkWidthScale;
            this._currentVisualParams.linkWidthScale = params.linkWidthScale;
        }
        if (params.curvedLinks !== undefined) {
            config.curvedLinks = params.curvedLinks;
            this._currentVisualParams.curvedLinks = params.curvedLinks;
        }
        if (params.curvedLinkWeight !== undefined) {
            config.curvedLinkWeight = params.curvedLinkWeight;
            this._currentVisualParams.curvedLinkWeight = params.curvedLinkWeight;
        }
        if (params.scalePointsOnZoom !== undefined) {
            config.scalePointsOnZoom = params.scalePointsOnZoom;
            this._currentVisualParams.scalePointsOnZoom = params.scalePointsOnZoom;
        }
        if (params.scaleLinksOnZoom !== undefined) {
            config.scaleLinksOnZoom = params.scaleLinksOnZoom;
            this._currentVisualParams.scaleLinksOnZoom = params.scaleLinksOnZoom;
        }
        if (params.showFPSMonitor !== undefined) {
            config.showFPSMonitor = params.showFPSMonitor;
            this._currentVisualParams.showFPSMonitor = params.showFPSMonitor;
        }
        if (params.linkOpacity !== undefined) {
            config.linkOpacity = params.linkOpacity;
            this._currentVisualParams.linkOpacity = params.linkOpacity;
        }
        if (params.pointOpacity !== undefined) {
            config.pointOpacity = params.pointOpacity;
            this._currentVisualParams.pointOpacity = params.pointOpacity;
        }
        
        try {
            this.graph.setConfig(config);
            this.graph.render();
            console.log('[CosmosAdapter] Visual params updated:', config);
            this.emit('visualParamsChanged', { params: this._currentVisualParams });
            return true;
        } catch (e) {
            console.error('[CosmosAdapter] Failed to set visual params:', e);
            return false;
        }
    }
    
    /**
     * Set a single visual parameter
     * @param {string} name - Parameter name
     * @param {any} value - Parameter value
     * @returns {boolean} Success
     */
    setVisualParam(name, value) {
        return this.setVisualParams({ [name]: value });
    }
    
    /**
     * Get current visual parameters
     * @returns {Object}
     */
    getVisualParams() {
        return { ...this._currentVisualParams };
    }
    
    // ============================================================================
    // REAL-TIME UPDATE MODE
    // ============================================================================
    
    /**
     * Enable/disable real-time parameter updates
     * @param {boolean} enabled
     */
    setRealTimeUpdates(enabled) {
        this._realTimeUpdates = enabled;
    }
    
    /**
     * Check if real-time updates are enabled
     * @returns {boolean}
     */
    isRealTimeUpdatesEnabled() {
        return this._realTimeUpdates;
    }
    
    /**
     * Set debounce delay for real-time updates
     * @param {number} delay - Delay in milliseconds
     */
    setUpdateDebounceDelay(delay) {
        this._updateDebounceDelay = delay;
    }
    
    // ============================================================================
    // LAYOUT SNAPSHOT METHODS
    // ============================================================================
    
    /**
     * Create a named snapshot of current layout
     * @param {string} name - Snapshot name
     * @returns {boolean} Success
     */
    createSnapshot(name = 'default') {
        const positions = this.capturePositions();
        if (!positions) return false;
        
        this._layoutSnapshots.set(name, new Float32Array(positions));
        console.log('[CosmosAdapter] Created snapshot:', name, 'with', positions.length / 2, 'positions');
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
        this.restorePositions(positions);
        console.log('[CosmosAdapter] Restored snapshot:', name);
        return true;
    }
    
    /**
     * Delete a named snapshot
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
     * Check if a snapshot exists
     * @param {string} name - Snapshot name
     * @returns {boolean}
     */
    hasSnapshot(name) {
        return this._layoutSnapshots.has(name);
    }
    
    /**
     * Export positions as JSON-serializable object
     * @returns {Object} - { nodeId: {x, y}, ... }
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
     * @param {Object} positionsObj - { nodeId: {x, y}, ... }
     * @param {boolean} pauseSimulation - Whether to pause simulation after import
     */
    importPositions(positionsObj, pauseSimulation = true) {
        const posArray = new Float32Array(this.nodeIds.length * 2);
        let importedCount = 0;
        
        this.nodeIds.forEach((id, index) => {
            const pos = positionsObj[id];
            if (pos && typeof pos.x === 'number' && typeof pos.y === 'number') {
                posArray[index * 2] = pos.x;
                posArray[index * 2 + 1] = pos.y;
                importedCount++;
            } else {
                // Keep current position
                posArray[index * 2] = this.positions[index * 2];
                posArray[index * 2 + 1] = this.positions[index * 2 + 1];
            }
        });
        
        if (pauseSimulation) {
            this.graph.pause();
        }
        
        this.positions = posArray;
        this.graph.setPointPositions(posArray, true);
        this.graph.render();
        
        console.log('[CosmosAdapter] Imported', importedCount, 'positions');
    }
    
    // ============================================================================
    // SELECTION METHODS
    // ============================================================================
    
    handleClick(pointIndex) {
        this._lastClickTime = Date.now();
        
        if (pointIndex === null || pointIndex === undefined || pointIndex < 0) {
            // Background click
            console.log('[CosmosAdapter] Background click detected');
            if (this.selectedIndices.size > 0) {
                this.clearSelection();
            }
            this.emit('backgroundClick', {});
            return;
        }
        
        const nodeId = this.nodeIds[pointIndex];
        if (!nodeId) return;
        
        const nodeData = this.nodeDataMap.get(nodeId);
        
        console.log('[CosmosAdapter] Node clicked:', nodeId, 'index:', pointIndex);
        
        // Toggle selection
        if (this.selectedIndices.has(pointIndex)) {
            this.selectedIndices.delete(pointIndex);
        } else {
            this.selectedIndices.add(pointIndex);
        }
        
        this.updateSelectionVisuals();
        
        // Emit nodeClick with both naming conventions for compatibility
        this.emit('nodeClick', { 
            id: nodeId,           // For compatibility with original code
            nodeId: nodeId,       // Explicit name
            data: nodeData,       // For compatibility with original code
            node: nodeData,       // Explicit name
            index: pointIndex,
            selected: this.selectedIndices.has(pointIndex),
            position: this.getNodePosition(nodeId)
        });
        
        // Also emit selectionChange for multi-select handling
        this.emit('selectionChange', {
            nodes: this.getSelectedNodes(),
            edges: []
        });
    }
    
    handleHover(pointIndex) {
        const prevHovered = this._hoveredIndex;
        this._hoveredIndex = pointIndex;
        
        if (pointIndex === prevHovered) return;
        
        if (prevHovered !== null && prevHovered !== undefined) {
            const prevId = this.nodeIds[prevHovered];
            if (prevId) {
                this.emit('nodeMouseOut', { nodeId: prevId });
            }
        }
        
        if (pointIndex !== null && pointIndex !== undefined && pointIndex >= 0) {
            const nodeId = this.nodeIds[pointIndex];
            if (nodeId) {
                const nodeData = this.nodeDataMap.get(nodeId);
                this.emit('nodeHover', { nodeId, node: nodeData });
                this.emit('nodeMouseOver', { nodeId, node: nodeData });
            }
        }
    }
    
    selectNode(nodeId) {
        const actualId = this.findNodeId(nodeId);
        const index = actualId ? this.nodeIndices.get(actualId) : undefined;
        if (index === undefined) {
            console.warn('[CosmosAdapter] selectNode: Node not found:', nodeId);
            return;
        }
        
        console.log('[CosmosAdapter] selectNode:', actualId, 'at index:', index);
        this.selectedIndices.add(index);
        
        // Only highlight this specific node - don't touch others
        // Get current colors from the graph and just change the selected node
        const numNodes = this.nodeIds.length;
        const colors = new Float32Array(numNodes * 4);
        
        // Re-build colors: metric or default + selection
        if (this._currentColorMetric && this._currentColorScale) {
            const gradientName = typeof this._currentColorScale === 'object' 
                ? (this._currentColorScale.gradient || 'spectral')
                : this._currentColorScale;
            const gradient = ColorGradients.gradients[gradientName] || ColorGradients.gradients['spectral'];
            
            // Get min/max
            let min = Infinity, max = -Infinity;
            const values = {};
            this.nodeIds.forEach(id => {
                const node = this.nodeDataMap.get(id);
                const val = node?.[this._currentColorMetric];
                if (typeof val === 'number' && !isNaN(val)) {
                    values[id] = val;
                    min = Math.min(min, val);
                    max = Math.max(max, val);
                }
            });
            if (min === Infinity) { min = 0; max = 1; }
            const range = max - min || 1;
            
            // Apply gradient colors
            this.nodeIds.forEach((id, i) => {
                const offset = i * 4;
                const val = values[id];
                
                if (typeof val === 'number') {
                    const normalized = (val - min) / range;
                    const rgba = ColorGradients.interpolateRgba(gradient, normalized, 1.0);
                    colors[offset] = rgba[0];
                    colors[offset + 1] = rgba[1];
                    colors[offset + 2] = rgba[2];
                    colors[offset + 3] = 1.0;
                } else {
                    const defaultColor = RendererSettings.getDefaultNodeColorRgba();
                    colors[offset] = defaultColor[0];
                    colors[offset + 1] = defaultColor[1];
                    colors[offset + 2] = defaultColor[2];
                    colors[offset + 3] = 1.0;
                }
            });
        } else {
            const defaultColor = RendererSettings.getDefaultNodeColorRgba();
            for (let i = 0; i < numNodes; i++) {
                const offset = i * 4;
                colors[offset] = defaultColor[0];
                colors[offset + 1] = defaultColor[1];
                colors[offset + 2] = defaultColor[2];
                colors[offset + 3] = 1.0;
            }
        }
        
        // Now apply RED to selected nodes
        this.selectedIndices.forEach(idx => {
            const offset = idx * 4;
            colors[offset] = 1.0;     // R
            colors[offset + 1] = 0.0; // G
            colors[offset + 2] = 0.0; // B
            colors[offset + 3] = 1.0; // A
        });
        
        this.graph.setPointColors(colors);
        this.graph.render();
        
        this.emit('selectionChange', { 
            nodes: this.getSelectedNodes(),
            edges: this.getSelectedEdges()
        });
    }
    
    selectNodes(nodeIds, additive = false) {
        if (!additive) {
            this.selectedIndices.clear();
        }
        
        const mappedIds = this.findNodeIds(nodeIds);
        
        mappedIds.forEach(id => {
            const index = this.nodeIndices.get(id);
            if (index !== undefined) {
                this.selectedIndices.add(index);
            }
        });
        
        this._applySelectionColors();
        
        this.emit('selectionChange', { 
            nodes: this.getSelectedNodes(),
            edges: this.getSelectedEdges()
        });
    }
    
    deselectNode(nodeId) {
        const index = this.nodeIndices.get(nodeId);
        if (index === undefined) return;
        
        this.selectedIndices.delete(index);
        this._applySelectionColors();
        
        this.emit('selectionChange', { 
            nodes: this.getSelectedNodes(),
            edges: this.getSelectedEdges()
        });
    }
    
    clearSelection() {
        console.log('[CosmosAdapter] clearSelection called, clearing', this.selectedIndices.size, 'nodes');
        this.selectedIndices.clear();
        this.highlightedIndices.clear();
        
        // Re-apply the current coloring from scratch (not from cache)
        if (this._currentColorMetric && this._currentColorScale) {
            // Re-apply metric coloring
            console.log('[CosmosAdapter] Re-applying metric coloring:', this._currentColorMetric);
            const values = {};
            this.nodeIds.forEach(id => {
                const node = this.nodeDataMap.get(id);
                if (node && node[this._currentColorMetric] !== undefined) {
                    values[id] = node[this._currentColorMetric];
                }
            });
            this.colorNodesByMetric(this._currentColorMetric, values, this._currentColorScale);
        } else {
            // Apply default colors
            console.log('[CosmosAdapter] Applying default colors');
            this.applyDefaultColors();
        }
        
        this.emit('selectionChange', { nodes: [], edges: [] });
    }
    
    /**
     * Apply colors: recompute current coloring + selection overlay
     * This recomputes colors fresh to avoid any stored state issues
     */
    _applySelectionColors() {
        console.log('[CosmosAdapter] _applySelectionColors called, selected:', this.selectedIndices.size);
        
        const colors = new Float32Array(this.nodeIds.length * 4);
        
        // Determine what coloring to use
        if (this._currentColorMetric && this._currentColorScale) {
            // Re-apply metric coloring fresh
            console.log('[CosmosAdapter] Re-applying metric coloring:', this._currentColorMetric);
            
            const gradientName = typeof this._currentColorScale === 'object' 
                ? (this._currentColorScale.gradient || 'spectral')
                : this._currentColorScale;
            const gradient = ColorGradients.gradients[gradientName] || ColorGradients.gradients['spectral'];
            
            // Build values map and get min/max
            const values = {};
            let min = Infinity, max = -Infinity;
            this.nodeIds.forEach(id => {
                const node = this.nodeDataMap.get(id);
                const val = node?.[this._currentColorMetric];
                if (typeof val === 'number' && !isNaN(val)) {
                    values[id] = val;
                    min = Math.min(min, val);
                    max = Math.max(max, val);
                }
            });
            
            if (min === Infinity) { min = 0; max = 1; }
            const range = max - min || 1;
            
            // Apply gradient colors
            this.nodeIds.forEach((id, i) => {
                const offset = i * 4;
                const val = values[id];
                
                if (typeof val === 'number') {
                    const normalized = (val - min) / range;
                    const rgba = ColorGradients.interpolateRgba(gradient, normalized, 1.0);
                    colors[offset] = rgba[0];
                    colors[offset + 1] = rgba[1];
                    colors[offset + 2] = rgba[2];
                    colors[offset + 3] = 1.0;
                } else {
                    const defaultColor = RendererSettings.getDefaultNodeColorRgba();
                    colors[offset] = defaultColor[0];
                    colors[offset + 1] = defaultColor[1];
                    colors[offset + 2] = defaultColor[2];
                    colors[offset + 3] = 1.0;
                }
            });
        } else {
            // Use default node color
            console.log('[CosmosAdapter] Re-applying default colors');
            const defaultColor = RendererSettings.getDefaultNodeColorRgba();
            for (let i = 0; i < this.nodeIds.length; i++) {
                const offset = i * 4;
                colors[offset] = defaultColor[0];
                colors[offset + 1] = defaultColor[1];
                colors[offset + 2] = defaultColor[2];
                colors[offset + 3] = 1.0;
            }
        }
        
        // Apply selection color (red) to selected nodes - AFTER base coloring
        if (this.selectedIndices.size > 0) {
            console.log('[CosmosAdapter] Applying red to', this.selectedIndices.size, 'selected nodes');
            this.selectedIndices.forEach(index => {
                const offset = index * 4;
                colors[offset] = 1.0;     // R
                colors[offset + 1] = 0.0; // G
                colors[offset + 2] = 0.0; // B
                colors[offset + 3] = 1.0; // A
            });
        }
        
        this.graph.setPointColors(colors);
        this.graph.render();
    }
    
    getSelectedNodes() {
        const nodes = Array.from(this.selectedIndices).map(idx => this.nodeIds[idx]).filter(Boolean);
        console.log('[CosmosAdapter] getSelectedNodes:', nodes.length, 'nodes selected, indices:', Array.from(this.selectedIndices));
        return nodes;
    }
    
    getSelectedEdges() {
        return [];
    }
    
    isNodeSelected(nodeId) {
        const index = this.nodeIndices.get(nodeId);
        return index !== undefined && this.selectedIndices.has(index);
    }
    
    /**
     * Update all node colors based on current state
     * Simplified to use the base color system
     */
    updateSelectionVisuals() {
        // If isolation mode is active, don't interfere with it
        if (this._isIsolationMode) {
            return;
        }
        
        // If path highlighting is active, use the visibility method which respects it
        if (this._isPathHighlightActive) {
            this.updateNodeVisibility();
            return;
        }
        
        // Use the standard selection color application
        this._applySelectionColors();
    }
    
    // ============================================================================
    // PATH / FLOW HIGHLIGHTING
    // ============================================================================
    
    /**
     * Highlight path nodes with specific colors
     * @param {Map} nodeColorMap - Map of nodeId -> { color: '#hex', type: 'source'|'target'|'intermediate' }
     */
    highlightPathNodes(nodeColorMap) {
        console.log('[CosmosAdapter] Highlighting path nodes:', nodeColorMap.size);
        
        this._pathNodeColors.clear();
        this._isPathHighlightActive = true;
        
        // Map path nodes to actual node IDs using the findNodeId helper
        // Also store the config for each actual node ID
        const actualIdToConfig = new Map();
        let foundCount = 0;
        
        nodeColorMap.forEach((config, nodeId) => {
            const actualNodeId = this.findNodeId(nodeId);
            
            if (actualNodeId) {
                const rgba = RendererSettings.hexToRgba(config.color, 1.0);
                this._pathNodeColors.set(actualNodeId, rgba);
                actualIdToConfig.set(actualNodeId, config);
                foundCount++;
            } else {
                console.warn('[CosmosAdapter] Path node not found:', nodeId);
            }
        });
        
        console.log('[CosmosAdapter] Found', foundCount, 'of', nodeColorMap.size, 'path nodes');
        
        if (foundCount === 0) {
            console.log('[CosmosAdapter] Sample input node IDs:', Array.from(nodeColorMap.keys()).slice(0, 3));
            console.log('[CosmosAdapter] Sample renderer node IDs:', this.nodeIds.slice(0, 5));
        }
        
        // Adjust point sizes for path nodes
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const defaultSize = cosmosConfig.pointSize || 4;
        const sizes = new Float32Array(this.nodeIds.length);
        
        this.nodeIds.forEach((id, index) => {
            const config = actualIdToConfig.get(id);
            if (config) {
                if (config.type === 'source' || config.type === 'target') {
                    sizes[index] = defaultSize * 3;  // Source/target larger
                } else {
                    sizes[index] = defaultSize * 2;  // Intermediate nodes
                }
            } else {
                sizes[index] = defaultSize;
            }
        });
        
        this.graph.setPointSizes(sizes);
        this.updateNodeVisibility();
    }
    
    /**
     * Highlight path edges with specific color
     * @param {Array} edgePairs - Array of { source, target } objects
     * @param {string} color - Hex color for edges
     * @param {number} opacity - Edge opacity
     */
    highlightPathEdges(edgePairs, color = '#00d4ff', opacity = 1.0) {
        console.log('[CosmosAdapter] Highlighting path edges:', edgePairs.length);
        
        this._pathEdgeColors.clear();
        this._isPathHighlightActive = true;
        
        const pathColor = RendererSettings.hexToRgba(color, opacity);
        
        // Build a set of path edges (both directions) using actual node IDs from graph
        const pathEdgeSet = new Set();
        edgePairs.forEach(pair => {
            const source = this.findNodeId(pair.source) || pair.source;
            const target = this.findNodeId(pair.target) || pair.target;
            pathEdgeSet.add(`${source}-${target}`);
            pathEdgeSet.add(`${target}-${source}`);
            // Also add lowercase versions
            pathEdgeSet.add(`${source.toLowerCase()}-${target.toLowerCase()}`);
            pathEdgeSet.add(`${target.toLowerCase()}-${source.toLowerCase()}`);
            
            this._pathEdgeColors.set(`${source}-${target}`, pathColor);
            this._pathEdgeColors.set(`${target}-${source}`, pathColor);
        });
        
        console.log('[CosmosAdapter] Path edge set size:', pathEdgeSet.size);
        
        // Now iterate through ALL edges in edgeDataMap and color them
        const edgeCount = this.edgeDataMap.size;
        if (edgeCount === 0) {
            console.warn('[CosmosAdapter] No edges in edgeDataMap');
            return;
        }
        
        const colors = new Float32Array(edgeCount * 4);
        const widths = new Float32Array(edgeCount);
        
        const styleConfig = RendererSettings.getStyleConfig();
        const dimColor = RendererSettings.hexToRgba(
            styleConfig.defaultEdgeColor || '#ffffff',
            0.1 // Dim opacity for non-path edges
        );
        
        let edgeIndex = 0;
        let pathEdgeCount = 0;
        
        this.edgeDataMap.forEach((edge, edgeId) => {
            const offset = edgeIndex * 4;
            
            // Check all possible key formats
            const key1 = `${edge.source}-${edge.target}`;
            const key2 = `${edge.target}-${edge.source}`;
            const key3 = `${edge.source.toLowerCase()}-${edge.target.toLowerCase()}`;
            const key4 = `${edge.target.toLowerCase()}-${edge.source.toLowerCase()}`;
            
            const isPathEdge = pathEdgeSet.has(key1) || pathEdgeSet.has(key2) || 
                               pathEdgeSet.has(key3) || pathEdgeSet.has(key4);
            
            if (isPathEdge) {
                colors[offset] = pathColor[0];
                colors[offset + 1] = pathColor[1];
                colors[offset + 2] = pathColor[2];
                colors[offset + 3] = pathColor[3];
                widths[edgeIndex] = 4.0; // Thicker for path edges
                pathEdgeCount++;
            } else {
                colors[offset] = dimColor[0];
                colors[offset + 1] = dimColor[1];
                colors[offset + 2] = dimColor[2];
                colors[offset + 3] = dimColor[3];
                widths[edgeIndex] = 1.0;
            }
            
            edgeIndex++;
        });
        
        console.log('[CosmosAdapter] Colored', pathEdgeCount, 'path edges out of', edgeCount, 'total');
        
        if (typeof this.graph.setLinkColors === 'function') {
            this.graph.setLinkColors(colors);
        }
        if (typeof this.graph.setLinkWidths === 'function') {
            this.graph.setLinkWidths(widths);
        }
        
        this.graph.render();
    }
    
    /**
     * Clear all path highlights and restore normal appearance
     */
    clearPathHighlights() {
        console.log('[CosmosAdapter] Clearing path highlights');
        
        // Clear path color maps
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        this._isPathHighlightActive = false;
        
        // Check if we were in isolation mode
        if (this._isIsolationMode) {
            this._isIsolationMode = false;
            this._isolatedNodes = null;
            this._isolatedEdges = null;
            // Restore all edges when exiting isolation mode
            this.restoreAllEdges();
        } else {
            // Not in isolation mode - just restore edge colors
            // Use edgeDataMap.size since those are the currently active edges
            const edgeCount = this.edgeDataMap.size;
            if (edgeCount > 0) {
                this._applyEdgeColorsForCount(edgeCount);
            }
        }
        
        // Clear hidden nodes
        this._hiddenNodes.clear();
        
        // Reset point sizes to default
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const sizes = new Float32Array(this.nodeIds.length);
        sizes.fill(cosmosConfig.pointSize || 4);
        this.graph.setPointSizes(sizes);
        
        // Reset node colors (will apply metric if active, otherwise default)
        if (this._currentColorMetric && this._currentColorScale) {
            this.applyNodeColors(this._currentColorMetric, this._currentColorScale);
        } else {
            this.applyDefaultColors();
        }
        
        this.graph.render();
        console.log('[CosmosAdapter] Path highlights cleared, styles restored');
    }
    
    /**
     * Check if path highlighting is active
     */
    isPathHighlightActive() {
        return this._isPathHighlightActive;
    }
    
    /**
     * Update edge colors respecting path highlighting
     */
    updateEdgeColors() {
        if (!this._edgeLinkData || this._edgeLinkData.length === 0) return;
        
        const edgeCount = this._edgeLinkData.length / 2;
        const colors = new Float32Array(edgeCount * 4);
        
        const styleConfig = RendererSettings.getStyleConfig();
        const defaultEdgeColor = RendererSettings.hexToRgba(
            styleConfig.defaultEdgeColor, 
            styleConfig.defaultEdgeOpacity
        );
        
        let edgeIndex = 0;
        this.edgeDataMap.forEach((edge, edgeId) => {
            const offset = edgeIndex * 4;
            
            // Check if this edge is in path highlight - try multiple key formats
            let isPathEdge = false;
            let pathColor = null;
            
            if (this._isPathHighlightActive && this._pathEdgeColors.size > 0) {
                // Try original keys
                const edgeKey = `${edge.source}-${edge.target}`;
                const reverseKey = `${edge.target}-${edge.source}`;
                
                // Try lowercase keys (in case path was normalized)
                const lowerKey = `${edge.source.toLowerCase()}-${edge.target.toLowerCase()}`;
                const lowerReverseKey = `${edge.target.toLowerCase()}-${edge.source.toLowerCase()}`;
                
                // Check all variations
                if (this._pathEdgeColors.has(edgeKey)) {
                    isPathEdge = true;
                    pathColor = this._pathEdgeColors.get(edgeKey);
                } else if (this._pathEdgeColors.has(reverseKey)) {
                    isPathEdge = true;
                    pathColor = this._pathEdgeColors.get(reverseKey);
                } else if (this._pathEdgeColors.has(lowerKey)) {
                    isPathEdge = true;
                    pathColor = this._pathEdgeColors.get(lowerKey);
                } else if (this._pathEdgeColors.has(lowerReverseKey)) {
                    isPathEdge = true;
                    pathColor = this._pathEdgeColors.get(lowerReverseKey);
                }
            }
            
            if (isPathEdge && pathColor) {
                colors[offset] = pathColor[0];
                colors[offset + 1] = pathColor[1];
                colors[offset + 2] = pathColor[2];
                colors[offset + 3] = pathColor[3];
            } else {
                // Default edge color - dim if path is active
                const alpha = this._isPathHighlightActive ? 0.1 : defaultEdgeColor[3];
                colors[offset] = defaultEdgeColor[0];
                colors[offset + 1] = defaultEdgeColor[1];
                colors[offset + 2] = defaultEdgeColor[2];
                colors[offset + 3] = alpha;
            }
            
            edgeIndex++;
        });
        
        if (typeof this.graph.setLinkColors === 'function') {
            this.graph.setLinkColors(colors);
        }
    }
    
    // ============================================================================
    // HIGHLIGHT METHODS
    // ============================================================================
    
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
    
    // ============================================================================
    // COLORING METHODS
    // ============================================================================
    
    // Hidden nodes tracking for visibility control
    _hiddenNodes = new Set();
    
    // Isolation mode tracking
    _isIsolationMode = false;
    _isolatedNodes = null;
    _isolatedEdges = null;
    
    // ============================================================================
    // NODE ID MAPPING HELPER
    // ============================================================================
    
    /**
     * Find the actual node ID in the renderer, handling various ID formats
     * (prefixes like a_, t_, case variations, etc.)
     * @param {string} nodeId - The node ID to find
     * @returns {string|null} The actual node ID in the renderer, or null if not found
     */
    findNodeId(nodeId) {
        if (!nodeId) return null;
        
        // Try direct match first
        if (this.nodeIndices.has(nodeId)) {
            return nodeId;
        }
        
        // Try lowercase
        const lower = nodeId.toLowerCase();
        if (this.nodeIndices.has(lower)) {
            return lower;
        }
        
        // Try with prefixes
        const prefixes = ['', 'a_', 't_'];
        const cleanId = nodeId.replace(/^[at]_/i, '').toLowerCase();
        
        for (const prefix of prefixes) {
            const testId = prefix + cleanId;
            if (this.nodeIndices.has(testId)) {
                return testId;
            }
        }
        
        // Try without prefix if it has one
        if (this.nodeIndices.has(cleanId)) {
            return cleanId;
        }
        
        // Try uppercase variations
        const upper = nodeId.toUpperCase();
        if (this.nodeIndices.has(upper)) {
            return upper;
        }
        
        return null; // Not found
    }
    
    /**
     * Find multiple node IDs, returning only those that exist in the renderer
     * @param {string[]} nodeIds - Array of node IDs to find
     * @returns {string[]} Array of actual node IDs found in the renderer
     */
    findNodeIds(nodeIds) {
        const found = [];
        for (const id of nodeIds) {
            const actualId = this.findNodeId(id);
            if (actualId) {
                found.push(actualId);
            }
        }
        return found;
    }
    
    // ============================================================================
    // NODE VISIBILITY (via alpha channel)
    // ============================================================================
    
    /**
     * Show only specified nodes (hide all others completely)
     * Handles node ID variations automatically
     * @param {string[]} nodeIdsToShow - IDs of nodes to keep visible
     * @param {Array} pathEdges - Optional array of {source, target} for path edges to highlight
     */
    showOnlyNodes(nodeIdsToShow, pathEdges = null) {
        // Map input IDs to actual renderer IDs
        const mappedIds = this.findNodeIds(nodeIdsToShow);
        
        console.log('[CosmosAdapter] showOnlyNodes: requested', nodeIdsToShow.length, 
                    'nodes, found', mappedIds.length, 'in renderer');
        
        if (mappedIds.length === 0) {
            console.warn('[CosmosAdapter] showOnlyNodes: No matching nodes found!');
            console.log('[CosmosAdapter] Sample input IDs:', nodeIdsToShow.slice(0, 3));
            console.log('[CosmosAdapter] Sample renderer IDs:', this.nodeIds.slice(0, 5));
            return;
        }
        
        // Store isolation state
        this._isolatedNodes = new Set(mappedIds);
        this._isIsolationMode = true;
        
        // Hide non-isolated nodes (set alpha to 0)
        const colors = new Float32Array(this.nodeIds.length * 4);
        const styleConfig = RendererSettings.getStyleConfig();
        const defaultColor = RendererSettings.hexToRgba(styleConfig.defaultNodeColor);
        
        this.nodeIds.forEach((id, index) => {
            const offset = index * 4;
            const isIsolated = this._isolatedNodes.has(id);
            
            colors[offset] = defaultColor[0];
            colors[offset + 1] = defaultColor[1];
            colors[offset + 2] = defaultColor[2];
            colors[offset + 3] = isIsolated ? 1.0 : 0.0;  // Completely hide non-isolated
        });
        
        this.graph.setPointColors(colors);
        
        // Show only path edges if provided
        if (pathEdges && pathEdges.length > 0) {
            this.setPathEdgesOnly(pathEdges);
        } else {
            // Show edges only between isolated nodes
            this.setIsolatedEdges(this._isolatedNodes);
        }
        
        this.graph.render();
        console.log('[CosmosAdapter] Isolated', mappedIds.length, 'nodes');
    }
    
    /**
     * Set only path edges visible (hide all others)
     * @param {Array} pathEdges - Array of {source, target} objects
     */
    setPathEdgesOnly(pathEdges) {
        if (!pathEdges || pathEdges.length === 0) {
            this.graph.setLinks(new Float32Array(0));
            return;
        }
        
        console.log('[CosmosAdapter] setPathEdgesOnly:', pathEdges.length, 'edges');
        
        // Create link data for the path edges
        const linkData = new Float32Array(pathEdges.length * 2);
        let validCount = 0;
        
        pathEdges.forEach(edge => {
            const sourceId = this.findNodeId(edge.source) || edge.source;
            const targetId = this.findNodeId(edge.target) || edge.target;
            
            const sourceIndex = this.nodeIndices.get(sourceId);
            const targetIndex = this.nodeIndices.get(targetId);
            
            if (sourceIndex !== undefined && targetIndex !== undefined) {
                linkData[validCount * 2] = sourceIndex;
                linkData[validCount * 2 + 1] = targetIndex;
                validCount++;
            }
        });
        
        if (validCount > 0) {
            const finalLinkData = linkData.slice(0, validCount * 2);
            this.graph.setLinks(finalLinkData);
            
            // Set cyan color for path edges - same as Cytoscape #00d4ff
            if (typeof this.graph.setLinkColors === 'function') {
                const linkColors = new Float32Array(validCount * 4);
                // #00d4ff = rgb(0, 212, 255)
                for (let i = 0; i < validCount; i++) {
                    const offset = i * 4;
                    linkColors[offset] = 0;           // R
                    linkColors[offset + 1] = 0.831;   // G (212/255)
                    linkColors[offset + 2] = 1.0;     // B (255/255)
                    linkColors[offset + 3] = 1.0;     // A
                }
                this.graph.setLinkColors(linkColors);
            }
            
            // Set thicker widths for path edges (matching Cytoscape's width: 4-5)
            if (typeof this.graph.setLinkWidths === 'function') {
                const linkWidths = new Float32Array(validCount);
                linkWidths.fill(4.0);  // Match Cytoscape path edge width
                this.graph.setLinkWidths(linkWidths);
            }
            
            console.log('[CosmosAdapter] Set', validCount, 'path edges (cyan #00d4ff, width 4)');
        } else {
            this.graph.setLinks(new Float32Array(0));
        }
    }
    
    /**
     * Set edges only between isolated nodes
     * @param {Set<string>} isolatedNodes - Set of isolated node IDs
     */
    setIsolatedEdges(isolatedNodes) {
        if (!this._storedEdgeData || this._storedEdgeData.length === 0) {
            console.log('[CosmosAdapter] No stored edges');
            return;
        }
        
        // Filter edges where both endpoints are in isolated nodes
        const isolatedEdges = this._storedEdgeData.filter(edge => {
            return isolatedNodes.has(edge.source) && isolatedNodes.has(edge.target);
        });
        
        console.log('[CosmosAdapter] setIsolatedEdges:', isolatedEdges.length, 'of', this._storedEdgeData.length);
        
        if (isolatedEdges.length > 0) {
            const linkData = new Float32Array(isolatedEdges.length * 2);
            let validCount = 0;
            
            isolatedEdges.forEach(edge => {
                const sourceIndex = this.nodeIndices.get(edge.source);
                const targetIndex = this.nodeIndices.get(edge.target);
                
                if (sourceIndex !== undefined && targetIndex !== undefined) {
                    linkData[validCount * 2] = sourceIndex;
                    linkData[validCount * 2 + 1] = targetIndex;
                    validCount++;
                }
            });
            
            const finalLinkData = linkData.slice(0, validCount * 2);
            this.graph.setLinks(finalLinkData);
            
            // Set cyan color - same as Cytoscape #00d4ff
            if (typeof this.graph.setLinkColors === 'function') {
                const linkColors = new Float32Array(validCount * 4);
                // #00d4ff = rgb(0, 212, 255)
                for (let i = 0; i < validCount; i++) {
                    const offset = i * 4;
                    linkColors[offset] = 0;           // R
                    linkColors[offset + 1] = 0.831;   // G (212/255)
                    linkColors[offset + 2] = 1.0;     // B (255/255)
                    linkColors[offset + 3] = 1.0;     // A
                }
                this.graph.setLinkColors(linkColors);
            }
            
            // Set thicker widths (matching Cytoscape)
            if (typeof this.graph.setLinkWidths === 'function') {
                const linkWidths = new Float32Array(validCount);
                linkWidths.fill(4.0);
                this.graph.setLinkWidths(linkWidths);
            }
        } else {
            this.graph.setLinks(new Float32Array(0));
        }
    }
    
    /**
     * Update edge visibility based on visible nodes
     * @param {Set<string>} visibleNodeIds - Set of visible node IDs
     */
    updateEdgeVisibility(visibleNodeIds) {
        if (!this._storedEdgeData || this._storedEdgeData.length === 0) {
            console.log('[CosmosAdapter] No stored edges to filter');
            return;
        }
        
        // Filter edges to only show those connecting visible nodes
        const visibleEdges = this._storedEdgeData.filter(edge => {
            const sourceVisible = visibleNodeIds.has(edge.source);
            const targetVisible = visibleNodeIds.has(edge.target);
            return sourceVisible && targetVisible;
        });
        
        console.log('[CosmosAdapter] Edge visibility: showing', visibleEdges.length, 'of', this._storedEdgeData.length);
        
        // Update the graph with filtered edges
        if (visibleEdges.length > 0 && typeof this.graph.setLinks === 'function') {
            try {
                // Create Float32Array for cosmos.gl
                const linkData = new Float32Array(visibleEdges.length * 2);
                let validCount = 0;
                
                visibleEdges.forEach((edge, i) => {
                    const sourceIndex = this.nodeIndices.get(edge.source);
                    const targetIndex = this.nodeIndices.get(edge.target);
                    
                    if (sourceIndex !== undefined && targetIndex !== undefined) {
                        linkData[validCount * 2] = sourceIndex;
                        linkData[validCount * 2 + 1] = targetIndex;
                        validCount++;
                    }
                });
                
                // Trim to actual valid count
                const finalLinkData = linkData.slice(0, validCount * 2);
                this.graph.setLinks(finalLinkData);
                this._visibleEdgeData = visibleEdges;
            } catch (err) {
                console.warn('[CosmosAdapter] Error updating edge visibility:', err);
            }
        } else if (visibleEdges.length === 0) {
            // Hide all edges
            try {
                this.graph.setLinks(new Float32Array(0));
            } catch (err) {
                console.warn('[CosmosAdapter] Error hiding edges:', err);
            }
        }
        
        this.graph.render();
    }
    
    /**
     * Hide specified nodes
     * Handles node ID variations automatically
     * Also hides edges connected to hidden nodes (matching Cytoscape behavior)
     * @param {string[]} nodeIdsToHide - IDs of nodes to hide
     */
    hideNodes(nodeIdsToHide) {
        console.log('[CosmosAdapter] hideNodes called with', nodeIdsToHide?.length, 'nodes:', nodeIdsToHide?.slice(0, 3));
        
        if (!nodeIdsToHide || nodeIdsToHide.length === 0) {
            console.warn('[CosmosAdapter] hideNodes: No nodes provided');
            return;
        }
        
        const mappedIds = this.findNodeIds(nodeIdsToHide);
        console.log('[CosmosAdapter] hideNodes: mapped to', mappedIds.length, 'actual IDs');
        
        if (mappedIds.length === 0) {
            console.warn('[CosmosAdapter] hideNodes: No matching nodes found in graph');
            return;
        }
        
        mappedIds.forEach(id => this._hiddenNodes.add(id));
        
        // Update node visibility (sets alpha to 0 for hidden nodes)
        this.updateNodeVisibility();
        
        // Update edge visibility - hide edges connected to hidden nodes
        this._updateEdgeVisibilityForHiddenNodes();
        
        console.log('[CosmosAdapter] Hidden', mappedIds.length, 'nodes, total hidden:', this._hiddenNodes.size);
    }
    
    /**
     * Update edge visibility based on hidden nodes
     * Hides edges where either endpoint is hidden
     */
    _updateEdgeVisibilityForHiddenNodes() {
        if (!this._storedEdgeData || this._storedEdgeData.length === 0) return;
        
        // If no hidden nodes, show all edges
        if (this._hiddenNodes.size === 0) {
            this.restoreAllEdges();
            return;
        }
        
        // Filter edges to only show those where BOTH endpoints are visible
        const visibleEdges = this._storedEdgeData.filter(edge => {
            return !this._hiddenNodes.has(edge.source) && !this._hiddenNodes.has(edge.target);
        });
        
        console.log('[CosmosAdapter] Edge visibility update:', visibleEdges.length, 'of', this._storedEdgeData.length, 'edges visible');
        
        if (visibleEdges.length > 0) {
            const linkData = new Float32Array(visibleEdges.length * 2);
            let validCount = 0;
            
            visibleEdges.forEach(edge => {
                const sourceIndex = this.nodeIndices.get(edge.source);
                const targetIndex = this.nodeIndices.get(edge.target);
                
                if (sourceIndex !== undefined && targetIndex !== undefined) {
                    linkData[validCount * 2] = sourceIndex;
                    linkData[validCount * 2 + 1] = targetIndex;
                    validCount++;
                }
            });
            
            const finalLinkData = linkData.slice(0, validCount * 2);
            this.graph.setLinks(finalLinkData);
            
            // Apply default colors to the visible edges
            this._applyEdgeColorsForCount(validCount);
        } else {
            // No visible edges
            this.graph.setLinks(new Float32Array(0));
        }
        
        this.graph.render();
    }
    
    /**
     * Show all nodes (reset visibility)
     */
    showAllNodes() {
        // Clear isolation mode
        this._isIsolationMode = false;
        this._isolatedNodes = null;
        this._isolatedEdges = null;
        
        // Clear path highlighting state
        this._isPathHighlightActive = false;
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        
        // Clear hidden nodes
        this._hiddenNodes.clear();
        
        // Restore normal node visibility (with metric coloring if active)
        this.updateNodeVisibility();
        
        // Restore all edges (this now handles coloring internally)
        this.restoreAllEdges();
        
        this.graph.render();
        console.log('[CosmosAdapter] All nodes visible, isolation cleared');
    }
    
    /**
     * Restore all edges to the graph
     */
    restoreAllEdges() {
        if (!this._storedEdgeData || this._storedEdgeData.length === 0) {
            console.log('[CosmosAdapter] No stored edges to restore');
            return;
        }
        
        if (typeof this.graph.setLinks === 'function') {
            try {
                // Create Float32Array for cosmos.gl
                const linkData = new Float32Array(this._storedEdgeData.length * 2);
                let validCount = 0;
                
                this._storedEdgeData.forEach((edge, i) => {
                    const sourceIndex = this.nodeIndices.get(edge.source);
                    const targetIndex = this.nodeIndices.get(edge.target);
                    
                    if (sourceIndex !== undefined && targetIndex !== undefined) {
                        linkData[validCount * 2] = sourceIndex;
                        linkData[validCount * 2 + 1] = targetIndex;
                        validCount++;
                    }
                });
                
                const finalLinkData = linkData.slice(0, validCount * 2);
                this.graph.setLinks(finalLinkData);
                this._visibleEdgeData = this._storedEdgeData;
                
                // Store the current link count for color operations
                this._currentLinkCount = validCount;
                
                // Apply default edge colors using the actual link count
                this._applyEdgeColorsForCount(validCount);
                
                console.log('[CosmosAdapter] Restored', validCount, 'edges');
            } catch (err) {
                console.warn('[CosmosAdapter] Error restoring edges:', err);
            }
        }
        
        this.graph.render();
    }
    
    /**
     * Apply edge colors for a specific count of edges
     * Uses stored edge style if available, otherwise defaults
     * @param {number} edgeCount - Number of edges to color
     */
    _applyEdgeColorsForCount(edgeCount) {
        if (edgeCount <= 0) return;
        
        // Use stored edge style if available, otherwise use defaults from settings
        let color, opacity, width;
        if (this._edgeStyle) {
            color = this._edgeStyle.color || '#ffffff';
            opacity = this._edgeStyle.opacity !== undefined ? this._edgeStyle.opacity : 0.5;
            width = this._edgeStyle.width || 1;
        } else {
            const styleConfig = RendererSettings.getStyleConfig();
            color = styleConfig.defaultEdgeColor || '#ffffff';
            opacity = styleConfig.defaultEdgeOpacity || 0.5;
            width = 1;
        }
        
        const rgba = RendererSettings.hexToRgba(color, opacity);
        
        const colors = new Float32Array(edgeCount * 4);
        for (let i = 0; i < edgeCount; i++) {
            colors[i * 4] = rgba[0];
            colors[i * 4 + 1] = rgba[1];
            colors[i * 4 + 2] = rgba[2];
            colors[i * 4 + 3] = rgba[3];
        }
        
        if (typeof this.graph.setLinkColors === 'function') {
            this.graph.setLinkColors(colors);
        }
        
        // Also set widths
        const widths = new Float32Array(edgeCount);
        widths.fill(width);
        if (typeof this.graph.setLinkWidths === 'function') {
            this.graph.setLinkWidths(widths);
        }
        
        console.log('[CosmosAdapter] Applied edge colors to', edgeCount, 'edges (color:', color, 'opacity:', opacity, 'width:', width, ')');
    }
    
    /**
     * Check if a node is hidden
     * @param {string} nodeId
     * @returns {boolean}
     */
    isNodeHidden(nodeId) {
        return this._hiddenNodes.has(nodeId);
    }
    
    /**
     * Get all hidden node IDs
     * @returns {string[]}
     */
    getHiddenNodeIds() {
        return Array.from(this._hiddenNodes);
    }
    
    /**
     * Get all node IDs
     * @returns {string[]}
     */
    getAllNodeIds() {
        return [...this.nodeIds];
    }
    
    /**
     * Update node visibility using alpha channel
     * Respects current coloring (metric/path) while adjusting alpha for hidden nodes
     */
    updateNodeVisibility() {
        const colors = new Float32Array(this.nodeIds.length * 4);
        const styleConfig = RendererSettings.getStyleConfig();
        const defaultColor = RendererSettings.hexToRgba(styleConfig.defaultNodeColor);
        const selectionColor = RendererSettings.getSelectionColorRgba();
        
        let gradient = null;
        let min = 0, max = 1;
        
        // Check if we have metric coloring active
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
            const offset = index * 4;
            let color;
            let alpha = 1.0;
            
            // Check if node is in path highlight
            if (this._isPathHighlightActive && this._pathNodeColors.has(id)) {
                color = this._pathNodeColors.get(id);
                alpha = color[3] || 1.0;
            }
            // If path highlighting is active but node is NOT in path, dim it
            else if (this._isPathHighlightActive) {
                color = defaultColor;
                alpha = 0.15;  // Dim non-path nodes
            }
            // Check if node is selected
            else if (this.selectedIndices.has(index)) {
                color = selectionColor;
                alpha = color[3] || 1.0;
            }
            // Check if we have metric coloring
            else if (gradient && this._currentColorMetric) {
                const nodeData = this.nodeDataMap.get(id);
                const value = nodeData?.[this._currentColorMetric];
                
                if (typeof value === 'number' && !isNaN(value)) {
                    const normalized = max > min ? (value - min) / (max - min) : 0.5;
                    // Use ColorGradients.interpolateRgba to get color from gradient
                    color = ColorGradients.interpolateRgba(gradient, Math.max(0, Math.min(1, normalized)));
                } else {
                    color = defaultColor;
                }
                alpha = color[3] || 1.0;
            }
            // Default color
            else {
                color = defaultColor;
                alpha = color[3] || 1.0;
            }
            
            // Apply color
            colors[offset] = color[0];
            colors[offset + 1] = color[1];
            colors[offset + 2] = color[2];
            
            // Set alpha based on visibility (hidden nodes override path dimming)
            if (this._hiddenNodes.has(id)) {
                colors[offset + 3] = 0.0; // Completely invisible for hidden nodes
            } else {
                colors[offset + 3] = alpha;
            }
        });
        
        this.graph.setPointColors(colors);
        this.graph.render();
    }
    
    applyDefaultColors() {
        const defaultNodeColor = RendererSettings.getDefaultNodeColorRgba();
        console.log('[CosmosAdapter] applyDefaultColors called');
        console.log('[CosmosAdapter] applyDefaultColors - defaultNodeColor RGBA:', JSON.stringify(defaultNodeColor));
        
        // Set node colors with FULL ALPHA
        const nodeColors = new Float32Array(this.nodeIds.length * 4);
        for (let i = 0; i < this.nodeIds.length; i++) {
            const offset = i * 4;
            nodeColors[offset] = defaultNodeColor[0];
            nodeColors[offset + 1] = defaultNodeColor[1];
            nodeColors[offset + 2] = defaultNodeColor[2];
            nodeColors[offset + 3] = 1.0;  // Force full alpha
        }
        
        // Store as base colors for selection restoration
        this._baseNodeColors = new Float32Array(nodeColors);
        console.log('[CosmosAdapter] applyDefaultColors - stored _baseNodeColors length:', this._baseNodeColors.length);
        console.log('[CosmosAdapter] applyDefaultColors - first node RGBA:', 
            nodeColors[0], nodeColors[1], nodeColors[2], nodeColors[3]);
        
        this.graph.setPointColors(nodeColors);
        this.graph.render();
    }
    
    colorNodesByMetric(metricName, values, colorScale = 'spectral') {
        if (!values || Object.keys(values).length === 0) {
            console.warn('[CosmosAdapter] No values provided for coloring');
            return;
        }
        
        this._currentColorMetric = metricName;
        this._currentColorScale = colorScale;
        
        // Get the gradient array from the name
        const gradient = ColorGradients.gradients[colorScale];
        if (!gradient) {
            console.warn(`[CosmosAdapter] Unknown color scale: ${colorScale}, using spectral`);
            const fallbackGradient = ColorGradients.gradients['spectral'];
            if (!fallbackGradient) {
                console.error('[CosmosAdapter] No gradients available');
                return;
            }
        }
        const gradientToUse = gradient || ColorGradients.gradients['spectral'];
        
        // Get min/max for normalization
        const numericValues = Object.values(values).filter(v => typeof v === 'number' && !isNaN(v));
        if (numericValues.length === 0) return;
        
        const min = Math.min(...numericValues);
        const max = Math.max(...numericValues);
        const range = max - min || 1;
        
        const colorArray = new Float32Array(this.nodeIds.length * 4);
        const defaultColor = RendererSettings.getDefaultNodeColorRgba();
        
        this.nodeIds.forEach((id, i) => {
            const value = values[id];
            const offset = i * 4;
            
            if (typeof value === 'number' && !isNaN(value)) {
                const normalized = (value - min) / range;
                // interpolateRgba returns [r, g, b, a] in 0-1 range
                const rgba = ColorGradients.interpolateRgba(gradientToUse, normalized);
                colorArray[offset] = rgba[0];
                colorArray[offset + 1] = rgba[1];
                colorArray[offset + 2] = rgba[2];
                colorArray[offset + 3] = 1.0;  // Force full alpha
            } else {
                colorArray[offset] = defaultColor[0];
                colorArray[offset + 1] = defaultColor[1];
                colorArray[offset + 2] = defaultColor[2];
                colorArray[offset + 3] = 1.0;  // Force full alpha
            }
        });
        
        // Store as base colors for selection restoration
        this._baseNodeColors = new Float32Array(colorArray);
        console.log('[CosmosAdapter] colorNodesByMetric - stored _baseNodeColors, alpha:', colorArray[3]);
        
        this.graph.setPointColors(colorArray);
        this.graph.render();
    }
    
    clearColoring() {
        this._currentColorMetric = null;
        this._currentColorScale = null;
        this.applyDefaultColors();
        this.graph.render();
    }
    
    // ============================================================================
    // SIZING METHODS
    // ============================================================================
    
    sizeNodesByMetric(metricName, values, minSize = 4, maxSize = 20) {
        if (!values || Object.keys(values).length === 0) return;
        
        const numericValues = Object.values(values).filter(v => typeof v === 'number' && !isNaN(v));
        if (numericValues.length === 0) return;
        
        const min = Math.min(...numericValues);
        const max = Math.max(...numericValues);
        const range = max - min || 1;
        
        const sizeArray = new Float32Array(this.nodeIds.length);
        const defaultSize = (minSize + maxSize) / 2;
        
        this.nodeIds.forEach((id, i) => {
            const value = values[id];
            if (typeof value === 'number' && !isNaN(value)) {
                const normalized = (value - min) / range;
                sizeArray[i] = minSize + normalized * (maxSize - minSize);
            } else {
                sizeArray[i] = defaultSize;
            }
        });
        
        this.graph.setPointSizes(sizeArray);
        this.graph.render();
    }
    
    resetNodeSizes() {
        const config = RendererSettings.getCosmosConfig();
        const defaultSize = config.pointSize || 6;
        
        const sizeArray = new Float32Array(this.nodeIds.length);
        sizeArray.fill(defaultSize);
        
        this.graph.setPointSizes(sizeArray);
        this.graph.render();
    }
    
    // ============================================================================
    // PATH/FLOW HIGHLIGHTING
    // ============================================================================
    
    highlightPath(nodeIds, edgeKeys, options = {}) {
        const {
            nodeColor = [1, 0.5, 0, 1],  // Orange
            edgeColor = [1, 0.5, 0, 0.8],
            dimOthers = true,
            dimOpacity = 0.2
        } = options;
        
        this._isPathHighlightActive = true;
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        
        // Store path colors
        nodeIds.forEach(id => {
            this._pathNodeColors.set(id, nodeColor);
        });
        
        edgeKeys.forEach(key => {
            this._pathEdgeColors.set(key, edgeColor);
        });
        
        // Apply node colors
        const colorArray = new Float32Array(this.nodeIds.length * 4);
        const defaultColor = RendererSettings.getDefaultNodeColorRgba();
        
        this.nodeIds.forEach((id, i) => {
            const offset = i * 4;
            if (this._pathNodeColors.has(id)) {
                const c = this._pathNodeColors.get(id);
                colorArray[offset] = c[0];
                colorArray[offset + 1] = c[1];
                colorArray[offset + 2] = c[2];
                colorArray[offset + 3] = c[3];
            } else if (dimOthers) {
                colorArray[offset] = defaultColor[0];
                colorArray[offset + 1] = defaultColor[1];
                colorArray[offset + 2] = defaultColor[2];
                colorArray[offset + 3] = dimOpacity;
            } else {
                colorArray[offset] = defaultColor[0];
                colorArray[offset + 1] = defaultColor[1];
                colorArray[offset + 2] = defaultColor[2];
                colorArray[offset + 3] = defaultColor[3];
            }
        });
        
        this.graph.setPointColors(colorArray);
        
        // Apply edge colors if supported
        if (typeof this.graph.setLinkColors === 'function') {
            const edgeArray = Array.from(this.edgeDataMap.values());
            const linkColors = new Float32Array(edgeArray.length * 4);
            const defaultEdgeColor = RendererSettings.getDefaultEdgeColorRgba();
            
            edgeArray.forEach((edge, i) => {
                const key = `${edge.source}-${edge.target}`;
                const offset = i * 4;
                
                if (this._pathEdgeColors.has(key)) {
                    const c = this._pathEdgeColors.get(key);
                    linkColors[offset] = c[0];
                    linkColors[offset + 1] = c[1];
                    linkColors[offset + 2] = c[2];
                    linkColors[offset + 3] = c[3];
                } else if (dimOthers) {
                    linkColors[offset] = defaultEdgeColor[0];
                    linkColors[offset + 1] = defaultEdgeColor[1];
                    linkColors[offset + 2] = defaultEdgeColor[2];
                    linkColors[offset + 3] = dimOpacity * 0.5;
                } else {
                    linkColors[offset] = defaultEdgeColor[0];
                    linkColors[offset + 1] = defaultEdgeColor[1];
                    linkColors[offset + 2] = defaultEdgeColor[2];
                    linkColors[offset + 3] = defaultEdgeColor[3];
                }
            });
            
            this.graph.setLinkColors(linkColors);
        }
        
        this.graph.render();
    }
    
    clearPathHighlight() {
        this._isPathHighlightActive = false;
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        
        // Also clear isolation mode if active
        if (this._isIsolationMode) {
            this._isIsolationMode = false;
            this._isolatedNodes = null;
            this._isolatedEdges = null;
            // Restore all edges
            this.restoreAllEdges();
        }
        
        this.applyDefaultColors();
        this.applyDefaultEdgeColors();
        this.graph.render();
    }
    
    // ============================================================================
    // VIEW CONTROL
    // ============================================================================
    
    fitView(nodeIds = null, padding = 0.1) {
        if (!nodeIds || nodeIds.length === 0) {
            this.graph.fitView();
            return;
        }
        
        // Map input IDs to actual renderer IDs using the helper
        const mappedIds = this.findNodeIds(nodeIds);
        
        // Fit to specific nodes
        const indices = mappedIds
            .map(id => this.nodeIndices.get(id))
            .filter(idx => idx !== undefined);
        
        if (indices.length === 0) {
            console.log('[CosmosAdapter] fitView: No matching nodes found for', nodeIds.slice(0, 3));
            this.graph.fitView();
            return;
        }
        
        // For a single node, zoom to it
        if (indices.length === 1) {
            if (typeof this.graph.zoomToPointByIndex === 'function') {
                this.graph.zoomToPointByIndex(indices[0], 500, 2);
            } else {
                this.graph.fitView();
            }
            return;
        }
        
        // For multiple nodes, calculate bounding box and fit
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        indices.forEach(idx => {
            if (this.positions) {
                const x = this.positions[idx * 2];
                const y = this.positions[idx * 2 + 1];
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
        });
        
        // Try to use cosmos.gl specific methods if available
        if (typeof this.graph.setConfig === 'function' && minX !== Infinity) {
            try {
                this.graph.setConfig({
                    fitViewByPointIndices: indices,
                    fitViewPadding: padding
                });
                this.graph.fitView();
                return;
            } catch (e) {
                console.warn('[CosmosAdapter] fitViewByPointIndices not supported:', e);
            }
        }
        
        // Fallback to general fit
        this.graph.fitView();
    }
    
    zoomToNodes(nodeIds, padding = 0.2) {
        // Use fitView with node IDs
        this.fitView(nodeIds, padding);
    }
    
    centerOnNode(nodeId) {
        const actualId = this.findNodeId(nodeId);
        const index = actualId ? this.nodeIndices.get(actualId) : undefined;
        if (index === undefined) return;
        
        // Try to zoom to specific point
        if (typeof this.graph.zoomToPointByIndex === 'function') {
            this.graph.zoomToPointByIndex(index, 500, 2);
        } else {
            // Fallback to fitView
            this.fitView([nodeId], 0.1);
        }
    }
    
    zoomToNode(nodeId, zoomLevel = 2, duration = 500) {
        const actualId = this.findNodeId(nodeId);
        const index = actualId ? this.nodeIndices.get(actualId) : undefined;
        if (index === undefined) return;
        
        if (typeof this.graph.zoomToPointByIndex === 'function') {
            this.graph.zoomToPointByIndex(index, duration, zoomLevel);
        } else {
            this.fitView([actualId], 0.1);
        }
    }
    
    getZoom() {
        // cosmos.gl doesn't expose zoom directly, estimate from view
        return 1;
    }
    
    setZoom(level, duration = 0) {
        if (typeof this.graph.zoom === 'function') {
            this.graph.zoom(level, duration);
        }
    }
    
    // ============================================================================
    // QUERY METHODS
    // ============================================================================
    
    getNodeData(nodeId) {
        return this.nodeDataMap.get(nodeId);
    }
    
    getEdgeData(edgeId) {
        return this.edgeDataMap.get(edgeId);
    }
    
    getAllNodes() {
        return Array.from(this.nodeDataMap.values());
    }
    
    getAllEdges() {
        return Array.from(this.edgeDataMap.values());
    }
    
    getNodeCount() {
        return this.nodeIds.length;
    }
    
    getEdgeCount() {
        return this.edgeDataMap.size;
    }
    
    getNeighbors(nodeId) {
        const incoming = this.incomingEdges.get(nodeId) || [];
        const outgoing = this.outgoingEdges.get(nodeId) || [];
        return {
            incoming: [...incoming],
            outgoing: [...outgoing],
            all: [...new Set([...incoming, ...outgoing])]
        };
    }
    
    /**
     * Get incoming neighbors for a node
     * @param {string} nodeId
     * @returns {string[]} Array of neighbor node IDs
     */
    getIncomingNeighbors(nodeId) {
        const actualId = this.findNodeId(nodeId);
        if (!actualId) return [];
        return [...(this.incomingEdges.get(actualId) || [])];
    }
    
    /**
     * Get outgoing neighbors for a node
     * @param {string} nodeId
     * @returns {string[]} Array of neighbor node IDs
     */
    getOutgoingNeighbors(nodeId) {
        const actualId = this.findNodeId(nodeId);
        if (!actualId) return [];
        return [...(this.outgoingEdges.get(actualId) || [])];
    }
    
    getNodePosition(nodeId) {
        const index = this.nodeIndices.get(nodeId);
        if (index === undefined || !this.positions) return null;
        
        return {
            x: this.positions[index * 2],
            y: this.positions[index * 2 + 1]
        };
    }
    
    getAllPositions() {
        const positions = {};
        this.nodeIds.forEach((id, index) => {
            if (this.positions) {
                positions[id] = {
                    x: this.positions[index * 2],
                    y: this.positions[index * 2 + 1]
                };
            }
        });
        return positions;
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
    // POSITION PRESERVATION SETTINGS
    // ============================================================================
    
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
    
    // ============================================================================
    // INDEX HELPERS
    // ============================================================================
    
    getNodeIndex(nodeId) {
        return this.nodeIndices.get(nodeId);
    }
    
    getNodeIdByIndex(index) {
        return this.nodeIds[index];
    }
    
    // ============================================================================
    // REQUIRED INTERFACE METHODS
    // ============================================================================
    
    /**
     * Get renderer type
     * @returns {string} 'cosmos'
     */
    getType() {
        return 'cosmos';
    }
    
    /**
     * Clear all graph data
     */
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
        this.positions = null;
        this._pathNodeColors.clear();
        this._pathEdgeColors.clear();
        this._isPathHighlightActive = false;
        this._storedEdgeData = [];
        this._edgeLinkData = null;
        
        // Clear cosmos.gl data
        this.graph.setPointPositions(new Float32Array(0));
        this.graph.setLinks(new Float32Array(0));
        this.graph.render();
    }
    
    /**
     * Remove elements from the graph
     * @param {Array} nodeIds - Node IDs to remove
     * @param {Array} edgeIds - Edge IDs to remove
     */
    removeElements(nodeIds = [], edgeIds = []) {
        // Remove edges first
        if (edgeIds.length > 0) {
            this.removeEdges(edgeIds);
        }
        
        // Removing nodes requires rebuilding - cosmos.gl doesn't support incremental node removal
        if (nodeIds.length > 0) {
            const nodeIdSet = new Set(nodeIds);
            const remainingNodes = Array.from(this.nodeDataMap.values())
                .filter(node => !nodeIdSet.has(node.id));
            const remainingEdges = Array.from(this.edgeDataMap.values())
                .filter(edge => !nodeIdSet.has(edge.source) && !nodeIdSet.has(edge.target));
            
            this.setData(remainingNodes, remainingEdges);
        }
    }
    
    /**
     * Get selected edges
     * @returns {Array} Selected edge IDs
     */
    getSelectedEdges() {
        // cosmos.gl doesn't have native edge selection
        return [];
    }
    
    /**
     * Apply node colors based on metric values
     * @param {string} metricName - Name of the metric
     * @param {Object} colorScale - Color scale configuration
     */
    applyNodeColors(metricName, colorScale = {}) {
        const gradient = colorScale.gradient || 'spectral';
        const values = {};
        
        // Build values map from node data
        this.nodeIds.forEach(id => {
            const node = this.nodeDataMap.get(id);
            if (node && node[metricName] !== undefined) {
                values[id] = node[metricName];
            }
        });
        
        this.colorNodesByMetric(metricName, values, gradient);
    }
    
    /**
     * Apply node sizes based on metric values
     * @param {string} metricName - Name of the metric
     * @param {Object} sizeScale - Size scale configuration {min, max}
     */
    applyNodeSizes(metricName, sizeScale = {}) {
        const minSize = sizeScale.min || 4;
        const maxSize = sizeScale.max || 20;
        const values = {};
        
        // Build values map from node data
        this.nodeIds.forEach(id => {
            const node = this.nodeDataMap.get(id);
            if (node && node[metricName] !== undefined) {
                values[id] = node[metricName];
            }
        });
        
        this.sizeNodesByMetric(metricName, values, minSize, maxSize);
    }
    
    /**
     * Export graph as PNG
     * @returns {Promise<Blob>}
     */
    async exportPNG() {
        // cosmos.gl renders to WebGL canvas
        const canvas = this.container.querySelector('canvas');
        if (!canvas) {
            throw new Error('No canvas found for export');
        }
        
        return new Promise((resolve, reject) => {
            try {
                canvas.toBlob(blob => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Failed to create PNG blob'));
                    }
                }, 'image/png');
            } catch (e) {
                reject(e);
            }
        });
    }
    
    // ============================================================================
    // MISSING INTERFACE METHODS
    // ============================================================================
    
    /**
     * Set background color for the graph
     * @param {string} color - Hex color string
     */
    setBackgroundColor(color) {
        if (!color || !this.graph) return;
        
        console.log('[CosmosAdapter] Setting background color:', color);
        
        try {
            // cosmos.gl expects backgroundColor as hex string or rgba array
            // Try both approaches
            if (typeof this.graph.setConfig === 'function') {
                // First try with hex string
                this.graph.setConfig({ backgroundColor: color });
            }
            
            // Also update the container and canvas background as fallback
            if (this.container) {
                this.container.style.backgroundColor = color;
                // Find the canvas inside and update it too
                const canvas = this.container.querySelector('canvas');
                if (canvas) {
                    canvas.style.backgroundColor = color;
                }
            }
            
            this.graph.render();
        } catch (err) {
            console.error('[CosmosAdapter] Error setting background color:', err);
        }
    }
    
    /**
     * Set edge styling
     * @param {Object} style - Style configuration {color, opacity, width}
     */
    setEdgeStyle(style) {
        if (!this._edgeStyle) {
            this._edgeStyle = { color: '#ffffff', opacity: 0.5, width: 1 };
        }
        
        if (style.color !== undefined) this._edgeStyle.color = style.color;
        if (style.opacity !== undefined) this._edgeStyle.opacity = style.opacity;
        if (style.width !== undefined) this._edgeStyle.width = style.width;
        
        try {
            const edgeCount = this.edgeDataMap.size;
            console.log('[CosmosAdapter] setEdgeStyle - edge count:', edgeCount, 'style:', this._edgeStyle);
            
            if (edgeCount > 0) {
                const opacity = this._edgeStyle.opacity !== undefined ? this._edgeStyle.opacity : 0.5;
                const rgba = RendererSettings.hexToRgba(this._edgeStyle.color, opacity);
                console.log('[CosmosAdapter] Edge RGBA:', rgba);
                
                // Set link colors - Float32Array with RGBA per edge
                const colors = new Float32Array(edgeCount * 4);
                for (let i = 0; i < edgeCount; i++) {
                    colors[i * 4] = rgba[0];
                    colors[i * 4 + 1] = rgba[1];
                    colors[i * 4 + 2] = rgba[2];
                    colors[i * 4 + 3] = rgba[3];
                }
                
                if (typeof this.graph.setLinkColors === 'function') {
                    this.graph.setLinkColors(colors);
                    console.log('[CosmosAdapter] setLinkColors called');
                } else {
                    console.warn('[CosmosAdapter] setLinkColors not available, using setConfig');
                    this.graph.setConfig({ linkColor: rgba });
                }
                
                // Set link widths - Float32Array with width per edge (cosmos.gl uses setLinkWidths plural)
                if (this._edgeStyle.width !== undefined) {
                    const width = Math.max(0.1, Math.min(10, this._edgeStyle.width));
                    
                    if (typeof this.graph.setLinkWidths === 'function') {
                        const widths = new Float32Array(edgeCount);
                        widths.fill(width);
                        this.graph.setLinkWidths(widths);
                        console.log('[CosmosAdapter] setLinkWidths called with width:', width);
                    } else {
                        console.warn('[CosmosAdapter] setLinkWidths not available, using setConfig');
                        this.graph.setConfig({ linkWidth: width });
                    }
                }
            }
            
            this.graph.render();
            console.log('[CosmosAdapter] Edge style applied successfully');
        } catch (err) {
            console.error('[CosmosAdapter] Error setting edge style:', err);
        }
    }
    
    /**
     * Apply default edge colors
     */
    applyDefaultEdgeColors() {
        const styleConfig = RendererSettings.getStyleConfig();
        this.setEdgeStyle({
            color: styleConfig.defaultEdgeColor || '#ffffff',
            opacity: styleConfig.defaultEdgeOpacity || 0.5,
            width: 1
        });
    }
    
    /**
     * Reset all styling to defaults
     */
    resetStyle() {
        this._currentColorMetric = null;
        this._currentColorScale = null;
        this.clearPathHighlight();
        this.applyDefaultColors();
        this.applyDefaultEdgeColors();
        
        // Reset node sizes to default
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const sizes = new Float32Array(this.nodeIds.length);
        sizes.fill(cosmosConfig.pointSize || 4);
        this.graph.setPointSizes(sizes);
        
        this.graph.render();
        console.log('[CosmosAdapter] Style reset to defaults');
    }
    
    /**
     * Set performance mode
     * @param {boolean} enabled - Enable performance mode
     */
    setPerformanceMode(enabled) {
        // cosmos.gl is already GPU-accelerated, but we can adjust some settings
        if (this.graph) {
            this.graph.setConfig({
                // Reduce quality for performance
                renderLinks: !enabled || this._storedEdgeData.length < 100000
            });
        }
        console.log('[CosmosAdapter] Performance mode:', enabled);
    }
    
    /**
     * Select edges by ID
     * @param {Array} edgeIds - Edge IDs to select
     */
    selectEdges(edgeIds) {
        // cosmos.gl doesn't support edge selection natively
        // We can highlight them instead
        console.log('[CosmosAdapter] selectEdges not natively supported, highlighting instead');
        // Could implement via edge coloring if needed
    }
    
    /**
     * Add CSS class to elements (no-op for cosmos.gl)
     * @param {Array} elementIds - Element IDs
     * @param {string} className - Class name to add
     */
    addClass(elementIds, className) {
        // cosmos.gl doesn't use CSS classes
        console.log('[CosmosAdapter] addClass not supported in cosmos.gl');
    }
    
    /**
     * Remove CSS class from elements (no-op for cosmos.gl)
     * @param {Array} elementIds - Element IDs
     * @param {string} className - Class name to remove
     */
    removeClass(elementIds, className) {
        // cosmos.gl doesn't use CSS classes
        console.log('[CosmosAdapter] removeClass not supported in cosmos.gl');
    }
    
    /**
     * Center view on specific position
     */
    center() {
        if (this.graph) {
            this.graph.setZoomLevel(1);
            this.graph.render();
        }
    }
    
    /**
     * Get current viewport
     * @returns {Object} Viewport info
     */
    getViewport() {
        if (!this.graph) return { x: 0, y: 0, zoom: 1 };
        
        return {
            x: 0,
            y: 0,
            zoom: this.graph.getZoomLevel() || 1
        };
    }
    
    /**
     * Set viewport
     * @param {Object} viewport - Viewport configuration
     */
    setViewport(viewport) {
        if (!this.graph) return;
        
        if (viewport.zoom !== undefined) {
            this.graph.setZoomLevel(viewport.zoom);
        }
        this.graph.render();
    }
    
    /**
     * Set zoom level
     * @param {number} zoom - Zoom level
     */
    setZoom(zoom) {
        if (this.graph) {
            this.graph.setZoomLevel(zoom);
            this.graph.render();
        }
    }
    
    /**
     * Start batch update (no-op for cosmos.gl)
     */
    startBatch() {
        // cosmos.gl doesn't need batching
    }
    
    /**
     * End batch update (no-op for cosmos.gl)
     */
    endBatch() {
        // cosmos.gl doesn't need batching
        this.graph?.render();
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
        
        if (this._updateDebounceTimer) {
            clearTimeout(this._updateDebounceTimer);
        }
        
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
    // COSMOS-SPECIFIC GETTERS
    // ============================================================================
    
    getGraph() {
        return this.graph;
    }
}

// Make available globally
window.CosmosAdapter = CosmosAdapter;