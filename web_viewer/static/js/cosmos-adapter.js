/**
 * Cosmos Adapter
 * 
 * Wraps @cosmograph/cosmos to implement GraphRendererInterface.
 * Provides GPU-accelerated graph visualization for large graphs.
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
            
            // Events (v2.0 uses index-based events)
            onClick: (pointIndex) => this.handleClick(pointIndex),
            onMouseMove: (pointIndex) => this.handleHover(pointIndex),
            onZoom: (zoom) => this.emit('viewportChange', { zoom })
        };
        
        this.graph = new this.CosmosGraph(this.container, config);
        
        // Setup WebGL context loss handling
        this.setupContextLossHandling();
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
        // Store current data
        const nodes = Array.from(this.nodeDataMap.values());
        const edges = Array.from(this.edgeDataMap.values());
        
        // Recreate graph
        if (this.graph) {
            this.graph.destroy();
        }
        this.initialize();
        
        // Restore data
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
        
        // Build node index mapping
        nodes.forEach((node, index) => {
            this.nodeIndices.set(node.id, index);
            this.nodeIds[index] = node.id;
            this.nodeDataMap.set(node.id, node);
            
            // Initialize edge tracking
            this.incomingEdges.set(node.id, []);
            this.outgoingEdges.set(node.id, []);
        });
        
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
                
                // Track incoming/outgoing edges
                this.incomingEdges.get(edge.target)?.push(edge.source);
                this.outgoingEdges.get(edge.source)?.push(edge.target);
            }
        });
        
        // Set data to cosmos
        this.graph.setPointPositions(this.positions);
        this.graph.setLinks(linkData);
        
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
                
                // Update stored data
                const nodeData = this.nodeDataMap.get(id);
                if (nodeData) {
                    nodeData.x = pos.x;
                    nodeData.y = pos.y;
                }
            }
        });
        
        this.positions = posArray;
        this.graph.setPointPositions(posArray);
        this.graph.render();
    }
    
    addNodes(nodes) {
        // cosmos.gl requires full rebuild for node additions
        const existingNodes = Array.from(this.nodeDataMap.values());
        const existingEdges = Array.from(this.edgeDataMap.values());
        this.setData([...existingNodes, ...nodes], existingEdges);
    }
    
    addEdges(edges) {
        // Add edges to tracking
        edges.forEach(edge => {
            const edgeId = edge.id || `${edge.source}-${edge.target}`;
            this.edgeDataMap.set(edgeId, edge);
            this.incomingEdges.get(edge.target)?.push(edge.source);
            this.outgoingEdges.get(edge.source)?.push(edge.target);
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
        
        this.graph.setLinks(linkData);
        this.graph.render();
    }
    
    removeElements(nodeIds = [], edgeIds = []) {
        // Remove nodes and rebuild
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
        
        this.graph.setPointPositions(new Float32Array(0));
        this.graph.setLinks(new Float32Array(0));
        this.graph.render();
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
        
        // Calculate range if not provided
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
            
            // Check if node is selected or highlighted
            if (this.selectedIndices.has(index)) {
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
        
        // Calculate value range
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
        // cosmos.gl uses uniform edge styling or per-edge arrays
        if (style.color) {
            const rgba = RendererSettings.hexToRgba(style.color, style.opacity || 1);
            // Would need to set per-edge colors if cosmos supports it
        }
        if (style.width !== undefined) {
            // Set link width globally
            // Note: cosmos.gl may not support dynamic width changes
        }
    }
    
    resetStyle() {
        this._currentColorMetric = null;
        this._currentColorScale = null;
        this.applyDefaultColors();
        
        // Reset sizes
        const cosmosConfig = RendererSettings.getCosmosConfig();
        const sizes = new Float32Array(this.nodeIds.length);
        sizes.fill(cosmosConfig.pointSize);
        this.graph.setPointSizes(sizes);
        
        this.graph.render();
    }
    
    setPerformanceMode(enabled) {
        this._performanceMode = enabled;
        // cosmos.gl is already optimized for performance
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
        // cosmos.gl doesn't support edge selection directly
        // We track this but can't visually show it
        console.warn('[CosmosAdapter] Edge selection not visually supported');
    }
    
    getSelectedNodes() {
        return Array.from(this.selectedIndices).map(idx => this.nodeIds[idx]);
    }
    
    getSelectedEdges() {
        return []; // cosmos.gl doesn't support edge selection
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
    
    /**
     * Update node colors to show selection/highlight state
     */
    updateSelectionVisuals() {
        // If we have a color metric applied, reapply it (it handles selection)
        if (this._currentColorMetric && this._currentColorScale) {
            this.applyNodeColors(this._currentColorMetric, this._currentColorScale);
            return;
        }
        
        // Otherwise, apply selection/highlight colors over defaults
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
        // cosmos.gl viewport info
        return {
            zoom: this.graph.getZoom?.() || 1,
            pan: { x: 0, y: 0 } // cosmos.gl manages this internally
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
    
    // ============================================================================
    // EVENT HANDLERS
    // ============================================================================
    
    handleClick(index) {
        if (index === undefined || index < 0 || index >= this.nodeIds.length) {
            this.emit('backgroundClick', {});
            return;
        }
        
        const nodeId = this.nodeIds[index];
        const nodeData = this.nodeDataMap.get(nodeId);
        
        // Toggle selection
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
            selectedNodes: this.selectedIndices.size,
            selectedEdges: 0
        };
    }
    
    /**
     * Get node position
     * @param {string} nodeId
     * @returns {Object|null} {x, y}
     */
    getNodePosition(nodeId) {
        const index = this.nodeIndices.get(nodeId);
        if (index === undefined || !this.positions) return null;
        
        return {
            x: this.positions[index * 2],
            y: this.positions[index * 2 + 1]
        };
    }
    
    async exportPNG() {
        // cosmos.gl renders to canvas, we can extract it
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
    }
    
    // ============================================================================
    // COSMOS-SPECIFIC METHODS
    // ============================================================================
    
    /**
     * Get the underlying cosmos.gl graph instance
     * @returns {Object}
     */
    getGraph() {
        return this.graph;
    }
    
    /**
     * Start force simulation
     */
    startSimulation() {
        this.graph.start();
    }
    
    /**
     * Pause force simulation
     */
    pauseSimulation() {
        this.graph.pause();
    }
    
    /**
     * Restart force simulation
     */
    restartSimulation() {
        this.graph.restart();
    }
    
    /**
     * Get node index by ID
     * @param {string} nodeId
     * @returns {number|undefined}
     */
    getNodeIndex(nodeId) {
        return this.nodeIndices.get(nodeId);
    }
    
    /**
     * Get node ID by index
     * @param {number} index
     * @returns {string|undefined}
     */
    getNodeIdByIndex(index) {
        return this.nodeIds[index];
    }
}

// Make available globally
window.CosmosAdapter = CosmosAdapter;