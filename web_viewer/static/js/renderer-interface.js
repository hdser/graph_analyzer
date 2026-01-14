/**
 * Renderer Interface Module
 * 
 * Abstract base class that defines the interface for graph renderers.
 * Both CosmosAdapter and CytoscapeAdapter implement this interface.
 */

class GraphRendererInterface {
    /**
     * @param {HTMLElement} container - DOM container for the renderer
     * @param {Object} options - Renderer options
     */
    constructor(container, options = {}) {
        if (new.target === GraphRendererInterface) {
            throw new Error('GraphRendererInterface is abstract and cannot be instantiated directly');
        }
        
        this.container = container;
        this.options = options;
        this.eventHandlers = new Map();
        this.nodeDataMap = new Map();  // id -> node data
        this.edgeDataMap = new Map();  // id -> edge data
        this._disposed = false;
    }
    
    // ============================================================================
    // DATA METHODS
    // ============================================================================
    
    /**
     * Set graph data
     * @param {Array} nodes - Array of node objects {id, x, y, ...metrics}
     * @param {Array} edges - Array of edge objects {source, target, ...}
     */
    setData(nodes, edges) {
        throw new Error('setData() must be implemented by subclass');
    }
    
    /**
     * Update node positions (for layout results)
     * @param {Map|Object} positions - id -> {x, y}
     */
    updatePositions(positions) {
        throw new Error('updatePositions() must be implemented by subclass');
    }
    
    /**
     * Add nodes incrementally
     * @param {Array} nodes - New nodes to add
     */
    addNodes(nodes) {
        throw new Error('addNodes() must be implemented by subclass');
    }
    
    /**
     * Add edges incrementally (for progressive loading)
     * @param {Array} edges - New edges to add
     */
    addEdges(edges) {
        throw new Error('addEdges() must be implemented by subclass');
    }
    
    /**
     * Remove elements
     * @param {Array} nodeIds - Node IDs to remove
     * @param {Array} edgeIds - Edge IDs to remove
     */
    removeElements(nodeIds = [], edgeIds = []) {
        throw new Error('removeElements() must be implemented by subclass');
    }
    
    /**
     * Clear all data
     */
    clear() {
        throw new Error('clear() must be implemented by subclass');
    }
    
    // ============================================================================
    // VISUAL STYLING
    // ============================================================================
    
    /**
     * Apply node colors based on metric
     * @param {string} metricName - Metric to color by
     * @param {Object} colorScale - {gradient: 'spectral', min, max}
     */
    applyNodeColors(metricName, colorScale) {
        throw new Error('applyNodeColors() must be implemented by subclass');
    }
    
    /**
     * Apply node sizes based on metric
     * @param {string} metricName - Metric to size by
     * @param {Object} sizeScale - {min: 5, max: 30}
     */
    applyNodeSizes(metricName, sizeScale) {
        throw new Error('applyNodeSizes() must be implemented by subclass');
    }
    
    /**
     * Set edge styling
     * @param {Object} style - {color, opacity, width}
     */
    setEdgeStyle(style) {
        throw new Error('setEdgeStyle() must be implemented by subclass');
    }
    
    /**
     * Reset all styling to defaults
     */
    resetStyle() {
        throw new Error('resetStyle() must be implemented by subclass');
    }
    
    /**
     * Toggle performance mode (simplified rendering)
     * @param {boolean} enabled
     */
    setPerformanceMode(enabled) {
        throw new Error('setPerformanceMode() must be implemented by subclass');
    }
    
    // ============================================================================
    // SELECTION & HIGHLIGHTING
    // ============================================================================
    
    /**
     * Select nodes
     * @param {Array} nodeIds - IDs to select
     * @param {boolean} additive - Add to existing selection?
     */
    selectNodes(nodeIds, additive = false) {
        throw new Error('selectNodes() must be implemented by subclass');
    }
    
    /**
     * Select edges
     * @param {Array} edgeIds - IDs to select
     * @param {boolean} additive - Add to existing selection?
     */
    selectEdges(edgeIds, additive = false) {
        throw new Error('selectEdges() must be implemented by subclass');
    }
    
    /**
     * Get currently selected node IDs
     * @returns {Array} Selected node IDs
     */
    getSelectedNodes() {
        throw new Error('getSelectedNodes() must be implemented by subclass');
    }
    
    /**
     * Get currently selected edge IDs
     * @returns {Array} Selected edge IDs
     */
    getSelectedEdges() {
        throw new Error('getSelectedEdges() must be implemented by subclass');
    }
    
    /**
     * Clear all selections
     */
    clearSelection() {
        throw new Error('clearSelection() must be implemented by subclass');
    }
    
    /**
     * Highlight specific nodes
     * @param {Array} nodeIds - IDs to highlight
     * @param {string} className - Optional class name for the highlight
     */
    highlightNodes(nodeIds, className = 'highlighted') {
        throw new Error('highlightNodes() must be implemented by subclass');
    }
    
    /**
     * Highlight node neighbors
     * @param {string} nodeId - Center node
     * @param {string} direction - 'in', 'out', 'both'
     */
    highlightNeighbors(nodeId, direction = 'both') {
        throw new Error('highlightNeighbors() must be implemented by subclass');
    }
    
    /**
     * Clear all highlights
     */
    clearHighlights() {
        throw new Error('clearHighlights() must be implemented by subclass');
    }
    
    /**
     * Add a CSS class to elements
     * @param {Array} elementIds - IDs of elements
     * @param {string} className - Class name to add
     * @param {string} type - 'nodes' or 'edges'
     */
    addClass(elementIds, className, type = 'nodes') {
        throw new Error('addClass() must be implemented by subclass');
    }
    
    /**
     * Remove a CSS class from elements
     * @param {Array} elementIds - IDs of elements
     * @param {string} className - Class name to remove
     * @param {string} type - 'nodes' or 'edges'
     */
    removeClass(elementIds, className, type = 'nodes') {
        throw new Error('removeClass() must be implemented by subclass');
    }
    
    // ============================================================================
    // VIEWPORT CONTROL
    // ============================================================================
    
    /**
     * Fit view to show all elements or specific nodes
     * @param {Array} nodeIds - Optional specific nodes to fit
     * @param {number} padding - Padding factor
     */
    fitView(nodeIds = null, padding = 50) {
        throw new Error('fitView() must be implemented by subclass');
    }
    
    /**
     * Center the viewport
     */
    center() {
        throw new Error('center() must be implemented by subclass');
    }
    
    /**
     * Zoom to specific node
     * @param {string} nodeId
     * @param {number} zoomLevel
     * @param {number} duration - Animation duration ms
     */
    zoomToNode(nodeId, zoomLevel = 3, duration = 500) {
        throw new Error('zoomToNode() must be implemented by subclass');
    }
    
    /**
     * Get current viewport state
     * @returns {Object} {zoom, pan: {x, y}}
     */
    getViewport() {
        throw new Error('getViewport() must be implemented by subclass');
    }
    
    /**
     * Set viewport state
     * @param {Object} viewport - {zoom, pan: {x, y}}
     */
    setViewport(viewport) {
        throw new Error('setViewport() must be implemented by subclass');
    }
    
    /**
     * Set zoom level
     * @param {number} level
     */
    setZoom(level) {
        throw new Error('setZoom() must be implemented by subclass');
    }
    
    // ============================================================================
    // GRAPH QUERIES
    // ============================================================================
    
    /**
     * Get incoming neighbors of a node
     * @param {string} nodeId
     * @returns {Array} Array of neighbor node IDs
     */
    getIncomingNeighbors(nodeId) {
        throw new Error('getIncomingNeighbors() must be implemented by subclass');
    }
    
    /**
     * Get outgoing neighbors of a node
     * @param {string} nodeId
     * @returns {Array} Array of neighbor node IDs
     */
    getOutgoingNeighbors(nodeId) {
        throw new Error('getOutgoingNeighbors() must be implemented by subclass');
    }
    
    /**
     * Get all neighbors of a node
     * @param {string} nodeId
     * @returns {Object} {incoming: [], outgoing: []}
     */
    getNeighbors(nodeId) {
        return {
            incoming: this.getIncomingNeighbors(nodeId),
            outgoing: this.getOutgoingNeighbors(nodeId)
        };
    }
    
    /**
     * Get edges connected to a node
     * @param {string} nodeId
     * @param {string} direction - 'in', 'out', 'both'
     * @returns {Array} Array of edge objects
     */
    getConnectedEdges(nodeId, direction = 'both') {
        throw new Error('getConnectedEdges() must be implemented by subclass');
    }
    
    // ============================================================================
    // EVENTS
    // ============================================================================
    
    /**
     * Register event handler
     * Events: 'nodeClick', 'nodeHover', 'nodeSelect', 'edgeClick', 
     *         'backgroundClick', 'viewportChange', 'selectionChange'
     * @param {string} eventType
     * @param {Function} handler
     * @returns {this}
     */
    on(eventType, handler) {
        if (!this.eventHandlers.has(eventType)) {
            this.eventHandlers.set(eventType, []);
        }
        this.eventHandlers.get(eventType).push(handler);
        return this;
    }
    
    /**
     * Unregister event handler
     * @param {string} eventType
     * @param {Function} handler
     * @returns {this}
     */
    off(eventType, handler) {
        const handlers = this.eventHandlers.get(eventType);
        if (handlers) {
            const idx = handlers.indexOf(handler);
            if (idx >= 0) handlers.splice(idx, 1);
        }
        return this;
    }
    
    /**
     * Emit event to handlers
     * @param {string} eventType
     * @param {*} data
     */
    emit(eventType, data) {
        const handlers = this.eventHandlers.get(eventType) || [];
        handlers.forEach(h => {
            try {
                h(data);
            } catch (err) {
                console.error(`[Renderer] Error in ${eventType} handler:`, err);
            }
        });
    }
    
    // ============================================================================
    // UTILITY
    // ============================================================================
    
    /**
     * Get node data by ID
     * @param {string} nodeId
     * @returns {Object|null}
     */
    getNodeData(nodeId) {
        return this.nodeDataMap.get(nodeId) || null;
    }
    
    /**
     * Get edge data by ID
     * @param {string} edgeId
     * @returns {Object|null}
     */
    getEdgeData(edgeId) {
        return this.edgeDataMap.get(edgeId) || null;
    }
    
    /**
     * Get all node IDs
     * @returns {Array}
     */
    getAllNodeIds() {
        return Array.from(this.nodeDataMap.keys());
    }
    
    /**
     * Get all edge IDs
     * @returns {Array}
     */
    getAllEdgeIds() {
        return Array.from(this.edgeDataMap.keys());
    }
    
    /**
     * Get renderer type
     * @returns {string} 'cosmos' or 'cytoscape'
     */
    getType() {
        throw new Error('getType() must be implemented by subclass');
    }
    
    /**
     * Get renderer statistics
     * @returns {Object}
     */
    getStats() {
        return {
            type: this.getType(),
            nodeCount: this.nodeDataMap.size,
            edgeCount: this.edgeDataMap.size,
            selectedNodes: this.getSelectedNodes().length,
            selectedEdges: this.getSelectedEdges().length
        };
    }
    
    /**
     * Export as PNG
     * @returns {Promise<Blob>}
     */
    exportPNG() {
        throw new Error('exportPNG() must be implemented by subclass');
    }
    
    // ============================================================================
    // BATCH OPERATIONS
    // ============================================================================
    
    /**
     * Start a batch operation (disables rendering until endBatch)
     */
    startBatch() {
        throw new Error('startBatch() must be implemented by subclass');
    }
    
    /**
     * End a batch operation and apply all changes
     */
    endBatch() {
        throw new Error('endBatch() must be implemented by subclass');
    }
    
    /**
     * Execute function within a batch
     * @param {Function} fn - Function to execute
     */
    batch(fn) {
        this.startBatch();
        try {
            fn();
        } finally {
            this.endBatch();
        }
    }
    
    // ============================================================================
    // LIFECYCLE
    // ============================================================================
    
    /**
     * Render/refresh the visualization
     */
    render() {
        throw new Error('render() must be implemented by subclass');
    }
    
    /**
     * Handle container resize
     */
    resize() {
        throw new Error('resize() must be implemented by subclass');
    }
    
    /**
     * Check if renderer has been disposed
     * @returns {boolean}
     */
    isDisposed() {
        return this._disposed;
    }
    
    /**
     * Cleanup and dispose
     */
    dispose() {
        this._disposed = true;
        this.nodeDataMap.clear();
        this.edgeDataMap.clear();
        this.eventHandlers.clear();
    }
}

// Make available globally
window.GraphRendererInterface = GraphRendererInterface;