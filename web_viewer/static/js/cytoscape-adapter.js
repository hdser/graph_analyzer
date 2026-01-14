/**
 * Cytoscape Adapter
 * 
 * Wraps Cytoscape.js to implement GraphRendererInterface.
 * Uses existing CytoscapeManager for initialization and styling.
 */

class CytoscapeAdapter extends GraphRendererInterface {
    /**
     * @param {HTMLElement} container - DOM container
     * @param {Object} options - Renderer options
     */
    constructor(container, options = {}) {
        super(container, options);
        
        this.cy = null;
        this._performanceMode = true;
        this._batchCount = 0;
        
        this.initialize();
    }
    
    /**
     * Initialize Cytoscape instance using CytoscapeManager
     * This ensures all event handlers (including InfoPanel) are properly set up
     */
    initialize() {
        // Use CytoscapeManager to create the Cytoscape instance
        // This sets up all the event handlers including InfoPanel click handlers
        this.cy = CytoscapeManager.initializeCytoscape(this.container);
    }
    
    /**
     * Get performance-optimized style
     */
    getPerformanceStyle() {
        const styleConfig = RendererSettings.getStyleConfig();
        
        return [
            {
                selector: 'node',
                style: {
                    'background-color': styleConfig.defaultNodeColor,
                    'width': 13,
                    'height': 13,
                    'label': '',
                    'border-width': 0
                }
            },
            {
                selector: 'edge',
                style: {
                    'line-color': styleConfig.defaultEdgeColor,
                    'width': 1,
                    'opacity': styleConfig.defaultEdgeOpacity,
                    'curve-style': 'straight',
                    'target-arrow-shape': 'none'
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'background-color': styleConfig.selectionColor,
                    'border-width': 3,
                    'border-color': styleConfig.selectionColor,
                    'z-index': 999
                }
            },
            {
                selector: 'edge:selected',
                style: {
                    'line-color': styleConfig.selectionColor,
                    'width': 2,
                    'opacity': 1,
                    'z-index': 999
                }
            },
            {
                selector: '.highlighted',
                style: {
                    'background-color': styleConfig.highlightColor,
                    'line-color': styleConfig.highlightColor,
                    'opacity': 0.8,
                    'z-index': 998
                }
            },
            {
                selector: '.searched',
                style: {
                    'background-color': '#00FF00',
                    'border-width': 2,
                    'border-color': '#00FF00',
                    'z-index': 997
                }
            },
            {
                selector: '.anomaly',
                style: {
                    'background-color': '#FF4444',
                    'border-width': 2,
                    'border-color': '#FF0000',
                    'z-index': 996
                }
            },
            {
                selector: '.new-node',
                style: {
                    'background-color': '#00FFFF',
                    'border-width': 2,
                    'border-color': '#00CCCC'
                }
            },
            {
                selector: '.hidden',
                style: {
                    'display': 'none'
                }
            }
        ];
    }
    
    /**
     * Setup Cytoscape event handlers
     */
    setupEvents() {
        // Debounce helper
        const debounce = (func, wait) => {
            let timeout;
            return (...args) => {
                clearTimeout(timeout);
                timeout = setTimeout(() => func.apply(this, args), wait);
            };
        };
        
        // Node tap
        this.cy.on('tap', 'node', (e) => {
            const node = e.target;
            this.emit('nodeClick', {
                id: node.id(),
                data: node.data(),
                position: node.position()
            });
        });
        
        // Edge tap
        this.cy.on('tap', 'edge', (e) => {
            const edge = e.target;
            this.emit('edgeClick', {
                id: edge.id(),
                source: edge.source().id(),
                target: edge.target().id(),
                data: edge.data()
            });
        });
        
        // Background tap
        this.cy.on('tap', (e) => {
            if (e.target === this.cy) {
                this.emit('backgroundClick', {});
            }
        });
        
        // Node hover
        this.cy.on('mouseover', 'node', (e) => {
            const node = e.target;
            this.emit('nodeHover', {
                id: node.id(),
                data: node.data()
            });
        });
        
        this.cy.on('mouseout', 'node', () => {
            this.emit('nodeHover', { id: null, data: null });
        });
        
        // Selection changes (debounced)
        this.cy.on('select unselect', debounce(() => {
            this.emit('selectionChange', {
                nodes: this.getSelectedNodes(),
                edges: this.getSelectedEdges()
            });
        }, 100));
        
        // Viewport changes
        this.cy.on('pan zoom', debounce(() => {
            this.emit('viewportChange', this.getViewport());
        }, 50));
    }
    
    // ============================================================================
    // DATA METHODS
    // ============================================================================
    
    setData(nodes, edges) {
        // Clear existing data
        this.cy.elements().remove();
        this.nodeDataMap.clear();
        this.edgeDataMap.clear();
        
        // Convert to Cytoscape format
        const elements = [];
        
        nodes.forEach(node => {
            elements.push({
                group: 'nodes',
                data: { id: node.id, ...node },
                position: { x: node.x || 0, y: node.y || 0 }
            });
            this.nodeDataMap.set(node.id, node);
        });
        
        edges.forEach(edge => {
            const edgeId = edge.id || `${edge.source}-${edge.target}`;
            elements.push({
                group: 'edges',
                data: { id: edgeId, source: edge.source, target: edge.target, ...edge }
            });
            this.edgeDataMap.set(edgeId, edge);
        });
        
        this.cy.add(elements);
        this.cy.fit();
    }
    
    updatePositions(positions) {
        this.cy.batch(() => {
            const posMap = positions instanceof Map ? positions : new Map(Object.entries(positions));
            posMap.forEach((pos, id) => {
                const node = this.cy.getElementById(id);
                if (node.length) {
                    node.position(pos);
                }
            });
        });
    }
    
    addNodes(nodes) {
        const elements = nodes.map(node => ({
            group: 'nodes',
            data: { id: node.id, ...node },
            position: { x: node.x || 0, y: node.y || 0 }
        }));
        
        this.cy.batch(() => {
            this.cy.add(elements);
        });
        
        nodes.forEach(node => {
            this.nodeDataMap.set(node.id, node);
        });
    }
    
    addEdges(edges) {
        const elements = edges.map(edge => {
            const edgeId = edge.id || `${edge.source}-${edge.target}`;
            return {
                group: 'edges',
                data: { id: edgeId, source: edge.source, target: edge.target, ...edge }
            };
        });
        
        this.cy.batch(() => {
            this.cy.add(elements);
        });
        
        edges.forEach(edge => {
            const edgeId = edge.id || `${edge.source}-${edge.target}`;
            this.edgeDataMap.set(edgeId, edge);
        });
    }
    
    removeElements(nodeIds = [], edgeIds = []) {
        this.cy.batch(() => {
            nodeIds.forEach(id => {
                this.cy.getElementById(id).remove();
                this.nodeDataMap.delete(id);
            });
            edgeIds.forEach(id => {
                this.cy.getElementById(id).remove();
                this.edgeDataMap.delete(id);
            });
        });
    }
    
    clear() {
        this.cy.elements().remove();
        this.nodeDataMap.clear();
        this.edgeDataMap.clear();
    }
    
    // ============================================================================
    // VISUAL STYLING
    // ============================================================================
    
    applyNodeColors(metricName, colorScale) {
        const gradient = ColorGradients.get(colorScale.gradient || 'spectral');
        
        // Calculate range if not provided
        let min = colorScale.min;
        let max = colorScale.max;
        
        if (min === undefined || max === undefined) {
            const values = this.cy.nodes()
                .map(n => n.data(metricName))
                .filter(v => typeof v === 'number' && !isNaN(v));
            
            if (values.length > 0) {
                min = min !== undefined ? min : Math.min(...values);
                max = max !== undefined ? max : Math.max(...values);
            }
        }
        
        this.cy.batch(() => {
            this.cy.nodes().forEach(node => {
                const val = node.data(metricName);
                if (typeof val === 'number' && !isNaN(val) && max > min) {
                    const norm = (val - min) / (max - min);
                    const color = ColorGradients.interpolate(gradient, norm);
                    node.style('background-color', color);
                }
            });
        });
    }
    
    applyNodeSizes(metricName, sizeScale) {
        const { min: sizeMin, max: sizeMax } = sizeScale;
        
        // Calculate value range
        const values = this.cy.nodes()
            .map(n => n.data(metricName))
            .filter(v => typeof v === 'number' && !isNaN(v));
        
        if (values.length === 0) return;
        
        const valueMin = Math.min(...values);
        const valueMax = Math.max(...values);
        
        this.cy.batch(() => {
            this.cy.nodes().forEach(node => {
                const val = node.data(metricName);
                if (typeof val === 'number' && !isNaN(val) && valueMax > valueMin) {
                    const norm = (val - valueMin) / (valueMax - valueMin);
                    const size = sizeMin + norm * (sizeMax - sizeMin);
                    node.style({ width: size, height: size });
                }
            });
        });
    }
    
    setEdgeStyle(style) {
        this.cy.batch(() => {
            this.cy.edges().forEach(edge => {
                const styleObj = {};
                if (style.color) styleObj['line-color'] = style.color;
                if (style.opacity !== undefined) styleObj['opacity'] = style.opacity;
                if (style.width !== undefined) styleObj['width'] = style.width;
                edge.style(styleObj);
            });
        });
    }
    
    resetStyle() {
        const styleConfig = RendererSettings.getStyleConfig();
        
        this.cy.batch(() => {
            this.cy.nodes().forEach(node => {
                node.style({
                    'background-color': styleConfig.defaultNodeColor,
                    'width': 13,
                    'height': 13
                });
            });
            this.cy.edges().forEach(edge => {
                edge.style({
                    'line-color': styleConfig.defaultEdgeColor,
                    'opacity': styleConfig.defaultEdgeOpacity,
                    'width': 1
                });
            });
        });
    }
    
    setPerformanceMode(enabled) {
        this._performanceMode = enabled;
        if (enabled) {
            this.resetStyle();
        }
    }
    
    // ============================================================================
    // SELECTION & HIGHLIGHTING
    // ============================================================================
    
    selectNodes(nodeIds, additive = false) {
        if (!additive) {
            this.cy.elements().unselect();
        }
        nodeIds.forEach(id => {
            this.cy.getElementById(id).select();
        });
    }
    
    selectEdges(edgeIds, additive = false) {
        if (!additive) {
            this.cy.elements().unselect();
        }
        edgeIds.forEach(id => {
            this.cy.getElementById(id).select();
        });
    }
    
    getSelectedNodes() {
        return this.cy.nodes(':selected').map(n => n.id());
    }
    
    getSelectedEdges() {
        return this.cy.edges(':selected').map(e => e.id());
    }
    
    clearSelection() {
        this.cy.elements().unselect();
    }
    
    highlightNodes(nodeIds, className = 'highlighted') {
        this.cy.batch(() => {
            nodeIds.forEach(id => {
                this.cy.getElementById(id).addClass(className);
            });
        });
    }
    
    highlightNeighbors(nodeId, direction = 'both') {
        this.cy.elements().removeClass('highlighted');
        const node = this.cy.getElementById(nodeId);
        
        if (!node.length) return;
        
        let neighbors;
        switch (direction) {
            case 'in':
                neighbors = node.incomers();
                break;
            case 'out':
                neighbors = node.outgoers();
                break;
            default:
                neighbors = node.neighborhood();
        }
        
        neighbors.addClass('highlighted');
    }
    
    clearHighlights() {
        this.cy.elements().removeClass('highlighted');
    }
    
    addClass(elementIds, className, type = 'nodes') {
        this.cy.batch(() => {
            elementIds.forEach(id => {
                this.cy.getElementById(id).addClass(className);
            });
        });
    }
    
    removeClass(elementIds, className, type = 'nodes') {
        this.cy.batch(() => {
            elementIds.forEach(id => {
                this.cy.getElementById(id).removeClass(className);
            });
        });
    }
    
    // ============================================================================
    // VIEWPORT CONTROL
    // ============================================================================
    
    fitView(nodeIds = null, padding = 50) {
        if (nodeIds && nodeIds.length) {
            const collection = this.cy.collection(
                nodeIds.map(id => this.cy.getElementById(id))
            );
            this.cy.fit(collection, padding);
        } else {
            this.cy.fit(padding);
        }
    }
    
    center() {
        this.cy.center();
    }
    
    zoomToNode(nodeId, zoomLevel = 3, duration = 500) {
        const node = this.cy.getElementById(nodeId);
        if (node.length) {
            this.cy.animate({
                center: { eles: node },
                zoom: zoomLevel,
                duration
            });
        }
    }
    
    getViewport() {
        return {
            zoom: this.cy.zoom(),
            pan: this.cy.pan()
        };
    }
    
    setViewport(viewport) {
        this.cy.viewport(viewport);
    }
    
    setZoom(level) {
        this.cy.zoom(level);
    }
    
    // ============================================================================
    // GRAPH QUERIES
    // ============================================================================
    
    getIncomingNeighbors(nodeId) {
        const node = this.cy.getElementById(nodeId);
        if (!node.length) return [];
        return node.incomers('node').map(n => n.id());
    }
    
    getOutgoingNeighbors(nodeId) {
        const node = this.cy.getElementById(nodeId);
        if (!node.length) return [];
        return node.outgoers('node').map(n => n.id());
    }
    
    getConnectedEdges(nodeId, direction = 'both') {
        const node = this.cy.getElementById(nodeId);
        if (!node.length) return [];
        
        let edges;
        switch (direction) {
            case 'in':
                edges = node.incomers('edge');
                break;
            case 'out':
                edges = node.outgoers('edge');
                break;
            default:
                edges = node.connectedEdges();
        }
        
        return edges.map(e => ({
            id: e.id(),
            source: e.source().id(),
            target: e.target().id(),
            data: e.data()
        }));
    }
    
    // ============================================================================
    // UTILITY
    // ============================================================================
    
    getType() {
        return 'cytoscape';
    }
    
    getStats() {
        return {
            type: 'cytoscape',
            nodeCount: this.cy.nodes().length,
            edgeCount: this.cy.edges().length,
            selectedNodes: this.cy.nodes(':selected').length,
            selectedEdges: this.cy.edges(':selected').length
        };
    }
    
    async exportPNG() {
        return this.cy.png({ output: 'blob', full: true });
    }
    
    // ============================================================================
    // BATCH OPERATIONS
    // ============================================================================
    
    startBatch() {
        if (this._batchCount === 0) {
            this.cy.startBatch();
        }
        this._batchCount++;
    }
    
    endBatch() {
        this._batchCount--;
        if (this._batchCount === 0) {
            this.cy.endBatch();
        }
    }
    
    // ============================================================================
    // LIFECYCLE
    // ============================================================================
    
    render() {
        this.cy.resize();
        this.cy.fit();
    }
    
    resize() {
        this.cy.resize();
    }
    
    dispose() {
        super.dispose();
        if (this.cy) {
            this.cy.destroy();
            this.cy = null;
        }
    }
    
    // ============================================================================
    // CYTOSCAPE-SPECIFIC METHODS
    // ============================================================================
    
    /**
     * Get the underlying Cytoscape instance
     * @returns {Object} Cytoscape instance
     */
    getCy() {
        return this.cy;
    }
    
    /**
     * Get a node element by ID
     * @param {string} nodeId
     * @returns {Object} Cytoscape node element
     */
    getNodeElement(nodeId) {
        return this.cy.getElementById(nodeId);
    }
    
    /**
     * Run a Cytoscape layout
     * @param {Object} options - Layout options
     */
    runLayout(options) {
        return this.cy.layout(options).run();
    }
    
    /**
     * Get nodes matching a selector
     * @param {string} selector - Cytoscape selector
     * @returns {Object} Cytoscape collection
     */
    nodes(selector = '') {
        return this.cy.nodes(selector);
    }
    
    /**
     * Get edges matching a selector
     * @param {string} selector - Cytoscape selector
     * @returns {Object} Cytoscape collection
     */
    edges(selector = '') {
        return this.cy.edges(selector);
    }
}

// Make available globally
window.CytoscapeAdapter = CytoscapeAdapter;