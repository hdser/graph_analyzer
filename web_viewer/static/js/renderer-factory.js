/**
 * Renderer Factory Module
 * 
 * Creates appropriate renderer based on WebGL capabilities and graph size.
 * Implements runtime renderer selection with graceful fallback.
 */

const RendererFactory = {
    // Current renderer instance
    _renderer: null,
    
    // Initialization state
    _initialized: false,
    
    /**
     * Initialize the factory
     */
    init() {
        if (this._initialized) return;
        
        // Detect WebGL capabilities
        const caps = WebGLDetector.detect();
        console.log('[RendererFactory] WebGL capabilities:', caps);
        
        // Check if cosmos.gl is available
        this._cosmosAvailable = this.checkCosmosLibrary();
        if (!this._cosmosAvailable) {
            console.warn('[RendererFactory] cosmos.gl library not loaded');
        }
        
        this._initialized = true;
    },
    
    /**
     * Check if cosmos.gl library is loaded
     * Local bundle exposes window.cosmosgl with { Graph, ... }
     */
    checkCosmosLibrary() {
        // Check if our local bundle loaded correctly
        if (typeof window.cosmosgl !== 'undefined' && typeof window.cosmosgl.Graph === 'function') {
            console.log('[RendererFactory] cosmos.gl loaded successfully');
            return true;
        }
        
        console.log('[RendererFactory] cosmos.gl not available:', {
            cosmosgl: typeof window.cosmosgl,
            Graph: typeof window.cosmosgl?.Graph
        });
        
        return false;
    },
    
    /**
     * Create a renderer for the given container
     * @param {HTMLElement} container - DOM container
     * @param {Object} options - Options including expectedNodeCount and rendererPreference
     * @returns {GraphRendererInterface} Renderer instance
     */
    create(container, options = {}) {
        this.init();
        
        // Dispose existing renderer if any
        if (this._renderer) {
            this._renderer.dispose();
            this._renderer = null;
        }
        
        const graphSize = options.expectedNodeCount || 0;
        const userPreference = options.rendererPreference || RendererSettings.getPreference();
        
        // Determine which renderer to use
        const selectedRenderer = this.selectRenderer(graphSize, userPreference);
        
        console.log(
            `[RendererFactory] Creating ${selectedRenderer} renderer ` +
            `(nodes: ${graphSize}, preference: ${userPreference})`
        );
        
        try {
            if (selectedRenderer === 'cosmos') {
                this._renderer = new CosmosAdapter(container, options);
            } else {
                this._renderer = new CytoscapeAdapter(container, options);
            }
        } catch (error) {
            console.error(`[RendererFactory] Failed to create ${selectedRenderer}:`, error);
            
            // Fallback to cytoscape if cosmos fails
            if (selectedRenderer === 'cosmos') {
                console.log('[RendererFactory] Falling back to Cytoscape');
                this._renderer = new CytoscapeAdapter(container, options);
            } else {
                throw error;
            }
        }
        
        return this._renderer;
    },
    
    /**
     * Select renderer based on capabilities, graph size, and preference
     * @param {number} graphSize - Number of nodes
     * @param {string} userPreference - 'auto', 'cosmos', or 'cytoscape'
     * @returns {string} 'cosmos' or 'cytoscape'
     */
    selectRenderer(graphSize, userPreference) {
        const caps = WebGLDetector.detect();
        const thresholds = RendererSettings.getThresholds();
        
        // User explicitly chose cytoscape
        if (userPreference === 'cytoscape') {
            return 'cytoscape';
        }
        
        // User explicitly chose cosmos
        if (userPreference === 'cosmos') {
            if (!this._cosmosAvailable) {
                console.warn('[RendererFactory] cosmos.gl not available - library not loaded');
                return 'cytoscape';
            }
            if (!caps.supported) {
                console.warn('[RendererFactory] cosmos.gl not supported:', caps.reason);
                return 'cytoscape';
            }
            return 'cosmos';
        }
        
        // Auto selection
        
        // cosmos.gl not available
        if (!this._cosmosAvailable || !caps.supported) {
            if (graphSize >= thresholds.cosmosPreferredNodes) {
                console.warn(
                    `[RendererFactory] Large graph (${graphSize} nodes) but cosmos.gl unavailable. ` +
                    `Cytoscape.js may be slow.`
                );
            }
            return 'cytoscape';
        }
        
        // Small graphs: Cytoscape (more features, better for small graphs)
        if (graphSize < thresholds.cosmosMinNodes) {
            return 'cytoscape';
        }
        
        // Medium to large graphs: cosmos.gl
        if (graphSize >= thresholds.cosmosMinNodes) {
            // Warn if exceeding estimated capacity
            if (graphSize > caps.details.estimatedMaxNodes) {
                console.warn(
                    `[RendererFactory] Graph size (${graphSize}) may exceed device capacity ` +
                    `(~${caps.details.estimatedMaxNodes})`
                );
            }
            return 'cosmos';
        }
        
        return 'cytoscape';
    },
    
    /**
     * Get the current renderer instance
     * @returns {GraphRendererInterface|null}
     */
    getRenderer() {
        return this._renderer;
    },
    
    /**
     * Get the current renderer type
     * @returns {string|null} 'cosmos' or 'cytoscape'
     */
    getRendererType() {
        return this._renderer?.getType() || null;
    },
    
    /**
     * Check if cosmos.gl would be available for the given graph size
     * @param {number} graphSize - Number of nodes
     * @returns {boolean}
     */
    wouldUseCosmos(graphSize) {
        const selected = this.selectRenderer(graphSize, 'auto');
        return selected === 'cosmos';
    },
    
    /**
     * Get capability summary for UI display
     * @returns {Object}
     */
    getCapabilitySummary() {
        const caps = WebGLDetector.getSummary();
        const cosmosLibraryLoaded = this._cosmosAvailable;
        const cosmosAvailable = cosmosLibraryLoaded && caps.cosmosAvailable;
        
        // If cosmos isn't available (library not loaded or WebGL insufficient),
        // show Cytoscape.js practical limits instead
        let maxNodes = caps.maxNodes;
        let reason = caps.reason;
        
        if (!cosmosAvailable) {
            // Cytoscape.js practical limits with performance mode
            maxNodes = 30000; // ~30k nodes is practical limit with WebGL canvas
            if (!cosmosLibraryLoaded) {
                reason = 'cosmos.gl library not loaded - using Cytoscape.js';
            }
        }
        
        return {
            ...caps,
            cosmosAvailable,
            cosmosLibraryLoaded,
            maxNodes,
            reason
        };
    },
    
    /**
     * Check if cosmos.gl is available (both library and WebGL)
     * @returns {boolean}
     */
    isCosmosAvailable() {
        this.init();
        return this._cosmosAvailable && WebGLDetector.isCosmosAvailable();
    },
    
    /**
     * Get recommended renderer for a graph size
     * @param {number} graphSize
     * @returns {Object} {renderer: string, reason: string}
     */
    getRecommendation(graphSize) {
        const selected = this.selectRenderer(graphSize, 'auto');
        const thresholds = RendererSettings.getThresholds();
        
        let reason;
        if (selected === 'cytoscape') {
            if (graphSize < thresholds.cosmosMinNodes) {
                reason = `Small graph (< ${thresholds.cosmosMinNodes} nodes) - Cytoscape.js offers more features`;
            } else if (!this._cosmosAvailable) {
                reason = 'cosmos.gl library not loaded';
            } else if (!WebGLDetector.isCosmosAvailable()) {
                reason = WebGLDetector.detect().reason;
            }
        } else {
            reason = `Large graph (>= ${thresholds.cosmosMinNodes} nodes) - cosmos.gl provides better performance`;
        }
        
        return { renderer: selected, reason };
    },
    
    /**
     * Dispose the current renderer
     */
    dispose() {
        if (this._renderer) {
            this._renderer.dispose();
            this._renderer = null;
        }
    },
    
    /**
     * Force recreation of renderer with same options
     * @param {HTMLElement} container
     * @param {Object} options
     */
    recreate(container, options = {}) {
        const currentType = this.getRendererType();
        this.dispose();
        
        // Keep the same renderer type if specified
        if (currentType && !options.rendererPreference) {
            options.rendererPreference = currentType;
        }
        
        return this.create(container, options);
    }
};

// Make available globally
window.RendererFactory = RendererFactory;