/**
 * Renderer Settings Module
 * 
 * Client-side renderer configuration loaded from backend.
 * All settings are configurable via environment variables on the server.
 * 
 * Includes cosmos.gl simulation parameter presets, visual configuration,
 * and extended configuration options with descriptions for UI display.
 */

const RendererSettings = {
    // Default settings (used before config is loaded from server)
    defaults: {
        preference: 'auto',
        thresholds: {
            cosmosMinNodes: 5000,
            cosmosPreferredNodes: 50000
        },
        cosmos: {
            spaceSize: 8192,
            pointSize: 6,
            linkWidth: 1,
            backgroundColor: '#1a1a1a',
            curvedLinks: true,
            enableDrag: true,
            enableRightClickRepulsion: false,
            simulation: {
                // Core parameters - cosmos.gl defaults from documentation
                friction: 0.85,
                gravity: 0.25,
                repulsion: 1.0,
                linkDistance: 10,
                linkSpring: 1.0,
                // Extended parameters
                decay: 5000,
                center: 0,
                repulsionTheta: 1.15,
                cluster: 0.1,
                repulsionFromMouse: 2.0
            }
        },
        style: {
            nodeSizeMin: 5,
            nodeSizeMax: 30,
            defaultGradient: 'spectral',
            selectionColor: '#FF0000',
            highlightColor: '#FFA500',
            defaultNodeColor: '#999999',
            defaultEdgeColor: '#ffffff',
            defaultEdgeOpacity: 0.5
        }
    },
    
    // Simulation presets for quick configuration with descriptions
    simulationPresets: {
        default: {
            name: 'Default',
            description: 'Balanced settings suitable for most graphs',
            icon: 'default',
            repulsion: 1.0,
            gravity: 0.25,
            linkDistance: 10,
            linkSpring: 1.0,
            friction: 0.85,
            decay: 5000,
            center: 0,
            repulsionTheta: 1.15,
            cluster: 0.1
        },
        dense: {
            name: 'Dense Networks',
            description: 'For highly connected graphs - spreads nodes apart more',
            icon: 'dense',
            repulsion: 1.5,
            gravity: 0.05,
            linkDistance: 5,
            linkSpring: 0.5,
            friction: 0.9,
            decay: 3000,
            center: 0.1,
            repulsionTheta: 1.0,
            cluster: 0.05
        },
        sparse: {
            name: 'Sparse Networks',
            description: 'For loosely connected graphs - brings nodes closer',
            icon: 'sparse',
            repulsion: 0.3,
            gravity: 0.15,
            linkDistance: 30,
            linkSpring: 0.2,
            friction: 0.8,
            decay: 6000,
            center: 0.2,
            repulsionTheta: 1.3,
            cluster: 0.1
        },
        clustered: {
            name: 'Clustered (Communities)',
            description: 'Emphasizes community structure and group separation',
            icon: 'clustered',
            repulsion: 0.8,
            gravity: 0.2,
            linkDistance: 15,
            linkSpring: 0.4,
            friction: 0.85,
            decay: 4000,
            center: 0.3,
            repulsionTheta: 1.0,
            cluster: 0.8
        },
        hierarchical: {
            name: 'Hierarchical',
            description: 'For tree-like structures with clear levels',
            icon: 'hierarchical',
            repulsion: 0.4,
            gravity: 0.05,
            linkDistance: 50,
            linkSpring: 0.8,
            friction: 0.7,
            decay: 7000,
            center: 0,
            repulsionTheta: 1.5,
            cluster: 0
        },
        radial: {
            name: 'Radial',
            description: 'Pulls nodes toward center with concentric arrangement',
            icon: 'radial',
            repulsion: 0.6,
            gravity: 0.3,
            linkDistance: 20,
            linkSpring: 0.5,
            friction: 0.85,
            decay: 5000,
            center: 0.5,
            repulsionTheta: 1.15,
            cluster: 0.2
        },
        fast: {
            name: 'Fast Layout',
            description: 'Quick convergence - lower quality but faster',
            icon: 'fast',
            repulsion: 0.3,
            gravity: 0.2,
            linkDistance: 10,
            linkSpring: 0.4,
            friction: 0.6,
            decay: 2000,
            center: 0.1,
            repulsionTheta: 1.5,
            cluster: 0.1
        },
        quality: {
            name: 'High Quality',
            description: 'Slow convergence - better layout quality',
            icon: 'quality',
            repulsion: 0.8,
            gravity: 0.1,
            linkDistance: 15,
            linkSpring: 0.3,
            friction: 0.95,
            decay: 10000,
            center: 0,
            repulsionTheta: 0.8,
            cluster: 0.1
        },
        organic: {
            name: 'Organic',
            description: 'Natural-looking layout with smooth curves',
            icon: 'organic',
            repulsion: 0.5,
            gravity: 0.1,
            linkDistance: 25,
            linkSpring: 0.3,
            friction: 0.9,
            decay: 6000,
            center: 0.05,
            repulsionTheta: 1.0,
            cluster: 0.3
        },
        compact: {
            name: 'Compact',
            description: 'Minimizes space usage - tight layout',
            icon: 'compact',
            repulsion: 0.2,
            gravity: 0.4,
            linkDistance: 5,
            linkSpring: 0.8,
            friction: 0.85,
            decay: 4000,
            center: 0.3,
            repulsionTheta: 1.2,
            cluster: 0.2
        }
    },
    
    // Parameter definitions with ranges and descriptions for UI
    parameterDefinitions: {
        // Core Force Parameters
        repulsion: {
            name: 'Repulsion',
            description: 'How strongly nodes push each other apart',
            category: 'forces',
            min: 0,
            max: 2,
            step: 0.05,
            default: 1.0,
            unit: ''
        },
        gravity: {
            name: 'Gravity',
            description: 'Pull strength toward the center',
            category: 'forces',
            min: 0,
            max: 1,
            step: 0.05,
            default: 0.25,
            unit: ''
        },
        center: {
            name: 'Center Force',
            description: 'Additional centering force coefficient',
            category: 'forces',
            min: 0,
            max: 1,
            step: 0.05,
            default: 0,
            unit: ''
        },
        repulsionTheta: {
            name: 'Repulsion Theta',
            description: 'Barnes-Hut approximation (lower = more accurate, slower)',
            category: 'forces',
            min: 0.3,
            max: 2,
            step: 0.05,
            default: 1.15,
            unit: ''
        },
        cluster: {
            name: 'Cluster Coefficient',
            description: 'How strongly connected nodes cluster together',
            category: 'forces',
            min: 0,
            max: 1,
            step: 0.05,
            default: 0.1,
            unit: ''
        },
        
        // Link Parameters
        linkDistance: {
            name: 'Link Distance',
            description: 'Minimum distance between connected nodes',
            category: 'links',
            min: 1,
            max: 100,
            step: 1,
            default: 10,
            unit: 'px'
        },
        linkSpring: {
            name: 'Link Spring',
            description: 'Spring force pulling connected nodes together',
            category: 'links',
            min: 0,
            max: 2,
            step: 0.1,
            default: 1.0,
            unit: ''
        },
        
        // Behavior Parameters
        friction: {
            name: 'Friction',
            description: 'Movement damping (0 = stops quickly, 1 = no friction)',
            category: 'behavior',
            min: 0,
            max: 1,
            step: 0.05,
            default: 0.85,
            unit: ''
        },
        decay: {
            name: 'Decay',
            description: 'How quickly the simulation cools down (lower = slower)',
            category: 'behavior',
            min: 100,
            max: 15000,
            step: 100,
            default: 5000,
            unit: ''
        },
        repulsionFromMouse: {
            name: 'Mouse Repulsion',
            description: 'Repulsion force when right-clicking',
            category: 'interaction',
            min: 0,
            max: 5,
            step: 0.1,
            default: 2.0,
            unit: ''
        }
    },
    
    // Visual parameter definitions
    visualParameterDefinitions: {
        pointSizeScale: {
            name: 'Node Size Scale',
            description: 'Scale factor for all node sizes',
            min: 0.1,
            max: 3,
            step: 0.1,
            default: 1.0
        },
        linkWidthScale: {
            name: 'Link Width Scale',
            description: 'Scale factor for all link widths',
            min: 0.1,
            max: 3,
            step: 0.1,
            default: 1.0
        },
        linkOpacity: {
            name: 'Link Opacity',
            description: 'Overall opacity of all links',
            min: 0,
            max: 1,
            step: 0.05,
            default: 1.0
        },
        pointOpacity: {
            name: 'Node Opacity',
            description: 'Overall opacity of all nodes',
            min: 0,
            max: 1,
            step: 0.05,
            default: 1.0
        },
        curvedLinkWeight: {
            name: 'Curve Weight',
            description: 'How curved the links are',
            min: 0,
            max: 1,
            step: 0.1,
            default: 0.8
        }
    },
    
    // Current configuration (updated from server)
    config: null,
    
    // Whether config has been loaded
    loaded: false,
    
    /**
     * Initialize settings from server configuration
     * @param {Object} serverConfig - Configuration from server API
     */
    init(serverConfig) {
        if (serverConfig && serverConfig.renderer) {
            this.config = this.mergeConfig(this.defaults, serverConfig.renderer);
            console.log('[RendererSettings] Loaded from server:', this.config);
        } else {
            this.config = this.deepClone(this.defaults);
            console.log('[RendererSettings] Using defaults');
        }
        this.loaded = true;
    },
    
    /**
     * Deep merge two configuration objects
     */
    mergeConfig(defaults, override) {
        const result = this.deepClone(defaults);
        
        for (const key of Object.keys(override)) {
            if (override[key] !== null && typeof override[key] === 'object' && !Array.isArray(override[key])) {
                result[key] = this.mergeConfig(result[key] || {}, override[key]);
            } else {
                result[key] = override[key];
            }
        }
        
        return result;
    },
    
    /**
     * Deep clone an object
     */
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },
    
    /**
     * Get the current configuration (or defaults if not loaded)
     */
    get() {
        return this.config || this.defaults;
    },
    
    /**
     * Get renderer preference
     * @returns {string} 'auto', 'cosmos', or 'cytoscape'
     */
    getPreference() {
        return this.get().preference;
    },
    
    /**
     * Get node count thresholds for renderer selection
     */
    getThresholds() {
        return this.get().thresholds;
    },
    
    /**
     * Get cosmos.gl specific configuration
     */
    getCosmosConfig() {
        return this.get().cosmos;
    },
    
    /**
     * Get style configuration
     */
    getStyleConfig() {
        return this.get().style;
    },
    
    /**
     * Convert hex color to RGBA array for cosmos.gl (0-1 range)
     * @param {string} hex - Hex color string
     * @param {number} alpha - Alpha value (0-1)
     * @returns {Array} [r, g, b, a]
     */
    hexToRgba(hex, alpha = 1.0) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        if (!result) return [0.5, 0.5, 0.5, alpha];
        
        return [
            parseInt(result[1], 16) / 255,
            parseInt(result[2], 16) / 255,
            parseInt(result[3], 16) / 255,
            alpha
        ];
    },
    
    /**
     * Convert RGBA array (0-1 range) to hex color
     * @param {Array} rgba - [r, g, b, a] array
     * @returns {string} Hex color string
     */
    rgbaToHex(rgba) {
        const r = Math.round(rgba[0] * 255);
        const g = Math.round(rgba[1] * 255);
        const b = Math.round(rgba[2] * 255);
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    },
    
    /**
     * Get selection color as RGBA for cosmos.gl
     */
    getSelectionColorRgba() {
        return this.hexToRgba(this.getStyleConfig().selectionColor);
    },
    
    /**
     * Get highlight color as RGBA for cosmos.gl
     */
    getHighlightColorRgba() {
        return this.hexToRgba(this.getStyleConfig().highlightColor);
    },
    
    /**
     * Get default node color as RGBA for cosmos.gl
     */
    getDefaultNodeColorRgba() {
        return this.hexToRgba(this.getStyleConfig().defaultNodeColor);
    },
    
    /**
     * Get default edge color as RGBA for cosmos.gl
     */
    getDefaultEdgeColorRgba() {
        return this.hexToRgba(
            this.getStyleConfig().defaultEdgeColor,
            this.getStyleConfig().defaultEdgeOpacity
        );
    },
    
    /**
     * Get simulation presets
     * @returns {Object} Named presets with simulation parameters
     */
    getSimulationPresets() {
        return this.simulationPresets;
    },
    
    /**
     * Get a specific simulation preset
     * @param {string} name - Preset name
     * @returns {Object|null} Preset parameters or null if not found
     */
    getSimulationPreset(name) {
        return this.simulationPresets[name] || null;
    },
    
    /**
     * Get list of available preset names
     * @returns {string[]}
     */
    getPresetNames() {
        return Object.keys(this.simulationPresets);
    },
    
    /**
     * Get preset info for UI display
     * @returns {Array} Array of {id, name, description, icon}
     */
    getPresetInfo() {
        return Object.entries(this.simulationPresets).map(([id, preset]) => ({
            id,
            name: preset.name,
            description: preset.description,
            icon: preset.icon
        }));
    },
    
    /**
     * Get parameter definition
     * @param {string} name - Parameter name
     * @returns {Object|null} Parameter definition
     */
    getParameterDefinition(name) {
        return this.parameterDefinitions[name] || null;
    },
    
    /**
     * Get all parameter definitions
     * @returns {Object}
     */
    getParameterDefinitions() {
        return this.parameterDefinitions;
    },
    
    /**
     * Get parameters grouped by category
     * @returns {Object} { category: [paramNames] }
     */
    getParametersByCategory() {
        const categories = {};
        
        for (const [name, def] of Object.entries(this.parameterDefinitions)) {
            const cat = def.category || 'other';
            if (!categories[cat]) {
                categories[cat] = [];
            }
            categories[cat].push(name);
        }
        
        return categories;
    },
    
    /**
     * Get visual parameter definition
     * @param {string} name - Parameter name
     * @returns {Object|null}
     */
    getVisualParameterDefinition(name) {
        return this.visualParameterDefinitions[name] || null;
    },
    
    /**
     * Get all visual parameter definitions
     * @returns {Object}
     */
    getVisualParameterDefinitions() {
        return this.visualParameterDefinitions;
    },
    
    /**
     * Validate a parameter value against its definition
     * @param {string} name - Parameter name
     * @param {number} value - Value to validate
     * @returns {number} Clamped value within valid range
     */
    validateParameter(name, value) {
        const def = this.parameterDefinitions[name] || this.visualParameterDefinitions[name];
        if (!def) return value;
        
        return Math.max(def.min, Math.min(def.max, value));
    }
};

// Make available globally
window.RendererSettings = RendererSettings;