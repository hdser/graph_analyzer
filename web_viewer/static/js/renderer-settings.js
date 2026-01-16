/**
 * Renderer Settings Module
 * 
 * Client-side renderer configuration loaded from backend.
 * All settings are configurable via environment variables on the server.
 * 
 * Includes cosmos.gl simulation parameter presets and extended configuration.
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
            defaultEdgeColor: '#f0f0f0',
            defaultEdgeOpacity: 0.3
        }
    },
    
    // Simulation presets for quick configuration
    simulationPresets: {
        default: {
            repulsion: 1.0,
            gravity: 0.25,
            linkDistance: 10,
            linkSpring: 1.0,
            friction: 0.85,
            decay: 5000,
            center: 0,
            cluster: 0.1
        },
        dense: {
            repulsion: 1.5,
            gravity: 0.05,
            linkDistance: 5,
            linkSpring: 0.5,
            friction: 0.9,
            decay: 3000,
            center: 0.1,
            cluster: 0.05
        },
        sparse: {
            repulsion: 0.3,
            gravity: 0.15,
            linkDistance: 30,
            linkSpring: 0.2,
            friction: 0.8,
            decay: 6000,
            center: 0.2,
            cluster: 0.1
        },
        clustered: {
            repulsion: 0.8,
            gravity: 0.2,
            linkDistance: 15,
            linkSpring: 0.4,
            friction: 0.85,
            decay: 4000,
            center: 0.3,
            cluster: 0.8
        },
        hierarchical: {
            repulsion: 0.4,
            gravity: 0.05,
            linkDistance: 50,
            linkSpring: 0.8,
            friction: 0.7,
            decay: 7000,
            center: 0,
            cluster: 0
        },
        radial: {
            repulsion: 0.6,
            gravity: 0.3,
            linkDistance: 20,
            linkSpring: 0.5,
            friction: 0.85,
            decay: 5000,
            center: 0.5,
            cluster: 0.2
        },
        fast: {
            repulsion: 0.3,
            gravity: 0.2,
            linkDistance: 10,
            linkSpring: 0.4,
            friction: 0.6,
            decay: 2000,
            center: 0.1,
            cluster: 0.1
        },
        quality: {
            repulsion: 0.8,
            gravity: 0.1,
            linkDistance: 15,
            linkSpring: 0.3,
            friction: 0.95,
            decay: 8000,
            center: 0,
            cluster: 0.1
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
    }
};

// Make available globally
window.RendererSettings = RendererSettings;