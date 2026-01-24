/**
 * Cytoscape Manager Module
 * Cytoscape initialization, styling, and event handling
 */

/**
 * Debounce helper - prevents function from firing too frequently
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Color gradient definitions
 */
const COLOR_GRADIENTS = {
    viridis: [
        { stop: 0, color: '#440154' },
        { stop: 0.25, color: '#3b528b' },
        { stop: 0.5, color: '#21918c' },
        { stop: 0.75, color: '#5ec962' },
        { stop: 1, color: '#fde725' }
    ],
    plasma: [
        { stop: 0, color: '#0d0887' },
        { stop: 0.25, color: '#7e03a8' },
        { stop: 0.5, color: '#cc4778' },
        { stop: 0.75, color: '#f89540' },
        { stop: 1, color: '#f0f921' }
    ],
    inferno: [
        { stop: 0, color: '#000004' },
        { stop: 0.25, color: '#57106e' },
        { stop: 0.5, color: '#bc3754' },
        { stop: 0.75, color: '#f98e09' },
        { stop: 1, color: '#fcffa4' }
    ],
    spectral: [
        { stop: 0, color: '#5e4fa2' },
        { stop: 0.25, color: '#3288bd' },
        { stop: 0.5, color: '#66c2a5' },
        { stop: 0.75, color: '#fee08b' },
        { stop: 1, color: '#d53e4f' }
    ],
    coolwarm: [
        { stop: 0, color: '#3b4cc0' },
        { stop: 0.25, color: '#7b9ff9' },
        { stop: 0.5, color: '#f7f7f7' },
        { stop: 0.75, color: '#f4a582' },
        { stop: 1, color: '#b40426' }
    ],
    greens: [
        { stop: 0, color: '#f7fcf5' },
        { stop: 0.25, color: '#c7e9c0' },
        { stop: 0.5, color: '#74c476' },
        { stop: 0.75, color: '#238b45' },
        { stop: 1, color: '#00441b' }
    ],
    blues: [
        { stop: 0, color: '#f7fbff' },
        { stop: 0.25, color: '#c6dbef' },
        { stop: 0.5, color: '#6baed6' },
        { stop: 0.75, color: '#2171b5' },
        { stop: 1, color: '#084594' }
    ],
    reds: [
        { stop: 0, color: '#fff5f0' },
        { stop: 0.25, color: '#fcbba1' },
        { stop: 0.5, color: '#fb6a4a' },
        { stop: 0.75, color: '#cb181d' },
        { stop: 1, color: '#67000d' }
    ],
    purples: [
        { stop: 0, color: '#fcfbfd' },
        { stop: 0.25, color: '#dadaeb' },
        { stop: 0.5, color: '#9e9ac8' },
        { stop: 0.75, color: '#6a51a3' },
        { stop: 1, color: '#3f007d' }
    ]
};

const CytoscapeManager = {
    /**
     * Initialize Cytoscape instance
     */
    initializeCytoscape(container) {
        State.cy = cytoscape({
            container: container,
            style: this.getPerformanceStyle(),
            layout: { name: 'preset' },
            
            // Enable WebGL Renderer
            renderer: {
                name: 'canvas',
                webgl: true,
                webglTexSize: 1024,
                showFps: false
            },
            
            minZoom: 0.01,
            maxZoom: 10,
            wheelSensitivity: 0.3,
            boxSelectionEnabled: true,
            selectionType: 'additive',
            autounselectify: false,
            autoungrabify: false,
            // Performance optimizations
            textureOnViewport: true,
            hideEdgesOnViewport: true,
            hideLabelsOnViewport: true,
            pixelRatio: 1,
            motionBlur: true
        });
        
        this.setupEvents();
        return State.cy;
    },

    /**
     * Get performance-optimized style
     * Uses RendererSettings for consistent defaults across both renderers
     */
    getPerformanceStyle() {
        // Get style config from RendererSettings for consistency with Cosmos
        const styleConfig = typeof RendererSettings !== 'undefined'
            ? RendererSettings.getStyleConfig()
            : {};

        // Use RendererSettings values with fallbacks to original defaults
        const defaultNodeColor = styleConfig.defaultNodeColor || '#c8c8c8';
        const defaultNodeSize = styleConfig.defaultNodeSize || 13;
        const defaultEdgeColor = styleConfig.defaultEdgeColor || '#ffffff';
        const defaultEdgeOpacity = styleConfig.defaultEdgeOpacity || 0.3;
        const selectionColor = styleConfig.selectionColor || '#FF0000';
        const highlightColor = styleConfig.highlightColor || '#FFA500';

        return [
            {
                selector: 'node',
                style: {
                    'background-color': defaultNodeColor,
                    'width': defaultNodeSize,
                    'height': defaultNodeSize,
                    'label': '',
                    'border-width': 0
                }
            },
            {
                selector: 'edge',
                style: {
                    'line-color': defaultEdgeColor,
                    'width': 1,
                    'opacity': defaultEdgeOpacity,
                    'curve-style': 'straight',
                    'target-arrow-shape': 'none'
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'background-color': selectionColor,
                    'border-width': 3,
                    'border-color': selectionColor,
                    'z-index': 999
                }
            },
            {
                selector: 'edge:selected',
                style: {
                    'line-color': selectionColor,
                    'width': 2,
                    'opacity': 1,
                    'z-index': 999
                }
            },
            {
                selector: '.highlighted',
                style: {
                    'background-color': highlightColor,
                    'line-color': highlightColor,
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
            }
        ];
    },

    /**
     * Setup Cytoscape event handlers
     */
    setupEvents() {
        const cy = State.cy;
        
        // Node tap
        cy.on('tap', 'node', (e) => {
            const selected = cy.nodes(':selected');
            if (selected.length > 1) {
                InfoPanel.showMultiSelect(selected);
            } else {
                InfoPanel.showNode(e.target);
            }
        });
        
        // Edge tap
        cy.on('tap', 'edge', (e) => {
            InfoPanel.showEdge(e.target);
        });
        
        // Background tap - close panel and clear navigation
        cy.on('tap', (e) => {
            if (e.target === cy) {
                DOMCache.infoPanel.style.display = 'none';
                InfoPanel.clearNavigation();  // Reset origin for next selection
            }
        });
        
        // Selection changes - DEBOUNCED to prevent freezing when selecting many nodes
        cy.on('select unselect', debounce(() => {
            const selected = cy.$(':selected');
            if (selected.length > 1) {
                InfoPanel.showMultiSelect(selected);
            } else if (selected.length === 1) {
                if (selected.isNode()) {
                    InfoPanel.showNode(selected[0]);
                } else {
                    InfoPanel.showEdge(selected[0]);
                }
            }
            // Update distributions window with new selection
            DistributionsComm.sendSelectionUpdate();
        }, 100));  // 100ms debounce - matches original app.js
        
        // Box selection - DEBOUNCED
        cy.on('boxend', debounce(() => {
            const selected = cy.nodes(':selected');
            if (selected.length > 1) {
                InfoPanel.showMultiSelect(selected);
            }
            DistributionsComm.sendSelectionUpdate();
        }, 100));
    },

    /**
     * Apply performance mode style
     */
    applyPerformanceMode() {
        if (!State.cy) return;
        State.cy.elements().removeStyle();
        State.cy.style().fromJson(this.getPerformanceStyle()).update();
        Toast.info('Performance mode enabled');
    },

    /**
     * Update styling based on current metric selections
     */
    updateStyle() {
        if (!State.cy || State.performanceMode) return;
        
        const sizeMetric = document.getElementById('node-size-metric')?.value;
        const colorMetric = document.getElementById('node-color-metric')?.value;
        const gradientName = document.getElementById('node-color-gradient')?.value || 'spectral';
        
        // Update size range
        if (sizeMetric) {
            const values = State.cy.nodes()
                .map(n => n.data(sizeMetric))
                .filter(v => typeof v === 'number' && !isNaN(v));
            if (values.length > 0) {
                State.styleCache.sizeRange = {
                    min: Math.min(...values),
                    max: Math.max(...values)
                };
            }
        }
        
        // Update color range
        if (colorMetric) {
            const values = State.cy.nodes()
                .map(n => n.data(colorMetric))
                .filter(v => typeof v === 'number' && !isNaN(v));
            if (values.length > 0) {
                State.styleCache.colorRange = {
                    min: Math.min(...values),
                    max: Math.max(...values)
                };
            }
        }
        
        const gradient = COLOR_GRADIENTS[gradientName] || COLOR_GRADIENTS.spectral;
        
        // Apply styles in batch
        State.cy.batch(() => {
            State.cy.nodes().forEach(node => {
                const style = {};
                
                // Size based on metric
                if (sizeMetric) {
                    const val = node.data(sizeMetric);
                    if (typeof val === 'number' && !isNaN(val)) {
                        const range = State.styleCache.sizeRange;
                        const norm = range.max > range.min 
                            ? (val - range.min) / (range.max - range.min) 
                            : 0.5;
                        style.width = style.height = 5 + norm * 30;
                    }
                }
                
                // Color based on metric
                if (colorMetric) {
                    const val = node.data(colorMetric);
                    if (typeof val === 'number' && !isNaN(val)) {
                        const range = State.styleCache.colorRange;
                        const norm = range.max > range.min 
                            ? (val - range.min) / (range.max - range.min) 
                            : 0.5;
                        style['background-color'] = this.getGradientColor(norm, gradient);
                    }
                }
                
                if (Object.keys(style).length > 0) {
                    node.style(style);
                }
            });
        });
    },

    /**
     * Get color from gradient at position
     */
    getGradientColor(norm, gradient) {
        norm = Math.max(0, Math.min(1, norm));
        
        let lower = gradient[0];
        let upper = gradient[gradient.length - 1];
        
        for (let i = 0; i < gradient.length - 1; i++) {
            if (norm >= gradient[i].stop && norm <= gradient[i + 1].stop) {
                lower = gradient[i];
                upper = gradient[i + 1];
                break;
            }
        }
        
        const range = upper.stop - lower.stop;
        const ratio = range > 0 ? (norm - lower.stop) / range : 0;
        
        const c1 = this.hexToRgb(lower.color);
        const c2 = this.hexToRgb(upper.color);
        
        const r = Math.round(c1.r + (c2.r - c1.r) * ratio);
        const g = Math.round(c1.g + (c2.g - c1.g) * ratio);
        const b = Math.round(c1.b + (c2.b - c1.b) * ratio);
        
        return this.rgbToHex(r, g, b);
    },

    /**
     * Convert hex to RGB
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    },

    /**
     * Convert RGB to hex
     */
    rgbToHex(r, g, b) {
        return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    },

    /**
     * Toggle neighbor highlighting
     */
    toggleNeighborHighlight() {
        if (!State.cy) return;
        
        const selected = State.cy.nodes(':selected');
        if (selected.length === 0) {
            Toast.info('Select a node first');
            return;
        }
        
        // Cycle through states: 0=none, 1=incoming, 2=outgoing, 3=all
        State.neighborHighlightState = (State.neighborHighlightState + 1) % 4;
        State.cy.elements().removeClass('highlighted');
        
        switch (State.neighborHighlightState) {
            case 1:
                selected.incomers().addClass('highlighted');
                Toast.info('Showing incoming neighbors');
                break;
            case 2:
                selected.outgoers().addClass('highlighted');
                Toast.info('Showing outgoing neighbors');
                break;
            case 3:
                selected.neighborhood().addClass('highlighted');
                Toast.info('Showing all neighbors');
                break;
            default:
                Toast.info('Neighbor highlight cleared');
        }
    },

    /**
     * Toggle between performance and styled render modes
     */
    toggleRenderMode(toPerformance) {
        State.performanceMode = toPerformance;
        if (toPerformance) {
            this.applyPerformanceMode();
        } else {
            this.updateStyle();
            Toast.info('Style mode enabled');
        }
    },

    /**
     * Apply full style including node and edge settings
     */
    applyFullStyle() {
        if (!State.cy) return;
        
        // Get all style settings
        const sizeMetric = document.getElementById('node-size-metric')?.value;
        const colorMetric = document.getElementById('node-color-metric')?.value;
        const gradientName = document.getElementById('node-color-gradient')?.value || 'spectral';
        const sizeMin = parseFloat(document.getElementById('node-size-min')?.value) || 8;
        const sizeMax = parseFloat(document.getElementById('node-size-max')?.value) || 25;
        const edgeWidthMin = parseFloat(document.getElementById('edge-width-min')?.value) || 2;
        const edgeWidthMax = parseFloat(document.getElementById('edge-width-max')?.value) || 5;
        const edgeOpacity = (parseFloat(document.getElementById('edge-opacity')?.value) || 20) / 100;
        const edgeColor = document.getElementById('edge-color')?.value || '#fcfafa';
        
        const gradient = COLOR_GRADIENTS[gradientName] || COLOR_GRADIENTS.spectral;
        
        // Calculate ranges for metrics
        if (sizeMetric) {
            const values = State.cy.nodes()
                .map(n => n.data(sizeMetric))
                .filter(v => typeof v === 'number' && !isNaN(v));
            if (values.length > 0) {
                State.styleCache.sizeRange = {
                    min: Math.min(...values),
                    max: Math.max(...values)
                };
            }
        }
        
        if (colorMetric) {
            const values = State.cy.nodes()
                .map(n => n.data(colorMetric))
                .filter(v => typeof v === 'number' && !isNaN(v));
            if (values.length > 0) {
                State.styleCache.colorRange = {
                    min: Math.min(...values),
                    max: Math.max(...values)
                };
            }
        }
        
        // Apply styles in batch
        State.cy.batch(() => {
            // Apply node styles
            State.cy.nodes().forEach(node => {
                const style = {};
                
                // Size based on metric
                if (sizeMetric) {
                    const val = node.data(sizeMetric);
                    if (typeof val === 'number' && !isNaN(val)) {
                        const range = State.styleCache.sizeRange;
                        const norm = range.max > range.min 
                            ? (val - range.min) / (range.max - range.min) 
                            : 0.5;
                        const size = sizeMin + norm * (sizeMax - sizeMin);
                        style.width = style.height = size;
                    }
                }
                
                // Color based on metric
                if (colorMetric) {
                    const val = node.data(colorMetric);
                    if (typeof val === 'number' && !isNaN(val)) {
                        const range = State.styleCache.colorRange;
                        const norm = range.max > range.min 
                            ? (val - range.min) / (range.max - range.min) 
                            : 0.5;
                        style['background-color'] = this.getGradientColor(norm, gradient);
                    }
                }
                
                if (Object.keys(style).length > 0) {
                    node.style(style);
                }
            });
            
            // Apply edge styles
            State.cy.edges().forEach(edge => {
                edge.style({
                    'line-color': edgeColor,
                    'opacity': edgeOpacity,
                    'width': edgeWidthMin  // Could be made dynamic based on edge weight
                });
            });
        });
        
        // Switch to style mode if in performance mode
        if (State.performanceMode) {
            State.performanceMode = false;
            // Update radio button
            const styleRadio = document.querySelector('input[name="render-mode"][value="style"]');
            if (styleRadio) styleRadio.checked = true;
        }
        
        Toast.success('Style applied');
    }
};