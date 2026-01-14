/**
 * Color Gradients Module
 * 
 * Shared color gradient definitions and interpolation functions
 * used by both CytoscapeAdapter and CosmosAdapter.
 */

const ColorGradients = {
    /**
     * Available color gradients with stop points
     */
    gradients: {
        viridis: [
            { stop: 0, color: '#440154', rgb: [68, 1, 84] },
            { stop: 0.25, color: '#3b528b', rgb: [59, 82, 139] },
            { stop: 0.5, color: '#21918c', rgb: [33, 145, 140] },
            { stop: 0.75, color: '#5ec962', rgb: [94, 201, 98] },
            { stop: 1, color: '#fde725', rgb: [253, 231, 37] }
        ],
        plasma: [
            { stop: 0, color: '#0d0887', rgb: [13, 8, 135] },
            { stop: 0.25, color: '#7e03a8', rgb: [126, 3, 168] },
            { stop: 0.5, color: '#cc4778', rgb: [204, 71, 120] },
            { stop: 0.75, color: '#f89540', rgb: [248, 149, 64] },
            { stop: 1, color: '#f0f921', rgb: [240, 249, 33] }
        ],
        inferno: [
            { stop: 0, color: '#000004', rgb: [0, 0, 4] },
            { stop: 0.25, color: '#57106e', rgb: [87, 16, 110] },
            { stop: 0.5, color: '#bc3754', rgb: [188, 55, 84] },
            { stop: 0.75, color: '#f98e09', rgb: [249, 142, 9] },
            { stop: 1, color: '#fcffa4', rgb: [252, 255, 164] }
        ],
        spectral: [
            { stop: 0, color: '#5e4fa2', rgb: [94, 79, 162] },
            { stop: 0.25, color: '#3288bd', rgb: [50, 136, 189] },
            { stop: 0.5, color: '#66c2a5', rgb: [102, 194, 165] },
            { stop: 0.75, color: '#fee08b', rgb: [254, 224, 139] },
            { stop: 1, color: '#d53e4f', rgb: [213, 62, 79] }
        ],
        coolwarm: [
            { stop: 0, color: '#3b4cc0', rgb: [59, 76, 192] },
            { stop: 0.25, color: '#7b9ff9', rgb: [123, 159, 249] },
            { stop: 0.5, color: '#f7f7f7', rgb: [247, 247, 247] },
            { stop: 0.75, color: '#f4a582', rgb: [244, 165, 130] },
            { stop: 1, color: '#b40426', rgb: [180, 4, 38] }
        ],
        greens: [
            { stop: 0, color: '#f7fcf5', rgb: [247, 252, 245] },
            { stop: 0.25, color: '#c7e9c0', rgb: [199, 233, 192] },
            { stop: 0.5, color: '#74c476', rgb: [116, 196, 118] },
            { stop: 0.75, color: '#238b45', rgb: [35, 139, 69] },
            { stop: 1, color: '#00441b', rgb: [0, 68, 27] }
        ],
        blues: [
            { stop: 0, color: '#f7fbff', rgb: [247, 251, 255] },
            { stop: 0.25, color: '#c6dbef', rgb: [198, 219, 239] },
            { stop: 0.5, color: '#6baed6', rgb: [107, 174, 214] },
            { stop: 0.75, color: '#2171b5', rgb: [33, 113, 181] },
            { stop: 1, color: '#084594', rgb: [8, 69, 148] }
        ],
        reds: [
            { stop: 0, color: '#fff5f0', rgb: [255, 245, 240] },
            { stop: 0.25, color: '#fcbba1', rgb: [252, 187, 161] },
            { stop: 0.5, color: '#fb6a4a', rgb: [251, 106, 74] },
            { stop: 0.75, color: '#cb181d', rgb: [203, 24, 29] },
            { stop: 1, color: '#67000d', rgb: [103, 0, 13] }
        ],
        purples: [
            { stop: 0, color: '#fcfbfd', rgb: [252, 251, 253] },
            { stop: 0.25, color: '#dadaeb', rgb: [218, 218, 235] },
            { stop: 0.5, color: '#9e9ac8', rgb: [158, 154, 200] },
            { stop: 0.75, color: '#6a51a3', rgb: [106, 81, 163] },
            { stop: 1, color: '#3f007d', rgb: [63, 0, 125] }
        ],
        oranges: [
            { stop: 0, color: '#fff5eb', rgb: [255, 245, 235] },
            { stop: 0.25, color: '#fdd0a2', rgb: [253, 208, 162] },
            { stop: 0.5, color: '#fd8d3c', rgb: [253, 141, 60] },
            { stop: 0.75, color: '#d94801', rgb: [217, 72, 1] },
            { stop: 1, color: '#7f2704', rgb: [127, 39, 4] }
        ],
        magma: [
            { stop: 0, color: '#000004', rgb: [0, 0, 4] },
            { stop: 0.25, color: '#51127c', rgb: [81, 18, 124] },
            { stop: 0.5, color: '#b73779', rgb: [183, 55, 121] },
            { stop: 0.75, color: '#fc8961', rgb: [252, 137, 97] },
            { stop: 1, color: '#fcfdbf', rgb: [252, 253, 191] }
        ],
        cividis: [
            { stop: 0, color: '#002051', rgb: [0, 32, 81] },
            { stop: 0.25, color: '#395a75', rgb: [57, 90, 117] },
            { stop: 0.5, color: '#7a7b78', rgb: [122, 123, 120] },
            { stop: 0.75, color: '#bbb866', rgb: [187, 184, 102] },
            { stop: 1, color: '#fdea45', rgb: [253, 234, 69] }
        ]
    },
    
    /**
     * Get a gradient by name
     * @param {string} name - Gradient name
     * @returns {Array} Gradient stops
     */
    get(name) {
        return this.gradients[name] || this.gradients.spectral;
    },
    
    /**
     * Get all available gradient names
     * @returns {Array} List of gradient names
     */
    getNames() {
        return Object.keys(this.gradients);
    },
    
    /**
     * Interpolate color from gradient at position (returns hex)
     * @param {Array} gradient - Gradient stops
     * @param {number} t - Position (0-1)
     * @returns {string} Hex color
     */
    interpolate(gradient, t) {
        t = Math.max(0, Math.min(1, t));
        
        let lower = gradient[0];
        let upper = gradient[gradient.length - 1];
        
        for (let i = 0; i < gradient.length - 1; i++) {
            if (t >= gradient[i].stop && t <= gradient[i + 1].stop) {
                lower = gradient[i];
                upper = gradient[i + 1];
                break;
            }
        }
        
        const range = upper.stop - lower.stop;
        const ratio = range > 0 ? (t - lower.stop) / range : 0;
        
        const r = Math.round(lower.rgb[0] + (upper.rgb[0] - lower.rgb[0]) * ratio);
        const g = Math.round(lower.rgb[1] + (upper.rgb[1] - lower.rgb[1]) * ratio);
        const b = Math.round(lower.rgb[2] + (upper.rgb[2] - lower.rgb[2]) * ratio);
        
        return this.rgbToHex(r, g, b);
    },
    
    /**
     * Interpolate color from gradient at position (returns RGBA array 0-1)
     * @param {Array} gradient - Gradient stops
     * @param {number} t - Position (0-1)
     * @param {number} alpha - Alpha value (0-1)
     * @returns {Array} [r, g, b, a] in 0-1 range
     */
    interpolateRgba(gradient, t, alpha = 1.0) {
        t = Math.max(0, Math.min(1, t));
        
        let lower = gradient[0];
        let upper = gradient[gradient.length - 1];
        
        for (let i = 0; i < gradient.length - 1; i++) {
            if (t >= gradient[i].stop && t <= gradient[i + 1].stop) {
                lower = gradient[i];
                upper = gradient[i + 1];
                break;
            }
        }
        
        const range = upper.stop - lower.stop;
        const ratio = range > 0 ? (t - lower.stop) / range : 0;
        
        return [
            (lower.rgb[0] + (upper.rgb[0] - lower.rgb[0]) * ratio) / 255,
            (lower.rgb[1] + (upper.rgb[1] - lower.rgb[1]) * ratio) / 255,
            (lower.rgb[2] + (upper.rgb[2] - lower.rgb[2]) * ratio) / 255,
            alpha
        ];
    },
    
    /**
     * Convert RGB values to hex
     * @param {number} r - Red (0-255)
     * @param {number} g - Green (0-255)
     * @param {number} b - Blue (0-255)
     * @returns {string} Hex color
     */
    rgbToHex(r, g, b) {
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    },
    
    /**
     * Convert hex to RGB array
     * @param {string} hex - Hex color
     * @returns {Array} [r, g, b] in 0-255 range
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? [
            parseInt(result[1], 16),
            parseInt(result[2], 16),
            parseInt(result[3], 16)
        ] : [128, 128, 128];
    },
    
    /**
     * Convert hex to RGBA array (0-1 range)
     * @param {string} hex - Hex color
     * @param {number} alpha - Alpha value
     * @returns {Array} [r, g, b, a] in 0-1 range
     */
    hexToRgba(hex, alpha = 1.0) {
        const rgb = this.hexToRgb(hex);
        return [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha];
    },
    
    /**
     * Generate CSS gradient string for display
     * @param {string} name - Gradient name
     * @returns {string} CSS linear-gradient
     */
    toCssGradient(name) {
        const gradient = this.get(name);
        const stops = gradient.map(s => `${s.color} ${s.stop * 100}%`).join(', ');
        return `linear-gradient(to right, ${stops})`;
    },
    
    /**
     * Create a discrete color palette from a gradient
     * @param {string} name - Gradient name
     * @param {number} n - Number of colors
     * @returns {Array} Array of hex colors
     */
    toPalette(name, n = 10) {
        const gradient = this.get(name);
        const colors = [];
        
        for (let i = 0; i < n; i++) {
            const t = n === 1 ? 0.5 : i / (n - 1);
            colors.push(this.interpolate(gradient, t));
        }
        
        return colors;
    }
};

// Make available globally
window.ColorGradients = ColorGradients;