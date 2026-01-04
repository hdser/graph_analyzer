/**
 * Icons Module
 * Centralized SVG icons for the application
 * 
 * Usage:
 *   Icons.get('copy')           - Returns SVG string
 *   Icons.create('copy')        - Returns SVG DOM element
 *   Icons.inject()              - Auto-injects icons into elements with data-icon attribute
 */

const Icons = {
    // Icon definitions - clean, simple SVG paths
    // All icons are 16x16 viewBox by default
    _icons: {
        // Actions
        copy: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="5" y="5" width="9" height="10" rx="1"/>
            <path d="M3 11V3a1 1 0 0 1 1-1h7"/>
        </svg>`,
        
        save: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M13 14H3a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1h8l3 3v9a1 1 0 0 1-1 1z"/>
            <path d="M11 14v-4H5v4"/>
            <path d="M5 2v3h5"/>
        </svg>`,
        
        refresh: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 8a6 6 0 0 1 10.3-4.2"/>
            <path d="M14 8a6 6 0 0 1-10.3 4.2"/>
            <path d="M12 2v3h-3"/>
            <path d="M4 14v-3h3"/>
        </svg>`,
        
        search: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="7" cy="7" r="4.5"/>
            <path d="M10.5 10.5L14 14"/>
        </svg>`,
        
        close: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 4l8 8"/>
            <path d="M12 4l-8 8"/>
        </svg>`,
        
        export: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 2v8"/>
            <path d="M4 6l4-4 4 4"/>
            <path d="M2 10v3a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-3"/>
        </svg>`,
        
        download: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 2v8"/>
            <path d="M4 6l4 4 4-4"/>
            <path d="M2 10v3a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-3"/>
        </svg>`,
        
        delete: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 4h10"/>
            <path d="M6 4V2h4v2"/>
            <path d="M12 4v9a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V4"/>
            <path d="M7 7v4"/>
            <path d="M9 7v4"/>
        </svg>`,
        
        // Navigation
        chevronDown: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 6l4 4 4-4"/>
        </svg>`,
        
        chevronUp: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 10l4-4 4 4"/>
        </svg>`,
        
        chevronRight: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 4l4 4-4 4"/>
        </svg>`,
        
        chevronLeft: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10 4l-4 4 4 4"/>
        </svg>`,
        
        // Status
        check: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 8l3.5 3.5L13 4"/>
        </svg>`,
        
        warning: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 1.5l6.5 12H1.5L8 1.5z"/>
            <path d="M8 6v3"/>
            <circle cx="8" cy="11" r="0.5" fill="currentColor"/>
        </svg>`,
        
        info: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="8" cy="8" r="6"/>
            <path d="M8 7v4"/>
            <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
        </svg>`,
        
        error: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="8" cy="8" r="6"/>
            <path d="M5.5 5.5l5 5"/>
            <path d="M10.5 5.5l-5 5"/>
        </svg>`,
        
        // Graph/Data
        nodes: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="4" cy="4" r="2"/>
            <circle cx="12" cy="4" r="2"/>
            <circle cx="8" cy="12" r="2"/>
            <path d="M5.5 5.5l2 4.5"/>
            <path d="M10.5 5.5l-2 4.5"/>
        </svg>`,
        
        flow: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="3" cy="8" r="2"/>
            <circle cx="13" cy="8" r="2"/>
            <path d="M5 8h6"/>
            <path d="M8 5l3 3-3 3"/>
        </svg>`,
        
        paths: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="3" cy="3" r="2"/>
            <circle cx="13" cy="13" r="2"/>
            <path d="M5 5l6 6"/>
            <path d="M5 3h4"/>
            <path d="M13 7v4"/>
        </svg>`,
        
        subgraph: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="2" width="12" height="12" rx="2" stroke-dasharray="2 2"/>
            <circle cx="5" cy="8" r="1.5" fill="currentColor"/>
            <circle cx="11" cy="5" r="1.5" fill="currentColor"/>
            <circle cx="11" cy="11" r="1.5" fill="currentColor"/>
            <path d="M6.5 7.5l3-2"/>
            <path d="M6.5 8.5l3 2"/>
        </svg>`,
        
        filter: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 3h12"/>
            <path d="M4 7h8"/>
            <path d="M6 11h4"/>
        </svg>`,
        
        chart: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="8" width="3" height="6"/>
            <rect x="6.5" y="4" width="3" height="10"/>
            <rect x="11" y="6" width="3" height="8"/>
        </svg>`,
        
        // UI
        menu: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 4h12"/>
            <path d="M2 8h12"/>
            <path d="M2 12h12"/>
        </svg>`,
        
        settings: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="8" cy="8" r="2"/>
            <path d="M8 1v2M8 13v2M1 8h2M13 8h2"/>
            <path d="M3.05 3.05l1.4 1.4M11.55 11.55l1.4 1.4M3.05 12.95l1.4-1.4M11.55 4.45l1.4-1.4"/>
        </svg>`,
        
        play: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 3l9 5-9 5V3z" fill="currentColor"/>
        </svg>`,
        
        pause: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="4" y="3" width="3" height="10" fill="currentColor"/>
            <rect x="9" y="3" width="3" height="10" fill="currentColor"/>
        </svg>`,
        
        zoomIn: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="7" cy="7" r="4.5"/>
            <path d="M10.5 10.5L14 14"/>
            <path d="M7 5v4"/>
            <path d="M5 7h4"/>
        </svg>`,
        
        zoomOut: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="7" cy="7" r="4.5"/>
            <path d="M10.5 10.5L14 14"/>
            <path d="M5 7h4"/>
        </svg>`,
        
        fit: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 5V2h3"/>
            <path d="M14 5V2h-3"/>
            <path d="M2 11v3h3"/>
            <path d="M14 11v3h-3"/>
        </svg>`,
        
        center: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="8" cy="8" r="2"/>
            <path d="M8 2v3"/>
            <path d="M8 11v3"/>
            <path d="M2 8h3"/>
            <path d="M11 8h3"/>
        </svg>`,
        
        // Arrows
        arrowUp: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 14V2"/>
            <path d="M3 7l5-5 5 5"/>
        </svg>`,
        
        arrowDown: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 2v12"/>
            <path d="M3 9l5 5 5-5"/>
        </svg>`,
        
        arrowLeft: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 8H2"/>
            <path d="M7 3L2 8l5 5"/>
        </svg>`,
        
        arrowRight: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 8h12"/>
            <path d="M9 3l5 5-5 5"/>
        </svg>`,
        
        // Math operators
        plus: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 3v10"/>
            <path d="M3 8h10"/>
        </svg>`,
        
        minus: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 8h10"/>
        </svg>`,
        
        multiply: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 4l8 8"/>
            <path d="M12 4l-8 8"/>
        </svg>`,
        
        divide: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 8h10"/>
            <circle cx="8" cy="4" r="1" fill="currentColor"/>
            <circle cx="8" cy="12" r="1" fill="currentColor"/>
        </svg>`,
        
        // Misc
        link: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 10l4-4"/>
            <path d="M9 5l1-1a2.83 2.83 0 1 1 4 4l-1 1"/>
            <path d="M7 11l-1 1a2.83 2.83 0 1 1-4-4l1-1"/>
        </svg>`,
        
        externalLink: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 9v4a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h4"/>
            <path d="M9 2h5v5"/>
            <path d="M14 2L7 9"/>
        </svg>`,
        
        highlight: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3L5 10l-2 4 4-2 7-7-2-2z"/>
            <path d="M10 5l2 2"/>
        </svg>`,
        
        eye: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z"/>
            <circle cx="8" cy="8" r="2"/>
        </svg>`,
        
        eyeOff: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 2l12 12"/>
            <path d="M6.5 6.5a2 2 0 0 0 2.8 2.8"/>
            <path d="M4.2 4.2C2.4 5.4 1 8 1 8s2.5 5 7 5c1.3 0 2.5-.4 3.5-1"/>
            <path d="M12 12c1.5-1.2 3-3.7 3-4s-2.5-5-7-5c-.7 0-1.4.1-2 .3"/>
        </svg>`,
        
        loading: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 2v2"/>
            <path d="M8 12v2"/>
            <path d="M2 8h2"/>
            <path d="M12 8h2"/>
            <path d="M3.76 3.76l1.41 1.41"/>
            <path d="M10.83 10.83l1.41 1.41"/>
            <path d="M3.76 12.24l1.41-1.41"/>
            <path d="M10.83 5.17l1.41-1.41"/>
        </svg>`,
        
        more: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="8" cy="8" r="1" fill="currentColor"/>
            <circle cx="3" cy="8" r="1" fill="currentColor"/>
            <circle cx="13" cy="8" r="1" fill="currentColor"/>
        </svg>`,
        
        // Additional icons for sidebar navigation
        upload: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 10v3a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-3"/>
            <path d="M8 2v8"/>
            <path d="M4 6l4-4 4 4"/>
        </svg>`,
        
        camera: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="1" y="3" width="14" height="10" rx="1"/>
            <circle cx="8" cy="8" r="2.5"/>
            <path d="M5 3L6 1h4l1 2"/>
        </svg>`,
        
        palette: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 1.5c3.5 0 6.5 3 6.5 6.5 0 1.5-.5 2.5-1.5 2.5h-1c-.5 0-1 .5-1 1 0 .5.2 1 .5 1.3.3.3.5.7.5 1.2 0 1-1 1.5-2 1.5-3.5 0-6.5-3-6.5-6.5S4.5 1.5 8 1.5z"/>
            <circle cx="5.5" cy="6.5" r="0.8" fill="currentColor"/>
            <circle cx="8" cy="5" r="0.8" fill="currentColor"/>
            <circle cx="10.5" cy="6.5" r="0.8" fill="currentColor"/>
        </svg>`,
        
        layers: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 11l6 3 6-3"/>
            <path d="M2 8l6 3 6-3"/>
            <path d="M8 2L2 5l6 3 6-3-6-3z"/>
        </svg>`,
        
        database: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <ellipse cx="8" cy="3" rx="6" ry="2"/>
            <path d="M2 3v8c0 1.1 2.7 2 6 2s6-.9 6-2V3"/>
            <path d="M2 7c0 1.1 2.7 2 6 2s6-.9 6-2"/>
        </svg>`,
        
        // App logo - network pattern inspired design
        appLogo: `<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Thick C shape as a filled path -->
            <path d="M20 2C9.507 2 1 10.507 1 21C1 31.493 9.507 40 20 40C20 40 20 32 20 32C13.925 32 9 27.075 9 21C9 14.925 13.925 10 20 10C20 10 20 2 20 2Z" fill="#4A90E2"/>
            <!-- Network edges -->
            <line x1="22" y1="13" x2="22" y2="18.5" stroke="#FF6B35" stroke-width="1.5"/>
            <line x1="22" y1="23.5" x2="22" y2="29" stroke="#FF6B35" stroke-width="1.5"/>
            <line x1="24.5" y1="21" x2="30" y2="21" stroke="#FF6B35" stroke-width="1.5"/>
            <line x1="22" y1="13" x2="30" y2="21" stroke="#FF6B35" stroke-width="1.5" opacity="0.7"/>
            <line x1="30" y1="21" x2="22" y2="29" stroke="#FF6B35" stroke-width="1.5" opacity="0.7"/>
            <!-- Network nodes -->
            <circle cx="22" cy="13" r="3.5" fill="#FF6B35"/>
            <circle cx="30" cy="21" r="3.5" fill="#FF6B35"/>
            <circle cx="22" cy="29" r="3.5" fill="#FF6B35"/>
            <circle cx="22" cy="21" r="2.5" fill="#FF6B35"/>
        </svg>`
    },

    /**
     * Get SVG string for an icon
     */
    get(name, options = {}) {
        const svg = this._icons[name];
        if (!svg) {
            console.warn(`Icon "${name}" not found`);
            return '';
        }
        
        let result = svg;
        
        if (options.size && options.size !== 16) {
            result = result.replace(/width="16"/g, `width="${options.size}"`);
            result = result.replace(/height="16"/g, `height="${options.size}"`);
        }
        
        if (options.class) {
            result = result.replace('<svg ', `<svg class="${options.class}" `);
        }
        
        return result;
    },

    /**
     * Create SVG DOM element for an icon
     */
    create(name, options = {}) {
        const svgString = this.get(name, options);
        if (!svgString) return null;
        
        const template = document.createElement('template');
        template.innerHTML = svgString.trim();
        return template.content.firstChild;
    },

    /**
     * Auto-inject icons into elements with data-icon attribute
     */
    inject() {
        document.querySelectorAll('[data-icon]').forEach(el => {
            const iconName = el.dataset.icon;
            const size = el.dataset.iconSize || 16;
            const svg = this.get(iconName, { size: parseInt(size) });
            if (svg) {
                el.innerHTML = svg;
            }
        });
    },

    /**
     * Get list of all available icon names
     */
    list() {
        return Object.keys(this._icons);
    },

    /**
     * Math operation symbols (text-based for dropdowns)
     */
    mathSymbols: {
        multiply: '*',
        add: '+',
        subtract: '-',
        divide: '/',
        maximum: 'max',
        minimum: 'min',
        average: 'avg',
        gte: '>=',
        lte: '<=',
        gt: '>',
        lt: '<',
        eq: '='
    },

    /**
     * Get math symbol for operation
     */
    getMathSymbol(operation) {
        return this.mathSymbols[operation] || operation;
    }
};

// Make available globally
window.Icons = Icons;