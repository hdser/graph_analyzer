/**
 * Utils Module
 * Utility functions for the application
 */

const Utils = {
    /**
     * Debounce function execution
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    },

    /**
     * Throttle function execution
     */
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * Copy text to clipboard with fallback
     */
    async copyToClipboard(text) {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
                return true;
            } else {
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                const result = document.execCommand('copy');
                document.body.removeChild(textArea);
                return result;
            }
        } catch (err) {
            console.error('Copy failed:', err);
            return false;
        }
    },

    /**
     * Format number with decimals
     */
    formatNumber(value, decimals = 4) {
        if (value === null || value === undefined) return '-';
        if (typeof value !== 'number') return value;  // Return strings as-is BEFORE isNaN check
        if (isNaN(value) || !isFinite(value)) return '-';
        if (Number.isInteger(value)) return value.toLocaleString();
        return value.toFixed(decimals);
    },

    /**
     * Format number in compact form (K, M, B)
     */
    formatCompact(value) {
        if (value === null || value === undefined || isNaN(value)) return '-';
        if (Math.abs(value) >= 1e9) return (value / 1e9).toFixed(2) + 'B';
        if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(2) + 'M';
        if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(2) + 'K';
        return value.toFixed(2);
    },

    /**
     * Escape value for CSV
     */
    escapeCSV(val) {
        if (val === undefined || val === null) return '';
        const str = String(val);
        if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return '"' + str.replace(/"/g, '""') + '"';
        }
        return str;
    },

    /**
     * Get timestamp string for filenames
     */
    getTimestamp() {
        return new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
    },

    /**
     * Deep clone object
     */
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },

    /**
     * Check if value is numeric
     */
    isNumeric(value) {
        return typeof value === 'number' && !isNaN(value) && isFinite(value);
    },

    /**
     * Get unique values from array
     */
    unique(arr) {
        return [...new Set(arr)];
    }
};