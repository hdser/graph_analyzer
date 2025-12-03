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
        if (typeof value !== 'number') return value;  // Return strings, booleans, etc. as-is
        if (isNaN(value) || !isFinite(value)) return '-';  // Check NaN only for actual numbers
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
     * Abbreviate an address/hash (e.g., "0x1234...abcd")
     */
    abbreviateAddress(addr, prefixLen = 6, suffixLen = 4) {
        if (!addr || typeof addr !== 'string') return String(addr || '');
        if (addr.length <= prefixLen + suffixLen + 3) return addr;
        return `${addr.slice(0, prefixLen)}...${addr.slice(-suffixLen)}`;
    },

    /**
     * Format token array for display
     * @param {Array|string} tokens - Array of token addresses or JSON string
     * @param {Object} options - Display options
     * @returns {Object} { summary: string, full: string, count: number, items: Array }
     */
    formatTokens(tokens, options = {}) {
        const { maxShow = 3, abbreviate = true } = options;
        
        // Handle null/undefined
        if (tokens === null || tokens === undefined) {
            return { summary: '-', full: '-', count: 0, items: [] };
        }
        
        // Parse JSON string if needed
        let tokenArray = tokens;
        if (typeof tokens === 'string') {
            try {
                tokenArray = JSON.parse(tokens);
            } catch (e) {
                // Not JSON, treat as single token
                tokenArray = [tokens];
            }
        }
        
        // Handle non-array
        if (!Array.isArray(tokenArray)) {
            return { summary: String(tokenArray), full: String(tokenArray), count: 1, items: [tokenArray] };
        }
        
        const count = tokenArray.length;
        if (count === 0) {
            return { summary: '-', full: '-', count: 0, items: [] };
        }
        
        // Format items
        const items = tokenArray.map(t => String(t));
        const displayItems = abbreviate ? items.map(t => this.abbreviateAddress(t)) : items;
        
        // Create summary (first N items + count)
        let summary;
        if (count <= maxShow) {
            summary = displayItems.join(', ');
        } else {
            summary = `${displayItems.slice(0, maxShow).join(', ')} (+${count - maxShow})`;
        }
        
        // Full list
        const full = items.join('\n');
        
        return { summary, full, count, items };
    },

    /**
     * Format token balances (object or array of {token, balance} pairs)
     * @param {Object|Array|string} balances - Token balances
     * @param {Object} options - Display options
     * @returns {Object} { summary: string, full: string, count: number, total: number, items: Array }
     */
    formatTokenBalances(balances, options = {}) {
        const { maxShow = 2, abbreviate = true } = options;
        
        // Handle null/undefined
        if (balances === null || balances === undefined) {
            return { summary: '-', full: '-', count: 0, total: 0, items: [] };
        }
        
        // Parse JSON string if needed
        let balanceData = balances;
        if (typeof balances === 'string') {
            try {
                balanceData = JSON.parse(balances);
            } catch (e) {
                return { summary: String(balances), full: String(balances), count: 1, total: 0, items: [] };
            }
        }
        
        // Convert to array of {token, balance} pairs
        let items = [];
        if (Array.isArray(balanceData)) {
            // Array format: [{token: "0x...", balance: 100}, ...]
            items = balanceData.map(item => ({
                token: String(item.token || item.address || item.id || ''),
                balance: parseFloat(item.balance || item.amount || item.value || 0)
            }));
        } else if (typeof balanceData === 'object') {
            // Object format: {"0x...": 100, "0x...": 200}
            items = Object.entries(balanceData).map(([token, balance]) => ({
                token: String(token),
                balance: parseFloat(balance) || 0
            }));
        }
        
        const count = items.length;
        if (count === 0) {
            return { summary: '-', full: '-', count: 0, total: 0, items: [] };
        }
        
        // Sort by balance descending
        items.sort((a, b) => b.balance - a.balance);
        
        // Calculate total
        const total = items.reduce((sum, item) => sum + item.balance, 0);
        
        // Format for display
        const formatItem = (item) => {
            const addr = abbreviate ? this.abbreviateAddress(item.token) : item.token;
            const bal = this.formatCompact(item.balance);
            return `${addr}: ${bal}`;
        };
        
        // Create summary
        let summary;
        if (count === 1) {
            summary = formatItem(items[0]);
        } else if (count <= maxShow) {
            summary = items.map(formatItem).join(', ');
        } else {
            const shown = items.slice(0, maxShow).map(formatItem).join(', ');
            summary = `${shown} (+${count - maxShow} more, Σ${this.formatCompact(total)})`;
        }
        
        // Full list
        const full = items.map(item => `${item.token}: ${this.formatNumber(item.balance)}`).join('\n');
        
        return { summary, full, count, total, items };
    },

    /**
     * Format any complex value (array, object) for display
     * Automatically detects tokens vs balances vs generic data
     */
    formatComplexValue(value, key = '') {
        if (value === null || value === undefined) return '-';
        if (typeof value !== 'object') return String(value);
        
        // Detect type by key name or content
        const keyLower = key.toLowerCase();
        
        if (keyLower.includes('balance') || keyLower.includes('amount')) {
            return this.formatTokenBalances(value).summary;
        }
        
        if (keyLower.includes('token') || keyLower.includes('address')) {
            return this.formatTokens(value).summary;
        }
        
        // Generic array
        if (Array.isArray(value)) {
            if (value.length === 0) return '-';
            if (value.length <= 3) {
                return value.map(v => 
                    typeof v === 'object' ? JSON.stringify(v) : String(v)
                ).join(', ');
            }
            return `[${value.length} items]`;
        }
        
        // Generic object
        const keys = Object.keys(value);
        if (keys.length === 0) return '-';
        if (keys.length <= 2) {
            return keys.map(k => `${k}: ${this.formatNumber(value[k])}`).join(', ');
        }
        return `{${keys.length} fields}`;
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