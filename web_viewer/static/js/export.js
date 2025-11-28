/**
 * Export Module
 * Copy to clipboard and export functions
 */

const Export = {
    /**
     * Copy current node data to clipboard
     */
    async copyNodeData() {
        if (!State.currentNodeData) {
            updateStatus('No node selected', 'error');
            return;
        }
        
        const data = State.currentNodeData;
        const lines = Object.entries(data)
            .filter(([k]) => k !== 'label' && k !== 'isNew')
            .map(([k, v]) => `${k}: ${Utils.formatNumber(v)}`);
        
        const text = `Node: ${data.id}\n${'='.repeat(40)}\n${lines.join('\n')}`;
        
        const success = await Utils.copyToClipboard(text);
        updateStatus(success ? 'Node data copied to clipboard' : 'Copy failed', success ? 'success' : 'error');
    },

    /**
     * Copy current node ID to clipboard
     */
    async copyNodeId() {
        if (!State.currentNodeData) {
            updateStatus('No node selected', 'error');
            return;
        }
        
        const success = await Utils.copyToClipboard(State.currentNodeData.id);
        updateStatus(success ? 'Node ID copied' : 'Copy failed', success ? 'success' : 'error');
    },

    /**
     * Copy current edge data to clipboard
     */
    async copyEdgeData() {
        if (!State.currentEdgeData) {
            updateStatus('No edge selected', 'error');
            return;
        }
        
        const data = State.currentEdgeData;
        const lines = Object.entries(data).map(([k, v]) => `${k}: ${v}`);
        
        const text = `Edge: ${data.source} â†’ ${data.target}\n${'='.repeat(40)}\n${lines.join('\n')}`;
        
        const success = await Utils.copyToClipboard(text);
        updateStatus(success ? 'Edge data copied to clipboard' : 'Copy failed', success ? 'success' : 'error');
    },

    /**
     * Copy selected nodes' IDs to clipboard
     */
    async copySelectedIds() {
        if (!State.cy) return;
        
        const selected = State.cy.nodes(':selected');
        if (selected.length === 0) {
            updateStatus('No nodes selected', 'error');
            return;
        }
        
        const ids = selected.map(n => n.id()).join('\n');
        const success = await Utils.copyToClipboard(ids);
        
        updateStatus(success ? `Copied ${selected.length} node IDs` : 'Copy failed', success ? 'success' : 'error');
    },

    /**
     * Copy selected nodes' data as JSON
     */
    async copySelectedAsJson() {
        if (!State.cy) return;
        
        const selected = State.cy.nodes(':selected');
        if (selected.length === 0) {
            updateStatus('No nodes selected', 'error');
            return;
        }
        
        const data = selected.map(n => {
            const nodeData = n.data();
            const clean = {};
            Object.entries(nodeData).forEach(([k, v]) => {
                if (k !== 'label' && k !== 'isNew' && typeof v !== 'object') {
                    clean[k] = v;
                }
            });
            return clean;
        });
        
        const json = JSON.stringify(data, null, 2);
        const success = await Utils.copyToClipboard(json);
        
        updateStatus(success ? `Copied ${selected.length} nodes as JSON` : 'Copy failed', success ? 'success' : 'error');
    },

    /**
     * Export selected nodes as CSV
     */
    exportSelectedAsCsv() {
        if (!State.cy) return;
        
        const selected = State.cy.nodes(':selected');
        if (selected.length === 0) {
            updateStatus('No nodes selected', 'error');
            return;
        }
        
        // Collect all keys
        const allKeys = new Set();
        selected.forEach(node => {
            const data = node.data();
            Object.keys(data).forEach(k => {
                if (!['label', 'isNew'].includes(k) && typeof data[k] !== 'object') {
                    allKeys.add(k);
                }
            });
        });
        
        const keys = Array.from(allKeys);
        
        // Build CSV
        let csv = keys.map(Utils.escapeCSV).join(',') + '\n';
        
        selected.forEach(node => {
            const data = node.data();
            const row = keys.map(k => {
                const v = data[k];
                if (v === undefined || v === null) return '';
                if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(6);
                return Utils.escapeCSV(v);
            });
            csv += row.join(',') + '\n';
        });
        
        this.downloadFile(csv, `selected_nodes_${Utils.getTimestamp()}.csv`, 'text/csv;charset=utf-8;');
        updateStatus(`Exported ${selected.length} nodes to CSV`, 'success');
    },

    /**
     * Export all nodes as CSV
     */
    exportAllAsCsv() {
        if (!State.cy) return;
        
        const nodes = State.cy.nodes();
        if (nodes.length === 0) {
            updateStatus('No nodes to export', 'error');
            return;
        }
        
        // Collect all keys
        const allKeys = new Set();
        nodes.forEach(node => {
            const data = node.data();
            Object.keys(data).forEach(k => {
                if (!['label', 'isNew'].includes(k) && typeof data[k] !== 'object') {
                    allKeys.add(k);
                }
            });
        });
        
        const keys = Array.from(allKeys);
        
        // Build CSV
        let csv = keys.map(Utils.escapeCSV).join(',') + '\n';
        
        nodes.forEach(node => {
            const data = node.data();
            const row = keys.map(k => {
                const v = data[k];
                if (v === undefined || v === null) return '';
                if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(6);
                return Utils.escapeCSV(v);
            });
            csv += row.join(',') + '\n';
        });
        
        this.downloadFile(csv, `all_nodes_${Utils.getTimestamp()}.csv`, 'text/csv;charset=utf-8;');
        updateStatus(`Exported ${nodes.length} nodes to CSV`, 'success');
    },

    /**
     * Export graph as JSON
     */
    exportGraphAsJson() {
        if (!State.cy) return;
        
        const json = State.cy.json();
        const data = JSON.stringify(json, null, 2);
        
        this.downloadFile(data, `graph_${Utils.getTimestamp()}.json`, 'application/json');
        updateStatus('Exported graph as JSON', 'success');
    },

    /**
     * Trigger file download
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
};