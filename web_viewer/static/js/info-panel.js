/**
 * Info Panel Module
 * Node and edge information panel management
 */

const InfoPanel = {
    /**
     * Show node information
     */
    showNode(node) {
        const data = node.data();
        State.currentNodeData = data;
        State.currentEdgeData = null;
        
        // Show node panel, hide others
        DOMCache.nodeInfo.style.display = 'block';
        DOMCache.edgeInfo.style.display = 'none';
        DOMCache.multiInfo.style.display = 'none';
        DOMCache.infoPanel.style.display = 'flex';
        
        // Set node ID
        DOMCache.nodeId.textContent = data.id || 'N/A';
        
        // Build metrics HTML
        const metricsHtml = Object.entries(data)
            .filter(([k, v]) => !['id', 'label', 'isNew'].includes(k) && typeof v !== 'object')
            .map(([k, v]) => {
                const formattedValue = Utils.formatNumber(v);
                return `<div class="metric-row">
                    <span class="metric-label">${k}</span>
                    <span class="metric-value">${formattedValue}</span>
                </div>`;
            })
            .join('');
        
        DOMCache.allMetrics.innerHTML = metricsHtml || '<div class="no-metrics">No metrics computed</div>';
        
        // Update neighbors
        if (State.cy) {
            const incoming = node.incomers('node');
            const outgoing = node.outgoers('node');
            
            DOMCache.inCount.textContent = incoming.length;
            DOMCache.outCount.textContent = outgoing.length;
            
            const maxDisplay = 20;
            
            // Incoming neighbors
            DOMCache.neighborInList.innerHTML = incoming.slice(0, maxDisplay)
                .map(n => `<div class="neighbor-item" data-id="${n.id()}">${n.id()}</div>`)
                .join('');
            if (incoming.length > maxDisplay) {
                DOMCache.neighborInList.innerHTML += `<div class="neighbor-more">+${incoming.length - maxDisplay} more</div>`;
            }
            
            // Outgoing neighbors
            DOMCache.neighborOutList.innerHTML = outgoing.slice(0, maxDisplay)
                .map(n => `<div class="neighbor-item" data-id="${n.id()}">${n.id()}</div>`)
                .join('');
            if (outgoing.length > maxDisplay) {
                DOMCache.neighborOutList.innerHTML += `<div class="neighbor-more">+${outgoing.length - maxDisplay} more</div>`;
            }
        }
        
        this.switchTab('metrics');
    },

    /**
     * Show edge information
     */
    showEdge(edge) {
        const data = edge.data();
        State.currentEdgeData = data;
        State.currentNodeData = null;
        
        // Show edge panel, hide others
        DOMCache.nodeInfo.style.display = 'none';
        DOMCache.edgeInfo.style.display = 'block';
        DOMCache.multiInfo.style.display = 'none';
        DOMCache.infoPanel.style.display = 'flex';
        
        // Build metrics HTML
        const metricsHtml = `
            <div class="metric-row">
                <span class="metric-label">Source</span>
                <span class="metric-value">${data.source}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Target</span>
                <span class="metric-value">${data.target}</span>
            </div>
        ` + Object.entries(data)
            .filter(([k]) => !['id', 'source', 'target'].includes(k))
            .map(([k, v]) => `<div class="metric-row">
                <span class="metric-label">${k}</span>
                <span class="metric-value">${Utils.formatNumber(v)}</span>
            </div>`)
            .join('');
        
        DOMCache.edgeMetrics.innerHTML = metricsHtml;
    },

    /**
     * Show multi-select information (aggregated)
     */
    showMultiSelect(selected) {
        DOMCache.nodeInfo.style.display = 'none';
        DOMCache.edgeInfo.style.display = 'none';
        DOMCache.multiInfo.style.display = 'block';
        DOMCache.infoPanel.style.display = 'flex';
        
        // Update count
        document.getElementById('multi-count').textContent = selected.length;
        
        // Aggregate metrics
        const metricSums = {};
        const metricCounts = {};
        const metricMins = {};
        const metricMaxs = {};
        
        selected.forEach(node => {
            Object.entries(node.data()).forEach(([k, v]) => {
                if (!['id', 'label', 'isNew'].includes(k) && typeof v === 'number' && !isNaN(v)) {
                    if (!(k in metricSums)) {
                        metricSums[k] = 0;
                        metricCounts[k] = 0;
                        metricMins[k] = Infinity;
                        metricMaxs[k] = -Infinity;
                    }
                    metricSums[k] += v;
                    metricCounts[k]++;
                    metricMins[k] = Math.min(metricMins[k], v);
                    metricMaxs[k] = Math.max(metricMaxs[k], v);
                }
            });
        });
        
        // Build HTML
        const html = Object.keys(metricSums).map(k => `
            <div class="multi-metric-group">
                <div class="multi-metric-name">${k}</div>
                <div class="multi-metric-stats">
                    <span>Avg: ${Utils.formatNumber(metricSums[k] / metricCounts[k])}</span>
                    <span>Min: ${Utils.formatNumber(metricMins[k])}</span>
                    <span>Max: ${Utils.formatNumber(metricMaxs[k])}</span>
                    <span>Sum: ${Utils.formatNumber(metricSums[k])}</span>
                </div>
            </div>
        `).join('');
        
        DOMCache.multiMetricsList.innerHTML = html || '<div class="no-metrics">No numeric metrics</div>';
    },

    /**
     * Switch between tabs
     */
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });
    },

    /**
     * Close info panel
     */
    close() {
        DOMCache.infoPanel.style.display = 'none';
    },

    /**
     * Setup neighbor click handlers
     */
    setupNeighborClicks() {
        // Incoming neighbors
        DOMCache.neighborInList?.addEventListener('click', (e) => {
            const item = e.target.closest('.neighbor-item');
            if (item) {
                Search.focusNode(item.dataset.id);
            }
        });
        
        // Outgoing neighbors
        DOMCache.neighborOutList?.addEventListener('click', (e) => {
            const item = e.target.closest('.neighbor-item');
            if (item) {
                Search.focusNode(item.dataset.id);
            }
        });
    }
};