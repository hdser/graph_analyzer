/**
 * Info Panel Module
 * Node information display and neighbor navigation
 */

const InfoPanel = {
    // Navigation state
    history: [],           // Stack of previous selections [{ids: [...], hop: n}, ...]
    originNodes: null,     // Original starting nodes (for reference)
    currentHop: 0,         // How many hops from origin (positive = out, negative = in)
    
    /**
     * Show single node information
     */
    showNode(node) {
        const data = node.data();
        State.currentNodeData = data;
        State.currentEdgeData = null;
        
        DOMCache.nodeInfo.style.display = 'block';
        DOMCache.edgeInfo.style.display = 'none';
        DOMCache.multiInfo.style.display = 'none';
        DOMCache.infoPanel.style.display = 'flex';
        
        DOMCache.nodeId.textContent = data.id || 'N/A';
        
        // Build metrics HTML
        const metricsHtml = Object.entries(data)
            .filter(([k, v]) => !['id', 'label', 'isNew'].includes(k) && typeof v !== 'object')
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([k, v]) => `<div class="metric-row">
                <span class="metric-label">${k.replace(/_/g, ' ')}</span>
                <span class="metric-value">${Utils.formatNumber(v)}</span>
            </div>`)
            .join('');
        
        DOMCache.allMetrics.innerHTML = metricsHtml || '<div class="no-metrics">No metrics computed</div>';
        
        // Set as origin if not navigating
        if (!this.originNodes) {
            this.originNodes = [node.id()];
            this.currentHop = 0;
        }
        
        // Update neighbor counts (from loaded edges or will be 0)
        const incoming = node.incomers('node');
        const outgoing = node.outgoers('node');
        DOMCache.inCount.textContent = incoming.length;
        DOMCache.outCount.textContent = outgoing.length;
        
        this.buildNeighborList(DOMCache.neighborInList, incoming);
        this.buildNeighborList(DOMCache.neighborOutList, outgoing);
        
        this.updateNavState();
        this.switchTab('metrics');
    },

    /**
     * Build neighbor list HTML
     */
    buildNeighborList(container, neighbors) {
        const maxDisplay = 20;
        if (neighbors.length > 0) {
            container.innerHTML = neighbors.slice(0, maxDisplay)
                .map(n => `<div class="neighbor-item" data-id="${n.id()}">${n.id()}</div>`)
                .join('');
            if (neighbors.length > maxDisplay) {
                container.innerHTML += `<div class="neighbor-more">+${neighbors.length - maxDisplay} more</div>`;
            }
        } else {
            container.innerHTML = '<div class="no-neighbors">Load edges to see neighbors</div>';
        }
    },

    /**
     * Show edge information
     */
    showEdge(edge) {
        const data = edge.data();
        State.currentEdgeData = data;
        State.currentNodeData = null;
        
        DOMCache.nodeInfo.style.display = 'none';
        DOMCache.edgeInfo.style.display = 'block';
        DOMCache.multiInfo.style.display = 'none';
        DOMCache.infoPanel.style.display = 'flex';
        
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
                <span class="metric-label">${k.replace(/_/g, ' ')}</span>
                <span class="metric-value">${Utils.formatNumber(v)}</span>
            </div>`)
            .join('');
        
        DOMCache.edgeMetrics.innerHTML = metricsHtml;
    },

    /**
     * Show multi-select information
     */
    showMultiSelect(selected) {
        DOMCache.nodeInfo.style.display = 'none';
        DOMCache.edgeInfo.style.display = 'none';
        DOMCache.multiInfo.style.display = 'flex';  // MUST be flex for scrolling to work
        DOMCache.infoPanel.style.display = 'flex';
        
        const nodes = selected.filter('node');
        const edges = selected.filter('edge');
        
        document.getElementById('multi-node-count').textContent = nodes.length.toLocaleString();
        document.getElementById('multi-edge-count').textContent = edges.length.toLocaleString();
        
        // Set as origin if not navigating
        if (!this.originNodes && nodes.length > 0) {
            this.originNodes = nodes.map(n => n.id());
            this.currentHop = 0;
        }
        
        this.updateNavState();
        
        // Build selected nodes list
        const nodesList = document.getElementById('selected-nodes-list');
        if (nodesList) {
            const maxShow = 50;
            const nodeIds = nodes.map(n => n.id());
            nodesList.innerHTML = nodeIds.slice(0, maxShow)
                .map(id => `<div class="selected-node-item" data-id="${id}">${id}</div>`)
                .join('');
            if (nodeIds.length > maxShow) {
                nodesList.innerHTML += `<div class="neighbor-more">+${nodeIds.length - maxShow} more</div>`;
            }
        }
        
        if (nodes.length === 0) {
            const metricsList = document.getElementById('multi-metrics-list');
            if (metricsList) metricsList.innerHTML = '<div class="no-metrics">No nodes selected</div>';
            return;
        }
        
        // Aggregate metrics
        const stats = {};
        nodes.forEach(node => {
            Object.entries(node.data()).forEach(([k, v]) => {
                if (!['id', 'label', 'isNew'].includes(k) && typeof v === 'number' && !isNaN(v)) {
                    if (!stats[k]) stats[k] = { sum: 0, count: 0, min: Infinity, max: -Infinity };
                    stats[k].sum += v;
                    stats[k].count++;
                    stats[k].min = Math.min(stats[k].min, v);
                    stats[k].max = Math.max(stats[k].max, v);
                }
            });
        });
        
        const html = Object.keys(stats).sort().map(k => {
            const s = stats[k];
            return `<div class="multi-metric-group">
                <div class="multi-metric-name">${k.replace(/_/g, ' ')}</div>
                <div class="multi-metric-stats">
                    <span>Avg: ${Utils.formatNumber(s.sum / s.count)}</span>
                    <span>Min: ${Utils.formatNumber(s.min)}</span>
                    <span>Max: ${Utils.formatNumber(s.max)}</span>
                </div>
            </div>`;
        }).join('');
        
        const metricsList = document.getElementById('multi-metrics-list');
        if (metricsList) {
            metricsList.innerHTML = html || '<div class="no-metrics">No numeric metrics</div>';
        }
    },

    /**
     * Update navigation state display
     */
    updateNavState() {
        // Update back buttons
        const backBtns = document.querySelectorAll('.nav-back-btn');
        backBtns.forEach(btn => {
            btn.disabled = this.history.length === 0;
            btn.textContent = this.history.length > 0 ? `← Back (${this.history.length})` : '← Back';
        });
        
        // Update ALL origin info boxes (both in node-info and multi-info)
        const originInfoBoxes = document.querySelectorAll('.origin-info-box');
        originInfoBoxes.forEach(box => {
            if (this.originNodes && this.originNodes.length > 0) {
                const hopText = this.currentHop === 0 ? 'Origin' : 
                               (this.currentHop > 0 ? `+${this.currentHop} hops out` : `${Math.abs(this.currentHop)} hops in`);
                const nodeText = this.originNodes.length === 1 ? 
                               `From: ${this.originNodes[0].substring(0, 16)}...` : 
                               `From: ${this.originNodes.length} nodes`;
                box.innerHTML = `<span>${nodeText}</span><span class="hop-label">${hopText}</span>`;
                box.style.display = 'flex';
            } else {
                box.style.display = 'none';
            }
        });
    },

    /**
     * Push current selection to history
     */
    pushHistory() {
        if (!State.cy) return;
        const selected = State.cy.nodes(':selected');
        if (selected.length > 0) {
            this.history.push({
                ids: selected.map(n => n.id()),
                hop: this.currentHop
            });
            if (this.history.length > 20) this.history.shift();
        }
    },

    /**
     * Go back to previous selection
     */
    goBack() {
        if (!State.cy || this.history.length === 0) return;
        
        const prev = this.history.pop();
        this.currentHop = prev.hop;
        
        State.cy.batch(() => {
            State.cy.nodes().unselect();
            prev.ids.forEach(id => {
                const node = State.cy.getElementById(id);
                if (node.length) node.select();
            });
        });
        
        updateStatus(`Back (${prev.ids.length} nodes)`, 'info');
        this.updateNavState();
    },

    /**
     * Reset to origin
     */
    resetToOrigin() {
        if (!State.cy || !this.originNodes) return;
        
        this.pushHistory();
        this.currentHop = 0;
        
        State.cy.batch(() => {
            State.cy.nodes().unselect();
            this.originNodes.forEach(id => {
                const node = State.cy.getElementById(id);
                if (node.length) node.select();
            });
        });
        
        updateStatus(`Reset to origin (${this.originNodes.length} nodes)`, 'info');
        this.updateNavState();
    },

    /**
     * Navigate to neighbors - EXPANDS selection cumulatively
     * Each click adds more neighbors to the current selection
     */
    async goToNeighbors(direction) {
        if (!State.cy) return;
        
        const selected = State.cy.nodes(':selected');
        if (selected.length === 0) {
            updateStatus('Select node(s) first', 'info');
            return;
        }
        
        const currentCount = selected.length;
        
        // Try to get neighbors from loaded edges first
        let neighbors;
        if (direction === 'in') {
            neighbors = selected.incomers('node');
        } else {
            neighbors = selected.outgoers('node');
        }
        
        // If we have neighbors from loaded edges, use them
        if (neighbors.length > 0) {
            // Filter to only new neighbors (not already selected)
            const newNeighbors = neighbors.filter(n => !n.selected());
            
            if (newNeighbors.length === 0) {
                updateStatus(`No new ${direction === 'in' ? 'incoming' : 'outgoing'} neighbors to add`, 'info');
                return;
            }
            
            this.pushHistory();
            this.currentHop += (direction === 'out' ? 1 : -1);
            
            // ADD to selection (don't unselect existing)
            State.cy.batch(() => {
                newNeighbors.select();
            });
            
            const newTotal = State.cy.nodes(':selected').length;
            updateStatus(`+${newNeighbors.length} ${direction === 'in' ? 'in' : 'out'} (${newTotal} total)`, 'success');
            this.updateNavState();
            return;
        }
        
        // Otherwise, query backend
        if (!State.currentGraph) {
            updateStatus('No graph loaded', 'error');
            return;
        }
        
        updateStatus(`Querying ${direction} neighbors...`, 'info');
        
        try {
            const nodeIds = selected.map(n => n.id());
            const response = await fetch(`/api/network/graphs/${State.currentGraph}/neighbors`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ node_ids: nodeIds, direction: direction })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to get neighbors: ${response.status}`);
            }
            
            const data = await response.json();
            const neighborIds = direction === 'in' ? data.incoming : data.outgoing;
            
            // Filter to only new neighbors (not already selected)
            const currentSelectedIds = new Set(nodeIds);
            const newNeighborIds = neighborIds.filter(id => !currentSelectedIds.has(id));
            
            if (newNeighborIds.length === 0) {
                updateStatus(`No new ${direction === 'in' ? 'incoming' : 'outgoing'} neighbors to add`, 'info');
                return;
            }
            
            this.pushHistory();
            this.currentHop += (direction === 'out' ? 1 : -1);
            
            // ADD to selection (don't unselect existing)
            State.cy.batch(() => {
                newNeighborIds.forEach(id => {
                    const node = State.cy.getElementById(id);
                    if (node.length) node.select();
                });
            });
            
            const newTotal = State.cy.nodes(':selected').length;
            updateStatus(`+${newNeighborIds.length} ${direction === 'in' ? 'in' : 'out'} (${newTotal} total)`, 'success');
            this.updateNavState();
            
        } catch (err) {
            console.error('Neighbor query failed:', err);
            updateStatus('Failed to get neighbors - load edges first', 'error');
        }
    },

    /**
     * Clear navigation state (call when selecting new nodes manually)
     */
    clearNavigation() {
        this.history = [];
        this.originNodes = null;
        this.currentHop = 0;
        this.updateNavState();
    },

    /**
     * Set current selection as new origin
     */
    setAsOrigin() {
        if (!State.cy) return;
        const selected = State.cy.nodes(':selected');
        if (selected.length > 0) {
            this.originNodes = selected.map(n => n.id());
            this.history = [];
            this.currentHop = 0;
            updateStatus(`Set ${selected.length} node(s) as origin`, 'success');
            this.updateNavState();
        }
    },

    /**
     * Copy all selected node IDs
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
     * Switch tabs
     */
    switchTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.style.display = content.id === `${tabName}-tab` ? 'block' : 'none';
        });
    },

    /**
     * Close panel
     */
    close() {
        DOMCache.infoPanel.style.display = 'none';
        // Clear navigation so next selection becomes new origin
        this.clearNavigation();
    },

    /**
     * Setup event handlers
     */
    setupNeighborClicks() {
        // Back buttons
        document.querySelectorAll('.nav-back-btn').forEach(btn => {
            btn.addEventListener('click', () => this.goBack());
        });
        
        // Origin/reset buttons
        document.querySelectorAll('.nav-origin-btn').forEach(btn => {
            btn.addEventListener('click', () => this.resetToOrigin());
        });
        
        // Set origin buttons
        document.querySelectorAll('.nav-set-origin-btn').forEach(btn => {
            btn.addEventListener('click', () => this.setAsOrigin());
        });
        
        // In/Out navigation buttons
        document.querySelectorAll('.nav-in-btn').forEach(btn => {
            btn.addEventListener('click', () => this.goToNeighbors('in'));
        });
        document.querySelectorAll('.nav-out-btn').forEach(btn => {
            btn.addEventListener('click', () => this.goToNeighbors('out'));
        });
        
        // Copy selected IDs button
        document.getElementById('copy-all-selected-btn')?.addEventListener('click', () => this.copySelectedIds());
        
        // Tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });
        
        // Neighbor/selected node clicks
        document.addEventListener('click', (e) => {
            const item = e.target.closest('.neighbor-item, .selected-node-item');
            if (item && item.dataset.id) {
                Search.focusNode(item.dataset.id);
            }
        });
        
        // Close button
        document.querySelector('.close-btn')?.addEventListener('click', () => this.close());
    }
};