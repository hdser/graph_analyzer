/**
 * Distributions Communication Module
 * Communication with distributions popup window
 */

const DistributionsComm = {
    /**
     * Setup message listener for popup requests
     */
    setup() {
        window.addEventListener('message', (event) => {
            if (event.data.type === 'REQUEST_DISTRIBUTION_DATA') {
                this.sendData();
            } else if (event.data.type === 'LOCATE_NODE') {
                // Handle both nodeId formats
                const nodeId = event.data.nodeId || (event.data.data && event.data.data.nodeId);
                if (nodeId) {
                    this.locateNode(nodeId);
                }
            } else if (event.data.type === 'HIGHLIGHT_ANOMALIES') {
                const nodeIds = event.data.nodeIds || (event.data.data && event.data.data.nodeIds);
                if (nodeIds) {
                    this.highlightAnomalies(nodeIds);
                }
            } else if (event.data.type === 'ANOMALY_APPLIED') {
                // Refresh graph data after anomaly scores applied
                this.handleAnomalyApplied(event.data.data);
            } else if (event.data.type === 'PCA_APPLIED') {
                // Handle PCA scores applied
                this.handlePCAApplied(event.data.data);
            } else if (event.data.type === 'COMPOSITE_CREATED') {
                // Handle composite metric created
                this.handleCompositeCreated(event.data.data);
            } else if (event.data.type === 'CLEAR_SELECTION') {
                // Clear all selections in main graph
                this.clearSelection();
            } else if (event.data.type === 'CLEAR_HIGHLIGHTS') {
                // Clear all highlight classes
                this.clearHighlights();
            } else if (event.data.type === 'SELECT_NODES') {
                // Select specific nodes
                const nodeIds = event.data.nodeIds || (event.data.data && event.data.data.nodeIds);
                if (nodeIds) {
                    this.selectNodes(nodeIds);
                }
            }
        });
        
        // Also expose functions globally for direct calls from popup
        window.highlightAnomalies = (nodeIds) => this.highlightAnomalies(nodeIds);
        window.clearGraphSelection = () => this.clearSelection();
        window.clearGraphHighlights = () => this.clearHighlights();
    },

    /**
     * Open distributions popup window
     */
    open() {
        if (!State.cy) {
            updateStatus('Please load a graph first', 'error');
            return;
        }
        
        // Check if window already open
        if (State.distributionsWindow && !State.distributionsWindow.closed) {
            State.distributionsWindow.focus();
            this.sendData();
            return;
        }
        
        // Calculate window position
        const width = 1400;
        const height = 900;
        const left = (screen.width - width) / 2;
        const top = (screen.height - height) / 2;
        
        // Open popup
        State.distributionsWindow = window.open(
            '/static/distributions.html',
            'distributionsAnalysis',
            `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
        );
        
        if (!State.distributionsWindow) {
            updateStatus('Popup blocked. Please allow popups.', 'error');
            return;
        }
        
        // Send data when loaded
        State.distributionsWindow.addEventListener('load', () => {
            this.sendData();
        });
    },

    /**
     * Send node data to distributions window
     */
    sendData() {
        if (!State.distributionsWindow || State.distributionsWindow.closed || !State.cy) return;
        
        // Collect node data
        const nodes = [];
        State.cy.nodes().forEach(node => {
            const data = node.data();
            const cleanData = { id: data.id };
            
            // Include only numeric metrics
            Object.keys(data).forEach(key => {
                if (typeof data[key] === 'number' && !isNaN(data[key])) {
                    cleanData[key] = data[key];
                }
            });
            
            nodes.push(cleanData);
        });
        
        // Get selected node IDs
        const selectedIds = State.cy.nodes(':selected').map(n => n.id());
        
        // Send message
        State.distributionsWindow.postMessage({
            type: 'DISTRIBUTION_DATA',
            data: {
                nodes,
                selectedIds,
                availableConfig: State.availableConfig
            }
        }, '*');
    },

    /**
     * Send selection update to distributions window
     */
    sendSelectionUpdate() {
        if (!State.distributionsWindow || State.distributionsWindow.closed || !State.cy) return;
        
        const selectedIds = State.cy.nodes(':selected').map(n => n.id());
        
        State.distributionsWindow.postMessage({
            type: 'SELECTION_UPDATE',
            data: { selectedIds }
        }, '*');
    },

    /**
     * Locate and focus on a specific node
     */
    locateNode(nodeId) {
        if (!State.cy) return;
        
        const node = State.cy.getElementById(nodeId);
        if (!node || node.empty()) {
            console.warn(`[DistributionsComm] Node not found: ${nodeId}`);
            updateStatus(`Node ${nodeId} not found`, 'error');
            return;
        }
        
        // Clear current selection
        State.cy.elements().unselect();
        
        // Select the target node
        node.select();
        
        // Fit view to the node with some padding
        State.cy.animate({
            fit: {
                eles: node,
                padding: 150
            },
            duration: 500,
            easing: 'ease-out-cubic'
        });
        
        // Flash the node for visibility
        this._flashNode(node);
        
        // Update info panel
        if (typeof InfoPanel !== 'undefined' && InfoPanel.show) {
            InfoPanel.show(nodeId);
        }
        
        console.log(`[DistributionsComm] Located node: ${nodeId}`);
    },

    /**
     * Highlight multiple anomaly nodes
     */
    highlightAnomalies(nodeIds) {
        if (!State.cy || !nodeIds || nodeIds.length === 0) return;
        
        console.log(`[DistributionsComm] Highlighting ${nodeIds.length} anomalies`);
        
        // Clear current selection and highlights
        State.cy.elements().unselect();
        State.cy.elements().removeClass('anomaly');
        
        // Convert nodeIds to strings for comparison
        const nodeIdSet = new Set(nodeIds.map(String));
        
        // Find and select all anomaly nodes
        const anomalyNodes = State.cy.nodes().filter(node => {
            const id = String(node.id());
            const dataId = String(node.data('id') || '');
            return nodeIdSet.has(id) || nodeIdSet.has(dataId);
        });
        
        if (anomalyNodes.empty()) {
            updateStatus('No anomaly nodes found in current view', 'warning');
            return;
        }
        
        // Add anomaly class and select
        anomalyNodes.addClass('anomaly');
        anomalyNodes.select();
        
        // Fit view to show all anomalies
        State.cy.animate({
            fit: {
                eles: anomalyNodes,
                padding: 50
            },
            duration: 500,
            easing: 'ease-out-cubic'
        });
        
        // Flash each node briefly
        anomalyNodes.forEach(node => {
            this._flashNode(node);
        });
        
        updateStatus(`Highlighted ${anomalyNodes.length} anomalies`, 'success');
    },

    /**
     * Select specific nodes by IDs
     */
    selectNodes(nodeIds) {
        if (!State.cy || !nodeIds || nodeIds.length === 0) return;
        
        console.log(`[DistributionsComm] Selecting ${nodeIds.length} nodes`);
        
        // Clear current selection
        State.cy.elements().unselect();
        
        // Convert nodeIds to strings
        const nodeIdSet = new Set(nodeIds.map(String));
        
        // Find and select nodes
        const nodesToSelect = State.cy.nodes().filter(node => {
            const id = String(node.id());
            const dataId = String(node.data('id') || '');
            return nodeIdSet.has(id) || nodeIdSet.has(dataId);
        });
        
        if (!nodesToSelect.empty()) {
            nodesToSelect.select();
            
            // Fit view if more than one node
            if (nodesToSelect.length > 1) {
                State.cy.animate({
                    fit: {
                        eles: nodesToSelect,
                        padding: 50
                    },
                    duration: 500,
                    easing: 'ease-out-cubic'
                });
            }
            
            updateStatus(`Selected ${nodesToSelect.length} nodes`, 'success');
        }
    },

    /**
     * Clear all selections in the graph
     */
    clearSelection() {
        if (!State.cy) return;
        
        State.cy.elements().unselect();
        updateStatus('Selection cleared', 'info');
        
        // Notify distributions window of the change
        this.sendSelectionUpdate();
        
        console.log('[DistributionsComm] Selection cleared');
    },

    /**
     * Clear all highlight classes from elements
     */
    clearHighlights() {
        if (!State.cy) return;
        
        // Remove all highlight-related classes
        State.cy.elements().removeClass('highlighted');
        State.cy.elements().removeClass('anomaly');
        State.cy.elements().removeClass('searched');
        State.cy.elements().removeClass('new-node');
        
        // Reset neighbor highlight state
        State.neighborHighlightState = 0;
        
        updateStatus('Highlights cleared', 'info');
        
        console.log('[DistributionsComm] Highlights cleared');
    },

    /**
     * Handle anomaly scores being applied to graph
     */
    handleAnomalyApplied(data) {
        if (!data || !data.node_updates) return;
        
        console.log(`[DistributionsComm] Applying ${data.node_updates.length} node updates`);
        
        // Update node data
        data.node_updates.forEach(update => {
            const node = State.cy.getElementById(update.id);
            if (node && !node.empty()) {
                // Update node data with new metrics
                Object.keys(update).forEach(key => {
                    if (key !== 'id') {
                        node.data(key, update[key]);
                    }
                });
            }
        });
        
        // Refresh distributions data
        this.sendData();
        
        updateStatus(`Applied ${data.metric_name} to graph`, 'success');
    },

    /**
     * Handle PCA scores being applied to graph
     */
    handlePCAApplied(data) {
        if (!data || !data.node_updates) return;
        
        console.log(`[DistributionsComm] Applying PCA scores to ${data.node_updates.length} nodes`);
        
        // Update node data
        data.node_updates.forEach(update => {
            const node = State.cy.getElementById(update.id);
            if (node && !node.empty()) {
                Object.keys(update).forEach(key => {
                    if (key !== 'id') {
                        node.data(key, update[key]);
                    }
                });
            }
        });
        
        // Refresh distributions data
        this.sendData();
        
        updateStatus(`Applied PCA scores (${data.n_components} components) to graph`, 'success');
    },

    /**
     * Handle composite metric being created
     */
    handleCompositeCreated(data) {
        if (!data || !data.node_updates) return;
        
        // Get the metric name from either field
        const metricName = data.name || data.metric_name;
        console.log(`[DistributionsComm] Composite created: ${metricName}`);
        
        // Update node data in Cytoscape
        let updatedCount = 0;
        data.node_updates.forEach(update => {
            const node = State.cy?.getElementById(update.id);
            if (node && !node.empty()) {
                Object.keys(update).forEach(key => {
                    if (key !== 'id') {
                        node.data(key, update[key]);
                        updatedCount++;
                    }
                });
            }
        });
        
        console.log(`[DistributionsComm] Updated ${updatedCount} node attributes`);
        
        // Refresh distributions data so popup has the new metric
        this.sendData();
        
        // Update metric dropdowns in main window (filter, style, etc.)
        if (typeof Metrics !== 'undefined' && Metrics.populateDropdowns && State.cy) {
            const nodes = State.cy.nodes().map(n => ({ data: n.data() }));
            Metrics.populateDropdowns(nodes, null);
            console.log('[DistributionsComm] Refreshed metric dropdowns');
        }
        
        // Refresh composite metrics panel if available
        if (typeof CompositeMetrics !== 'undefined' && CompositeMetrics.loadSaved) {
            CompositeMetrics.loadSaved();
        }
        
        updateStatus(`Created composite: ${metricName}`, 'success');
    },

    /**
     * Flash a node for visibility
     */
    _flashNode(node) {
        const originalStyle = {
            'border-width': node.style('border-width'),
            'border-color': node.style('border-color')
        };
        
        // Flash animation
        let flashCount = 0;
        const flashInterval = setInterval(() => {
            if (flashCount >= 6) {
                clearInterval(flashInterval);
                node.style({
                    'border-width': originalStyle['border-width'],
                    'border-color': originalStyle['border-color']
                });
                return;
            }
            
            const isHighlight = flashCount % 2 === 0;
            node.style({
                'border-width': isHighlight ? '4px' : originalStyle['border-width'],
                'border-color': isHighlight ? '#ff4d4f' : originalStyle['border-color']
            });
            
            flashCount++;
        }, 200);
    }
};