/**
 * Distributions Communication Module
 * Communication with distributions popup window
 * 
 * UPDATED: Now snapshot-aware - when viewing a snapshot, sends snapshot data
 * to the analysis popup instead of live network data.
 */

const DistributionsComm = {
    // Track active flash intervals so we can cancel them
    _activeFlashIntervals: [],

    /**
     * Setup message listener for popup requests
     */
    setup() {
        const self = this; // Capture reference to DistributionsComm
        
        window.addEventListener('message', (event) => {
            if (event.data.type === 'REQUEST_DISTRIBUTION_DATA') {
                self.sendData();
            } else if (event.data.type === 'LOCATE_NODE') {
                // Handle both nodeId formats
                const nodeId = event.data.nodeId || (event.data.data && event.data.data.nodeId);
                if (nodeId) {
                    self.locateNode(nodeId);
                }
            } else if (event.data.type === 'HIGHLIGHT_ANOMALIES') {
                const nodeIds = event.data.nodeIds || (event.data.data && event.data.data.nodeIds);
                if (nodeIds) {
                    self.highlightAnomalies(nodeIds);
                }
            } else if (event.data.type === 'ANOMALY_APPLIED') {
                // Refresh graph data after anomaly scores applied
                self.handleAnomalyApplied(event.data.data);
            } else if (event.data.type === 'PCA_APPLIED') {
                // Handle PCA scores applied
                self.handlePCAApplied(event.data.data);
            } else if (event.data.type === 'COMPOSITE_CREATED') {
                // Handle composite metric created
                self.handleCompositeCreated(event.data.data);
            } else if (event.data.type === 'TEMPORAL_APPLIED') {
                // Handle temporal metric applied
                self.handleTemporalApplied(event.data.data);
            } else if (event.data.type === 'CLEAR_SELECTION') {
                // Clear all selections in main graph
                console.log('[DistributionsComm] Received CLEAR_SELECTION message');
                self.clearSelection();
            } else if (event.data.type === 'CLEAR_HIGHLIGHTS') {
                // Clear all highlight classes
                console.log('[DistributionsComm] Received CLEAR_HIGHLIGHTS message');
                self.clearHighlights();
            } else if (event.data.type === 'SELECT_NODES') {
                // Select specific nodes
                const nodeIds = event.data.nodeIds || (event.data.data && event.data.data.nodeIds);
                if (nodeIds) {
                    self.selectNodes(nodeIds);
                }
            }
        });
        
        // Expose functions globally for direct calls from popup
        // Use regular functions bound to self to ensure correct 'this' context
        window.highlightAnomalies = function(nodeIds) { 
            self.highlightAnomalies(nodeIds); 
        };
        window.clearGraphSelection = function() { 
            console.log('[DistributionsComm] clearGraphSelection called directly');
            self.clearSelection(); 
        };
        window.clearGraphHighlights = function() { 
            console.log('[DistributionsComm] clearGraphHighlights called directly');
            self.clearHighlights(); 
        };
        
        console.log('[DistributionsComm] Setup complete, global functions exposed');
    },

    /**
     * Check if we're currently viewing a snapshot
     * @returns {string|null} Snapshot ID or null if viewing live network
     */
    getCurrentSnapshotId() {
        if (typeof Snapshots !== 'undefined' && Snapshots.getCurrentSnapshotId) {
            return Snapshots.getCurrentSnapshotId();
        }
        return null;
    },

    /**
     * Get snapshot info if viewing a snapshot
     * @returns {Object|null} Snapshot metadata or null
     */
    getCurrentSnapshotInfo() {
        if (typeof Snapshots !== 'undefined' && Snapshots.getCurrentSnapshot) {
            return Snapshots.getCurrentSnapshot();
        }
        return null;
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
     * Now snapshot-aware: sends snapshot info if viewing a snapshot
     */
    sendData() {
        if (!State.distributionsWindow || State.distributionsWindow.closed || !State.cy) return;
        
        // Check if we're viewing a snapshot
        const snapshotId = this.getCurrentSnapshotId();
        const snapshotInfo = this.getCurrentSnapshotInfo();
        
        // Collect node data from current Cytoscape view (works for both live and snapshot)
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
        
        // Build message payload
        const payload = {
            nodes,
            selectedIds,
            availableConfig: State.availableConfig,
            // Snapshot context
            isSnapshot: !!snapshotId,
            snapshotId: snapshotId,
            snapshotInfo: snapshotInfo
        };
        
        console.log('[DistributionsComm] Building payload, snapshotId:', snapshotId, 'isSnapshot:', payload.isSnapshot);
        
        // If we're viewing a snapshot, add additional context
        if (snapshotId && snapshotInfo) {
            // Extract base_sql_file from snapshot_id if not in info
            // Format: {base_sql_file}_block_{number}
            let baseSqlFile = snapshotInfo.base_sql_file;
            if (!baseSqlFile && snapshotId) {
                const match = snapshotId.match(/^(.+)_block_\d+$/);
                if (match) {
                    baseSqlFile = match[1];
                }
            }
            
            payload.snapshotContext = {
                blockNumber: snapshotInfo.block_number,
                blockTimestamp: snapshotInfo.block_timestamp,
                baseSqlFile: baseSqlFile,
                nodeCount: snapshotInfo.node_count,
                edgeCount: snapshotInfo.edge_count,
                metricsComputed: snapshotInfo.metrics_computed || [],
                label: snapshotInfo.label
            };
            console.log('[DistributionsComm] Sending snapshot data:', snapshotId);
        } else {
            console.log('[DistributionsComm] Sending live network data');
        }
        
        // Send message
        State.distributionsWindow.postMessage({
            type: 'DISTRIBUTION_DATA',
            data: payload
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
        
        // Flash the node for visibility (single node, so we can use flash)
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
        
        // Clear any existing flash intervals first
        this._clearAllFlashIntervals();
        
        // Clear current selection and highlights
        State.cy.elements().unselect();
        State.cy.elements().removeClass('anomaly');
        State.cy.nodes().removeStyle();
        
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
        
        // Don't flash nodes for large sets - it causes the style persistence issue
        // Only flash if there are few anomalies
        if (anomalyNodes.length <= 10) {
            anomalyNodes.forEach(node => {
                this._flashNode(node);
            });
        }
        
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
     * Clear all active flash intervals
     */
    _clearAllFlashIntervals() {
        console.log(`[DistributionsComm] Clearing ${this._activeFlashIntervals.length} flash intervals`);
        this._activeFlashIntervals.forEach(intervalId => {
            clearInterval(intervalId);
        });
        this._activeFlashIntervals = [];
    },

    /**
     * Clear all highlight classes from elements
     */
    clearHighlights() {
        console.log('[DistributionsComm] clearHighlights() called');
        
        if (!State.cy) {
            console.warn('[DistributionsComm] No Cytoscape instance available');
            return;
        }
        
        const cy = State.cy;
        
        // FIRST: Clear all active flash intervals to stop any ongoing animations
        this._clearAllFlashIntervals();
        
        // Get count of elements with highlight classes before clearing
        const anomalyCount = cy.elements('.anomaly').length;
        const highlightedCount = cy.elements('.highlighted').length;
        const searchedCount = cy.elements('.searched').length;
        console.log(`[DistributionsComm] Elements before clear: anomaly=${anomalyCount}, highlighted=${highlightedCount}, searched=${searchedCount}`);
        
        // Batch all operations for better performance
        cy.batch(() => {
            // Remove all highlight-related classes
            cy.elements().removeClass('highlighted');
            cy.elements().removeClass('anomaly');
            cy.elements().removeClass('searched');
            cy.elements().removeClass('new-node');
            
            // Remove any directly-set styles (from _flashNode, etc.)
            cy.nodes().removeStyle();
            cy.edges().removeStyle();
        });
        
        // Force style recalculation
        cy.style().update();
        
        // Force re-render by triggering a minimal viewport change
        // This invalidates the textureOnViewport cache
        const currentZoom = cy.zoom();
        cy.zoom(currentZoom * 1.0001);
        requestAnimationFrame(() => {
            cy.zoom(currentZoom);
        });
        
        // Reset neighbor highlight state
        State.neighborHighlightState = 0;
        
        // Verify classes were removed
        const anomalyCountAfter = cy.elements('.anomaly').length;
        console.log(`[DistributionsComm] Elements after clear: anomaly=${anomalyCountAfter}`);
        
        updateStatus('Highlights cleared', 'info');
        
        console.log('[DistributionsComm] Highlights cleared successfully');
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
     * Handle temporal metric being applied
     */
    handleTemporalApplied(data) {
        if (!data || !data.node_updates) return;
        
        const metricName = data.metric_name;
        console.log(`[DistributionsComm] Temporal metric applied: ${metricName}`);
        
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
        
        // Refresh distributions data
        this.sendData();
        
        // Update metric dropdowns
        if (typeof Metrics !== 'undefined' && Metrics.populateDropdowns && State.cy) {
            const nodes = State.cy.nodes().map(n => ({ data: n.data() }));
            Metrics.populateDropdowns(nodes, null);
        }
        
        updateStatus(`Applied temporal metric: ${metricName}`, 'success');
    },

    /**
     * Flash a node for visibility
     */
    _flashNode(node) {
        // Capture the style BEFORE any highlight classes are applied
        // We want the base style, not the highlighted style
        const baseStyle = {
            'border-width': '1px',
            'border-color': '#333333'
        };
        
        // Flash animation
        let flashCount = 0;
        const self = this;
        const flashInterval = setInterval(() => {
            if (flashCount >= 6) {
                clearInterval(flashInterval);
                // Remove from active intervals
                const idx = self._activeFlashIntervals.indexOf(flashInterval);
                if (idx > -1) {
                    self._activeFlashIntervals.splice(idx, 1);
                }
                // Don't set any style at the end - let the stylesheet handle it
                // Just remove any directly-set styles
                node.removeStyle();
                return;
            }
            
            const isHighlight = flashCount % 2 === 0;
            node.style({
                'border-width': isHighlight ? '4px' : baseStyle['border-width'],
                'border-color': isHighlight ? '#ff4d4f' : baseStyle['border-color']
            });
            
            flashCount++;
        }, 200);
        
        // Track this interval so we can cancel it if needed
        this._activeFlashIntervals.push(flashInterval);
    }
};