/**
 * Auto Reload Module
 * Auto-reload SSE system for background data refresh
 */

const AutoReload = {
    /**
     * Setup auto-reload event handlers
     */
    setup() {
        // Toggle handler
        DOMCache.autoReloadToggle?.addEventListener('change', (e) => this.handleToggle(e));
        
        // Interval change handler
        DOMCache.reloadInterval?.addEventListener('change', () => {
            if (State.autoReloadEnabled) {
                this.handleToggle({ target: { checked: true } });
            }
        });
        
        // Compute metrics toggle handler
        DOMCache.reloadComputeMetrics?.addEventListener('change', () => {
            if (State.autoReloadEnabled) {
                this.handleToggle({ target: { checked: true } });
            }
        });
        
        // Initialize indicator
        this.updateIndicator('disabled');
    },

    /**
     * Handle auto-reload toggle (admin mode)
     */
    async handleToggle(e) {
        const enabled = e.target.checked;
        
        if (enabled) {
            const selectedFiles = Array.from(document.querySelectorAll('input[name="sql-file"]:checked'))
                .map(cb => cb.value);
            
            if (selectedFiles.length === 0) {
                Toast.error('Select SQL files first');
                DOMCache.autoReloadToggle.checked = false;
                return;
            }
            
            const propertiesFiles = Array.from(
                document.querySelectorAll('input[name="properties-file"]:checked')
            ).map(cb => cb.value);
            
            try {
                const status = await API.startAutoReload({
                    enabled: true,
                    interval_seconds: parseInt(DOMCache.reloadInterval.value) || 300,
                    sql_files: selectedFiles,
                    node_properties_files: propertiesFiles,
                    preserve_layout: true,
                    compute_metrics: DOMCache.reloadComputeMetrics?.checked || false,
                    metrics_mode: 'basic'
                });
                
                State.autoReloadEnabled = true;
                this.updateUI(status);
                this.connectSSE();
                Toast.success('Auto-reload enabled');
                
            } catch (err) {
                console.error('Auto-reload start error:', err);
                Toast.error('Failed to start auto-reload: ' + err.message);
                DOMCache.autoReloadToggle.checked = false;
            }
            
        } else {
            try {
                await API.stopAutoReload();
                State.autoReloadEnabled = false;
                this.disconnectSSE();
                this.updateIndicator('disabled');
                if (DOMCache.reloadStatusText) DOMCache.reloadStatusText.textContent = 'Disabled';
                if (DOMCache.nextReloadTime) DOMCache.nextReloadTime.textContent = '-';
                Toast.info('Auto-reload disabled');
            } catch (err) {
                console.error('Auto-reload stop error:', err);
            }
        }
    },

    /**
     * Start auto-reload in production mode (called automatically)
     * NOTE: compute_metrics is FALSE - new nodes come in without metrics
     * Users can run metrics manually when needed
     */
    async startProductionMode(config) {
        if (!config.default_sql_files || config.default_sql_files.length === 0) {
            console.warn('No default SQL files configured for auto-reload');
            return;
        }
        
        try {
            const status = await API.startAutoReload({
                enabled: true,
                interval_seconds: config.auto_reload_interval || 300,
                sql_files: config.default_sql_files,
                node_properties_files: config.default_properties_files || [],
                preserve_layout: true,
                compute_metrics: false,  // Do NOT auto-compute metrics
                metrics_mode: 'basic'
            });
            
            State.autoReloadEnabled = true;
            this.connectSSE();
            console.log(`[AUTO-RELOAD] Started with ${config.auto_reload_interval}s interval (no metrics)`);
            
        } catch (err) {
            console.error('Auto-reload start error:', err);
        }
    },

    /**
     * Connect to SSE event stream
     */
    connectSSE() {
        if (State.autoReloadSSE) {
            State.autoReloadSSE.close();
        }
        
        State.autoReloadSSE = API.createAutoReloadSSE();
        
        State.autoReloadSSE.addEventListener('status_update', (e) => {
            this.updateUI(JSON.parse(e.data));
        });
        
        State.autoReloadSSE.addEventListener('reload_started', () => {
            this.updateIndicator('loading');
            if (DOMCache.reloadStatusText) {
                DOMCache.reloadStatusText.textContent = 'Reloading...';
            }
            Toast.info('Background reload started...');
        });
        
        State.autoReloadSSE.addEventListener('reload_complete', (e) => {
            this.handleComplete(JSON.parse(e.data));
        });
        
        State.autoReloadSSE.addEventListener('reload_error', (e) => {
            const data = JSON.parse(e.data);
            this.updateIndicator('error');
            Toast.error('Reload error: ' + data.error);
        });
        
        State.autoReloadSSE.onerror = () => {
            this.updateIndicator('error');
        };
    },

    /**
     * Disconnect from SSE event stream
     */
    disconnectSSE() {
        if (State.autoReloadSSE) {
            State.autoReloadSSE.close();
            State.autoReloadSSE = null;
        }
    },

    /**
     * Update UI from status object
     */
    updateUI(status) {
        if (!status) return;
        
        this.updateIndicator(status.enabled ? 'active' : 'disabled');
        
        if (DOMCache.reloadStatusText) {
            DOMCache.reloadStatusText.textContent = status.enabled ? 'Active' : 'Disabled';
        }
        
        if (status.last_reload_time && DOMCache.lastReloadTime) {
            const lastTime = new Date(status.last_reload_time);
            DOMCache.lastReloadTime.textContent = lastTime.toLocaleTimeString();
            
            const diff = Math.floor((Date.now() - lastTime.getTime()) / 1000);
            if (DOMCache.lastReloadDiff) {
                DOMCache.lastReloadDiff.textContent = this.formatTimeDiff(diff);
            }
        }
        
        if (status.next_reload_time && DOMCache.nextReloadTime) {
            DOMCache.nextReloadTime.textContent = new Date(status.next_reload_time).toLocaleTimeString();
        }
    },

    /**
     * Update reload indicator state
     */
    updateIndicator(state) {
        if (!DOMCache.reloadIndicator) return;
        DOMCache.reloadIndicator.className = 'auto-reload-indicator ' + state;
    },

    /**
     * Format time difference for display
     */
    formatTimeDiff(seconds) {
        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        return `${Math.floor(seconds / 3600)}h ago`;
    },

    /**
     * Handle reload complete event - refresh the graph display
     */
    async handleComplete(data) {
        this.updateIndicator('active');
        
        if (DOMCache.reloadStatusText) {
            DOMCache.reloadStatusText.textContent = 'Active';
        }
        if (DOMCache.lastReloadTime) {
            DOMCache.lastReloadTime.textContent = new Date().toLocaleTimeString();
        }
        
        const changeText = data.nodes_added > 0 || data.nodes_removed > 0 
            ? `+${data.nodes_added}/-${data.nodes_removed} nodes` 
            : 'No changes';
        
        Toast.success(`Reload complete: ${changeText}`);
        
        // Refresh the graph display
        if (State.currentGraph && State.cy) {
            await this.refreshGraphData(data);
        }
        
        // Update distributions if open
        DistributionsComm.sendData();
    },

    /**
     * Refresh graph data after auto-reload
     */
    async refreshGraphData(data) {
        try {
            if (data.nodes_added > 0 || data.nodes_removed > 0) {
                await this.fullGraphRefresh();
            } else {
                await this.incrementalMetricsUpdate();
            }
        } catch (error) {
            console.error('Auto-reload refresh error:', error);
            Toast.error('Failed to refresh graph display');
        }
    },

    /**
     * Full graph refresh (for structural changes)
     */
    async fullGraphRefresh() {
        Toast.info('Refreshing graph display...');
        
        try {
            const nodesData = await API.getGraphElements(State.currentGraph, 'nodes_only');
            
            const pan = State.cy.pan();
            const zoom = State.cy.zoom();
            const selectedIds = State.cy.nodes(':selected').map(n => n.id());
            
            const hadEdges = State.cy.edges().length > 0;
            const currentEdges = hadEdges ? State.cy.edges().jsons() : [];
            
            State.cy.batch(() => {
                State.cy.nodes().remove();
                State.cy.add(nodesData.elements);
                
                if (currentEdges.length > 0) {
                    const nodeIds = new Set(nodesData.elements.map(e => e.data.id));
                    const validEdges = currentEdges.filter(e => 
                        nodeIds.has(e.data.source) && nodeIds.has(e.data.target)
                    );
                    State.cy.add(validEdges);
                }
            });
            
            State.cy.viewport({ pan, zoom });
            
            if (selectedIds.length > 0) {
                State.cy.batch(() => {
                    selectedIds.forEach(id => {
                        const node = State.cy.getElementById(id);
                        if (node.length > 0) {
                            node.select();
                        }
                    });
                });
            }
            
            DOMCache.nodeCount.textContent = `${State.cy.nodes().length} nodes`;
            DOMCache.edgeCount.textContent = `${State.cy.edges().length} edges`;
            
            Metrics.populateDropdowns(nodesData.elements, null);
            
            if (!State.performanceMode) {
                CytoscapeManager.updateStyle();
            }
            
            Toast.success('Graph display refreshed');
            
        } catch (error) {
            console.error('Full graph refresh error:', error);
            throw error;
        }
    },

    /**
     * Incremental metrics update (no structural changes)
     */
    async incrementalMetricsUpdate() {
        try {
            const response = await API.getNodeUpdates(State.currentGraph);
            const updates = response.updates || [];
            
            if (updates.length === 0) return;
            
            State.cy.batch(() => {
                updates.forEach(update => {
                    const node = State.cy.getElementById(update.id);
                    if (node.length > 0) {
                        const { position, ...dataUpdates } = update;
                        node.data(dataUpdates);
                    }
                });
            });
            
            if (!State.performanceMode) {
                CytoscapeManager.updateStyle();
            }
            
            console.log(`[AUTO-RELOAD] Updated ${updates.length} nodes incrementally`);
            
        } catch (error) {
            console.error('Incremental update error:', error);
            await this.fullGraphRefresh();
        }
    }
};