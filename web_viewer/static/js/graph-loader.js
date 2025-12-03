/**
 * Graph Loader Module
 * Graph loading and edge management
 */

const GraphLoader = {
    /**
     * Load graphs from selected SQL files
     */
    async loadGraphs() {
        const selectedFiles = Array.from(document.querySelectorAll('input[name="sql-file"]:checked'))
            .map(cb => cb.value);
        
        if (selectedFiles.length === 0) {
            updateStatus('Select at least one SQL file', 'error');
            return;
        }
        
        // Get selected properties files
        const propertiesFiles = Array.from(
            document.querySelectorAll('input[name="properties-file"]:checked')
        ).map(cb => cb.value);
        
        const skipSql = document.getElementById('skip-sql')?.checked || false;
        const useCachedLayout = document.getElementById('use-cached-layout')?.checked !== false;
        const computeMetrics = document.getElementById('compute-metrics-on-load')?.checked !== false;
        const metricsMode = computeMetrics ? 'basic' : 'skip';
        
        const loadBtn = document.getElementById('load-btn');
        loadBtn.disabled = true;
        loadBtn.textContent = 'Loading...';
        
        DOMCache.loading.style.display = 'flex';
        updateStatus('Loading networks...', 'info');
        
        try {
            const result = await API.loadNetwork({
                sql_files: selectedFiles,
                node_properties_files: propertiesFiles,
                skip_sql: skipSql,
                use_cached_layout: useCachedLayout,
                metrics_mode: metricsMode
            });
            
            State.currentState = result;
            
            // Populate graph selector and show it
            const graphSelector = document.getElementById('graph-selector');
            const graphSelect = document.getElementById('graph-select');
            graphSelect.innerHTML = '<option value="">Select graph...</option>' +
                result.loaded_graphs.map(g => `<option value="${g}">${g}</option>`).join('');
            
            // Show the graph selector dropdown
            if (graphSelector) {
                graphSelector.style.display = 'block';
            }
            
            // Enable metrics button
            const metricsBtn = document.getElementById('metrics-btn');
            if (metricsBtn) {
                metricsBtn.disabled = false;
            }
            
            // Initialize graph data cache
            result.loaded_graphs.forEach(g => {
                State.graphData[g] = { loaded: false };
            });
            
            // Auto-select first graph
            if (result.loaded_graphs.length > 0) {
                graphSelect.value = result.loaded_graphs[0];
                await this.displayGraph(result.loaded_graphs[0]);
            }
            
            // Build status message
            let statusMsg = `Loaded ${result.node_count} nodes, ${result.edge_count} edges`;
            if (result.node_properties_loaded && result.node_properties_loaded.length > 0) {
                statusMsg += ` (+${result.node_properties_loaded.length} properties)`;
            }
            updateStatus(statusMsg, 'success');
            
        } catch (error) {
            console.error('Load error:', error);
            updateStatus('Load failed: ' + error.message, 'error');
        } finally {
            loadBtn.disabled = false;
            loadBtn.textContent = 'Load';
            DOMCache.loading.style.display = 'none';
        }
    },

    /**
     * Display a specific graph
     */
    async displayGraph(graphId) {
        State.currentGraph = graphId;
        State.edgesLoading = false;
        
        if (DOMCache.edgesProgress) {
            DOMCache.edgesProgress.textContent = '';
        }
        
        try {
            // Initialize Cytoscape if needed
            if (!State.cy) {
                CytoscapeManager.initializeCytoscape(DOMCache.cyContainer);
            }
            
            // Load nodes first (fast)
            const nodesData = await API.getGraphElements(graphId, 'nodes_only');
            
            // Clear and add nodes
            State.cy.elements().remove();
            State.cy.add(nodesData.elements);
            State.cy.fit();
            
            // Update counts
            DOMCache.nodeCount.textContent = `${State.cy.nodes().length} nodes`;
            DOMCache.edgeCount.textContent = '0 edges';
            
            // Update load edges button
            if (DOMCache.loadEdgesBtn) {
                DOMCache.loadEdgesBtn.textContent = 'Load Edges';
                DOMCache.loadEdgesBtn.disabled = false;
            }
            
            // Populate metric dropdowns
            Metrics.populateDropdowns(nodesData.elements, null);
            
            // Send data to distributions window
            DistributionsComm.sendData();
            
            updateStatus(`Graph displayed: ${State.cy.nodes().length} nodes (edges not loaded)`, 'success');
            
        } catch (error) {
            console.error('Display error:', error);
            updateStatus('Failed to display graph: ' + error.message, 'error');
        }
    },

    /**
     * Load edges incrementally in batches
     */
    async loadEdgesIncrementally(graphId) {
        if (!State.cy) return;
        
        // If edges already loaded, clear them instead
        if (State.cy.edges().length > 0) {
            this.clearEdges();
            return;
        }
        
        if (State.edgesLoading) return;
        
        State.edgesLoading = true;
        const BATCH_SIZE = 50000;
        let offset = 0;
        let totalLoaded = 0;
        let hasMore = true;
        
        if (DOMCache.loadEdgesBtn) {
            DOMCache.loadEdgesBtn.disabled = true;
            DOMCache.loadEdgesBtn.textContent = 'Loading...';
        }
        
        try {
            while (hasMore && State.edgesLoading) {
                const result = await API.getGraphEdges(graphId, offset, BATCH_SIZE);
                
                if (result.edges && result.edges.length > 0) {
                    // Add edges in batch
                    State.cy.batch(() => {
                        State.cy.add(result.edges);
                    });
                    
                    totalLoaded += result.edges.length;
                    offset = totalLoaded;
                    
                    // Update progress
                    if (DOMCache.edgesProgress) {
                        DOMCache.edgesProgress.textContent = 
                            `${totalLoaded.toLocaleString()} / ${result.total.toLocaleString()}`;
                    }
                    DOMCache.edgeCount.textContent = `${totalLoaded.toLocaleString()} edges`;
                    
                    hasMore = result.has_more;
                } else {
                    hasMore = false;
                }
            }
            
            DOMCache.edgeCount.textContent = `${totalLoaded.toLocaleString()} edges`;
            if (DOMCache.edgesProgress) {
                DOMCache.edgesProgress.textContent = '';
            }
            
            updateStatus(`Loaded ${totalLoaded.toLocaleString()} edges`, 'success');
            
        } catch (error) {
            console.error('Edge loading error:', error);
            updateStatus('Edge loading failed: ' + error.message, 'error');
        } finally {
            State.edgesLoading = false;
            if (DOMCache.loadEdgesBtn) {
                DOMCache.loadEdgesBtn.disabled = false;
                DOMCache.loadEdgesBtn.textContent = State.cy.edges().length > 0 ? 'Clear Edges' : 'Load Edges';
            }
        }
    },

    /**
     * Clear all edges from the graph
     */
    clearEdges() {
        if (!State.cy) return;
        
        const edgeCount = State.cy.edges().length;
        
        State.cy.batch(() => {
            State.cy.edges().remove();
        });
        
        DOMCache.edgeCount.textContent = '0 edges';
        if (DOMCache.edgesProgress) {
            DOMCache.edgesProgress.textContent = '';
        }
        
        if (DOMCache.loadEdgesBtn) {
            DOMCache.loadEdgesBtn.textContent = 'Load Edges';
        }
        
        updateStatus(`Cleared ${edgeCount.toLocaleString()} edges`, 'success');
    },

    /**
     * Stop edge loading
     */
    stopEdgeLoading() {
        State.edgesLoading = false;
    },

    /**
     * Refresh current graph
     */
    async refreshCurrentGraph() {
        if (State.currentGraph) {
            await this.displayGraph(State.currentGraph);
        }
    }
};