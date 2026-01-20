/**
 * Graph Loader Module
 * Graph loading and edge management with renderer abstraction
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
     * Display a specific graph using the appropriate renderer
     */
    async displayGraph(graphId) {
        State.currentGraph = graphId;
        State.edgesLoading = false;
        
        if (DOMCache.edgesProgress) {
            DOMCache.edgesProgress.textContent = '';
        }
        
        try {
            // Load nodes first (fast)
            const nodesData = await API.getGraphElements(graphId, 'nodes_only');
            
            // Count nodes to determine renderer
            const nodeElements = nodesData.elements.filter(e => e.group === 'nodes');
            const nodeCount = nodeElements.length;
            
            // Convert from Cytoscape format to unified format
            const nodes = nodeElements.map(e => ({
                id: e.data.id,
                x: e.position?.x || 0,
                y: e.position?.y || 0,
                ...e.data
            }));
            
            // Create or recreate renderer based on graph size
            const renderer = RendererFactory.create(DOMCache.cyContainer, {
                expectedNodeCount: nodeCount,
                rendererPreference: State.rendererPreference
            });
            
            // Update state
            State.setRenderer(renderer);
            
            // Check if nodes have pre-computed positions
            const hasPositions = nodes.some(n => 
                n.x !== undefined && n.y !== undefined && 
                (Math.abs(n.x) > 0.1 || Math.abs(n.y) > 0.1)
            );
            
            // Set data - use static mode if positions exist and cosmos renderer
            if (renderer.getType() === 'cosmos' && hasPositions) {
                // Check if simulation should be paused (from config)
                const simulationOnLoad = RendererSettings.getValue('cosmos.simulationOnLoad', false);
                
                if (!simulationOnLoad && typeof renderer.setDataWithPositions === 'function') {
                    console.log('[GraphLoader] Using pre-computed positions, simulation will be paused');
                    renderer.setDataWithPositions(nodes, [], { pauseSimulation: true, fitView: true });
                    State.cosmosSimulationPaused = true;
                } else {
                    renderer.setData(nodes, []);
                    State.cosmosSimulationPaused = false;
                }
            } else {
                renderer.setData(nodes, []);
                if (renderer.getType() === 'cosmos') {
                    State.cosmosSimulationPaused = false;
                }
            }
            
            // Setup event handlers based on renderer type
            // Cytoscape events are handled by cytoscape-manager.js
            // Cosmos events need explicit handling here
            if (renderer.getType() === 'cosmos') {
                this.setupCosmosEventHandlers(renderer);
                
                // Handle simulation state based on configuration
                if (State.cosmosSimulationPaused) {
                    // Simulation is paused (pre-computed positions mode)
                    // Just fit view to show the layout
                    setTimeout(() => {
                        renderer.fitView();
                        console.log('[GraphLoader] cosmos.gl fit complete, simulation paused (static mode)');
                    }, 300);
                } else {
                    // Simulation runs continuously - user controls via toolbar button
                    // Fit view after initial layout starts settling
                    setTimeout(() => {
                        renderer.fitView();
                        console.log('[GraphLoader] cosmos.gl initial fit complete, simulation running');
                    }, 1500);
                }
                
                // Update simulation button to reflect state
                this.updateSimulationButtonState();
            }
            
            // Update renderer indicator in UI
            this.updateRendererIndicator();
            
            // Update counts
            DOMCache.nodeCount.textContent = `${nodeCount} nodes`;
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
            
            // Dispatch event for other modules (like Snapshots)
            document.dispatchEvent(new CustomEvent('graphLoaded', { 
                detail: { graphId: graphId } 
            }));
            
            const rendererType = renderer.getType();
            updateStatus(
                `Graph displayed: ${nodeCount} nodes (edges not loaded) [${rendererType}]`, 
                'success'
            );
            
        } catch (error) {
            console.error('Display error:', error);
            updateStatus('Failed to display graph: ' + error.message, 'error');
        }
    },
    
    /**
     * Update the renderer indicator in the UI
     */
    updateRendererIndicator() {
        const indicator = document.getElementById('renderer-indicator');
        if (!indicator) return;
        
        const type = State.rendererType;
        const caps = RendererFactory.getCapabilitySummary();
        
        const maxNodesFormatted = caps.maxNodes >= 1000000 
            ? `${(caps.maxNodes / 1000000).toFixed(1)}M` 
            : caps.maxNodes >= 1000 
                ? `${Math.round(caps.maxNodes / 1000)}k`
                : `${caps.maxNodes}`;
        
        // Get appropriate icon
        const icon = type === 'cosmos' ? Icons.get('rocket') : Icons.get('canvas');
        const label = type === 'cosmos' ? 'cosmos.gl' : 'Cytoscape.js';
        
        // Build layout mode indicator for cosmos
        let layoutModeHtml = '';
        if (type === 'cosmos') {
            const isStatic = State.cosmosSimulationPaused;
            const layoutMode = isStatic ? 'Static' : 'Live';
            const layoutIcon = isStatic ? Icons.get('lock') : Icons.get('refresh');
            const layoutClass = isStatic ? 'layout-static' : 'layout-live';
            
            layoutModeHtml = `
                <span class="layout-mode-badge ${layoutClass}" title="Layout mode: ${layoutMode}">
                    <span class="layout-mode-icon">${layoutIcon}</span>
                    <span class="layout-mode-label">${layoutMode}</span>
                </span>
            `;
        }
        
        indicator.innerHTML = `
            <span class="renderer-badge renderer-${type}" title="${caps.reason}">
                <span class="renderer-icon">${icon}</span>
                <span class="renderer-label">${label}</span>
            </span>
            ${layoutModeHtml}
        `;
        
        // Show WebGL warning if cosmos not available
        const warning = document.getElementById('webgl-warning');
        if (warning) {
            if (!caps.cosmosAvailable && !caps.cosmosLibraryLoaded) {
                warning.style.display = 'block';
                const details = document.getElementById('webgl-warning-details');
                if (details) {
                    details.textContent = caps.cosmosLibraryLoaded 
                        ? caps.reason 
                        : 'cosmos.gl library not loaded. Using Cytoscape.js fallback.';
                }
            } else {
                warning.style.display = 'none';
            }
        }
    },

    /**
     * Setup event handlers for cosmos.gl renderer
     * Cytoscape handlers are set up by cytoscape-manager.js
     */
    setupCosmosEventHandlers(renderer) {
        // Node click - show info panel
        renderer.on('nodeClick', (e) => {
            console.log('[GraphLoader] cosmos nodeClick event:', e);
            // cosmos-adapter uses nodeId and node, not id and data
            const nodeId = e.nodeId || e.id;
            const nodeData = e.node || e.data;
            
            if (nodeId) {
                const selectedNodes = renderer.getSelectedNodes();
                if (selectedNodes.length > 1) {
                    // Multi-selection
                    if (typeof InfoPanel !== 'undefined' && InfoPanel.showMultiSelectFromIds) {
                        InfoPanel.showMultiSelectFromIds(selectedNodes);
                    } else {
                        console.warn('[GraphLoader] InfoPanel.showMultiSelectFromIds not available');
                    }
                } else {
                    // Single node
                    if (typeof InfoPanel !== 'undefined' && InfoPanel.showNodeFromData) {
                        InfoPanel.showNodeFromData(nodeId, nodeData);
                    } else {
                        console.warn('[GraphLoader] InfoPanel.showNodeFromData not available');
                    }
                }
            }
        });
        
        // Background click - hide info panel
        renderer.on('backgroundClick', () => {
            if (DOMCache.infoPanel) {
                DOMCache.infoPanel.style.display = 'none';
            }
            if (typeof InfoPanel !== 'undefined' && InfoPanel.clearNavigation) {
                InfoPanel.clearNavigation();
            }
        });
        
        // Selection change
        renderer.on('selectionChange', (e) => {
            if (e.nodes && e.nodes.length > 1) {
                if (typeof InfoPanel !== 'undefined' && InfoPanel.showMultiSelectFromIds) {
                    InfoPanel.showMultiSelectFromIds(e.nodes);
                }
            }
        });
        
        console.log('[GraphLoader] cosmos.gl event handlers set up');
    },
    
    /**
     * Apply saved layout positions to current renderer
     * Works with both Cytoscape.js and cosmos.gl
     */
    applyLayoutPositions(positions) {
        const renderer = State.renderer;
        if (!renderer) return;
        
        if (renderer.getType() === 'cosmos') {
            // For cosmos.gl, pause simulation first
            renderer.pauseSimulation();
            
            // Apply positions
            renderer.updatePositions(positions);
            renderer.fitView();
            
            console.log('[GraphLoader] Layout positions applied to cosmos.gl');
        } else if (State.cy) {
            // For Cytoscape.js, use batch update
            State.cy.batch(() => {
                State.cy.nodes().forEach(node => {
                    const pos = positions[node.id()];
                    if (pos) {
                        node.position(pos);
                    }
                });
            });
            State.cy.fit();
            
            console.log('[GraphLoader] Layout positions applied to Cytoscape.js');
        }
    },
    
    /**
     * Control simulation (cosmos.gl only)
     */
    toggleSimulation() {
        const renderer = State.renderer;
        if (!renderer || renderer.getType() !== 'cosmos') return;
        
        if (State.cosmosSimulationPaused) {
            renderer.startSimulation();
            State.cosmosSimulationPaused = false;
            console.log('[GraphLoader] cosmos.gl simulation started');
        } else {
            renderer.pauseSimulation();
            State.cosmosSimulationPaused = true;
            console.log('[GraphLoader] cosmos.gl simulation paused');
        }
        
        // Update renderer indicator to reflect layout mode change
        this.updateRendererIndicator();
        
        return State.cosmosSimulationPaused;
    },
    
    /**
     * Pause cosmos.gl simulation
     */
    pauseSimulation() {
        const renderer = State.renderer;
        if (renderer && renderer.getType() === 'cosmos') {
            renderer.pauseSimulation();
            State.cosmosSimulationPaused = true;
        }
    },
    
    /**
     * Start cosmos.gl simulation
     */
    startSimulation() {
        const renderer = State.renderer;
        if (renderer && renderer.getType() === 'cosmos') {
            renderer.startSimulation();
            State.cosmosSimulationPaused = false;
        }
    },

    /**
     * Load edges incrementally in batches
     */
    async loadEdgesIncrementally(graphId) {
        const renderer = State.renderer;
        if (!renderer) return;
        
        // Get current edge count from renderer
        const currentStats = renderer.getStats();
        
        // If edges already loaded, clear them instead
        if (currentStats.edgeCount > 0) {
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
        
        // CRITICAL: Disable pointer events during bulk add
        DOMCache.cyContainer.style.pointerEvents = 'none';
        
        try {
            while (hasMore && State.edgesLoading) {
                const result = await API.getGraphEdges(graphId, offset, BATCH_SIZE);
                
                if (result.edges && result.edges.length > 0) {
                    // Convert from Cytoscape format to unified format
                    const edges = result.edges.map(e => ({
                        source: e.data.source,
                        target: e.data.target,
                        id: e.data.id,
                        ...e.data
                    }));
                    
                    // Add edges to renderer
                    renderer.addEdges(edges);
                    
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
            
            // Re-enable pointer events after renderer settles
            setTimeout(() => {
                DOMCache.cyContainer.style.pointerEvents = 'auto';
            }, 500);
            
            if (DOMCache.loadEdgesBtn) {
                DOMCache.loadEdgesBtn.disabled = false;
                const stats = renderer.getStats();
                DOMCache.loadEdgesBtn.textContent = stats.edgeCount > 0 ? 'Clear Edges' : 'Load Edges';
            }
        }
    },

    /**
     * Clear all edges from the graph
     */
    clearEdges() {
        const renderer = State.renderer;
        if (!renderer) return;
        
        const stats = renderer.getStats();
        const edgeCount = stats.edgeCount || stats.visibleEdgeCount || 0;
        
        if (edgeCount === 0) {
            Toast.show('No edges to clear', 'info');
            return;
        }
        
        // Disable pointer events during removal
        DOMCache.cyContainer.style.pointerEvents = 'none';
        
        // Use renderer-specific clear method if available (cosmos-adapter has position-preserving clearEdges)
        if (renderer.getType() === 'cosmos' && typeof renderer.clearEdges === 'function') {
            renderer.clearEdges();
        } else {
            // Fallback: Get all edge IDs and remove them
            const edgeIds = renderer.getAllEdgeIds();
            renderer.removeElements([], edgeIds);
        }
        
        // Re-enable after a delay
        setTimeout(() => {
            DOMCache.cyContainer.style.pointerEvents = 'auto';
        }, 300);
        
        DOMCache.edgeCount.textContent = '0 edges';
        if (DOMCache.edgesProgress) {
            DOMCache.edgesProgress.textContent = '';
        }
        
        if (DOMCache.loadEdgesBtn) {
            DOMCache.loadEdgesBtn.textContent = 'Load Edges';
        }
        
        // Update edge visibility toggle if present
        const edgeToggle = document.getElementById('cosmos-edge-visibility');
        if (edgeToggle) edgeToggle.checked = true;
        
        updateStatus(`Cleared ${edgeCount.toLocaleString()} edges (layout preserved)`, 'success');
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
    },
    
    /**
     * Switch renderer preference and reload current graph
     */
    async switchRenderer(preference) {
        State.setRendererPreference(preference);
        
        if (State.currentGraph) {
            await this.displayGraph(State.currentGraph);
        } else {
            // Just update the indicator
            this.updateRendererIndicator();
        }
    },
    
    /**
     * Update the simulation button state in the toolbar
     * Called after graph load to reflect initial simulation state
     */
    updateSimulationButtonState() {
        const btn = document.getElementById('toolbar-sim-btn');
        const icon = document.getElementById('toolbar-sim-icon');
        if (!btn) return;
        
        const isRunning = !State.cosmosSimulationPaused;
        
        if (isRunning) {
            btn.classList.remove('paused');
            btn.classList.add('running');
            btn.title = 'Pause simulation';
            if (icon) icon.setAttribute('data-icon', 'pause');
        } else {
            btn.classList.remove('running');
            btn.classList.add('paused');
            btn.title = 'Resume simulation';
            if (icon) icon.setAttribute('data-icon', 'play');
        }
        
        // Re-inject icons if needed
        if (typeof Icons !== 'undefined' && Icons.inject) {
            Icons.inject();
        }
        
        console.log('[GraphLoader] Simulation button state updated:', isRunning ? 'running' : 'paused');
    }
};