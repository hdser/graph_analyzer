/**
 * Snapshots Module
 * Handles UI for viewing and managing historical network snapshots
 */
const Snapshots = (function() {
    // Private state
    let currentBaseSqlFile = null;
    let currentSnapshotId = null;
    let batchEventSource = null;
    
    // Comparison state
    let isComparing = false;
    let comparisonData = null;
    
    // Animation state
    let animationData = null;
    let animationFrames = [];
    let currentFrameIndex = 0;
    let isAnimating = false;
    let animationInterval = null;
    let animationSpeed = 1; // seconds per frame
    
    // DOM element references (cached after init)
    const elements = {};
    
    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
    
    /**
     * Initialize the module
     */
    function init() {
        console.log('[Snapshots] Initializing...');
        cacheElements();
        bindEvents();
        updateStatusIndicator(false);
        
        // Listen for graph load to update current SQL file
        document.addEventListener('graphLoaded', handleGraphLoaded);
        
        console.log('[Snapshots] Initialized');
    }
    
    /**
     * Cache DOM element references
     */
    function cacheElements() {
        Object.assign(elements, {
            section: document.getElementById('snapshots-section'),
            dropdown: document.getElementById('snapshot-select'),
            loadBtn: document.getElementById('load-snapshot-btn'),
            returnBtn: document.getElementById('return-live-btn'),
            blockInput: document.getElementById('snapshot-block-input'),
            createBtn: document.getElementById('create-snapshot-btn'),
            batchBtn: document.getElementById('batch-snapshots-btn'),
            suggestBtn: document.getElementById('suggest-blocks-btn'),
            progressContainer: document.getElementById('snapshot-progress'),
            progressBar: document.getElementById('snapshot-progress-bar'),
            progressText: document.getElementById('snapshot-progress-text'),
            statusIndicator: document.getElementById('snapshot-status'),
            intervalSelect: document.getElementById('snapshot-interval'),
            countInput: document.getElementById('snapshot-count'),
            createContent: document.getElementById('snapshot-create-content'),
            // Comparison elements
            compareFromSelect: document.getElementById('compare-from-select'),
            compareToSelect: document.getElementById('compare-to-select'),
            compareBtn: document.getElementById('compare-snapshots-btn'),
            exitCompareBtn: document.getElementById('exit-compare-btn'),
            compareResults: document.getElementById('compare-results'),
            compareAddedCount: document.getElementById('compare-added-count'),
            compareRemovedCount: document.getElementById('compare-removed-count'),
            compareRetainedCount: document.getElementById('compare-retained-count'),
            // Animation elements
            animPrevBtn: document.getElementById('anim-prev-btn'),
            animPlayBtn: document.getElementById('anim-play-btn'),
            animNextBtn: document.getElementById('anim-next-btn'),
            animSpeedSlider: document.getElementById('anim-speed'),
            animSpeedLabel: document.getElementById('anim-speed-label'),
            animTimeline: document.getElementById('anim-timeline'),
            animCurrentLabel: document.getElementById('anim-current-label'),
            animTotalLabel: document.getElementById('anim-total-label'),
            animBlockInfo: document.getElementById('anim-block-info'),
            animDateInfo: document.getElementById('anim-date-info'),
            animIncludeEdges: document.getElementById('anim-include-edges'),
            loadAnimationBtn: document.getElementById('load-animation-btn')
        });
    }
    
    /**
     * Bind event handlers
     */
    function bindEvents() {
        elements.loadBtn?.addEventListener('click', handleLoadSnapshot);
        elements.returnBtn?.addEventListener('click', handleReturnToLive);
        elements.createBtn?.addEventListener('click', handleCreateSnapshot);
        elements.batchBtn?.addEventListener('click', handleBatchCreate);
        elements.suggestBtn?.addEventListener('click', handleSuggestBlocks);
        elements.dropdown?.addEventListener('change', handleDropdownChange);
        
        // Comparison events
        elements.compareBtn?.addEventListener('click', handleCompare);
        elements.exitCompareBtn?.addEventListener('click', handleExitCompare);
        elements.compareFromSelect?.addEventListener('change', updateCompareButtonState);
        elements.compareToSelect?.addEventListener('change', updateCompareButtonState);
        
        // Animation events
        elements.animPlayBtn?.addEventListener('click', handleAnimPlayPause);
        elements.animPrevBtn?.addEventListener('click', handleAnimPrev);
        elements.animNextBtn?.addEventListener('click', handleAnimNext);
        // Use 'change' event (fires on release) for rendering frame
        // Use 'input' event for preview (just show frame info, no render)
        elements.animTimeline?.addEventListener('change', handleAnimTimelineChange);
        elements.animTimeline?.addEventListener('input', handleAnimTimelinePreview);
        elements.animSpeedSlider?.addEventListener('input', handleAnimSpeedChange);
        elements.loadAnimationBtn?.addEventListener('click', handleLoadAnimation);
    }
    
    /**
     * Handle graph loaded event
     */
    function handleGraphLoaded(event) {
        const graphId = event.detail?.graphId;
        if (graphId) {
            console.log('[Snapshots] Graph loaded:', graphId);
            currentBaseSqlFile = graphId;
            loadAvailableSnapshots();
        }
    }
    
    // ==========================================================================
    // SNAPSHOT LOADING
    // ==========================================================================
    
    /**
     * Load available snapshots for current SQL file
     */
    async function loadAvailableSnapshots() {
        if (!currentBaseSqlFile) {
            console.log('[Snapshots] No SQL file selected');
            return;
        }
        
        try {
            console.log('[Snapshots] Loading snapshots for:', currentBaseSqlFile);
            const result = await SnapshotAPI.listSnapshots(currentBaseSqlFile);
            
            State.setAvailableSnapshots(result.snapshots || []);
            populateDropdown(result.snapshots || []);
            populateComparisonDropdowns(result.snapshots || []);
            
            console.log(`[Snapshots] Loaded ${result.snapshots?.length || 0} snapshots`);
        } catch (error) {
            console.error('[Snapshots] Failed to load snapshots:', error);
            // Don't show toast - might just not have any snapshots yet
            populateDropdown([]);
            populateComparisonDropdowns([]);
        }
    }
    
    /**
     * Populate snapshot dropdown
     */
    function populateDropdown(snapshots) {
        if (!elements.dropdown) return;
        
        elements.dropdown.innerHTML = '<option value="">Select snapshot...</option>';
        
        if (!snapshots || snapshots.length === 0) {
            elements.dropdown.innerHTML += '<option value="" disabled>No snapshots available</option>';
            if (elements.loadBtn) elements.loadBtn.disabled = true;
            return;
        }
        
        // Sort by block number descending (newest first)
        const sorted = [...snapshots].sort((a, b) => b.block_number - a.block_number);
        
        sorted.forEach(snapshot => {
            const option = document.createElement('option');
            option.value = snapshot.snapshot_id;
            option.textContent = formatSnapshotLabel(snapshot);
            elements.dropdown.appendChild(option);
        });
        
        handleDropdownChange();
    }
    
    /**
     * Format snapshot for display in dropdown
     */
    function formatSnapshotLabel(snapshot) {
        let label = `Block ${snapshot.block_number.toLocaleString()}`;
        
        if (snapshot.block_timestamp) {
            const date = new Date(snapshot.block_timestamp);
            const dateStr = date.toLocaleDateString('en-US', { 
                month: 'short', 
                day: 'numeric',
                year: 'numeric'
            });
            label = `${dateStr}  -  ${label}`;
        }
        
        if (snapshot.node_count) {
            label += ` (${snapshot.node_count.toLocaleString()} nodes)`;
        }
        
        return label;
    }
    
    /**
     * Handle loading selected snapshot
     * Loads nodes only - user must click "Load Edges" for edges
     */
    async function handleLoadSnapshot() {
        const snapshotId = elements.dropdown?.value;
        if (!snapshotId) {
            Toast.show('Please select a snapshot', 'warning');
            return;
        }
        
        setLoading(true);
        showProgress(true, 0, 'Loading snapshot...');
        
        try {
            // Load nodes only (fast)
            showProgress(true, 30, 'Loading nodes...');
            console.log('[Snapshots] Loading nodes for:', snapshotId);
            
            const nodesData = await SnapshotAPI.getSnapshotNodes(snapshotId);
            
            console.log('[Snapshots] Received nodes:', {
                nodeCount: nodesData.elements?.length || 0,
                metadata: nodesData.metadata
            });
            
            if (!nodesData.elements || nodesData.elements.length === 0) {
                throw new Error('No nodes in snapshot data');
            }
            
            // Render nodes in Cytoscape
            showProgress(true, 60, 'Rendering nodes...');
            const cy = State.cy;
            if (!cy) {
                throw new Error('Cytoscape not initialized');
            }
            
            cy.batch(() => {
                cy.elements().remove();
                cy.add(nodesData.elements);
            });
            
            // Fit view
            cy.fit();
            
            console.log('[Snapshots] Nodes rendered:', cy.nodes().length);
            
            // Store current snapshot ID for edge loading
            currentSnapshotId = snapshotId;
            
            // Store metadata
            const snapshotMetadata = nodesData.metadata;
            
            // Update state
            State.setSnapshotActive(true, snapshotMetadata);
            updateStatusIndicator(true, snapshotMetadata);
            
            // Update counts - show pending edge count
            const nodeCountEl = document.getElementById('node-count');
            const edgeCountEl = document.getElementById('edge-count');
            if (nodeCountEl) {
                nodeCountEl.textContent = `${cy.nodes().length.toLocaleString()} nodes`;
            }
            if (edgeCountEl) {
                edgeCountEl.textContent = `0 / ${(snapshotMetadata?.edge_count || 0).toLocaleString()} edges`;
            }
            
            // Update Load Edges button
            const loadEdgesBtn = document.getElementById('load-edges-btn');
            if (loadEdgesBtn) {
                loadEdgesBtn.textContent = 'Load Edges';
                loadEdgesBtn.disabled = false;
            }
            
            // Populate metric dropdowns
            if (typeof Metrics !== 'undefined' && Metrics.populateDropdowns) {
                Metrics.populateDropdowns(nodesData.elements, snapshotMetadata?.metrics_computed);
            }
            
            showProgress(true, 100, 'Nodes loaded!');
            
            const label = snapshotMetadata?.label || `Block ${snapshotMetadata?.block_number}`;
            Toast.show(`Loaded snapshot: ${label} (click "Load Edges" for edges)`, 'success');
            
        } catch (error) {
            console.error('[Snapshots] Failed to load snapshot:', error);
            Toast.show('Failed to load snapshot: ' + error.message, 'error');
        } finally {
            setLoading(false);
            setTimeout(() => showProgress(false), 1000);
        }
    }
    
    /**
     * Load snapshot edges incrementally in batches
     * Called when user clicks "Load Edges" button
     */
    async function loadSnapshotEdges() {
        // Disable edge loading during animation - use the checkbox to include edges
        if (animationNodeSets && animationNodeSets.length > 0) {
            Toast.show('Use "Include Edges" checkbox before starting animation', 'info');
            return true; // Handled
        }
        
        if (!currentSnapshotId) {
            console.log('[Snapshots] No snapshot loaded, delegating to GraphLoader');
            return false; // Let GraphLoader handle it
        }
        
        const cy = State.cy;
        if (!cy) return false;
        
        // If edges already loaded, clear them
        if (cy.edges().length > 0) {
            clearSnapshotEdges();
            return true;
        }
        
        const BATCH_SIZE = 50000;
        let offset = 0;
        let totalLoaded = 0;
        let hasMore = true;
        
        const loadEdgesBtn = document.getElementById('load-edges-btn');
        if (loadEdgesBtn) {
            loadEdgesBtn.disabled = true;
            loadEdgesBtn.textContent = 'Loading...';
        }
        
        // CRITICAL: Disable pointer events during bulk add to prevent WebGL crash
        const container = document.getElementById('cy');
        container.style.pointerEvents = 'none';
        
        console.log('[Snapshots] Starting edge loading for:', currentSnapshotId);
        
        try {
            while (hasMore) {
                const result = await SnapshotAPI.getSnapshotEdges(currentSnapshotId, offset, BATCH_SIZE);
                
                if (result.edges && result.edges.length > 0) {
                    // Add edges in batch
                    cy.batch(() => {
                        cy.add(result.edges);
                    });
                    
                    totalLoaded += result.edges.length;
                    offset = totalLoaded;
                    
                    // Update edge count display
                    const edgeCountEl = document.getElementById('edge-count');
                    if (edgeCountEl) {
                        edgeCountEl.textContent = `${totalLoaded.toLocaleString()} / ${result.total.toLocaleString()} edges`;
                    }
                    
                    // Update progress in edges progress element
                    const edgesProgress = document.getElementById('edges-progress');
                    if (edgesProgress) {
                        edgesProgress.textContent = `${totalLoaded.toLocaleString()} / ${result.total.toLocaleString()}`;
                    }
                    
                    hasMore = result.has_more;
                    
                    console.log('[Snapshots] Loaded edges:', totalLoaded, '/', result.total);
                } else {
                    hasMore = false;
                }
            }
            
            // Final update
            const edgeCountEl = document.getElementById('edge-count');
            if (edgeCountEl) {
                edgeCountEl.textContent = `${totalLoaded.toLocaleString()} edges`;
            }
            
            const edgesProgress = document.getElementById('edges-progress');
            if (edgesProgress) {
                edgesProgress.textContent = '';
            }
            
            Toast.show(`Loaded ${totalLoaded.toLocaleString()} edges`, 'success');
            
        } catch (error) {
            console.error('[Snapshots] Edge loading error:', error);
            Toast.show('Edge loading failed: ' + error.message, 'error');
        } finally {
            // Re-enable pointer events after WebGL settles (500ms for large edge counts)
            setTimeout(() => {
                container.style.pointerEvents = 'auto';
            }, 500);
            
            if (loadEdgesBtn) {
                loadEdgesBtn.disabled = false;
                loadEdgesBtn.textContent = cy.edges().length > 0 ? 'Clear Edges' : 'Load Edges';
            }
        }
        
        return true;
    }
    
    /**
     * Clear edges from snapshot view
     */
    function clearSnapshotEdges() {
        const cy = State.cy;
        if (!cy) return;
        
        const edges = cy.edges();
        const edgeCount = edges.length;
        
        if (edgeCount === 0) {
            Toast.show('No edges to clear', 'info');
            return;
        }
        
        // Disable pointer events during removal
        const container = document.getElementById('cy');
        container.style.pointerEvents = 'none';
        
        // Remove all edges in one operation
        edges.remove();
        
        // Re-enable after a delay
        setTimeout(() => {
            container.style.pointerEvents = 'auto';
        }, 300);
        
        const edgeCountEl = document.getElementById('edge-count');
        const snapshotInfo = State.getCurrentSnapshot();
        if (edgeCountEl) {
            edgeCountEl.textContent = `0 / ${(snapshotInfo?.edge_count || 0).toLocaleString()} edges`;
        }
        
        const edgesProgress = document.getElementById('edges-progress');
        if (edgesProgress) {
            edgesProgress.textContent = '';
        }
        
        const loadEdgesBtn = document.getElementById('load-edges-btn');
        if (loadEdgesBtn) {
            loadEdgesBtn.textContent = 'Load Edges';
        }
        
        Toast.show(`Cleared ${edgeCount.toLocaleString()} edges`, 'success');
    }
    
    /**
     * Build Cytoscape elements from snapshot data
     */
    function buildCytoscapeElements(snapshotData) {
        const nodeSet = new Set();
        const cyElements = [];
        
        console.log('[Snapshots] Building elements from:', {
            edges: snapshotData.edges?.length,
            layout: typeof snapshotData.layout,
            metrics: typeof snapshotData.metrics
        });
        
        // Add edges and collect node IDs
        if (snapshotData.edges && Array.isArray(snapshotData.edges)) {
            snapshotData.edges.forEach((edge, idx) => {
                if (!edge.source || !edge.target) {
                    if (idx < 5) console.warn('[Snapshots] Invalid edge:', edge);
                    return;
                }
                
                const source = String(edge.source);
                const target = String(edge.target);
                
                nodeSet.add(source);
                nodeSet.add(target);
                
                cyElements.push({
                    group: 'edges',
                    data: {
                        id: `${source}->${target}`,
                        source: source,
                        target: target
                    }
                });
            });
        }
        
        console.log('[Snapshots] Collected nodes:', nodeSet.size, 'edges:', cyElements.length);
        
        // Add nodes with metrics and positions
        const metrics = snapshotData.metrics || {};
        const layout = snapshotData.layout || {};
        
        // Check if layout has positions
        const layoutKeys = Object.keys(layout);
        if (layoutKeys.length > 0) {
            const sampleKey = layoutKeys[0];
            const samplePos = layout[sampleKey];
            console.log('[Snapshots] Sample layout position:', sampleKey, samplePos);
        }
        
        let nodesWithPosition = 0;
        let nodesWithoutPosition = 0;
        
        nodeSet.forEach(nodeId => {
            const nodeMetrics = metrics[nodeId] || {};
            let position = layout[nodeId];
            
            // Handle position
            if (position && typeof position.x === 'number' && typeof position.y === 'number') {
                nodesWithPosition++;
            } else {
                nodesWithoutPosition++;
                // Assign random position if not found
                position = { 
                    x: Math.random() * 2000 - 1000, 
                    y: Math.random() * 2000 - 1000 
                };
            }
            
            cyElements.push({
                group: 'nodes',
                data: {
                    id: nodeId,
                    ...nodeMetrics
                },
                position: {
                    x: position.x,
                    y: position.y
                }
            });
        });
        
        console.log('[Snapshots] Nodes with position:', nodesWithPosition, 'without:', nodesWithoutPosition);
        
        return cyElements;
    }
    
    /**
     * Apply snapshot layout positions to graph
     */
    function applySnapshotLayout(cy, layout) {
        if (!layout || !cy) return;
        
        cy.batch(() => {
            Object.entries(layout).forEach(([nodeId, position]) => {
                const node = cy.getElementById(nodeId);
                if (node.length > 0 && position) {
                    node.position({
                        x: position.x,
                        y: position.y
                    });
                }
            });
        });
    }
    
    /**
     * Update node/edge count display
     */
    function updateGraphCounts(cy, totalEdges = null) {
        const nodeCount = document.getElementById('node-count');
        const edgeCount = document.getElementById('edge-count');
        
        if (nodeCount) {
            nodeCount.textContent = `${cy.nodes().length.toLocaleString()} nodes`;
        }
        if (edgeCount) {
            const currentEdges = cy.edges().length;
            if (totalEdges && totalEdges > currentEdges) {
                edgeCount.textContent = `${currentEdges.toLocaleString()} / ${totalEdges.toLocaleString()} edges`;
            } else {
                edgeCount.textContent = `${currentEdges.toLocaleString()} edges`;
            }
        }
    }
    
    // ==========================================================================
    // RETURN TO LIVE
    // ==========================================================================
    
    /**
     * Return to live graph view
     */
    async function handleReturnToLive() {
        console.log('[Snapshots] Returning to live view...');
        setLoading(true);
        showProgress(true, 0, 'Returning to live view...');
        
        try {
            showProgress(true, 20, 'Cleaning up...');
            
            // Stop any running animation first
            stopAnimation();
            
            // Clear animation state
            animationNodeSets = [];
            animationFrames = [];
            currentFrameIndex = 0;
            animationHasEdges = false;
            animationLoading = false;
            animationLoadTimestamp = 0;
            animationAllNodes = null;
            animationAllEdges = null;
            enableAnimationControls(false);
            
            // Re-enable Load Edges button
            const loadEdgesBtn = document.getElementById('load-edges-btn');
            if (loadEdgesBtn) {
                loadEdgesBtn.disabled = false;
                loadEdgesBtn.title = 'Load edges for this snapshot';
            }
            
            // Reset UI labels
            if (elements.animTimeline) { elements.animTimeline.value = 0; elements.animTimeline.max = 0; }
            if (elements.animCurrentLabel) elements.animCurrentLabel.textContent = '-';
            if (elements.animTotalLabel) elements.animTotalLabel.textContent = '-';
            if (elements.animBlockInfo) elements.animBlockInfo.textContent = 'Block: -';
            if (elements.animDateInfo) elements.animDateInfo.textContent = 'Date: -';
            
            // Clear comparison state
            isComparing = false;
            comparisonData = null;
            
            // Hide comparison UI
            if (elements.compareResults) {
                elements.compareResults.style.display = 'none';
            }
            if (elements.exitCompareBtn) {
                elements.exitCompareBtn.style.display = 'none';
            }
            if (elements.compareBtn) {
                elements.compareBtn.style.display = 'inline-block';
            }
            
            // Clear snapshot state
            currentSnapshotId = null;
            State.setSnapshotActive(false);
            updateStatusIndicator(false);
            
            showProgress(true, 40, 'Reinitializing graph renderer...');
            
            // Destroy current Cytoscape instance to force WebGL reinitialization
            // (Animation uses canvas-only renderer, live view uses WebGL)
            if (State.cy) {
                State.cy.destroy();
                State.cy = null;
            }
            
            showProgress(true, 60, 'Reloading live graph...');
            
            // Determine which graph to reload
            const graphToLoad = State.currentGraph || currentBaseSqlFile;
            console.log('[Snapshots] Graph to reload:', graphToLoad);
            
            if (graphToLoad && typeof GraphLoader !== 'undefined' && GraphLoader.displayGraph) {
                try {
                    await GraphLoader.displayGraph(graphToLoad);
                    showProgress(true, 100, 'Done!');
                    Toast.show('Returned to live view', 'success');
                } catch (loadError) {
                    console.error('[Snapshots] GraphLoader.displayGraph failed:', loadError);
                    // Try fallback approach
                    const graphSelect = document.getElementById('graph-select');
                    if (graphSelect && graphSelect.value) {
                        console.log('[Snapshots] Trying fallback with graph-select:', graphSelect.value);
                        await GraphLoader.displayGraph(graphSelect.value);
                        Toast.show('Returned to live view', 'success');
                    } else {
                        throw loadError;
                    }
                }
            } else {
                // Fallback: try graph selector
                const graphSelect = document.getElementById('graph-select');
                console.log('[Snapshots] Using fallback, graph-select value:', graphSelect?.value);
                if (graphSelect && graphSelect.value && typeof GraphLoader !== 'undefined') {
                    await GraphLoader.displayGraph(graphSelect.value);
                    showProgress(true, 100, 'Done!');
                    Toast.show('Returned to live view', 'success');
                } else {
                    showProgress(true, 100, 'Done!');
                    Toast.show('No graph to reload - please select a graph', 'warning');
                }
            }
            
        } catch (error) {
            console.error('[Snapshots] Failed to return to live view:', error);
            Toast.show('Failed to return to live view: ' + error.message, 'error');
        } finally {
            setLoading(false);
            setTimeout(() => showProgress(false), 500);
        }
    }
    
    // ==========================================================================
    // SNAPSHOT CREATION
    // ==========================================================================
    
    /**
     * Create a new snapshot
     */
    async function handleCreateSnapshot() {
        const blockNumber = parseInt(elements.blockInput?.value);
        
        if (!blockNumber || isNaN(blockNumber)) {
            Toast.show('Please enter a valid block number', 'warning');
            return;
        }
        
        if (!currentBaseSqlFile) {
            Toast.show('Please load a graph first', 'warning');
            return;
        }
        
        setLoading(true);
        showProgress(true, 0, 'Creating snapshot...');
        
        try {
            showProgress(true, 30, 'Querying database...');
            
            const result = await SnapshotAPI.createSnapshot({
                base_sql_file: currentBaseSqlFile,
                block_number: blockNumber,
                metrics_mode: 'standard'
            });
            
            showProgress(true, 100, 'Complete!');
            Toast.show(`Created snapshot: ${result.label || result.snapshot_id}`, 'success');
            
            // Reload snapshot list
            await loadAvailableSnapshots();
            
            // Select the newly created snapshot
            if (elements.dropdown && result.snapshot_id) {
                elements.dropdown.value = result.snapshot_id;
                handleDropdownChange();
            }
            
        } catch (error) {
            console.error('[Snapshots] Failed to create snapshot:', error);
            Toast.show('Failed to create snapshot: ' + error.message, 'error');
        } finally {
            setLoading(false);
            setTimeout(() => showProgress(false), 1000);
        }
    }
    
    /**
     * Batch create snapshots
     */
    async function handleBatchCreate() {
        if (!currentBaseSqlFile) {
            Toast.show('Please load a graph first', 'warning');
            return;
        }
        
        const interval = elements.intervalSelect?.value || 'daily';
        const count = parseInt(elements.countInput?.value) || 30;
        
        try {
            // First get suggested block numbers
            showProgress(true, 0, 'Getting block suggestions...');
            
            const suggestions = await SnapshotAPI.suggestBlockNumbers({
                base_sql_file: currentBaseSqlFile,
                interval: interval,
                count: count
            });
            
            if (!suggestions.suggestions || suggestions.suggestions.length === 0) {
                Toast.show('No block suggestions available', 'warning');
                showProgress(false);
                return;
            }
            
            // Confirm with user
            const numSnapshots = suggestions.suggestions.length;
            const confirmed = confirm(
                `Create ${numSnapshots} ${interval} snapshots?\n\n` +
                `This will create snapshots for:\n` +
                `- First: Block ${suggestions.suggestions[0].block_number}\n` +
                `- Last: Block ${suggestions.suggestions[numSnapshots-1].block_number}\n\n` +
                `This may take several minutes.`
            );
            
            if (!confirmed) {
                showProgress(false);
                return;
            }
            
            // Start batch creation with SSE progress
            setLoading(true);
            showProgress(true, 0, 'Starting batch creation...');
            
            const blockNumbers = suggestions.suggestions.map(s => s.block_number);
            
            await SnapshotAPI._createBatchWithFetch(
                {
                    base_sql_file: currentBaseSqlFile,
                    block_numbers: blockNumbers,
                    metrics_mode: 'standard'
                },
                // onProgress callback
                (data) => {
                    const percent = Math.round((data.current / data.total) * 100);
                    showProgress(true, percent, 
                        `Creating snapshot ${data.current}/${data.total} (Block ${data.block_number})...`
                    );
                },
                // onComplete callback (individual snapshot)
                (data) => {
                    console.log('[Snapshots] Created:', data.snapshot_id);
                },
                // onDone callback (all done)
                (data) => {
                    showProgress(true, 100, `Created ${data.total_created} snapshots!`);
                    setTimeout(() => {
                        showProgress(false);
                        setLoading(false);
                    }, 1500);
                    
                    Toast.show(`Created ${data.total_created} snapshots`, 'success');
                    loadAvailableSnapshots();
                },
                // onError callback
                (data) => {
                    console.error('[Snapshots] Batch error:', data);
                    showProgress(false);
                    setLoading(false);
                    Toast.show(`Error: ${data.error || data.message}`, 'error');
                }
            );
            
        } catch (error) {
            console.error('[Snapshots] Failed to start batch creation:', error);
            Toast.show('Failed to start batch creation: ' + error.message, 'error');
            setLoading(false);
            showProgress(false);
        }
    }
    
    /**
     * Suggest block numbers and populate input
     */
    async function handleSuggestBlocks() {
        if (!currentBaseSqlFile) {
            Toast.show('Please load a graph first', 'warning');
            return;
        }
        
        try {
            const result = await SnapshotAPI.suggestBlockNumbers({
                base_sql_file: currentBaseSqlFile,
                interval: 'daily',
                count: 10
            });
            
            if (result.suggestions && result.suggestions.length > 0) {
                // Get the most recent suggestion
                const suggestion = result.suggestions[result.suggestions.length - 1];
                
                if (elements.blockInput) {
                    elements.blockInput.value = suggestion.block_number;
                }
                
                // Format the date nicely
                const dateStr = suggestion.label || new Date(suggestion.timestamp).toLocaleDateString();
                Toast.show(`Suggested: Block ${suggestion.block_number.toLocaleString()} (${dateStr})`, 'info');
            } else {
                Toast.show('No suggestions available', 'warning');
            }
            
        } catch (error) {
            console.error('[Snapshots] Failed to get suggestions:', error);
            Toast.show('Failed to get block suggestions', 'error');
        }
    }
    
    // ==========================================================================
    // UI HELPERS
    // ==========================================================================
    
    /**
     * Update status indicator
     */
    function updateStatusIndicator(isActive, snapshotInfo = null, isCompareMode = false, isAnimMode = false, animFrame = null, animTotal = null) {
        if (!elements.statusIndicator) return;
        
        // Remove all state classes
        elements.statusIndicator.classList.remove('live', 'snapshot-active', 'viewing-snapshot', 'comparing', 'animating');
        
        if (isCompareMode) {
            elements.statusIndicator.textContent = '⇋ Comparison View';
            elements.statusIndicator.classList.add('comparing');
        } else if (isAnimMode) {
            elements.statusIndicator.textContent = `▶ Animation: ${animFrame}/${animTotal}`;
            elements.statusIndicator.classList.add('animating');
        } else if (isActive && snapshotInfo) {
            const label = snapshotInfo.label || `Block ${snapshotInfo.block_number?.toLocaleString()}`;
            elements.statusIndicator.textContent = `📊 ${label}`;
            elements.statusIndicator.classList.add('viewing-snapshot');
        } else {
            elements.statusIndicator.textContent = '● Live View';
            elements.statusIndicator.classList.add('live');
        }
        
        // Toggle button visibility
        const showReturn = isActive || isCompareMode || isAnimMode;
        if (elements.returnBtn) {
            elements.returnBtn.style.display = showReturn ? 'block' : 'none';
        }
        if (elements.loadBtn) {
            elements.loadBtn.style.display = showReturn ? 'none' : 'block';
        }
        
        // Disable create section when viewing snapshot/comparing/animating
        if (elements.createContent) {
            elements.createContent.style.opacity = showReturn ? '0.5' : '1';
            elements.createContent.style.pointerEvents = showReturn ? 'none' : 'auto';
        }
    }
    
    /**
     * Set loading state
     */
    function setLoading(isLoading) {
        State.setSnapshotLoading(isLoading);
        
        // Disable/enable controls
        const controls = [
            elements.loadBtn, 
            elements.returnBtn,
            elements.createBtn, 
            elements.batchBtn, 
            elements.suggestBtn,
            elements.dropdown
        ];
        
        controls.forEach(el => {
            if (el) el.disabled = isLoading;
        });
    }
    
    /**
     * Show/hide progress indicator
     */
    function showProgress(show, percent = 0, text = '') {
        if (elements.progressContainer) {
            elements.progressContainer.style.display = show ? 'block' : 'none';
        }
        if (elements.progressBar) {
            elements.progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        }
        if (elements.progressText) {
            elements.progressText.textContent = text;
            elements.progressText.classList.remove('error', 'success');
            if (percent >= 100) {
                elements.progressText.classList.add('success');
            }
        }
    }
    
    /**
     * Handle dropdown selection change
     */
    function handleDropdownChange() {
        const hasSelection = elements.dropdown?.value && elements.dropdown.value !== '';
        if (elements.loadBtn) {
            elements.loadBtn.disabled = !hasSelection;
        }
    }
    
    /**
     * Check if auto-reload should be prevented
     */
    function shouldPreventAutoReload() {
        return State.isViewingSnapshot();
    }
    
    // ==========================================================================
    // COMPARISON FUNCTIONS
    // ==========================================================================
    
    /**
     * Populate comparison dropdowns with snapshots
     */
    function populateComparisonDropdowns(snapshots) {
        const fromSelect = elements.compareFromSelect;
        const toSelect = elements.compareToSelect;
        
        if (!fromSelect || !toSelect) return;
        
        const defaultOption = '<option value="">Select snapshot...</option>';
        fromSelect.innerHTML = defaultOption;
        toSelect.innerHTML = defaultOption;
        
        if (!snapshots || snapshots.length < 2) {
            fromSelect.innerHTML += '<option value="" disabled>Need at least 2 snapshots</option>';
            toSelect.innerHTML += '<option value="" disabled>Need at least 2 snapshots</option>';
            return;
        }
        
        // Sort by block number descending
        const sorted = [...snapshots].sort((a, b) => b.block_number - a.block_number);
        
        sorted.forEach(snapshot => {
            const option = `<option value="${snapshot.snapshot_id}">${formatSnapshotLabel(snapshot)}</option>`;
            fromSelect.innerHTML += option;
            toSelect.innerHTML += option;
        });
        
        updateCompareButtonState();
    }
    
    /**
     * Update compare button enabled state
     */
    function updateCompareButtonState() {
        const fromId = elements.compareFromSelect?.value;
        const toId = elements.compareToSelect?.value;
        
        const canCompare = fromId && toId && fromId !== toId;
        
        if (elements.compareBtn) {
            elements.compareBtn.disabled = !canCompare;
        }
    }
    
    /**
     * Handle compare button click
     */
    async function handleCompare() {
        const fromId = elements.compareFromSelect?.value;
        const toId = elements.compareToSelect?.value;
        
        if (!fromId || !toId) {
            Toast.show('Please select two snapshots to compare', 'warning');
            return;
        }
        
        setLoading(true);
        showProgress(true, 0, 'Comparing snapshots...');
        
        try {
            console.log('[Snapshots] Comparing:', fromId, '->', toId);
            
            showProgress(true, 30, 'Loading comparison data...');
            comparisonData = await SnapshotAPI.compareSnapshots(fromId, toId);
            
            console.log('[Snapshots] Comparison result:', comparisonData);
            
            showProgress(true, 60, 'Rendering comparison...');
            await renderComparison(comparisonData);
            
            // Update UI state
            isComparing = true;
            updateStatusIndicator(false, null, true);
            
            // Show results and exit button
            if (elements.compareResults) {
                elements.compareResults.style.display = 'block';
                elements.compareAddedCount.textContent = comparisonData.diff.added_node_count;
                elements.compareRemovedCount.textContent = comparisonData.diff.removed_node_count;
                elements.compareRetainedCount.textContent = comparisonData.diff.retained_node_count;
            }
            
            if (elements.exitCompareBtn) {
                elements.exitCompareBtn.style.display = 'inline-block';
            }
            if (elements.compareBtn) {
                elements.compareBtn.style.display = 'none';
            }
            
            showProgress(true, 100, 'Complete!');
            Toast.show('Comparison loaded', 'success');
            
        } catch (error) {
            console.error('[Snapshots] Comparison failed:', error);
            Toast.show('Comparison failed: ' + error.message, 'error');
        } finally {
            setLoading(false);
            setTimeout(() => showProgress(false), 1000);
        }
    }
    
    /**
     * Render comparison visualization
     */
    async function renderComparison(data) {
        const cy = State.cy;
        if (!cy) return;
        
        // Build elements with comparison styling
        const elements_arr = [];
        
        // Added nodes (green)
        data.diff.added_nodes.forEach(nodeId => {
            const pos = data.layout[nodeId] || { x: Math.random() * 1000, y: Math.random() * 1000 };
            elements_arr.push({
                group: 'nodes',
                data: { id: nodeId, compareStatus: 'added' },
                position: pos,
                classes: 'compare-added'
            });
        });
        
        // Removed nodes (red)
        data.diff.removed_nodes.forEach(nodeId => {
            const pos = data.layout[nodeId] || { x: Math.random() * 1000, y: Math.random() * 1000 };
            elements_arr.push({
                group: 'nodes',
                data: { id: nodeId, compareStatus: 'removed' },
                position: pos,
                classes: 'compare-removed'
            });
        });
        
        // Retained nodes (blue)
        data.diff.retained_nodes.forEach(nodeId => {
            const pos = data.layout[nodeId] || { x: Math.random() * 1000, y: Math.random() * 1000 };
            elements_arr.push({
                group: 'nodes',
                data: { id: nodeId, compareStatus: 'retained' },
                position: pos,
                classes: 'compare-retained'
            });
        });
        
        // Clear and render
        cy.batch(() => {
            cy.elements().remove();
            cy.add(elements_arr);
        });
        
        // Apply comparison styles
        cy.style()
            .selector('.compare-added')
            .style({
                'background-color': '#4CAF50',
                'border-color': '#2E7D32',
                'border-width': 2
            })
            .selector('.compare-removed')
            .style({
                'background-color': '#EF5350',
                'border-color': '#C62828',
                'border-width': 2
            })
            .selector('.compare-retained')
            .style({
                'background-color': '#42A5F5',
                'border-color': '#1565C0',
                'border-width': 1
            })
            .update();
        
        cy.fit();
        
        // Update counts
        const nodeCount = document.getElementById('node-count');
        const edgeCount = document.getElementById('edge-count');
        if (nodeCount) {
            nodeCount.textContent = `${cy.nodes().length.toLocaleString()} nodes`;
        }
        if (edgeCount) {
            edgeCount.textContent = '0 edges (comparison view)';
        }
    }
    
    /**
     * Exit comparison mode
     */
    async function handleExitCompare() {
        isComparing = false;
        comparisonData = null;
        
        // Hide comparison UI
        if (elements.compareResults) {
            elements.compareResults.style.display = 'none';
        }
        if (elements.exitCompareBtn) {
            elements.exitCompareBtn.style.display = 'none';
        }
        if (elements.compareBtn) {
            elements.compareBtn.style.display = 'inline-block';
        }
        
        // Return to live view
        await handleReturnToLive();
    }
    
    
    // ==========================================================================
    // ANIMATION FUNCTIONS - Network Growth Animation
    // ==========================================================================
    
    // Animation state
    let animationLoading = false;
    let animationNodeSets = []; // Pre-loaded node/edge data for each frame
    let animationHasEdges = false;
    let animationLoadTimestamp = 0;
    let animationAllNodes = null; // All nodes data (for lookup)
    let animationAllEdges = null; // All edges as Map for fast lookup
    
    /**
     * Yield to browser - allows UI to remain responsive during heavy operations
     */
    function yieldToBrowser() {
        return new Promise(resolve => setTimeout(resolve, 0));
    }
    
    /**
     * Process items in chunks, yielding to browser between chunks
     * @param {Array} items - Items to process
     * @param {number} chunkSize - Items per chunk
     * @param {Function} processor - Function to call for each item
     * @param {Function} onProgress - Optional progress callback (called with percent)
     */
    async function processInChunks(items, chunkSize, processor, onProgress = null) {
        for (let i = 0; i < items.length; i += chunkSize) {
            const chunk = items.slice(i, i + chunkSize);
            for (const item of chunk) {
                processor(item);
            }
            if (onProgress) {
                onProgress(Math.round((i / items.length) * 100));
            }
            // Yield to browser every chunk
            await yieldToBrowser();
        }
    }
    
    /**
     * Load animation - new approach: load only visible elements, add/remove diffs
     */
    async function handleLoadAnimation() {
        if (!currentBaseSqlFile) {
            Toast.show('Please load a graph first', 'warning');
            return;
        }
        
        const now = Date.now();
        if (now - animationLoadTimestamp < 2000) {
            console.log('[Snapshots] Animation debounced');
            return;
        }
        
        if (animationLoading) {
            console.log('[Snapshots] Animation already loading, ignoring');
            return;
        }
        
        const snapshots = State.getAvailableSnapshots();
        
        if (!snapshots || snapshots.length < 2) {
            Toast.show('Need at least 2 snapshots for animation', 'warning');
            return;
        }
        
        const includeEdges = elements.animIncludeEdges?.checked || false;
        
        // Check if animation is already loaded with same parameters
        if (animationNodeSets && animationNodeSets.length > 0 && 
            animationFrames.length === snapshots.length &&
            animationHasEdges === includeEdges) {
            
            // Check if snapshots are the same
            const sortedNew = [...snapshots].sort((a, b) => a.block_number - b.block_number);
            let same = true;
            for (let i = 0; i < sortedNew.length && same; i++) {
                if (sortedNew[i].snapshot_id !== animationFrames[i].snapshot_id) {
                    same = false;
                }
            }
            
            if (same) {
                console.log('[Snapshots] Animation already loaded, restarting from frame 0');
                Toast.show('Animation already loaded - use Play button to start', 'info');
                
                // Just restart from frame 0
                await displayAnimationFrame(0);
                enableAnimationControls(true);
                return;
            }
        }
        
        animationLoadTimestamp = now;
        
        stopAnimation();
        
        animationFrames = [...snapshots].sort((a, b) => a.block_number - b.block_number);
        currentFrameIndex = 0;
        animationNodeSets = [];
        animationHasEdges = includeEdges;
        animationLoading = true;
        animationAllNodes = new Map();
        animationAllEdges = new Map();
        
        console.log('[Snapshots] Starting animation with', animationFrames.length, 'frames', 
            includeEdges ? '(with edges)' : '(nodes only)');
        
        setLoading(true);
        showProgress(true, 0, 'Loading animation data...');
        
        // Disable all animation controls during load to prevent interaction
        enableAnimationControls(false);
        if (elements.loadAnimationBtn) elements.loadAnimationBtn.disabled = true;
        
        const container = document.getElementById('cy');
        
        try {
            const cy = State.cy;
            if (!cy) {
                Toast.show('No graph instance available', 'error');
                return;
            }
            
            // Load the LAST snapshot to get all possible nodes/edges
            const lastFrame = animationFrames[animationFrames.length - 1];
            
            showProgress(true, 5, 'Loading all nodes...');
            const nodesData = await SnapshotAPI.getSnapshotNodes(lastFrame.snapshot_id);
            
            if (!nodesData.elements || nodesData.elements.length === 0) {
                Toast.show('Latest snapshot has no nodes', 'error');
                return;
            }
            
            // Store all nodes in a Map for fast lookup - NON-BLOCKING with chunked processing
            const nodeElements = nodesData.elements;
            const NODE_CHUNK_SIZE = 5000; // Process 5k nodes per chunk
            for (let i = 0; i < nodeElements.length; i += NODE_CHUNK_SIZE) {
                const chunk = nodeElements.slice(i, i + NODE_CHUNK_SIZE);
                for (const el of chunk) {
                    animationAllNodes.set(el.data.id, el);
                }
                // Yield to browser between chunks
                await yieldToBrowser();
                const pct = Math.round((i / nodeElements.length) * 5) + 5;
                showProgress(true, pct, `Indexing nodes ${Math.min(i + NODE_CHUNK_SIZE, nodeElements.length)}/${nodeElements.length}...`);
            }
            console.log('[Snapshots] Indexed', animationAllNodes.size, 'nodes');
            
            // Load edges if requested - NON-BLOCKING
            if (includeEdges) {
                showProgress(true, 10, 'Loading all edges...');
                
                try {
                    const response = await fetch(`/api/snapshots/${lastFrame.snapshot_id}/edges/lightweight`);
                    if (response.ok) {
                        const edgeData = await response.json();
                        
                        if (edgeData.edges && edgeData.edges.length > 0) {
                            const edges = edgeData.edges;
                            const EDGE_CHUNK_SIZE = 10000; // Process 10k edges per chunk
                            
                            for (let i = 0; i < edges.length; i += EDGE_CHUNK_SIZE) {
                                const chunk = edges.slice(i, i + EDGE_CHUNK_SIZE);
                                for (const [source, target] of chunk) {
                                    const id = `${source}->${target}`;
                                    animationAllEdges.set(id, {
                                        group: 'edges',
                                        data: { id, source, target }
                                    });
                                }
                                // Yield to browser between chunks
                                await yieldToBrowser();
                                const pct = Math.round((i / edges.length) * 5) + 10;
                                showProgress(true, pct, `Indexing edges ${Math.min(i + EDGE_CHUNK_SIZE, edges.length)}/${edges.length}...`);
                            }
                            console.log('[Snapshots] Indexed', animationAllEdges.size, 'edges');
                        }
                    }
                } catch (e) {
                    console.warn('[Snapshots] Failed to load edges:', e);
                    animationHasEdges = false;
                }
            }
            
            showProgress(true, 15, 'Pre-computing frames...');
            
            // Pre-compute which nodes/edges are visible in each frame
            // Process ONE frame at a time to allow UI updates
            for (let i = 0; i < animationFrames.length; i++) {
                const frame = animationFrames[i];
                
                try {
                    const response = await fetch(`/api/snapshots/${frame.snapshot_id}/layout`);
                    if (response.ok) {
                        const data = await response.json();
                        const nodeIds = new Set(Object.keys(data.positions || {}));
                        
                        // Compute visible edge IDs - chunked for large edge sets
                        let edgeIds = new Set();
                        if (animationHasEdges && animationAllEdges.size > 0) {
                            const edgeEntries = Array.from(animationAllEdges.entries());
                            const EDGE_VIS_CHUNK = 20000;
                            
                            for (let j = 0; j < edgeEntries.length; j += EDGE_VIS_CHUNK) {
                                const chunk = edgeEntries.slice(j, j + EDGE_VIS_CHUNK);
                                for (const [edgeId, edgeEl] of chunk) {
                                    if (nodeIds.has(edgeEl.data.source) && nodeIds.has(edgeEl.data.target)) {
                                        edgeIds.add(edgeId);
                                    }
                                }
                                // Yield only for very large edge sets
                                if (edgeEntries.length > EDGE_VIS_CHUNK) {
                                    await yieldToBrowser();
                                }
                            }
                        }
                        
                        animationNodeSets.push({
                            block_number: frame.block_number,
                            timestamp: frame.block_timestamp,
                            nodeIds,
                            edgeIds,
                            nodeCount: nodeIds.size,
                            edgeCount: edgeIds.size
                        });
                    } else {
                        animationNodeSets.push({
                            block_number: frame.block_number,
                            timestamp: frame.block_timestamp,
                            nodeIds: new Set(),
                            edgeIds: new Set(),
                            nodeCount: 0,
                            edgeCount: 0
                        });
                    }
                } catch (e) {
                    console.warn('[Snapshots] Failed to load frame:', frame.snapshot_id, e);
                    animationNodeSets.push({
                        block_number: frame.block_number,
                        timestamp: frame.block_timestamp,
                        nodeIds: new Set(),
                        edgeIds: new Set(),
                        nodeCount: 0,
                        edgeCount: 0
                    });
                }
                
                // Yield after each frame to keep UI responsive
                await yieldToBrowser();
                
                const progress = 15 + Math.round((animationNodeSets.length / animationFrames.length) * 80);
                showProgress(true, progress, `Pre-computing frame ${animationNodeSets.length}/${animationFrames.length}...`);
            }
            
            console.log('[Snapshots] Pre-computed', animationNodeSets.length, 'frames');
            
            // Clear current graph and load FIRST frame only
            showProgress(true, 95, 'Loading first frame...');
            
            // CRITICAL: Disable pointer events during bulk add to prevent WebGL crash
            const container = document.getElementById('cy');
            container.style.pointerEvents = 'none';
            
            const firstFrame = animationNodeSets[0];
            const firstNodes = [];
            const firstEdges = [];
            
            // Build first frame nodes in chunks - NON-BLOCKING
            const firstNodeIds = Array.from(firstFrame.nodeIds);
            const FIRST_NODE_CHUNK = 5000;
            for (let i = 0; i < firstNodeIds.length; i += FIRST_NODE_CHUNK) {
                const chunk = firstNodeIds.slice(i, i + FIRST_NODE_CHUNK);
                for (const nodeId of chunk) {
                    const nodeEl = animationAllNodes.get(nodeId);
                    if (nodeEl) firstNodes.push(nodeEl);
                }
                if (firstNodeIds.length > FIRST_NODE_CHUNK) {
                    await yieldToBrowser();
                }
            }
            
            if (animationHasEdges) {
                const firstEdgeIds = Array.from(firstFrame.edgeIds);
                const FIRST_EDGE_CHUNK = 10000;
                for (let i = 0; i < firstEdgeIds.length; i += FIRST_EDGE_CHUNK) {
                    const chunk = firstEdgeIds.slice(i, i + FIRST_EDGE_CHUNK);
                    for (const edgeId of chunk) {
                        const edgeEl = animationAllEdges.get(edgeId);
                        if (edgeEl) firstEdges.push(edgeEl);
                    }
                    if (firstEdgeIds.length > FIRST_EDGE_CHUNK) {
                        await yieldToBrowser();
                    }
                }
            }
            
            // Batch all operations together
            cy.startBatch();
            cy.elements().remove();
            cy.endBatch();
            
            await yieldToBrowser();
            
            // Add nodes in chunks to Cytoscape
            const CY_CHUNK = 5000;
            for (let i = 0; i < firstNodes.length; i += CY_CHUNK) {
                const chunk = firstNodes.slice(i, i + CY_CHUNK);
                cy.startBatch();
                cy.add(chunk);
                cy.endBatch();
                if (firstNodes.length > CY_CHUNK) {
                    await yieldToBrowser();
                }
            }
            
            // Add edges in chunks to Cytoscape
            if (animationHasEdges && firstEdges.length > 0) {
                for (let i = 0; i < firstEdges.length; i += CY_CHUNK) {
                    const chunk = firstEdges.slice(i, i + CY_CHUNK);
                    cy.startBatch();
                    cy.add(chunk);
                    cy.endBatch();
                    if (firstEdges.length > CY_CHUNK) {
                        await yieldToBrowser();
                    }
                }
            }
            
            // Fit after batch completes
            cy.fit();
            
            console.log('[Snapshots] Loaded first frame:', firstNodes.length, 'nodes,', firstEdges.length, 'edges');
            
            // Re-enable pointer events after WebGL settles (500ms for 247k elements)
            setTimeout(() => {
                container.style.pointerEvents = 'auto';
            }, 500);
            
            // Update UI
            if (elements.animTimeline) {
                elements.animTimeline.max = animationFrames.length - 1;
                elements.animTimeline.value = 0;
            }
            if (elements.animTotalLabel) {
                elements.animTotalLabel.textContent = `${animationFrames.length} frames`;
            }
            if (elements.animCurrentLabel) {
                elements.animCurrentLabel.textContent = 'Frame 1';
            }
            
            // Update counts
            document.getElementById('node-count').textContent = `${firstFrame.nodeCount.toLocaleString()} nodes`;
            document.getElementById('edge-count').textContent = animationHasEdges ? 
                `${firstFrame.edgeCount.toLocaleString()} edges` : 'edges hidden';
            
            currentSnapshotId = lastFrame.snapshot_id;
            State.setSnapshotActive(true, lastFrame);
            
            enableAnimationControls(true);
            updateAnimationUI(0);
            
            showProgress(true, 100, 'Ready!');
            Toast.show(`Animation ready: ${animationFrames.length} frames`, 'success');
            
        } catch (error) {
            console.error('[Snapshots] Failed to load animation:', error);
            Toast.show('Failed to load animation: ' + error.message, 'error');
            animationNodeSets = [];
            animationFrames = [];
            animationHasEdges = false;
        } finally {
            animationLoading = false;
            setLoading(false);
            if (elements.loadAnimationBtn) elements.loadAnimationBtn.disabled = false;
            
            // CRITICAL: Always re-enable pointer events, even on error
            if (container) {
                setTimeout(() => {
                    container.style.pointerEvents = 'auto';
                }, 100);
            }
            
            setTimeout(() => showProgress(false), 500);
        }
    }
    
    /**
     * Update animation UI labels
     */
    function updateAnimationUI(frameIndex) {
        const frame = animationNodeSets[frameIndex];
        if (!frame) return;
        
        if (elements.animTimeline) elements.animTimeline.value = frameIndex;
        if (elements.animCurrentLabel) elements.animCurrentLabel.textContent = `Frame ${frameIndex + 1}`;
        if (elements.animBlockInfo) elements.animBlockInfo.textContent = `Block: ${frame.block_number.toLocaleString()}`;
        if (elements.animDateInfo && frame.timestamp) {
            elements.animDateInfo.textContent = `Date: ${new Date(frame.timestamp).toLocaleDateString()}`;
        }
        
        document.getElementById('node-count').textContent = `${frame.nodeCount.toLocaleString()} nodes`;
        document.getElementById('edge-count').textContent = animationHasEdges ? 
            `${frame.edgeCount.toLocaleString()} edges` : 'edges hidden';
        
        updateStatusIndicator(false, null, false, true, frameIndex + 1, animationFrames.length);
    }
    
    /**
     * Display animation frame - uses add/remove for fast transitions
     * Made async to avoid blocking the UI during large operations
     */
    async function displayAnimationFrame(targetFrameIndex) {
        if (!animationNodeSets || targetFrameIndex < 0 || targetFrameIndex >= animationNodeSets.length) {
            console.warn('[Snapshots] displayAnimationFrame: invalid state', {
                hasNodeSets: !!animationNodeSets,
                length: animationNodeSets?.length,
                targetFrameIndex
            });
            return;
        }
        
        const cy = State.cy;
        if (!cy) {
            console.warn('[Snapshots] displayAnimationFrame: no cy instance');
            return;
        }
        
        const currFrame = animationNodeSets[currentFrameIndex];
        const targetFrame = animationNodeSets[targetFrameIndex];
        
        console.log('[Snapshots] displayAnimationFrame:', {
            from: currentFrameIndex,
            to: targetFrameIndex,
            currNodes: currFrame?.nodeIds?.size,
            targetNodes: targetFrame?.nodeIds?.size
        });
        
        if (currentFrameIndex === targetFrameIndex) {
            updateAnimationUI(targetFrameIndex);
            return;
        }
        
        const frameDiff = Math.abs(targetFrameIndex - currentFrameIndex);
        
        // For large jumps (more than 5 frames), use direct rebuild instead of incremental diff
        // This is much faster when jumping backwards to early frames with fewer nodes
        if (frameDiff > 5) {
            console.log('[Snapshots] Large jump detected, using direct rebuild');
            
            // Build arrays in chunks to avoid blocking - NON-BLOCKING
            const nodesToAdd = [];
            const edgesToAdd = [];
            
            const nodeIds = Array.from(targetFrame.nodeIds);
            const NODE_BUILD_CHUNK = 10000;
            
            for (let i = 0; i < nodeIds.length; i += NODE_BUILD_CHUNK) {
                const chunk = nodeIds.slice(i, i + NODE_BUILD_CHUNK);
                for (const nodeId of chunk) {
                    const nodeEl = animationAllNodes.get(nodeId);
                    if (nodeEl) nodesToAdd.push(nodeEl);
                }
                // Yield every chunk for large node sets
                if (nodeIds.length > NODE_BUILD_CHUNK) {
                    await yieldToBrowser();
                }
            }
            
            if (animationHasEdges) {
                const edgeIds = Array.from(targetFrame.edgeIds);
                const EDGE_BUILD_CHUNK = 20000;
                
                for (let i = 0; i < edgeIds.length; i += EDGE_BUILD_CHUNK) {
                    const chunk = edgeIds.slice(i, i + EDGE_BUILD_CHUNK);
                    for (const edgeId of chunk) {
                        const edgeEl = animationAllEdges.get(edgeId);
                        if (edgeEl) edgesToAdd.push(edgeEl);
                    }
                    // Yield every chunk for large edge sets
                    if (edgeIds.length > EDGE_BUILD_CHUNK) {
                        await yieldToBrowser();
                    }
                }
            }
            
            // Now apply to Cytoscape in chunks
            cy.startBatch();
            cy.elements().remove();
            cy.endBatch();
            
            await yieldToBrowser();
            
            // Add nodes in chunks
            const CY_ADD_CHUNK = 5000;
            for (let i = 0; i < nodesToAdd.length; i += CY_ADD_CHUNK) {
                const chunk = nodesToAdd.slice(i, i + CY_ADD_CHUNK);
                cy.startBatch();
                cy.add(chunk);
                cy.endBatch();
                if (nodesToAdd.length > CY_ADD_CHUNK) {
                    await yieldToBrowser();
                }
            }
            
            // Add edges in chunks
            for (let i = 0; i < edgesToAdd.length; i += CY_ADD_CHUNK) {
                const chunk = edgesToAdd.slice(i, i + CY_ADD_CHUNK);
                cy.startBatch();
                cy.add(chunk);
                cy.endBatch();
                if (edgesToAdd.length > CY_ADD_CHUNK) {
                    await yieldToBrowser();
                }
            }
            
            console.log('[Snapshots] Direct rebuild complete:', {
                nodesAdded: nodesToAdd.length,
                edgesAdded: edgesToAdd.length
            });
            
            currentFrameIndex = targetFrameIndex;
            updateAnimationUI(targetFrameIndex);
            return;
        }
        
        // For small jumps, use incremental diff (more efficient for 1-5 frame changes)
        const nodesToAdd = [];
        const nodesToRemove = [];
        const edgesToAdd = [];
        const edgesToRemove = [];
        
        // Nodes to remove (in current but not in target)
        for (const nodeId of currFrame.nodeIds) {
            if (!targetFrame.nodeIds.has(nodeId)) {
                nodesToRemove.push(nodeId);
            }
        }
        
        // Nodes to add (in target but not in current)
        for (const nodeId of targetFrame.nodeIds) {
            if (!currFrame.nodeIds.has(nodeId)) {
                const nodeEl = animationAllNodes.get(nodeId);
                if (nodeEl) nodesToAdd.push(nodeEl);
            }
        }
        
        // Same for edges
        if (animationHasEdges) {
            for (const edgeId of currFrame.edgeIds) {
                if (!targetFrame.edgeIds.has(edgeId)) {
                    edgesToRemove.push(edgeId);
                }
            }
            
            for (const edgeId of targetFrame.edgeIds) {
                if (!currFrame.edgeIds.has(edgeId)) {
                    const edgeEl = animationAllEdges.get(edgeId);
                    if (edgeEl) edgesToAdd.push(edgeEl);
                }
            }
        }
        
        console.log('[Snapshots] Frame diff:', {
            nodesToAdd: nodesToAdd.length,
            nodesToRemove: nodesToRemove.length,
            edgesToAdd: edgesToAdd.length,
            edgesToRemove: edgesToRemove.length
        });
        
        // Apply changes in batch
        cy.startBatch();
        
        // Remove elements
        for (const id of nodesToRemove) {
            cy.getElementById(id).remove();
        }
        for (const id of edgesToRemove) {
            cy.getElementById(id).remove();
        }
        
        // Add elements
        if (nodesToAdd.length > 0) cy.add(nodesToAdd);
        if (edgesToAdd.length > 0) cy.add(edgesToAdd);
        
        cy.endBatch();
        
        currentFrameIndex = targetFrameIndex;
        updateAnimationUI(targetFrameIndex);
    }
    
    /**
     * Play/pause toggle
     */
    function handleAnimPlayPause() {
        if (animationLoading || !animationNodeSets || animationNodeSets.length === 0) return;
        if (isAnimating) {
            stopAnimation();
        } else {
            startAnimation();
        }
    }
    
    /**
     * Start animation playback
     */
    async function startAnimation() {
        console.log('[Snapshots] startAnimation called', {
            animationLoading,
            hasNodeSets: !!animationNodeSets,
            length: animationNodeSets?.length,
            isAnimating,
            currentFrameIndex
        });
        
        if (animationLoading) {
            console.log('[Snapshots] Animation still loading, ignoring');
            return;
        }
        if (!animationNodeSets || animationNodeSets.length < 2) {
            console.log('[Snapshots] Not enough frames');
            return;
        }
        if (isAnimating) {
            console.log('[Snapshots] Already animating');
            return;
        }
        
        // If we're at the last frame, restart from the beginning
        if (currentFrameIndex >= animationNodeSets.length - 1) {
            console.log('[Snapshots] At last frame, restarting from beginning');
            await displayAnimationFrame(0);
        }
        
        isAnimating = true;
        
        // Disable pointer events during playback to prevent WebGL crashes
        const container = document.getElementById('cy');
        container.style.pointerEvents = 'none';
        
        if (elements.animPlayBtn) {
            elements.animPlayBtn.textContent = '||';
            elements.animPlayBtn.classList.add('playing');
        }
        
        const playNextFrame = async () => {
            if (!isAnimating) return;
            
            const nextFrame = currentFrameIndex + 1;
            if (nextFrame >= animationNodeSets.length) {
                stopAnimation();
                Toast.show('Animation complete', 'info');
                return;
            }
            
            await displayAnimationFrame(nextFrame);
            
            if (!isAnimating) return;
            
            // Fast - add/remove is instant
            const delayMs = Math.max(200, 800 / animationSpeed);
            animationInterval = setTimeout(playNextFrame, delayMs);
        };
        
        const initialDelay = Math.max(200, 800 / animationSpeed);
        animationInterval = setTimeout(playNextFrame, initialDelay);
    }
    
    /**
     * Stop animation
     */
    function stopAnimation() {
        isAnimating = false;
        if (animationInterval) {
            clearTimeout(animationInterval);
            animationInterval = null;
        }
        if (elements.animPlayBtn) {
            elements.animPlayBtn.textContent = 'v';
            elements.animPlayBtn.classList.remove('playing');
        }
        
        // Re-enable pointer events after stopping
        const container = document.getElementById('cy');
        if (container) {
            setTimeout(() => {
                container.style.pointerEvents = 'auto';
            }, 300);
        }
    }
    
    /**
     * Previous frame
     */
    async function handleAnimPrev() {
        if (animationLoading || !animationNodeSets || animationNodeSets.length === 0) return;
        stopAnimation();
        await displayAnimationFrame(Math.max(0, currentFrameIndex - 1));
    }
    
    /**
     * Next frame
     */
    async function handleAnimNext() {
        if (animationLoading || !animationNodeSets || animationNodeSets.length === 0) return;
        stopAnimation();
        await displayAnimationFrame(Math.min(animationNodeSets.length - 1, currentFrameIndex + 1));
    }
    
    /**
     * Timeline slider - preview on drag, render on release
     */
    async function handleAnimTimelineChange() {
        // Guard: don't process if animation is loading or no frames
        if (animationLoading || !animationNodeSets || animationNodeSets.length === 0) {
            return;
        }
        stopAnimation();
        const targetFrame = parseInt(elements.animTimeline.value);
        
        // Disable pointer events during frame transition
        const container = document.getElementById('cy');
        if (container) container.style.pointerEvents = 'none';
        
        await displayAnimationFrame(targetFrame);
        
        // Re-enable after transition
        if (container) {
            setTimeout(() => {
                container.style.pointerEvents = 'auto';
            }, 100);
        }
    }
    
    /**
     * Preview frame info while dragging (without rendering)
     */
    function handleAnimTimelinePreview() {
        if (!animationFrames || animationFrames.length === 0) return;
        const frameIndex = parseInt(elements.animTimeline.value);
        const frame = animationFrames[frameIndex];
        if (frame) {
            // Update only the labels, not the graph
            if (elements.animCurrentLabel) elements.animCurrentLabel.textContent = frameIndex + 1;
            if (elements.animBlockInfo) elements.animBlockInfo.textContent = `Block: ${frame.block_number}`;
            if (elements.animDateInfo) {
                const date = new Date(frame.block_timestamp);
                elements.animDateInfo.textContent = `Date: ${date.toLocaleDateString()}`;
            }
        }
    }
    
    /**
     * Restart animation from beginning
     */
    async function restartAnimation() {
        if (animationLoading || !animationNodeSets || animationNodeSets.length === 0) {
            return;
        }
        stopAnimation();
        
        // Jump to first frame
        await displayAnimationFrame(0);
        
        // Update slider
        if (elements.animTimeline) elements.animTimeline.value = 0;
    }
    
    /**
     * Speed slider
     */
    function handleAnimSpeedChange() {
        animationSpeed = parseFloat(elements.animSpeedSlider.value);
        if (elements.animSpeedLabel) {
            elements.animSpeedLabel.textContent = `${animationSpeed}x`;
        }
    }
    
    /**
     * Enable/disable controls
     */
    function enableAnimationControls(enabled) {
        [elements.animPrevBtn, elements.animPlayBtn, elements.animNextBtn, elements.animTimeline]
            .forEach(ctrl => { if (ctrl) ctrl.disabled = !enabled; });
        
        // Disable Load Edges button during animation (re-enable when animation ends)
        const loadEdgesBtn = document.getElementById('load-edges-btn');
        if (loadEdgesBtn) {
            loadEdgesBtn.disabled = enabled;
            loadEdgesBtn.title = enabled ? 'Disabled during animation' : 'Load edges for this snapshot';
        }
    }
    
    /**
     * Reset animation state
     */
    function resetAnimationState() {
        console.log('[Snapshots] Resetting animation state');
        stopAnimation();
        animationNodeSets = [];
        animationFrames = [];
        currentFrameIndex = 0;
        animationHasEdges = false;
        animationLoading = false;
        animationLoadTimestamp = 0;
        animationAllNodes = null;
        animationAllEdges = null;
        enableAnimationControls(false);
        
        // Re-enable Load Edges button
        const loadEdgesBtn = document.getElementById('load-edges-btn');
        if (loadEdgesBtn) {
            loadEdgesBtn.disabled = false;
            loadEdgesBtn.title = 'Load edges for this snapshot';
        }
        
        // Reset UI labels
        if (elements.animTimeline) { elements.animTimeline.value = 0; elements.animTimeline.max = 0; }
        if (elements.animCurrentLabel) elements.animCurrentLabel.textContent = '-';
        if (elements.animTotalLabel) elements.animTotalLabel.textContent = '-';
        if (elements.animBlockInfo) elements.animBlockInfo.textContent = 'Block: -';
        if (elements.animDateInfo) elements.animDateInfo.textContent = 'Date: -';
    }
    // ==========================================================================
    // PUBLIC API
    // ==========================================================================
    
    return {
        init,
        loadAvailableSnapshots,
        
        // Load specific snapshot by ID
        loadSnapshotById: async (snapshotId) => {
            if (elements.dropdown) {
                elements.dropdown.value = snapshotId;
            }
            await handleLoadSnapshot();
        },
        
        // Return to live view
        returnToLive: handleReturnToLive,
        
        // Check if viewing snapshot
        isActive: () => State.isViewingSnapshot(),
        
        // Get current snapshot info
        getCurrentSnapshot: () => State.getCurrentSnapshot(),
        
        // Get current snapshot ID (for edge loading)
        getCurrentSnapshotId: () => currentSnapshotId,
        
        // Load edges for current snapshot (called by Load Edges button)
        loadEdges: loadSnapshotEdges,
        
        // Clear edges from snapshot
        clearEdges: clearSnapshotEdges,
        
        // For GraphLoader/AutoReload integration
        shouldPreventAutoReload,
        
        // Refresh snapshot list
        refresh: loadAvailableSnapshots,
        
        // Comparison
        isComparing: () => isComparing,
        compare: handleCompare,
        exitCompare: handleExitCompare,
        
        // Animation
        isAnimating: () => isAnimating,
        loadAnimation: handleLoadAnimation,
        playAnimation: startAnimation,
        stopAnimation: stopAnimation,
        restartAnimation: restartAnimation,
        nextFrame: handleAnimNext,
        prevFrame: handleAnimPrev,
        gotoFrame: displayAnimationFrame
    };
})();