/**
 * Embedding Panel Module
 * 
 * UI for GIT-CD deep learning features:
 * - Model training (async with status polling)
 * - Community detection
 * - Similarity search
 */

const EmbeddingPanel = (function() {
    'use strict';
    
    // State
    let _initialized = false;
    let _pollingInterval = null;
    let _lastCommunities = null;
    let _lastSimilarNodes = null;
    let _trainingMonitorWindow = null;
    let _lastTrainingStatus = null;
    
    // Polling configuration
    const POLL_INTERVAL_MS = 2000;
    
    /**
     * Initialize the embedding panel.
     */
    async function init() {
        if (_initialized) return;
        
        console.log('[EmbeddingPanel] Initializing...');
        
        // Check availability
        const availability = await EmbeddingsAPI.checkAvailability();
        console.log('[EmbeddingPanel] Availability:', availability);
        
        updateAvailabilityUI(availability);
        
        if (availability.available) {
            setupTabs();
            setupTrainTab();
            setupCommunitiesTab();
            setupSimilarityTab();
            setupLoadModelButton();
            setupGraphNameDisplay();
            setupTrainingMonitorButton();
            
            // Update model name if already loaded
            if (availability.hasModel) {
                updateModelName('gitcd_model');
            }
        }
        
        _initialized = true;
        console.log('[EmbeddingPanel] Initialized');
    }
    
    /**
     * Setup graph name display and model name placeholder.
     */
    function setupGraphNameDisplay() {
        const graphNameEl = document.getElementById('training-graph-name');
        const modelNameInput = document.getElementById('model-name-input');
        
        function updateGraphInfo() {
            // State is a global variable, not window.State
            const currentGraph = (typeof State !== 'undefined' && State.currentGraph) ? State.currentGraph : null;
            
            // Update graph name display
            if (graphNameEl) {
                graphNameEl.textContent = currentGraph || 'No graph loaded';
            }
            
            // Update model name placeholder
            if (modelNameInput && currentGraph) {
                modelNameInput.placeholder = `gitcd_${currentGraph}`;
            }
        }
        
        // Initial update
        updateGraphInfo();
        
        // Update when graph changes (poll every 2 seconds as fallback)
        setInterval(updateGraphInfo, 2000);
    }
    
    /**
     * Setup training monitor button.
     */
    function setupTrainingMonitorButton() {
        const btn = document.getElementById('btn-open-training-monitor');
        if (!btn) return;
        
        btn.addEventListener('click', () => {
            openTrainingMonitor();
        });
    }
    
    /**
     * Open training monitor popup window.
     */
    function openTrainingMonitor() {
        // Close existing window if open
        if (_trainingMonitorWindow && !_trainingMonitorWindow.closed) {
            _trainingMonitorWindow.focus();
            return;
        }
        
        const width = 900;
        const height = 700;
        const left = (screen.width - width) / 2;
        const top = (screen.height - height) / 2;
        
        _trainingMonitorWindow = window.open(
            '/static/training-monitor.html',
            'TrainingMonitor',
            `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
        );
        
        // Listen for ready signal from popup
        window.addEventListener('message', function onReady(event) {
            if (event.data.type === 'training_monitor_ready') {
                console.log('[EmbeddingPanel] Training monitor ready, sending current status');
                // Send current status if we have one
                if (_lastTrainingStatus) {
                    updateTrainingMonitor(_lastTrainingStatus);
                }
                window.removeEventListener('message', onReady);
            }
        });
    }
    
    /**
     * Update training monitor with status.
     */
    function updateTrainingMonitor(status) {
        // Store last status for resending when popup opens
        _lastTrainingStatus = status;
        
        if (_trainingMonitorWindow && !_trainingMonitorWindow.closed) {
            try {
                _trainingMonitorWindow.postMessage({
                    type: 'training_update',
                    data: status
                }, '*');
            } catch (e) {
                console.warn('[EmbeddingPanel] Failed to post to training monitor:', e);
            }
        }
    }
    
    /**
     * Update UI based on availability.
     */
    function updateAvailabilityUI(availability) {
        const statusEl = document.getElementById('dl-status');
        const unavailableEl = document.getElementById('dl-unavailable');
        const availableEl = document.getElementById('dl-available');
        
        if (!statusEl) return;
        
        if (availability.available) {
            statusEl.classList.add('available');
            statusEl.classList.remove('unavailable');
            const statusText = statusEl.querySelector('.status-text');
            if (statusText) {
                const device = availability.cuda ? 'CUDA' : (availability.mps ? 'MPS' : 'CPU');
                statusText.textContent = `Available - ${device}`;
            }
            
            if (unavailableEl) unavailableEl.style.display = 'none';
            if (availableEl) availableEl.style.display = 'block';
        } else {
            statusEl.classList.add('unavailable');
            statusEl.classList.remove('available');
            const statusText = statusEl.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = 'Not Available';
            }
            
            if (unavailableEl) unavailableEl.style.display = 'block';
            if (availableEl) availableEl.style.display = 'none';
        }
    }
    
    /**
     * Setup tab navigation.
     */
    function setupTabs() {
        const tabBtns = document.querySelectorAll('.embed-tab-btn');
        
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;
                
                // Update active button
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Update active content
                const tabContents = document.querySelectorAll('.embed-tab-content');
                tabContents.forEach(content => {
                    if (content.id === tabId) {
                        content.style.display = 'block';
                        content.classList.add('active');
                    } else {
                        content.style.display = 'none';
                        content.classList.remove('active');
                    }
                });
            });
        });
    }
    
    /**
     * Update model name display.
     */
    function updateModelName(name) {
        const el = document.querySelector('#current-model-status .model-name');
        if (el) el.textContent = name || 'No model loaded';
    }
    
    /**
     * Setup load model button with dropdown.
     */
    function setupLoadModelButton() {
        const loadBtn = document.getElementById('btn-load-model');
        if (!loadBtn) return;
        
        loadBtn.addEventListener('click', async () => {
            try {
                // Remove existing dropdown if any
                const existingDropdown = document.getElementById('model-load-dropdown');
                if (existingDropdown) {
                    existingDropdown.remove();
                    return;
                }
                
                showToast('Loading available models...', 'info');
                
                const result = await EmbeddingsAPI.listModels();
                
                if (!result.models || result.models.length === 0) {
                    showToast('No saved models found', 'info');
                    return;
                }
                
                // Create dropdown
                const dropdown = document.createElement('div');
                dropdown.id = 'model-load-dropdown';
                dropdown.className = 'model-dropdown';
                dropdown.innerHTML = `
                    <div class="model-dropdown-header">Select Model</div>
                    <div class="model-dropdown-list">
                        ${result.models.map(m => `
                            <div class="model-dropdown-item" data-model="${m.name}">
                                <span class="model-item-name">${m.name}</span>
                                ${m.num_clusters ? `<span class="model-item-info">${m.num_clusters} clusters</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                `;
                
                // Position dropdown below the model status box
                const statusBox = document.getElementById('current-model-status');
                if (statusBox) {
                    statusBox.parentElement.appendChild(dropdown);
                } else {
                    loadBtn.parentElement.appendChild(dropdown);
                }
                
                // Add click handlers
                dropdown.querySelectorAll('.model-dropdown-item').forEach(item => {
                    item.addEventListener('click', async () => {
                        const modelName = item.dataset.model;
                        dropdown.remove();
                        
                        try {
                            showToast(`Loading model "${modelName}"...`, 'info');
                            const loadResult = await EmbeddingsAPI.loadModel(modelName);
                            if (loadResult.success) {
                                updateModelName(modelName);
                                showToast(`Model "${modelName}" loaded successfully`, 'success');
                            } else {
                                showToast(`Failed to load model: ${loadResult.message || 'Unknown error'}`, 'error');
                            }
                        } catch (error) {
                            console.error('[EmbeddingPanel] Load model error:', error);
                            showToast(`Failed: ${error.message}`, 'error');
                        }
                    });
                });
                
                // Close on click outside
                setTimeout(() => {
                    document.addEventListener('click', function closeDropdown(e) {
                        if (!dropdown.contains(e.target) && e.target !== loadBtn) {
                            dropdown.remove();
                            document.removeEventListener('click', closeDropdown);
                        }
                    });
                }, 100);
                
            } catch (error) {
                console.error('[EmbeddingPanel] List models error:', error);
                showToast(`Failed to load models: ${error.message}`, 'error');
            }
        });
    }
    
    /**
     * Setup train tab with async training.
     */
    function setupTrainTab() {
        const trainBtn = document.getElementById('btn-train-model');
        const progressEl = document.getElementById('training-progress');
        const resultsEl = document.getElementById('training-results');
        
        if (!trainBtn) return;
        
        trainBtn.addEventListener('click', async () => {
            // Get current graph from state (State is a global variable)
            const currentGraph = (typeof State !== 'undefined' && State.currentGraph) ? State.currentGraph : null;
            
            if (!currentGraph) {
                showToast('No graph loaded. Please load a network first.', 'error');
                return;
            }
            
            // Get model name - if empty, backend will auto-generate from graph name
            const modelNameInput = document.getElementById('model-name-input')?.value?.trim();
            
            // Get config from inputs
            const config = {
                graph_name: currentGraph,
                model_name: modelNameInput || null,  // null = auto-generate
                num_clusters: parseInt(document.getElementById('num-clusters-input')?.value) || 20,
                hidden_dim: parseInt(document.getElementById('hidden-dim-input')?.value) || 128,
                num_gnn_layers: parseInt(document.getElementById('gnn-layers-input')?.value) || 1,
                num_transformer_layers: parseInt(document.getElementById('transformer-layers-input')?.value) || 2,
                dropout: parseFloat(document.getElementById('dropout-input')?.value) || 0.5,
                max_epochs: parseInt(document.getElementById('max-epochs-input')?.value) || 200,
                learning_rate: parseFloat(document.getElementById('learning-rate-input')?.value) || 0.0003,
                patience: parseInt(document.getElementById('patience-input')?.value) || 5,
            };
            
            // Show progress
            if (progressEl) {
                progressEl.style.display = 'block';
                updateProgress(0, `Starting training on ${currentGraph}...`);
            }
            if (resultsEl) resultsEl.style.display = 'none';
            
            trainBtn.disabled = true;
            const originalText = trainBtn.textContent;
            trainBtn.textContent = 'Training...';
            
            try {
                showToast(`Starting training on ${currentGraph}...`, 'info');
                
                // Start async training
                const startResult = await EmbeddingsAPI.trainModel(config);
                
                if (startResult.task_id) {
                    // Poll for status
                    console.log('[EmbeddingPanel] Training started, task_id:', startResult.task_id, 'graph:', currentGraph);
                    
                    await pollTrainingStatus(startResult.task_id, progressEl, resultsEl);
                } else if (startResult.success) {
                    // Synchronous result
                    handleTrainingComplete(startResult, resultsEl);
                    if (progressEl) progressEl.style.display = 'none';
                } else {
                    showToast('Training failed to start', 'error');
                    if (progressEl) progressEl.style.display = 'none';
                }
            } catch (error) {
                console.error('[EmbeddingPanel] Training error:', error);
                showToast(`Training failed: ${error.message}`, 'error');
                if (progressEl) progressEl.style.display = 'none';
            } finally {
                trainBtn.disabled = false;
                trainBtn.textContent = originalText;
            }
        });
    }
    
    /**
     * Poll training status until complete.
     */
    async function pollTrainingStatus(taskId, progressEl, resultsEl) {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const maxAttempts = 600;
            
            _pollingInterval = setInterval(async () => {
                attempts++;
                
                try {
                    const status = await EmbeddingsAPI.getTrainingStatus(taskId);
                    
                    // Update progress
                    const progress = status.progress || (status.current_epoch / status.max_epochs * 100) || 0;
                    const message = status.message || `Epoch ${status.current_epoch || 0}/${status.max_epochs || '?'}`;
                    updateProgress(progress, message);
                    
                    // Send update to training monitor window
                    updateTrainingMonitor(status);
                    
                    if (status.status === 'completed') {
                        clearInterval(_pollingInterval);
                        _pollingInterval = null;
                        
                        showToast('Training completed!', 'success');
                        handleTrainingComplete(status.result, resultsEl);
                        if (progressEl) progressEl.style.display = 'none';
                        
                        resolve(status);
                    } else if (status.status === 'failed') {
                        clearInterval(_pollingInterval);
                        _pollingInterval = null;
                        
                        showToast(`Training failed: ${status.error}`, 'error');
                        if (progressEl) progressEl.style.display = 'none';
                        
                        reject(new Error(status.error));
                    } else if (attempts >= maxAttempts) {
                        clearInterval(_pollingInterval);
                        _pollingInterval = null;
                        
                        showToast('Training timed out', 'warning');
                        if (progressEl) progressEl.style.display = 'none';
                        
                        reject(new Error('Training timed out'));
                    }
                } catch (error) {
                    console.error('[EmbeddingPanel] Status poll error:', error);
                    if (attempts >= 5 && error.message.includes('404')) {
                        clearInterval(_pollingInterval);
                        _pollingInterval = null;
                        reject(error);
                    }
                }
            }, POLL_INTERVAL_MS);
        });
    }
    
    /**
     * Update progress display.
     */
    function updateProgress(percent, message) {
        const progressBar = document.querySelector('#training-progress .embed-progress-bar');
        const progressText = document.querySelector('#training-progress .progress-text');
        
        if (progressBar) {
            progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        }
        if (progressText) {
            progressText.textContent = message;
        }
    }
    
    /**
     * Handle training completion.
     */
    function handleTrainingComplete(result, resultsEl) {
        if (!result) return;
        
        console.log('[EmbeddingPanel] Training complete:', result);
        
        // Update model name
        if (result.model_name) {
            updateModelName(result.model_name);
        }
        
        // Show results
        if (resultsEl) {
            resultsEl.style.display = 'block';
            
            const epochsEl = document.getElementById('result-epochs');
            const lossEl = document.getElementById('result-loss');
            const silhouetteEl = document.getElementById('result-silhouette');
            const timeEl = document.getElementById('result-time');
            
            if (epochsEl) epochsEl.textContent = result.epochs_trained || '-';
            if (lossEl) lossEl.textContent = result.final_loss?.toFixed(4) || '-';
            if (silhouetteEl) silhouetteEl.textContent = result.silhouette_score?.toFixed(4) || '-';
            if (timeEl) {
                const time = result.training_time_seconds;
                if (time) {
                    timeEl.textContent = time > 60 ? `${(time/60).toFixed(1)}m` : `${time.toFixed(1)}s`;
                } else {
                    timeEl.textContent = '-';
                }
            }
        }
        
        showToast(`Model "${result.model_name}" trained successfully`, 'success');
    }
    
    /**
     * Setup communities tab.
     */
    function setupCommunitiesTab() {
        const detectBtn = document.getElementById('btn-detect-communities');
        const applyBtn = document.getElementById('btn-apply-communities');
        const clearBtn = document.getElementById('btn-clear-communities');
        
        if (detectBtn) {
            detectBtn.addEventListener('click', async () => {
                if (!EmbeddingsAPI.hasModel()) {
                    showToast('No model loaded. Train a model first.', 'warning');
                    return;
                }
                
                detectBtn.disabled = true;
                const originalText = detectBtn.textContent;
                detectBtn.textContent = 'Detecting...';
                
                try {
                    showToast('Detecting communities...', 'info');
                    
                    const result = await EmbeddingsAPI.getCommunities();
                    
                    if (result.success) {
                        _lastCommunities = result;
                        showToast(`Found ${result.num_communities} communities`, 'success');
                        displayCommunityResults(result);
                    }
                } catch (error) {
                    console.error('[EmbeddingPanel] Community detection error:', error);
                    showToast(`Failed: ${error.message}`, 'error');
                } finally {
                    detectBtn.disabled = false;
                    detectBtn.textContent = originalText;
                }
            });
        }
        
        if (applyBtn) {
            applyBtn.addEventListener('click', async () => {
                if (!_lastCommunities) {
                    showToast('Detect communities first', 'warning');
                    return;
                }
                
                applyBtn.disabled = true;
                
                try {
                    applyCommunityColors(_lastCommunities.assignments);
                    showToast('Communities applied to graph', 'success');
                } catch (error) {
                    console.error('[EmbeddingPanel] Apply communities error:', error);
                    showToast(`Failed: ${error.message}`, 'error');
                } finally {
                    applyBtn.disabled = false;
                }
            });
        }
        
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                clearCommunityStyles();
                _lastCommunities = null;
                
                // Hide results
                const resultsEl = document.getElementById('communities-results');
                if (resultsEl) resultsEl.style.display = 'none';
                
                showToast('Community colors cleared', 'info');
            });
        }
    }
    
    /**
     * Clear community styles from graph.
     */
    function clearCommunityStyles() {
        const cy = (typeof State !== 'undefined' && State.cy);
        if (!cy) return;
        
        cy.batch(() => {
            cy.nodes().forEach(node => {
                node.removeData('community');
                node.removeStyle('background-color');
            });
        });
    }
    
    /**
     * Display community detection results.
     */
    function displayCommunityResults(result) {
        const resultsEl = document.getElementById('communities-results');
        const countEl = document.getElementById('community-count');
        const listEl = document.getElementById('community-sizes-list');
        
        if (resultsEl) resultsEl.style.display = 'block';
        if (countEl) countEl.textContent = result.num_communities;
        
        if (listEl && result.community_sizes) {
            // Sort communities by size descending
            const sorted = Object.entries(result.community_sizes)
                .sort((a, b) => b[1] - a[1]);
            
            listEl.innerHTML = sorted.slice(0, 20).map(([id, size]) => `
                <div class="community-item" data-community="${id}">
                    <span class="community-id">Community ${id}</span>
                    <span class="community-size">${size} nodes</span>
                </div>
            `).join('');
            
            // Add click handlers
            listEl.querySelectorAll('.community-item').forEach(item => {
                item.addEventListener('click', () => {
                    const communityId = item.dataset.community;
                    highlightCommunity(communityId, result.assignments);
                });
            });
        }
    }
    
    /**
     * Apply community colors to graph.
     */
    function applyCommunityColors(assignments) {
        const cy = (typeof State !== 'undefined' && State.cy);
        if (!cy) {
            console.warn('[EmbeddingPanel] Cytoscape not available');
            return;
        }
        
        // Generate colors
        const communityIds = [...new Set(assignments.map(a => a.community))];
        const colors = generateCommunityColors(communityIds.length);
        const colorMap = {};
        communityIds.forEach((id, i) => colorMap[id] = colors[i]);
        
        // Apply to nodes
        cy.batch(() => {
            assignments.forEach(assignment => {
                const node = cy.getElementById(assignment.node_id);
                if (node.length > 0) {
                    node.data('community', assignment.community);
                    node.style('background-color', colorMap[assignment.community]);
                }
            });
        });
    }
    
    /**
     * Highlight nodes in a community.
     */
    function highlightCommunity(communityId, assignments) {
        const cy = (typeof State !== 'undefined' && State.cy);
        if (!cy) return;
        
        const nodeIds = assignments
            .filter(a => a.community === parseInt(communityId))
            .map(a => a.node_id);
        
        cy.batch(() => {
            cy.nodes().unselect();
            nodeIds.forEach(id => {
                const node = cy.getElementById(id);
                if (node.length > 0) node.select();
            });
        });
        
        const selected = cy.nodes(':selected');
        if (selected.length > 0) {
            cy.fit(selected, 50);
        }
        
        showToast(`Selected ${nodeIds.length} nodes in community ${communityId}`, 'info');
    }
    
    /**
     * Generate distinct colors.
     */
    function generateCommunityColors(count) {
        const colors = [];
        for (let i = 0; i < count; i++) {
            const hue = (i * 137.5) % 360;
            colors.push(`hsl(${hue}, 70%, 50%)`);
        }
        return colors;
    }
    
    /**
     * Setup similarity tab.
     */
    function setupSimilarityTab() {
        const findBtn = document.getElementById('btn-find-similar');
        const applyBtn = document.getElementById('btn-apply-similar');
        const clearBtn = document.getElementById('btn-clear-similar');
        
        if (findBtn) {
            findBtn.addEventListener('click', async () => {
                if (!EmbeddingsAPI.hasModel()) {
                    showToast('No model loaded. Train a model first.', 'warning');
                    return;
                }
                
                const queryNode = document.getElementById('similarity-query-input')?.value?.trim();
                if (!queryNode) {
                    showToast('Enter a node ID to search', 'warning');
                    return;
                }
                
                const k = parseInt(document.getElementById('similarity-k-input')?.value) || 10;
                const metric = document.getElementById('similarity-metric-select')?.value || 'cosine';
                
                findBtn.disabled = true;
                const originalText = findBtn.textContent;
                findBtn.textContent = 'Searching...';
                
                try {
                    showToast('Finding similar nodes...', 'info');
                    
                    const result = await EmbeddingsAPI.findSimilar(queryNode, k, metric);
                    
                    if (result.success) {
                        _lastSimilarNodes = result;
                        showToast(`Found ${result.similar_nodes.length} similar nodes`, 'success');
                        displaySimilarityResults(result);
                    }
                } catch (error) {
                    console.error('[EmbeddingPanel] Similarity search error:', error);
                    showToast(`Failed: ${error.message}`, 'error');
                } finally {
                    findBtn.disabled = false;
                    findBtn.textContent = originalText;
                }
            });
        }
        
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                if (!_lastSimilarNodes || !_lastSimilarNodes.similar_nodes) {
                    showToast('Find similar nodes first', 'warning');
                    return;
                }
                
                highlightSimilarNodes(_lastSimilarNodes);
                showToast('Similar nodes highlighted', 'success');
            });
        }
        
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                clearSimilarHighlights();
                _lastSimilarNodes = null;
                
                // Hide results
                const resultsEl = document.getElementById('similarity-results');
                if (resultsEl) resultsEl.style.display = 'none';
                
                // Clear input
                const queryInput = document.getElementById('similarity-query-input');
                if (queryInput) queryInput.value = '';
                
                showToast('Similarity highlights cleared', 'info');
            });
        }
        
        // Allow Enter key in query input
        const queryInput = document.getElementById('similarity-query-input');
        if (queryInput) {
            queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    findBtn?.click();
                }
            });
        }
    }
    
    /**
     * Highlight similar nodes in graph.
     */
    function highlightSimilarNodes(result) {
        const cy = (typeof State !== 'undefined' && State.cy);
        if (!cy) return;
        
        const nodeIds = result.similar_nodes.map(n => n.node_id);
        const queryNode = result.query_node;
        
        cy.batch(() => {
            // Dim all nodes first
            cy.nodes().style('opacity', 0.3);
            
            // Highlight similar nodes with gradient based on similarity
            result.similar_nodes.forEach(node => {
                const el = cy.getElementById(node.node_id);
                if (el.length > 0) {
                    el.style({
                        'opacity': 1,
                        'border-width': 3,
                        'border-color': '#f39c12'
                    });
                }
            });
            
            // Highlight query node
            const queryEl = cy.getElementById(queryNode);
            if (queryEl.length > 0) {
                queryEl.style({
                    'opacity': 1,
                    'border-width': 4,
                    'border-color': '#e74c3c',
                    'background-color': '#e74c3c'
                });
            }
        });
    }
    
    /**
     * Clear similarity highlights.
     */
    function clearSimilarHighlights() {
        const cy = (typeof State !== 'undefined' && State.cy);
        if (!cy) return;
        
        cy.batch(() => {
            cy.nodes().removeStyle('opacity border-width border-color');
        });
    }
    
    /**
     * Display similarity results.
     */
    function displaySimilarityResults(result) {
        const resultsEl = document.getElementById('similarity-results');
        const listEl = document.getElementById('similar-nodes-list');
        
        if (resultsEl) resultsEl.style.display = 'block';
        
        if (listEl && result.similar_nodes) {
            listEl.innerHTML = result.similar_nodes.map(node => `
                <div class="similar-item" data-node-id="${node.node_id}">
                    <span class="similar-id" title="${node.node_id}">${truncateId(node.node_id)}</span>
                    <span class="similar-score">${(node.similarity * 100).toFixed(1)}%</span>
                </div>
            `).join('');
            
            // Add click handlers
            listEl.querySelectorAll('.similar-item').forEach(item => {
                item.addEventListener('click', () => {
                    selectAndFocusNode(item.dataset.nodeId);
                });
            });
        }
    }
    
    /**
     * Truncate long IDs for display.
     */
    function truncateId(id, maxLen = 20) {
        if (!id || id.length <= maxLen) return id;
        return id.substring(0, 8) + '...' + id.substring(id.length - 8);
    }
    
    /**
     * Select and focus on a node.
     */
    function selectAndFocusNode(nodeId) {
        const cy = (typeof State !== 'undefined' && State.cy);
        if (!cy) return;
        
        const node = cy.getElementById(nodeId);
        if (node.length > 0) {
            cy.nodes().unselect();
            node.select();
            cy.animate({
                center: { eles: node },
                zoom: Math.min(cy.zoom() * 1.5, 2),
                duration: 300
            });
        } else {
            showToast(`Node ${nodeId} not found in graph`, 'warning');
        }
    }
    
    /**
     * Show toast notification.
     */
    function showToast(message, type = 'info') {
        if (typeof Toast !== 'undefined' && Toast.show) {
            Toast.show(message, type);
        } else {
            console.log(`[Toast ${type}] ${message}`);
        }
    }
    
    // Public API
    return {
        init,
        updateAvailabilityUI,
    };
})();

// Export to window
window.EmbeddingPanel = EmbeddingPanel;