/**
 * Metrics Module - Simplified Version
 * 
 * Simple 3-mode selection without fancy icons or badges
 */

const Metrics = {
    // State
    config: null,
    selectedMetrics: new Set(),
    metricParameters: {},
    
    // UI Elements
    elements: {},
    
    /**
     * Initialize metrics module
     */
    async init() {
        console.log('[Metrics] Initializing...');
        
        this.cacheElements();
        await this.loadConfig();
        this.setupCustomMode();
        this.setupParameterModal();
        this.setupActions();
        this.initFilterUI();
        
        console.log('[Metrics] Initialized');
    },
    
    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            metricsTree: document.getElementById('metrics-tree'),
            metricSearch: document.getElementById('metrics-search'),
            targetGraph: document.getElementById('metrics-graph'),
            computeBtn: document.getElementById('compute-metrics-btn'),
            selectionCount: document.getElementById('metrics-selection-count'),
            parameterModal: document.getElementById('parameter-modal'),
            parameterModalContent: document.getElementById('parameter-modal-content'),
            parameterModalClose: document.querySelectorAll('.parameter-modal-close'),
        };
    },
    
    /**
     * Load metrics configuration from API
     */
    async loadConfig() {
        try {
            const data = await API.listMetrics();
            this.config = data;
            console.log(`[Metrics] Loaded ${data.total_count} metrics`);
        } catch (error) {
            console.error('[Metrics] Failed to load config:', error);
        }
    },
    
    /**
     * Setup mode switching
     */
    
    /**
     * Setup custom mode
     */
    setupCustomMode() {
        if (!this.config) return;
        
        this.elements.metricSearch.addEventListener('input', (e) => {
            this.filterMetrics(e.target.value);
        });
        
        this.buildMetricsTree();
    },
    
    /**
     * Build metrics tree
     */
    buildMetricsTree() {
        this.elements.metricsTree.innerHTML = '';
        
        this.config.categories.forEach(category => {
            const categoryDiv = document.createElement('div');
            categoryDiv.style.marginBottom = '12px';
            
            const header = document.createElement('div');
            header.style.cssText = 'cursor: pointer; padding: 6px; background: #2a2a2a; border-radius: 3px; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;';
            header.innerHTML = `
                <input type="checkbox" class="category-select-checkbox" data-category="${category.name}" style="cursor: pointer;">
                <span class="expand-icon" data-icon="chevronRight" style="flex-shrink: 0;"></span>
                <span style="flex-grow: 1;">${this.formatName(category.name)} (${category.metric_count})</span>
            `;
            header.dataset.category = category.name;
            
            const metricsList = document.createElement('div');
            metricsList.style.display = 'none';
            metricsList.style.paddingLeft = '20px';
            
            const categoryMetrics = this.config.metrics.filter(m => m.category === category.name);
            categoryMetrics.forEach(metric => {
                const metricDiv = document.createElement('div');
                metricDiv.style.marginBottom = '4px';
                metricDiv.dataset.metric = metric.name;
                
                metricDiv.innerHTML = `
                    <label style="display: flex; align-items: center; cursor: pointer;">
                        <input type="checkbox" class="metric-checkbox" data-metric="${metric.name}" style="margin-right: 8px;">
                        <span>${this.formatName(metric.name)}</span>
                        ${metric.has_parameters ? '<button class="param-btn" style="margin-left: auto; padding: 2px 8px; background: #4A90E2; border: none; border-radius: 3px; color: white; cursor: pointer; font-size: 11px;" data-icon="settings"></button>' : ''}
                    </label>
                `;
                
                const checkbox = metricDiv.querySelector('.metric-checkbox');
                checkbox.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        this.selectedMetrics.add(metric.name);
                    } else {
                        this.selectedMetrics.delete(metric.name);
                    }
                    this.updateSelectionCount();
                    this.updateCategoryCheckbox(category.name);
                });
                
                const paramBtn = metricDiv.querySelector('.param-btn');
                if (paramBtn) {
                    paramBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        this.openParameterModal(metric.name);
                    });
                }
                
                metricsList.appendChild(metricDiv);
            });
            
            // Category checkbox handler - select/deselect all metrics in category
            const categoryCheckbox = header.querySelector('.category-select-checkbox');
            categoryCheckbox.addEventListener('change', (e) => {
                e.stopPropagation(); // Prevent header click
                const isChecked = e.target.checked;
                categoryMetrics.forEach(metric => {
                    const metricCheckbox = metricsList.querySelector(`input[data-metric="${metric.name}"]`);
                    if (metricCheckbox) {
                        metricCheckbox.checked = isChecked;
                        if (isChecked) {
                            this.selectedMetrics.add(metric.name);
                        } else {
                            this.selectedMetrics.delete(metric.name);
                        }
                    }
                });
                this.updateSelectionCount();
            });
            
            // Header click handler - expand/collapse
            header.addEventListener('click', (e) => {
                // Don't toggle if clicking on the checkbox
                if (e.target.classList.contains('category-select-checkbox')) return;
                
                const icon = header.querySelector('.expand-icon');
                if (metricsList.style.display === 'none') {
                    metricsList.style.display = 'block';
                    icon.setAttribute('data-icon', 'chevronDown');
                    Icons.inject(); // Re-inject icons
                } else {
                    metricsList.style.display = 'none';
                    icon.setAttribute('data-icon', 'chevronRight');
                    Icons.inject(); // Re-inject icons
                }
            });
            
            categoryDiv.appendChild(header);
            categoryDiv.appendChild(metricsList);
            this.elements.metricsTree.appendChild(categoryDiv);
        });
        
        // Inject SVG icons after building tree
        Icons.inject();
    },
    
    /**
     * Update category checkbox state based on selected metrics
     */
    updateCategoryCheckbox(categoryName) {
        const categoryMetrics = this.config.metrics.filter(m => m.category === categoryName);
        const selectedInCategory = categoryMetrics.filter(m => this.selectedMetrics.has(m.name)).length;
        const categoryCheckbox = document.querySelector(`.category-select-checkbox[data-category="${categoryName}"]`);
        
        if (categoryCheckbox) {
            if (selectedInCategory === 0) {
                categoryCheckbox.checked = false;
                categoryCheckbox.indeterminate = false;
            } else if (selectedInCategory === categoryMetrics.length) {
                categoryCheckbox.checked = true;
                categoryCheckbox.indeterminate = false;
            } else {
                categoryCheckbox.checked = false;
                categoryCheckbox.indeterminate = true;
            }
        }
    },
    
    /**
     * Filter metrics
     */
    filterMetrics(term) {
        const lower = term.toLowerCase();
        document.querySelectorAll('[data-metric]').forEach(item => {
            const matches = item.dataset.metric.toLowerCase().includes(lower);
            item.style.display = matches ? 'block' : 'none';
        });
    },
    
    /**
     * Setup parameter modal
     */
    setupParameterModal() {
        this.elements.parameterModalClose.forEach(el => {
            el.addEventListener('click', () => this.closeParameterModal());
        });
        
        this.elements.parameterModal.addEventListener('click', (e) => {
            if (e.target === this.elements.parameterModal) {
                this.closeParameterModal();
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.elements.parameterModal.style.display === 'flex') {
                this.closeParameterModal();
            }
        });
    },
    
    /**
     * Open parameter modal
     */
    async openParameterModal(metricName) {
        try {
            const params = await API.getMetricParameters(metricName);
            this.elements.parameterModalContent.innerHTML = `<h3>${this.formatName(metricName)}</h3>`;
            
            params.parameters.forEach(param => {
                const div = document.createElement('div');
                div.style.marginBottom = '16px';
                
                const currentValue = this.metricParameters[metricName]?.[param.name] ?? param.default;
                
                div.innerHTML = `
                    <label style="display: block; margin-bottom: 4px;">${this.formatName(param.name)}</label>
                    <small style="display: block; color: #888; margin-bottom: 6px;">${param.description}</small>
                `;
                
                if (param.type === 'bool') {
                    div.innerHTML += `<input type="checkbox" ${currentValue ? 'checked' : ''} data-param="${param.name}">`;
                } else if (param.type === 'int' || param.type === 'float') {
                    div.innerHTML += `
                        <input type="range" min="${param.min_value || 0}" max="${param.max_value || 100}" step="${param.step || 0.01}" value="${currentValue}" data-param="${param.name}" style="width: 70%;">
                        <input type="number" min="${param.min_value}" max="${param.max_value}" step="${param.step || 0.01}" value="${currentValue}" data-param="${param.name}" style="width: 25%; margin-left: 5%;">
                    `;
                    
                    const range = div.querySelector('input[type="range"]');
                    const number = div.querySelector('input[type="number"]');
                    range.addEventListener('input', () => number.value = range.value);
                    number.addEventListener('input', () => range.value = number.value);
                } else {
                    div.innerHTML += `<input type="text" value="${currentValue}" data-param="${param.name}" style="width: 100%;">`;
                }
                
                this.elements.parameterModalContent.appendChild(div);
            });
            
            const saveBtn = document.createElement('button');
            saveBtn.textContent = 'Save';
            saveBtn.style.cssText = 'background: #4A90E2; color: white; border: none; padding: 8px 20px; border-radius: 4px; cursor: pointer; margin-top: 16px;';
            saveBtn.addEventListener('click', () => {
                this.saveParameters(metricName);
                this.closeParameterModal();
            });
            this.elements.parameterModalContent.appendChild(saveBtn);
            
            this.elements.parameterModal.style.display = 'flex';
        } catch (error) {
            console.error('[Metrics] Failed to load parameters:', error);
        }
    },
    
    /**
     * Save parameters
     */
    saveParameters(metricName) {
        const params = {};
        this.elements.parameterModalContent.querySelectorAll('[data-param]').forEach(input => {
            const paramName = input.dataset.param;
            let value;
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number' || input.type === 'range') {
                value = parseFloat(input.value);
            } else {
                value = input.value;
            }
            params[paramName] = value;
        });
        
        if (Object.keys(params).length > 0) {
            this.metricParameters[metricName] = params;
        }
    },
    
    /**
     * Close parameter modal
     */
    closeParameterModal() {
        this.elements.parameterModal.style.display = 'none';
    },
    
    /**
     * Setup action buttons
     */
    setupActions() {
        this.elements.computeBtn.addEventListener('click', () => this.run());
    },
    
    /**
     * Update selection count
     */
    updateSelectionCount() {
        const count = this.selectedMetrics.size;
        this.elements.selectionCount.textContent = `${count} selected`;
        this.elements.computeBtn.disabled = count === 0;
    },
    
    /**
     * Build config for API
     */
    buildConfig() {
        const config = {
            metrics_graph_id: this.elements.targetGraph?.value || null,
            metrics: Array.from(this.selectedMetrics)
        };
        
        if (Object.keys(this.metricParameters).length > 0) {
            config.metric_parameters = this.metricParameters;
        }
        
        return config;
    },
    
    /**
     * Run metrics computation
     */
    async run() {
        const config = this.buildConfig();
        
        try {
            this.elements.computeBtn.disabled = true;
            this.elements.computeBtn.textContent = 'Computing...';
            
            const result = await API.computeMetrics(config);
            console.log('[Metrics] Complete:', result);
            
            if (typeof Toast !== 'undefined') {
                Toast.success('Metrics computed successfully');
            }
            
            // Reload current graph to fetch updated node data with new metrics
            // Note: This clears edges - user will need to click "Load Edges" again if needed
            if (State.cy && State.currentGraph && typeof GraphLoader !== 'undefined') {
                await GraphLoader.displayGraph(State.currentGraph);
            }
        } catch (error) {
            console.error('[Metrics] Failed:', error);
            if (typeof Toast !== 'undefined') {
                Toast.error('Failed to compute metrics: ' + error.message);
            }
        } finally {
            this.elements.computeBtn.disabled = false;
            this.elements.computeBtn.textContent = 'Compute Metrics';
        }
    },
    
    /**
     * Populate metric dropdowns with available metrics (called by other modules)
     * This is the original method signature that graph-loader, snapshots, etc expect
     */
    populateDropdowns(nodes, edges) {
        if (!nodes || nodes.length === 0) return;
        
        const numericMetrics = new Set();
        const arrayProperties = new Set();
        const stringProperties = new Set();
        
        // Initialize propertyTypes if not exists
        if (!this.propertyTypes) this.propertyTypes = {};
        
        // Extract metrics and properties from nodes
        nodes.forEach(node => {
            const data = node.data || node;
            Object.entries(data).forEach(([key, value]) => {
                if (['id', 'label', 'isNew', 'parent'].includes(key)) return;
                
                if (Array.isArray(value)) {
                    arrayProperties.add(key);
                    this.propertyTypes[key] = 'array';
                } else if (typeof value === 'number' && !isNaN(value)) {
                    numericMetrics.add(key);
                    this.propertyTypes[key] = 'number';
                } else if (typeof value === 'string' && value.length > 0) {
                    stringProperties.add(key);
                    if (!this.propertyTypes[key]) {
                        this.propertyTypes[key] = 'string';
                    }
                }
            });
        });
        
        const sortedNumeric = Array.from(numericMetrics).sort();
        const sortedArrays = Array.from(arrayProperties).sort();
        const sortedStrings = Array.from(stringProperties).sort();
        
        // Build options HTML for numeric-only dropdowns
        const numericOptionsHtml = '<option value="">Uniform</option>' + 
            sortedNumeric.map(m => `<option value="${m}">${m}</option>`).join('');
        
        // Update numeric-only dropdowns (size, color, composite)
        ['node-size-metric', 'node-color-metric', 'composite-metric-1', 'composite-metric-2'].forEach(id => {
            const select = document.getElementById(id);
            if (select) {
                const currentValue = select.value;
                select.innerHTML = numericOptionsHtml;
                if (sortedNumeric.includes(currentValue)) {
                    select.value = currentValue;
                }
            }
        });
        
        // Build options HTML for filter dropdown (includes all types)
        let filterOptionsHtml = '<option value="">Select Property...</option>';
        
        if (sortedNumeric.length > 0) {
            filterOptionsHtml += '<optgroup label="Numeric">';
            filterOptionsHtml += sortedNumeric.map(m => `<option value="${m}">${m}</option>`).join('');
            filterOptionsHtml += '</optgroup>';
        }
        
        if (sortedArrays.length > 0) {
            filterOptionsHtml += '<optgroup label="Arrays">';
            filterOptionsHtml += sortedArrays.map(m => `<option value="${m}">${m}</option>`).join('');
            filterOptionsHtml += '</optgroup>';
        }
        
        if (sortedStrings.length > 0) {
            filterOptionsHtml += '<optgroup label="Text">';
            filterOptionsHtml += sortedStrings.map(m => `<option value="${m}">${m}</option>`).join('');
            filterOptionsHtml += '</optgroup>';
        }
        
        const filterSelect = document.getElementById('filter-metric');
        if (filterSelect) {
            const currentValue = filterSelect.value;
            filterSelect.innerHTML = filterOptionsHtml;
            if (this.propertyTypes[currentValue]) {
                filterSelect.value = currentValue;
            }
        }
    },
    
    /**
     * Initialize filter UI handlers
     */
    initFilterUI() {
        const filterMetric = document.getElementById('filter-metric');
        if (filterMetric) {
            filterMetric.addEventListener('change', () => this.updateFilterInputs());
        }
    },
    
    /**
     * Update filter inputs based on selected property type
     */
    updateFilterInputs() {
        const property = document.getElementById('filter-metric')?.value;
        const type = (this.propertyTypes && this.propertyTypes[property]) || 'number';
        
        const numericInputs = document.getElementById('numeric-filter-inputs');
        const arrayInputs = document.getElementById('array-filter-inputs');
        const stringInputs = document.getElementById('string-filter-inputs');
        
        // Hide all
        if (numericInputs) numericInputs.style.display = 'none';
        if (arrayInputs) arrayInputs.style.display = 'none';
        if (stringInputs) stringInputs.style.display = 'none';
        
        // Show appropriate inputs
        if (!property) {
            if (numericInputs) numericInputs.style.display = 'flex';
        } else if (type === 'array') {
            if (arrayInputs) arrayInputs.style.display = 'flex';
        } else if (type === 'string') {
            if (stringInputs) stringInputs.style.display = 'flex';
        } else {
            if (numericInputs) numericInputs.style.display = 'flex';
        }
    },
    
    /**
     * Filter nodes by property criteria
     */
    filter() {
        const renderer = State.renderer;
        if (!renderer) {
            if (typeof Toast !== 'undefined') Toast.error('Load a graph first');
            return;
        }
        
        const property = document.getElementById('filter-metric')?.value;
        if (!property) {
            if (typeof Toast !== 'undefined') Toast.error('Please select a property');
            return;
        }
        
        const type = (this.propertyTypes && this.propertyTypes[property]) || 'number';
        let matchingNodeIds = [];
        
        // Get all node IDs and data from the renderer
        const allNodeIds = renderer.getAllNodeIds();
        
        if (type === 'number') {
            const operator = document.getElementById('filter-operator').value;
            const filterValue = parseFloat(document.getElementById('filter-value').value);
            
            if (isNaN(filterValue)) {
                if (typeof Toast !== 'undefined') Toast.error('Enter a valid number');
                return;
            }
            
            matchingNodeIds = allNodeIds.filter(nodeId => {
                const nodeData = renderer.getNodeData(nodeId);
                const nodeValue = nodeData?.[property];
                if (nodeValue === undefined || nodeValue === null) return false;
                
                switch (operator) {
                    case 'gt': return nodeValue > filterValue;
                    case 'lt': return nodeValue < filterValue;
                    case 'eq': return nodeValue === filterValue;
                    case 'gte': return nodeValue >= filterValue;
                    case 'lte': return nodeValue <= filterValue;
                    default: return false;
                }
            });
        } else if (type === 'array') {
            const operator = document.getElementById('filter-array-operator').value;
            const filterValue = document.getElementById('filter-array-value').value;
            
            if (!filterValue) {
                if (typeof Toast !== 'undefined') Toast.error('Enter a value to search for');
                return;
            }
            
            matchingNodeIds = allNodeIds.filter(nodeId => {
                const nodeData = renderer.getNodeData(nodeId);
                const nodeValue = nodeData?.[property];
                if (!Array.isArray(nodeValue)) return false;
                
                if (operator === 'regex') {
                    try {
                        const match = filterValue.match(/^\/(.+)\/([gimsuy]*)$/);
                        const pattern = match ? match[1] : filterValue;
                        const flags = match ? match[2] : 'i';
                        const regex = new RegExp(pattern, flags);
                        return nodeValue.some(v => regex.test(String(v)));
                    } catch (e) {
                        if (typeof Toast !== 'undefined') Toast.error('Invalid regex pattern');
                        return false;
                    }
                } else {
                    const contains = nodeValue.some(v => String(v).includes(filterValue));
                    return operator === 'contains' ? contains : !contains;
                }
            });
        } else if (type === 'string') {
            const operator = document.getElementById('filter-string-operator').value;
            const filterValue = document.getElementById('filter-string-value').value;
            
            if (!filterValue) {
                if (typeof Toast !== 'undefined') Toast.error('Enter a value');
                return;
            }
            
            matchingNodeIds = allNodeIds.filter(nodeId => {
                const nodeData = renderer.getNodeData(nodeId);
                const nodeValue = String(nodeData?.[property] || '');
                
                if (operator === 'regex') {
                    try {
                        const match = filterValue.match(/^\/(.+)\/([gimsuy]*)$/);
                        const pattern = match ? match[1] : filterValue;
                        const flags = match ? match[2] : 'i';
                        const regex = new RegExp(pattern, flags);
                        return regex.test(nodeValue);
                    } catch (e) {
                        if (typeof Toast !== 'undefined') Toast.error('Invalid regex pattern');
                        return false;
                    }
                } else if (operator === 'eq') {
                    return nodeValue === filterValue;
                } else if (operator === 'neq') {
                    return nodeValue !== filterValue;
                } else if (operator === 'contains') {
                    return nodeValue.includes(filterValue);
                }
                return false;
            });
        }
        
        if (matchingNodeIds && matchingNodeIds.length > 0) {
            // Clear current selection and select matching nodes
            renderer.clearSelection();
            renderer.selectNodes(matchingNodeIds);
            
            if (typeof Toast !== 'undefined') Toast.success(`Selected ${matchingNodeIds.length} nodes`);
            
            // Fit view to selected nodes if reasonable count
            if (matchingNodeIds.length <= 100) {
                renderer.fitView(matchingNodeIds);
            }
        } else {
            if (typeof Toast !== 'undefined') Toast.error('No matching nodes found');
        }
    },
    
    /**
     * Reset selection
     */
    reset() {
        const renderer = State.renderer;
        if (renderer) {
            renderer.clearSelection();
            if (typeof Toast !== 'undefined') Toast.success('Selection cleared');
        }
    },
    
    /**
     * Show only selected nodes (hide all others)
     * Note: Full hide/show only works with Cytoscape. With Cosmos, we highlight selected nodes.
     */
    showOnlySelected() {
        const renderer = State.renderer;
        if (!renderer) {
            if (typeof Toast !== 'undefined') Toast.error('Load a graph first');
            return;
        }
        
        const selectedIds = renderer.getSelectedNodes();
        if (selectedIds.length === 0) {
            if (typeof Toast !== 'undefined') Toast.error('No nodes selected');
            return;
        }
        
        if (State.rendererType === 'cytoscape' && State.cy) {
            const selected = State.cy.nodes(':selected');
            // Hide all non-selected nodes and their edges
            State.cy.batch(() => {
                State.cy.nodes().not(':selected').style('display', 'none');
                State.cy.edges().style('display', 'none');
                
                // Show edges between visible nodes
                selected.connectedEdges().filter(edge => {
                    const source = edge.source();
                    const target = edge.target();
                    return source.selected() && target.selected();
                }).style('display', 'element');
            });
            
            // Fit view to selected nodes
            State.cy.fit(selected, 50);
            
            if (typeof Toast !== 'undefined') {
                const hiddenCount = State.cy.nodes().length - selected.length;
                Toast.success(`Showing only ${selected.length} selected nodes (${hiddenCount} hidden)`);
            }
        } else if (State.rendererType === 'cosmos' && renderer.showOnlyNodes) {
            // Cosmos: Use alpha channel visibility
            renderer.showOnlyNodes(selectedIds);
            renderer.fitView(selectedIds);
            
            if (typeof Toast !== 'undefined') {
                const totalCount = renderer.getAllNodeIds().length;
                const hiddenCount = totalCount - selectedIds.length;
                Toast.success(`Showing only ${selectedIds.length} selected nodes (${hiddenCount} hidden)`);
            }
        } else {
            // Fallback: Fit view to selected nodes
            renderer.fitView(selectedIds);
            if (typeof Toast !== 'undefined') {
                Toast.info(`Focused on ${selectedIds.length} selected nodes`);
            }
        }
    },
    
    /**
     * Hide selected nodes (show only non-selected)
     */
    hideSelected() {
        const renderer = State.renderer;
        console.log('[Metrics] hideSelected called, rendererType:', State.rendererType, 'renderer exists:', !!renderer);
        
        if (!renderer) {
            if (typeof Toast !== 'undefined') Toast.error('Load a graph first');
            return;
        }
        
        const selectedIds = renderer.getSelectedNodes();
        console.log('[Metrics] Selected nodes:', selectedIds?.length, selectedIds?.slice(0, 3));
        
        if (selectedIds.length === 0) {
            if (typeof Toast !== 'undefined') Toast.error('No nodes selected');
            return;
        }
        
        if (State.rendererType === 'cytoscape' && State.cy) {
            const selected = State.cy.nodes(':selected');
            // Hide selected nodes and their edges
            State.cy.batch(() => {
                selected.style('display', 'none');
                
                // Hide edges connected to hidden nodes
                selected.connectedEdges().style('display', 'none');
                
                // Show edges between visible nodes
                const visibleNodes = State.cy.nodes().not(':selected');
                visibleNodes.connectedEdges().filter(edge => {
                    const source = edge.source();
                    const target = edge.target();
                    return !source.selected() && !target.selected();
                }).style('display', 'element');
            });
            
            // Clear selection
            State.cy.nodes().unselect();
            
            // Fit view to remaining visible nodes
            const visibleNodes = State.cy.nodes('[display != "none"]');
            if (visibleNodes.length > 0) {
                State.cy.fit(visibleNodes, 50);
            }
            
            if (typeof Toast !== 'undefined') {
                const visibleCount = State.cy.nodes().length - selectedIds.length;
                Toast.success(`Hidden ${selectedIds.length} nodes (${visibleCount} visible)`);
            }
        } else if (State.rendererType === 'cosmos' && renderer.hideNodes) {
            console.log('[Metrics] Using Cosmos hideNodes');
            // Cosmos: Use alpha channel visibility
            renderer.hideNodes(selectedIds);
            renderer.clearSelection();
            
            // Fit view to visible nodes
            const allIds = renderer.getAllNodeIds();
            const visibleIds = allIds.filter(id => !renderer.isNodeHidden(id));
            if (visibleIds.length > 0) {
                renderer.fitView(visibleIds);
            }
            
            if (typeof Toast !== 'undefined') {
                Toast.success(`Hidden ${selectedIds.length} nodes (${visibleIds.length} visible)`);
            }
        } else {
            console.log('[Metrics] Fallback: clearing selection, hideNodes exists:', typeof renderer.hideNodes);
            // Fallback: Just clear selection
            renderer.clearSelection();
            if (typeof Toast !== 'undefined') {
                Toast.info(`Cleared ${selectedIds.length} selected nodes`);
            }
        }
    },
    
    /**
     * Show all nodes (reset visibility)
     */
    showAllNodes() {
        const renderer = State.renderer;
        if (!renderer) {
            if (typeof Toast !== 'undefined') Toast.error('Load a graph first');
            return;
        }
        
        if (State.rendererType === 'cytoscape' && State.cy) {
            // Show all nodes and edges
            State.cy.batch(() => {
                State.cy.nodes().style('display', 'element');
                State.cy.edges().style('display', 'element');
            });
            
            // Fit to all
            State.cy.fit();
            
            if (typeof Toast !== 'undefined') {
                Toast.success(`All ${State.cy.nodes().length} nodes visible`);
            }
        } else if (State.rendererType === 'cosmos' && renderer.showAllNodes) {
            // Cosmos: Reset visibility
            renderer.showAllNodes();
            renderer.fitView();
            
            const nodeCount = renderer.getAllNodeIds().length;
            if (typeof Toast !== 'undefined') {
                Toast.success(`All ${nodeCount} nodes visible`);
            }
        } else {
            // Fallback: Just fit view to show all
            renderer.fitView();
            const nodeCount = renderer.getAllNodeIds().length;
            if (typeof Toast !== 'undefined') {
                Toast.success(`Fit view to all ${nodeCount} nodes`);
            }
        }
    },
    
    /**
     * Populate metric dropdowns after metrics are computed
     */
    populateMetricDropdowns() {
        if (!State.cy) return;
        
        // Get all node properties (metrics) from first node
        const nodes = State.cy.nodes();
        if (nodes.length === 0) return;
        
        const firstNode = nodes[0].data();
        const metrics = [];
        
        // Collect all numeric properties
        for (const key in firstNode) {
            if (typeof firstNode[key] === 'number' && key !== 'id') {
                metrics.push(key);
            }
        }
        
        // Populate size dropdown
        const sizeSelect = document.getElementById('node-size-metric');
        if (sizeSelect) {
            const currentValue = sizeSelect.value;
            sizeSelect.innerHTML = '<option value="">Uniform</option>' +
                metrics.map(m => `<option value="${m}">${this.formatName(m)}</option>`).join('');
            if (currentValue && metrics.includes(currentValue)) {
                sizeSelect.value = currentValue;
            }
        }
        
        // Populate color dropdown  
        const colorSelect = document.getElementById('node-color-metric');
        if (colorSelect) {
            const currentValue = colorSelect.value;
            colorSelect.innerHTML = '<option value="">Uniform</option>' +
                metrics.map(m => `<option value="${m}">${this.formatName(m)}</option>`).join('');
            if (currentValue && metrics.includes(currentValue)) {
                colorSelect.value = currentValue;
            }
        }
        
        // Populate filter dropdown
        const filterSelect = document.getElementById('filter-metric');
        if (filterSelect) {
            const currentValue = filterSelect.value;
            filterSelect.innerHTML = '<option value="">Select Property...</option>' +
                metrics.map(m => `<option value="${m}">${this.formatName(m)}</option>`).join('');
            if (currentValue && metrics.includes(currentValue)) {
                filterSelect.value = currentValue;
            }
        }
        
        console.log('[Metrics] Populated dropdowns with', metrics.length, 'metrics');
    },
    
    /**
     * Format name
     */
    formatName(name) {
        return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }
};