/**
 * Metrics Module
 * Metrics computation and dropdown management
 */

const Metrics = {
    // Track property types for filter UI
    propertyTypes: {},  // { propertyName: 'number' | 'array' | 'string' }
    
    /**
     * Run metrics computation
     */
    async run() {
        const selectedCategories = Array.from(document.querySelectorAll('input[name="custom-metric"]:checked'))
            .map(cb => cb.value);
        
        if (selectedCategories.length === 0) {
            updateStatus('Select at least one metric category', 'error');
            return;
        }
        
        const config = {
            metrics_mode: selectedCategories.join(','),
            metrics_graph_id: document.getElementById('metrics-graph').value || null
        };
        
        const btn = document.getElementById('metrics-btn');
        const status = document.getElementById('metrics-status');
        
        btn.disabled = true;
        btn.textContent = 'Running...';
        if (status) status.style.display = 'block';
        
        updateStatus('Calculating metrics...', 'info');
        
        try {
            const result = await API.computeMetrics(config);
            
            updateStatus(`Computed ${result.metrics_computed.length} metrics`, 'success');
            
            // Update node data in Cytoscape
            if (State.cy && result.node_data) {
                State.cy.batch(() => {
                    result.node_data.forEach(data => {
                        const node = State.cy.getElementById(data.id);
                        if (node.length > 0) {
                            node.data(data);
                        }
                    });
                });
                
                // Update dropdowns
                this.populateDropdowns(State.cy.nodes().map(n => ({ data: n.data() })), null);
                
                // Reset style cache
                State.styleCache = {
                    sizeRange: { min: 0, max: 1 },
                    colorRange: { min: 0, max: 1 },
                    widthRange: { min: 0, max: 1 }
                };
                
                // Update styling if not in performance mode
                if (!State.performanceMode) {
                    CytoscapeManager.updateStyle();
                }
                
                // Send updated data to distributions
                DistributionsComm.sendData();
            }
            
        } catch (err) {
            console.error('Metrics error:', err);
            updateStatus(err.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Run Metrics';
            if (status) status.style.display = 'none';
        }
    },

    /**
     * Populate metric dropdowns with available metrics
     */
    populateDropdowns(nodes, edges) {
        if (!nodes || nodes.length === 0) return;
        
        const numericMetrics = new Set();
        const arrayProperties = new Set();
        const stringProperties = new Set();
        
        // Reset property types
        this.propertyTypes = {};
        
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
        const numericOptionsHtml = '<option value="">None</option>' + 
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
        const property = document.getElementById('filter-metric').value;
        const type = this.propertyTypes[property] || 'number';
        
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
     * Supports numeric, array, and string filtering
     */
    filter() {
        if (!State.cy) return;
        
        const property = document.getElementById('filter-metric').value;
        if (!property) {
            return updateStatus('Please select a property', 'error');
        }
        
        const type = this.propertyTypes[property] || 'number';
        let matches;
        
        updateStatus('Applying filter...', 'info');
        
        State.cy.batch(() => {
            State.cy.elements().unselect();
            
            if (type === 'array') {
                // Array filtering
                const op = document.getElementById('filter-array-operator').value;
                const searchValue = document.getElementById('filter-array-value').value.trim();
                
                if (!searchValue) {
                    updateStatus('Please enter a value to search for', 'error');
                    return;
                }
                
                // For regex, parse the pattern
                let regex = null;
                if (op === 'regex') {
                    try {
                        const regexMatch = searchValue.match(/^\/(.+)\/([gimsuy]*)$/);
                        if (regexMatch) {
                            regex = new RegExp(regexMatch[1], regexMatch[2]);
                        } else {
                            regex = new RegExp(searchValue, 'i');
                        }
                    } catch (e) {
                        updateStatus(`Invalid regex: ${e.message}`, 'error');
                        return;
                    }
                }
                
                const searchLower = searchValue.toLowerCase();
                
                matches = State.cy.nodes().filter(n => {
                    const arr = n.data(property);
                    if (!Array.isArray(arr)) return false;
                    
                    if (op === 'regex') {
                        // Check if any element matches regex
                        return arr.some(item => regex.test(String(item)));
                    } else {
                        // Check if any element contains the search value
                        const found = arr.some(item => 
                            String(item).toLowerCase().includes(searchLower)
                        );
                        return op === 'contains' ? found : !found;
                    }
                });
                
            } else if (type === 'string') {
                // String filtering
                const op = document.getElementById('filter-string-operator').value;
                const searchValue = document.getElementById('filter-string-value').value.trim();
                
                if (!searchValue) {
                    updateStatus('Please enter a value to search for', 'error');
                    return;
                }
                
                // For regex, parse the pattern
                let regex = null;
                if (op === 'regex') {
                    try {
                        // Support /pattern/flags format or plain pattern
                        const regexMatch = searchValue.match(/^\/(.+)\/([gimsuy]*)$/);
                        if (regexMatch) {
                            regex = new RegExp(regexMatch[1], regexMatch[2]);
                        } else {
                            regex = new RegExp(searchValue, 'i'); // Default case-insensitive
                        }
                    } catch (e) {
                        updateStatus(`Invalid regex: ${e.message}`, 'error');
                        return;
                    }
                }
                
                const searchLower = searchValue.toLowerCase();
                
                matches = State.cy.nodes().filter(n => {
                    const val = n.data(property);
                    if (val === undefined || val === null) return false;
                    const strVal = String(val);
                    
                    switch(op) {
                        case 'eq': return strVal.toLowerCase() === searchLower;
                        case 'neq': return strVal.toLowerCase() !== searchLower;
                        case 'contains': return strVal.toLowerCase().includes(searchLower);
                        case 'regex': return regex.test(strVal);
                        default: return false;
                    }
                });
                
            } else {
                // Numeric filtering
                const op = document.getElementById('filter-operator').value;
                const rawVal = document.getElementById('filter-value').value;
                const val = parseFloat(rawVal);
                
                if (isNaN(val)) {
                    updateStatus('Please enter a numeric value', 'error');
                    return;
                }
                
                matches = State.cy.nodes().filter(n => {
                    const d = n.data(property);
                    if (d === undefined || typeof d !== 'number') return false;
                    switch(op) {
                        case 'gt': return d > val;
                        case 'lt': return d < val;
                        case 'eq': return d == val;
                        case 'gte': return d >= val;
                        case 'lte': return d <= val;
                        case 'neq': return d != val;
                        default: return false;
                    }
                });
            }
            
            if (matches && matches.length > 0) {
                matches.select();
                updateStatus(`Selected ${matches.length} matching nodes`, 'success');
            } else {
                updateStatus('No nodes match criteria', 'info');
            }
        });
    },

    /**
     * Reset all filters and selections
     */
    reset() {
        if (!State.cy) return;
        
        State.cy.elements().unselect();
        State.cy.elements().removeClass('highlighted anomaly searched');
        
        updateStatus('Selection reset', 'info');
    },

    /**
     * Select nodes by percentile
     */
    selectPercentile(metric, percentile, top = true) {
        if (!State.cy || !metric) return;
        
        const values = State.cy.nodes()
            .map(n => ({ node: n, value: n.data(metric) }))
            .filter(item => typeof item.value === 'number' && !isNaN(item.value));
        
        if (values.length === 0) return;
        
        // Sort by value
        values.sort((a, b) => top ? b.value - a.value : a.value - b.value);
        
        // Calculate cutoff
        const cutoffIndex = Math.ceil(values.length * percentile / 100);
        const toSelect = values.slice(0, cutoffIndex);
        
        State.cy.batch(() => {
            State.cy.nodes().unselect();
            toSelect.forEach(item => item.node.select());
        });
        
        updateStatus(`Selected ${top ? 'top' : 'bottom'} ${percentile}% (${toSelect.length} nodes)`, 'success');
    }
};