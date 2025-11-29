/**
 * Metrics Module
 * Metrics computation and dropdown management
 */

const Metrics = {
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
        
        const metricNames = new Set();
        
        // Extract numeric metrics from nodes
        nodes.forEach(node => {
            const data = node.data || node;
            Object.entries(data).forEach(([key, value]) => {
                if (!['id', 'label', 'isNew', 'parent'].includes(key) && 
                    typeof value === 'number' && !isNaN(value)) {
                    metricNames.add(key);
                }
            });
        });
        
        const sortedMetrics = Array.from(metricNames).sort();
        const optionsHtml = '<option value="">None</option>' + 
            sortedMetrics.map(m => `<option value="${m}">${m}</option>`).join('');
        
        // Update all metric dropdowns
        const dropdownIds = [
            'node-size-metric',
            'node-color-metric',
            'filter-metric',
            'composite-metric-1',
            'composite-metric-2'
        ];
        
        dropdownIds.forEach(id => {
            const select = document.getElementById(id);
            if (select) {
                const currentValue = select.value;
                select.innerHTML = optionsHtml;
                // Restore selection if still valid
                if (sortedMetrics.includes(currentValue)) {
                    select.value = currentValue;
                }
            }
        });
    },

    /**
     * Filter nodes by metric criteria
     * Uses the same pattern as the original working implementation
     */
    filter() {
        if (!State.cy) return;
        
        const metric = document.getElementById('filter-metric').value;
        const op = document.getElementById('filter-operator').value;
        const rawVal = document.getElementById('filter-value').value;
        const val = parseFloat(rawVal);

        if (!metric || isNaN(val)) {
            return updateStatus('Please select a metric and enter a numeric value', 'error');
        }
        
        updateStatus('Applying filter...', 'info');

        State.cy.batch(() => {
            State.cy.elements().unselect();
            
            // Use cy.nodes().filter() which returns a collection directly
            const matches = State.cy.nodes().filter(n => {
                const d = n.data(metric);
                if (d === undefined) return false;
                switch(op) {
                    case 'gt': return d > val;
                    case 'lt': return d < val;
                    case 'eq': return d == val;  // Use loose equality for type coercion
                    case 'gte': return d >= val;
                    case 'lte': return d <= val;
                    case 'neq': return d != val;
                    default: return false;
                }
            });
            
            if (matches.length > 0) {
                matches.select();  // Select entire collection at once
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