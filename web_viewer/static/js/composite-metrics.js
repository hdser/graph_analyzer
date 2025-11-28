/**
 * Composite Metrics Module
 * Composite metric creation and management
 */

const CompositeMetrics = {
    /**
     * Setup composite metrics UI
     */
    setup() {
        // Create button
        DOMCache.createCompositeBtn?.addEventListener('click', () => this.create());
        
        // Refresh button
        DOMCache.refreshCompositesBtn?.addEventListener('click', () => this.loadSaved());
        
        // Auto-name updates
        DOMCache.compositeMetric1?.addEventListener('change', () => this.updateAutoName());
        DOMCache.compositeMetric2?.addEventListener('change', () => this.updateAutoName());
        DOMCache.compositeOperation?.addEventListener('change', () => this.updateAutoName());
        
        // Load saved composites
        this.loadSaved();
    },

    /**
     * Create new composite metric
     */
    async create() {
        const metric1 = DOMCache.compositeMetric1?.value;
        const metric2 = DOMCache.compositeMetric2?.value;
        const operation = DOMCache.compositeOperation?.value || 'multiply';
        const name = DOMCache.compositeName?.value?.trim();
        const normalize = DOMCache.compositeNormalize?.checked || false;
        
        if (!metric1 || !metric2) {
            updateStatus('Select two metrics to combine', 'error');
            return;
        }
        
        if (!name) {
            updateStatus('Enter a name for the composite metric', 'error');
            return;
        }
        
        DOMCache.createCompositeBtn.disabled = true;
        DOMCache.createCompositeBtn.textContent = 'Creating...';
        
        try {
            const result = await API.createCompositeMetric({
                name,
                metrics: [metric1, metric2],
                operation,
                normalize,
                save: true
            });
            
            updateStatus(`Created composite metric: ${name}`, 'success');
            
            // Update node data
            if (State.cy && result.node_data) {
                State.cy.batch(() => {
                    result.node_data.forEach(item => {
                        const node = State.cy.getElementById(item.id);
                        if (node.length > 0) {
                            node.data(name, item[name]);
                        }
                    });
                });
                
                // Update dropdowns
                Metrics.populateDropdowns(State.cy.nodes().map(n => ({ data: n.data() })), null);
                
                // Update styling
                if (!State.performanceMode) {
                    CytoscapeManager.updateStyle();
                }
                
                // Update distributions
                DistributionsComm.sendData();
            }
            
            // Refresh saved list
            this.loadSaved();
            
            // Clear name input
            if (DOMCache.compositeName) {
                DOMCache.compositeName.value = '';
            }
            
        } catch (err) {
            console.error('Composite creation error:', err);
            updateStatus('Failed to create composite: ' + err.message, 'error');
        } finally {
            DOMCache.createCompositeBtn.disabled = false;
            DOMCache.createCompositeBtn.textContent = 'Create';
        }
    },

    /**
     * Load saved composite metrics
     */
    async loadSaved() {
        if (!DOMCache.savedCompositesList) return;
        
        try {
            const result = await API.getSavedComposites();
            const composites = result.composites || [];
            
            if (composites.length === 0) {
                DOMCache.savedCompositesList.innerHTML = '<div class="no-composites">No saved composites</div>';
                return;
            }
            
            // Build list HTML
            DOMCache.savedCompositesList.innerHTML = composites.map(comp => `
                <div class="saved-composite-item" data-name="${comp.name}">
                    <div class="composite-info">
                        <span class="composite-name">${comp.name}</span>
                        <span class="composite-formula">${comp.source_metrics[0]} ${Icons.getMathSymbol(comp.operation)} ${comp.source_metrics[1]}</span>
                    </div>
                    <div class="composite-actions">
                        <button class="btn-small apply-composite">Apply</button>
                        <button class="btn-small delete-composite" data-icon="close"></button>
                    </div>
                </div>
            `).join('');
            
            // Inject icons into newly created elements
            DOMCache.savedCompositesList.querySelectorAll('[data-icon]').forEach(el => {
                const iconName = el.dataset.icon;
                el.innerHTML = Icons.get(iconName);
            });
            
            // Add event listeners
            DOMCache.savedCompositesList.querySelectorAll('.apply-composite').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const name = e.target.closest('.saved-composite-item').dataset.name;
                    this.apply(name);
                });
            });
            
            DOMCache.savedCompositesList.querySelectorAll('.delete-composite').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const name = e.target.closest('.saved-composite-item').dataset.name;
                    this.delete(name);
                });
            });
            
        } catch (err) {
            console.error('Error loading saved composites:', err);
            DOMCache.savedCompositesList.innerHTML = '<div class="no-composites">Failed to load</div>';
        }
    },

    /**
     * Apply saved composite to current graph
     */
    async apply(name) {
        try {
            // Get composite definition
            const result = await API.getSavedComposites();
            const comp = result.composites?.find(c => c.name === name);
            
            if (!comp) {
                updateStatus('Composite not found', 'error');
                return;
            }
            
            // Create without saving
            const createResult = await API.createCompositeMetric({
                name: comp.name,
                metrics: comp.source_metrics,
                operation: comp.operation,
                normalize: comp.normalize || false,
                save: false
            });
            
            // Update nodes
            if (State.cy && createResult.node_data) {
                State.cy.batch(() => {
                    createResult.node_data.forEach(item => {
                        const node = State.cy.getElementById(item.id);
                        if (node.length > 0) {
                            node.data(name, item[name]);
                        }
                    });
                });
                
                Metrics.populateDropdowns(State.cy.nodes().map(n => ({ data: n.data() })), null);
                DistributionsComm.sendData();
            }
            
            updateStatus(`Applied composite: ${name}`, 'success');
            
        } catch (err) {
            console.error('Error applying composite:', err);
            updateStatus('Failed to apply composite: ' + err.message, 'error');
        }
    },

    /**
     * Delete saved composite
     */
    async delete(name) {
        if (!confirm(`Delete composite "${name}"?`)) return;
        
        try {
            await API.deleteCompositeMetric(name);
            updateStatus(`Deleted composite: ${name}`, 'success');
            this.loadSaved();
        } catch (err) {
            console.error('Error deleting composite:', err);
            updateStatus('Failed to delete composite: ' + err.message, 'error');
        }
    },

    /**
     * Update auto-generated name placeholder
     */
    updateAutoName() {
        if (!DOMCache.compositeName || DOMCache.compositeName.value.trim()) return;
        
        const m1 = DOMCache.compositeMetric1?.value;
        const m2 = DOMCache.compositeMetric2?.value;
        const op = DOMCache.compositeOperation?.value;
        
        if (m1 && m2 && op) {
            const opShort = {
                multiply: 'x',
                add: '+',
                subtract: '-',
                divide: '/',
                maximum: 'max',
                minimum: 'min',
                average: 'avg'
            }[op] || op;
            
            DOMCache.compositeName.placeholder = `${m1}_${opShort}_${m2}`;
        }
    },

    /**
     * Get display symbol for operation
     */
    getOperationSymbol(operation) {
        return Icons.getMathSymbol(operation);
    }
};