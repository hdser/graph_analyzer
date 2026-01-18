/**
 * Search Module
 * Node search functionality - works with both Cytoscape and cosmos.gl renderers
 */

const Search = {
    /**
     * Search for node by ID or partial match
     */
    search() {
        const query = document.getElementById('node-search')?.value?.trim();
        if (!query) {
            updateStatus('Enter a search term', 'info');
            return;
        }
        
        // Check which renderer is active
        if (State.rendererType === 'cosmos' && State.renderer) {
            this.searchCosmos(query);
        } else if (State.cy) {
            this.searchCytoscape(query);
        } else {
            updateStatus('No graph loaded', 'error');
        }
    },
    
    /**
     * Search in cosmos.gl renderer
     */
    searchCosmos(query) {
        const renderer = State.renderer;
        if (!renderer) return;
        
        // Get all node IDs from the renderer
        const nodeIds = renderer.nodeIds || [];
        const lowerQuery = query.toLowerCase();
        
        console.log('[Search] Cosmos search for:', query);
        console.log('[Search] Renderer has', nodeIds.length, 'nodes');
        
        // First, try using the renderer's findNodeId helper if available (handles prefixes, case)
        let found = null;
        if (typeof renderer.findNodeId === 'function') {
            found = renderer.findNodeId(query);
            console.log('[Search] findNodeId result:', found);
        }
        
        // If not found with helper, try direct exact match
        if (!found) {
            found = nodeIds.find(id => id === query);
        }
        
        // If still not found, try partial match (case-insensitive)
        if (!found) {
            const matches = nodeIds.filter(id => 
                id.toLowerCase().includes(lowerQuery)
            );
            
            console.log('[Search] Partial matches:', matches.length);
            
            if (matches.length === 1) {
                found = matches[0];
            } else if (matches.length > 1) {
                // Multiple matches - highlight all and zoom to fit
                renderer.clearSelection();
                matches.forEach(id => renderer.selectNode(id));
                renderer.fitView(matches);
                updateStatus(`Found ${matches.length} matching nodes`, 'info');
                return;
            }
        }
        
        if (found) {
            console.log('[Search] Found node:', found);
            
            // Select and center on the node
            renderer.clearSelection();
            renderer.selectNode(found);
            
            // Center on the node - use zoomToNode which uses findNodeId internally
            if (typeof renderer.zoomToNode === 'function') {
                renderer.zoomToNode(found, 2, 400);
            } else if (typeof renderer.centerOnNode === 'function') {
                renderer.centerOnNode(found);
            }
            
            // Show info panel
            const nodeData = renderer.nodeDataMap?.get(found);
            if (typeof InfoPanel !== 'undefined' && InfoPanel.showNodeFromData) {
                InfoPanel.showNodeFromData(found, nodeData);
            }
            
            updateStatus(`Found: ${found}`, 'success');
        } else {
            console.log('[Search] Node not found. Sample IDs:', nodeIds.slice(0, 5));
            updateStatus('Node not found', 'error');
        }
    },
    
    /**
     * Search in Cytoscape renderer
     */
    searchCytoscape(query) {
        // Clear previous highlights
        State.cy.nodes().removeClass('searched');
        
        // Try exact match first
        let found = State.cy.getElementById(query);
        
        if (found.length === 0) {
            // Try partial match (case-insensitive)
            const lowerQuery = query.toLowerCase();
            const matches = State.cy.nodes().filter(n => 
                n.id().toLowerCase().includes(lowerQuery)
            );
            
            if (matches.length === 1) {
                found = matches;
            } else if (matches.length > 1) {
                // Multiple matches - highlight all
                matches.addClass('searched');
                updateStatus(`Found ${matches.length} matching nodes`, 'info');
                State.cy.fit(matches, 50);
                return;
            }
        }
        
        if (found.length > 0) {
            found.addClass('searched');
            found.select();
            
            State.cy.animate({
                center: { eles: found },
                zoom: 2
            }, { duration: 300 });
            
            InfoPanel.showNode(found);
            updateStatus(`Found: ${found.id()}`, 'success');
        } else {
            updateStatus('Node not found', 'error');
        }
    },

    /**
     * Clear search highlights
     */
    clear() {
        const searchInput = document.getElementById('node-search');
        if (searchInput) {
            searchInput.value = '';
        }
        
        if (State.rendererType === 'cosmos' && State.renderer) {
            State.renderer.clearSelection();
        } else if (State.cy) {
            State.cy.nodes().removeClass('searched');
        }
        
        updateStatus('Search cleared', 'info');
    },

    /**
     * Focus on specific node by ID
     */
    focusNode(nodeId) {
        if (State.rendererType === 'cosmos' && State.renderer) {
            this.focusNodeCosmos(nodeId);
        } else if (State.cy) {
            this.focusNodeCytoscape(nodeId);
        }
    },
    
    /**
     * Focus on node in cosmos.gl
     */
    focusNodeCosmos(nodeId) {
        const renderer = State.renderer;
        if (!renderer) return;
        
        // Use findNodeId helper to handle ID variations
        let actualId = nodeId;
        if (typeof renderer.findNodeId === 'function') {
            actualId = renderer.findNodeId(nodeId);
        }
        
        if (!actualId) {
            updateStatus('Node not found', 'error');
            return;
        }
        
        // Check if node exists
        const nodeData = renderer.nodeDataMap?.get(actualId);
        if (nodeData) {
            renderer.clearSelection();
            renderer.selectNode(actualId);
            
            // Center on the node
            if (typeof renderer.zoomToNode === 'function') {
                renderer.zoomToNode(actualId, 2, 400);
            } else if (typeof renderer.centerOnNode === 'function') {
                renderer.centerOnNode(actualId);
            }
            
            // Show info panel
            if (typeof InfoPanel !== 'undefined' && InfoPanel.showNodeFromData) {
                InfoPanel.showNodeFromData(actualId, nodeData);
            }
        } else {
            updateStatus('Node not found', 'error');
        }
    },
    
    /**
     * Focus on node in Cytoscape
     */
    focusNodeCytoscape(nodeId) {
        const node = State.cy.getElementById(nodeId);
        if (node.length > 0) {
            State.cy.animate({
                center: { eles: node },
                zoom: 2
            }, { duration: 300 });
            
            node.select();
            InfoPanel.showNode(node);
        } else {
            updateStatus('Node not found', 'error');
        }
    },

    /**
     * Search nodes by attribute value
     */
    searchByAttribute(attribute, value, operator = 'eq') {
        if (State.rendererType === 'cosmos' && State.renderer) {
            return this.searchByAttributeCosmos(attribute, value, operator);
        } else if (State.cy) {
            return this.searchByAttributeCytoscape(attribute, value, operator);
        }
        return [];
    },
    
    /**
     * Search by attribute in cosmos.gl
     */
    searchByAttributeCosmos(attribute, value, operator = 'eq') {
        const renderer = State.renderer;
        if (!renderer) return [];
        
        const matches = [];
        
        renderer.nodeDataMap?.forEach((nodeData, nodeId) => {
            const nodeValue = nodeData[attribute];
            if (nodeValue === undefined) return;
            
            let match = false;
            switch (operator) {
                case 'eq':
                    match = nodeValue === value;
                    break;
                case 'contains':
                    match = String(nodeValue).toLowerCase().includes(String(value).toLowerCase());
                    break;
                case 'gt':
                    match = typeof nodeValue === 'number' && nodeValue > value;
                    break;
                case 'lt':
                    match = typeof nodeValue === 'number' && nodeValue < value;
                    break;
                case 'gte':
                    match = typeof nodeValue === 'number' && nodeValue >= value;
                    break;
                case 'lte':
                    match = typeof nodeValue === 'number' && nodeValue <= value;
                    break;
                default:
                    match = nodeValue === value;
            }
            
            if (match) {
                matches.push(nodeId);
            }
        });
        
        if (matches.length > 0) {
            renderer.clearSelection();
            matches.forEach(id => renderer.selectNode(id));
            updateStatus(`Found ${matches.length} nodes`, 'success');
            
            if (matches.length === 1) {
                if (typeof renderer.zoomToNode === 'function') {
                    renderer.zoomToNode(matches[0], 2, 400);
                } else if (typeof renderer.centerOnNode === 'function') {
                    renderer.centerOnNode(matches[0]);
                }
                const nodeData = renderer.nodeDataMap?.get(matches[0]);
                if (typeof InfoPanel !== 'undefined' && InfoPanel.showNodeFromData) {
                    InfoPanel.showNodeFromData(matches[0], nodeData);
                }
            } else {
                renderer.fitView(matches);
            }
        } else {
            updateStatus('No matching nodes found', 'info');
        }
        
        return matches;
    },
    
    /**
     * Search by attribute in Cytoscape
     */
    searchByAttributeCytoscape(attribute, value, operator = 'eq') {
        State.cy.nodes().removeClass('searched');
        
        const matches = State.cy.nodes().filter(node => {
            const nodeValue = node.data(attribute);
            if (nodeValue === undefined) return false;
            
            switch (operator) {
                case 'eq':
                    return nodeValue === value;
                case 'contains':
                    return String(nodeValue).toLowerCase().includes(String(value).toLowerCase());
                case 'gt':
                    return typeof nodeValue === 'number' && nodeValue > value;
                case 'lt':
                    return typeof nodeValue === 'number' && nodeValue < value;
                case 'gte':
                    return typeof nodeValue === 'number' && nodeValue >= value;
                case 'lte':
                    return typeof nodeValue === 'number' && nodeValue <= value;
                default:
                    return nodeValue === value;
            }
        });
        
        if (matches.length > 0) {
            matches.addClass('searched');
            updateStatus(`Found ${matches.length} nodes`, 'success');
            
            if (matches.length === 1) {
                State.cy.animate({
                    center: { eles: matches },
                    zoom: 2
                }, { duration: 300 });
                InfoPanel.showNode(matches[0]);
            } else {
                State.cy.fit(matches, 50);
            }
        } else {
            updateStatus('No matching nodes found', 'info');
        }
        
        return matches;
    }
};