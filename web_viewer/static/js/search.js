/**
 * Search Module
 * Node search functionality
 */

const Search = {
    /**
     * Search for node by ID or partial match
     */
    search() {
        if (!State.cy) return;
        
        const query = document.getElementById('node-search')?.value?.trim();
        if (!query) {
            updateStatus('Enter a search term', 'info');
            return;
        }
        
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
        if (!State.cy) return;
        
        State.cy.nodes().removeClass('searched');
        
        const searchInput = document.getElementById('node-search');
        if (searchInput) {
            searchInput.value = '';
        }
        
        updateStatus('Search cleared', 'info');
    },

    /**
     * Focus on specific node by ID
     */
    focusNode(nodeId) {
        if (!State.cy) return;
        
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
        if (!State.cy) return;
        
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