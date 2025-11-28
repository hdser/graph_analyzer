/**
 * Distributions Communication Module
 * Communication with distributions popup window
 */

const DistributionsComm = {
    /**
     * Setup message listener for popup requests
     */
    setup() {
        window.addEventListener('message', (event) => {
            if (event.data.type === 'REQUEST_DISTRIBUTION_DATA') {
                this.sendData();
            }
        });
    },

    /**
     * Open distributions popup window
     */
    open() {
        if (!State.cy) {
            updateStatus('Please load a graph first', 'error');
            return;
        }
        
        // Check if window already open
        if (State.distributionsWindow && !State.distributionsWindow.closed) {
            State.distributionsWindow.focus();
            this.sendData();
            return;
        }
        
        // Calculate window position
        const width = 1400;
        const height = 900;
        const left = (screen.width - width) / 2;
        const top = (screen.height - height) / 2;
        
        // Open popup
        State.distributionsWindow = window.open(
            '/static/distributions.html',
            'distributionsAnalysis',
            `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
        );
        
        if (!State.distributionsWindow) {
            updateStatus('Popup blocked. Please allow popups.', 'error');
            return;
        }
        
        // Send data when loaded
        State.distributionsWindow.addEventListener('load', () => {
            this.sendData();
        });
    },

    /**
     * Send node data to distributions window
     */
    sendData() {
        if (!State.distributionsWindow || State.distributionsWindow.closed || !State.cy) return;
        
        // Collect node data
        const nodes = [];
        State.cy.nodes().forEach(node => {
            const data = node.data();
            const cleanData = { id: data.id };
            
            // Include only numeric metrics
            Object.keys(data).forEach(key => {
                if (typeof data[key] === 'number' && !isNaN(data[key])) {
                    cleanData[key] = data[key];
                }
            });
            
            nodes.push(cleanData);
        });
        
        // Get selected node IDs
        const selectedIds = State.cy.nodes(':selected').map(n => n.id());
        
        // Send message
        State.distributionsWindow.postMessage({
            type: 'DISTRIBUTION_DATA',
            data: {
                nodes,
                selectedIds,
                availableConfig: State.availableConfig
            }
        }, '*');
    },

    /**
     * Send selection update to distributions window
     */
    sendSelectionUpdate() {
        if (!State.distributionsWindow || State.distributionsWindow.closed || !State.cy) return;
        
        const selectedIds = State.cy.nodes(':selected').map(n => n.id());
        
        State.distributionsWindow.postMessage({
            type: 'SELECTION_UPDATE',
            data: { selectedIds }
        }, '*');
    }
};