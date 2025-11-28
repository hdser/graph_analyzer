/**
 * Auto Reload Module
 * Auto-reload SSE system for background data refresh
 */

const AutoReload = {
    /**
     * Setup auto-reload event handlers
     */
    setup() {
        // Toggle handler
        DOMCache.autoReloadToggle?.addEventListener('change', (e) => this.handleToggle(e));
        
        // Interval change handler
        DOMCache.reloadInterval?.addEventListener('change', () => {
            if (State.autoReloadEnabled) {
                this.handleToggle({ target: { checked: true } });
            }
        });
        
        // Compute metrics toggle handler
        DOMCache.reloadComputeMetrics?.addEventListener('change', () => {
            if (State.autoReloadEnabled) {
                this.handleToggle({ target: { checked: true } });
            }
        });
        
        // Initialize indicator
        this.updateIndicator('disabled');
    },

    /**
     * Handle auto-reload toggle
     */
    async handleToggle(e) {
        const enabled = e.target.checked;
        
        if (enabled) {
            const selectedFiles = Array.from(document.querySelectorAll('input[name="sql-file"]:checked'))
                .map(cb => cb.value);
            
            if (selectedFiles.length === 0) {
                Toast.error('Select SQL files first');
                DOMCache.autoReloadToggle.checked = false;
                return;
            }
            
            try {
                const status = await API.startAutoReload({
                    enabled: true,
                    interval_seconds: parseInt(DOMCache.reloadInterval.value) || 300,
                    sql_files: selectedFiles,
                    preserve_layout: true,
                    compute_metrics: DOMCache.reloadComputeMetrics?.checked || false,
                    metrics_mode: 'basic'
                });
                
                State.autoReloadEnabled = true;
                this.updateUI(status);
                this.connectSSE();
                Toast.success('Auto-reload enabled');
                
            } catch (err) {
                console.error('Auto-reload start error:', err);
                Toast.error('Failed to start auto-reload: ' + err.message);
                DOMCache.autoReloadToggle.checked = false;
            }
            
        } else {
            try {
                await API.stopAutoReload();
                State.autoReloadEnabled = false;
                this.disconnectSSE();
                this.updateIndicator('disabled');
                DOMCache.reloadStatusText.textContent = 'Disabled';
                DOMCache.nextReloadTime.textContent = '-';
                Toast.info('Auto-reload disabled');
            } catch (err) {
                console.error('Auto-reload stop error:', err);
            }
        }
    },

    /**
     * Connect to SSE event stream
     */
    connectSSE() {
        // Close existing connection
        if (State.autoReloadSSE) {
            State.autoReloadSSE.close();
        }
        
        State.autoReloadSSE = API.createAutoReloadSSE();
        
        // Status update event
        State.autoReloadSSE.addEventListener('status_update', (e) => {
            this.updateUI(JSON.parse(e.data));
        });
        
        // Reload started event
        State.autoReloadSSE.addEventListener('reload_started', () => {
            this.updateIndicator('loading');
            DOMCache.reloadStatusText.textContent = 'Reloading...';
            Toast.info('Background reload started...');
        });
        
        // Reload complete event
        State.autoReloadSSE.addEventListener('reload_complete', (e) => {
            this.handleComplete(JSON.parse(e.data));
        });
        
        // Error event
        State.autoReloadSSE.addEventListener('error_event', (e) => {
            const data = JSON.parse(e.data);
            this.updateIndicator('error');
            Toast.error('Reload error: ' + data.error);
        });
        
        // Connection error
        State.autoReloadSSE.onerror = () => {
            this.updateIndicator('error');
        };
    },

    /**
     * Disconnect from SSE event stream
     */
    disconnectSSE() {
        if (State.autoReloadSSE) {
            State.autoReloadSSE.close();
            State.autoReloadSSE = null;
        }
    },

    /**
     * Update UI from status object
     */
    updateUI(status) {
        if (!status) return;
        
        this.updateIndicator(status.active ? 'active' : 'disabled');
        DOMCache.reloadStatusText.textContent = status.active ? 'Active' : 'Disabled';
        
        if (status.last_reload_time) {
            const lastTime = new Date(status.last_reload_time);
            DOMCache.lastReloadTime.textContent = lastTime.toLocaleTimeString();
            
            const diff = Math.floor((Date.now() - lastTime.getTime()) / 1000);
            DOMCache.lastReloadDiff.textContent = this.formatTimeDiff(diff);
        }
        
        if (status.next_reload_time) {
            DOMCache.nextReloadTime.textContent = new Date(status.next_reload_time).toLocaleTimeString();
        }
    },

    /**
     * Update reload indicator state
     */
    updateIndicator(state) {
        if (!DOMCache.reloadIndicator) return;
        DOMCache.reloadIndicator.className = 'reload-indicator ' + state;
    },

    /**
     * Format time difference for display
     */
    formatTimeDiff(seconds) {
        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        return `${Math.floor(seconds / 3600)}h ago`;
    },

    /**
     * Handle reload complete event
     */
    async handleComplete(data) {
        this.updateIndicator('active');
        DOMCache.reloadStatusText.textContent = 'Active';
        DOMCache.lastReloadTime.textContent = new Date().toLocaleTimeString();
        
        // Show change summary
        const changeText = data.nodes_added > 0 || data.nodes_removed > 0 
            ? `+${data.nodes_added}/-${data.nodes_removed} nodes` 
            : 'No changes';
        
        Toast.success(`Reload complete: ${changeText}`);
        
        // Update distributions if open
        DistributionsComm.sendData();
    }
};