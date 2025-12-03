/**
 * Data Explorer
 * 
 * Opens a detached window for exploring all node data.
 * Communication with main window via postMessage.
 */

const DataExplorer = {
    window: null,

    /**
     * Initialize - just setup the button
     */
    init() {
        document.getElementById('data-explorer-btn')?.addEventListener('click', () => this.open());
    },

    /**
     * Open data explorer in new window
     */
    open() {
        // If window exists and is open, focus it
        if (this.window && !this.window.closed) {
            this.window.focus();
            return;
        }

        // Open new window
        this.window = window.open(
            '/data-explorer',
            'dataExplorer',
            'width=1400,height=800,menubar=no,toolbar=no,location=no,status=no'
        );

        if (!this.window) {
            Toast.show('Failed to open Data Explorer (popup blocked?)', 'error');
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    DataExplorer.init();
});