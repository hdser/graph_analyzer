/**
 * Data Explorer
 *
 * Opens data explorer inline in the analysis panel tab, or falls back
 * to a detached window if the analysis panel is not available.
 */

const DataExplorer = {
    window: null,
    url: '/data-explorer?v=locate-fix-20260227',

    /**
     * Open data explorer — prefer inline analysis panel tab
     */
    open() {
        // Prefer inline tab in analysis panel
        if (typeof AnalysisPanel !== 'undefined') {
            AnalysisPanel.open('data-explorer-tab');
            return;
        }

        // Fallback: open in new window
        if (this.window && !this.window.closed) {
            this.window.focus();
            return;
        }

        this.window = window.open(
            this.url,
            'dataExplorer',
            'width=1400,height=800,menubar=no,toolbar=no,location=no,status=no'
        );

        if (!this.window) {
            Toast.show('Failed to open Data Explorer (popup blocked?)', 'error');
        }
    }
};
