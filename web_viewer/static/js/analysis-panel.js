/**
 * Analysis Panel Module
 * IDE-style bottom panel with tabs for Snapshots, Metrics, Embeddings, and Query.
 * Replaces individual sidebar panels for "analysis" features.
 */
const AnalysisPanel = {
    DATA_EXPLORER_URL: '/data-explorer?v=locate-fix-20260227',

    // =========================================================================
    // State
    // =========================================================================

    isOpen: false,
    activeTab: 'metrics-tab',
    panelHeight: 300,
    minHeight: 40,
    maxHeightRatio: 0.7,
    isResizing: false,

    STORAGE_KEY: 'analysis_panel_state',

    // DOM refs (cached on init)
    panel: null,
    resizeHandle: null,
    toggleBtn: null,
    closeBtn: null,
    tabs: null,
    tabContents: null,

    // Query tab state
    _queryInitialized: false,
    _queryTablesLoaded: false,
    _queryState: {
        lastResult: null,
        tables: [],
    },

    // Popout state
    _popoutWindow: null,

    // =========================================================================
    // Initialization
    // =========================================================================

    init() {
        this.panel = document.getElementById('analysis-panel');
        this.resizeHandle = document.getElementById('analysis-resize-handle');
        this.toggleBtn = document.getElementById('analysis-panel-toggle');
        this.closeBtn = document.getElementById('analysis-close-btn');

        if (!this.panel || !this.toggleBtn) {
            console.warn('[AnalysisPanel] Required DOM elements not found');
            return;
        }

        this.tabs = this.panel.querySelectorAll('.analysis-tab');
        this.tabContents = this.panel.querySelectorAll('.analysis-tab-content');

        this._bindEvents();
        this._setupResize();
        this._setupPopout();
        this._initQueryTab();
        this._loadState();

        console.log('[AnalysisPanel] Initialized');
    },

    // =========================================================================
    // Toggle / Open / Close
    // =========================================================================

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    },

    open(tabName) {
        this.isOpen = true;
        this.panel.classList.add('open');
        this.panel.style.height = this.panelHeight + 'px';
        this.resizeHandle.classList.add('visible');
        this.toggleBtn.classList.add('active');

        if (tabName) {
            this.switchTab(tabName);
        }

        this._saveState();
        this._notifyViewportChange();

        // Re-inject icons for newly visible content
        if (typeof Icons !== 'undefined') {
            setTimeout(() => Icons.inject(), 50);
        }
    },

    close() {
        this.isOpen = false;
        this.panel.classList.remove('open');
        this.resizeHandle.classList.remove('visible');
        this.toggleBtn.classList.remove('active');

        this._saveState();
        this._notifyViewportChange();
    },

    // =========================================================================
    // Tab Switching
    // =========================================================================

    switchTab(tabName) {
        this.activeTab = tabName;

        // Update tab buttons
        this.tabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.analysisTab === tabName);
        });

        // Update tab content panels
        this.tabContents.forEach(content => {
            content.classList.toggle('active', content.dataset.analysisTab === tabName);
        });

        this._saveState();

        // Lazy-load query tables on first activation
        if (tabName === 'query-tab' && !this._queryTablesLoaded) {
            this._queryTablesLoaded = true;
            this._loadQueryTables();
        }

        // Lazy-load data explorer iframe on first activation
        if (tabName === 'data-explorer-tab') {
            const iframe = document.getElementById('data-explorer-iframe');
            if (iframe && !iframe.src.includes(this.DATA_EXPLORER_URL)) {
                iframe.src = this.DATA_EXPLORER_URL;
            }
        }

        // Re-inject icons for the newly shown tab
        if (typeof Icons !== 'undefined') {
            setTimeout(() => Icons.inject(), 50);
        }
    },

    // =========================================================================
    // Events
    // =========================================================================

    _bindEvents() {
        // Toggle button
        this.toggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.toggle();
        });

        // Close button
        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.close());
        }

        // Tab switching
        this.tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.switchTab(tab.dataset.analysisTab);
            });
        });

        // ESC to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                // Only close if no modal/overlay is open
                const modal = document.querySelector('.modal.show, .overlay.show, #parameter-modal.show');
                if (!modal) {
                    this.close();
                }
            }
        });
    },

    // =========================================================================
    // Resize
    // =========================================================================

    _setupResize() {
        if (!this.resizeHandle) return;

        let startY, startHeight;

        const onMouseDown = (e) => {
            e.preventDefault();
            this.isResizing = true;
            startY = e.clientY;
            startHeight = this.panel.offsetHeight;
            this.resizeHandle.classList.add('dragging');
            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        };

        const onMouseMove = (e) => {
            if (!this.isResizing) return;
            const delta = startY - e.clientY;
            const maxHeight = window.innerHeight * this.maxHeightRatio;
            const newHeight = Math.min(maxHeight, Math.max(this.minHeight, startHeight + delta));

            this.panelHeight = newHeight;
            this.panel.style.height = newHeight + 'px';
        };

        const onMouseUp = () => {
            this.isResizing = false;
            this.resizeHandle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';

            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);

            this._saveState();
            this._notifyViewportChange();
        };

        this.resizeHandle.addEventListener('mousedown', onMouseDown);
    },

    // =========================================================================
    // Viewport notification (tell renderer canvas resized)
    // =========================================================================

    _notifyViewportChange() {
        requestAnimationFrame(() => {
            // Cosmos.gl adapter
            if (typeof State !== 'undefined' && State.renderer && State.renderer.resize) {
                State.renderer.resize();
            }
            // Cytoscape
            if (typeof State !== 'undefined' && State.cy) {
                State.cy.resize();
            }
        });
    },

    // =========================================================================
    // State persistence
    // =========================================================================

    _saveState() {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify({
                isOpen: this.isOpen,
                activeTab: this.activeTab,
                panelHeight: this.panelHeight,
            }));
        } catch (e) { /* ignore */ }
    },

    _loadState() {
        try {
            const raw = localStorage.getItem(this.STORAGE_KEY);
            if (!raw) return;
            const state = JSON.parse(raw);

            this.panelHeight = state.panelHeight || 300;
            this.activeTab = state.activeTab || 'snapshots-tab';

            // Restore active tab visually
            this.switchTab(this.activeTab);

            // Restore open state
            if (state.isOpen) {
                this.open(this.activeTab);
            }
        } catch (e) { /* ignore */ }
    },

    // =========================================================================
    // Query Tab
    // =========================================================================

    _initQueryTab() {
        const editor = document.getElementById('querySqlEditor');
        if (!editor) return;

        // Keyboard shortcuts
        editor.addEventListener('keydown', (e) => {
            // Ctrl+Enter to run
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this._runQuery();
            }
            // Tab inserts spaces
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = editor.selectionStart;
                editor.value = editor.value.substring(0, start) + '  ' + editor.value.substring(editor.selectionEnd);
                editor.selectionStart = editor.selectionEnd = start + 2;
            }
        });

        // Button handlers
        document.getElementById('queryRunBtn')?.addEventListener('click', () => this._runQuery());
        document.getElementById('queryExportBtn')?.addEventListener('click', () => this._exportQueryCSV());
        document.getElementById('queryClearBtn')?.addEventListener('click', () => this._clearQueryResults());

        this._queryInitialized = true;
    },

    async _loadQueryTables() {
        try {
            const resp = await fetch('/api/query/tables');
            if (!resp.ok) throw new Error('Failed to load tables');
            this._queryState.tables = await resp.json();
            this._renderQueryTables();
            this._renderQueryExamples();
        } catch (e) {
            const list = document.getElementById('queryTableList');
            if (list) {
                list.innerHTML = '<div class="query-tables-header">Tables</div>' +
                    '<div class="query-empty-state" style="font-size:11px;">No tables available</div>';
            }
        }
    },

    _renderQueryTables() {
        const container = document.getElementById('queryTableList');
        if (!container) return;

        const tables = this._queryState.tables;
        let html = '<div class="query-tables-header">Tables</div>';
        html += '<div class="query-tables-content">';

        if (!tables.length) {
            html += '<div class="query-empty-state" style="font-size:11px;">Load data first</div>';
        } else {
            for (const t of tables) {
                html += `<div class="query-table-item" onclick="AnalysisPanel._insertTableSelect('${t.name.replace(/'/g, "\\'")}')" title="${t.columns.map(c => c.name + ': ' + c.type).join('\\n')}">` +
                    `<span>${t.name}</span>` +
                    `<span class="table-rows">${t.row_count.toLocaleString()}</span>` +
                    `</div>`;
            }
        }

        html += '</div>';
        container.innerHTML = html;
    },

    _renderQueryExamples() {
        const container = document.getElementById('queryExamples');
        if (!container) return;

        const tables = this._queryState.tables;
        const examples = [];

        if (tables.length > 0) {
            examples.push({
                label: `SELECT * FROM "${tables[0].name}" LIMIT 10`,
                sql: `SELECT * FROM "${tables[0].name}" LIMIT 10`,
            });
        }

        const metricsTable = tables.find(t => t.name.includes('metrics'));
        if (metricsTable) {
            examples.push({
                label: 'Top 10 by PageRank',
                sql: `SELECT * FROM "${metricsTable.name}" ORDER BY pagerank DESC LIMIT 10`,
            });
        }

        const edgesTable = tables.find(t => t.name.includes('edges'));
        if (edgesTable) {
            examples.push({
                label: 'Edge count by source',
                sql: `SELECT source, COUNT(*) AS cnt FROM "${edgesTable.name}" GROUP BY source ORDER BY cnt DESC LIMIT 20`,
            });
        }

        examples.push({ label: 'Show tables', sql: 'SHOW TABLES' });

        container.innerHTML = examples.map(ex =>
            `<span class="query-example-chip" onclick="AnalysisPanel._setAndRunQuery(\`${ex.sql.replace(/`/g, '\\`')}\`)">${ex.label}</span>`
        ).join('');
    },

    _insertTableSelect(tableName) {
        const editor = document.getElementById('querySqlEditor');
        if (editor) {
            editor.value = `SELECT * FROM "${tableName}" LIMIT 100`;
            editor.focus();
        }
    },

    _setQuery(sql) {
        const editor = document.getElementById('querySqlEditor');
        if (editor) {
            editor.value = sql;
            editor.focus();
        }
    },

    _setAndRunQuery(sql) {
        this._setQuery(sql);
        this._runQuery();
    },

    async _runQuery() {
        const editor = document.getElementById('querySqlEditor');
        const statusBar = document.getElementById('queryStatusBar');
        const runBtn = document.getElementById('queryRunBtn');
        if (!editor || !statusBar) return;

        const sql = editor.value.trim();
        if (!sql) return;

        if (runBtn) runBtn.disabled = true;
        statusBar.innerHTML = 'Executing...';

        try {
            const resp = await fetch('/api/query/sql', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sql }),
            });

            const data = await resp.json();

            if (!resp.ok) {
                statusBar.innerHTML = `<span class="error">Error: ${data.detail || 'Query failed'}</span>`;
                return;
            }

            this._queryState.lastResult = data;
            this._renderQueryResults(data);

            statusBar.innerHTML =
                `<span class="success">${data.row_count.toLocaleString()} rows</span> · ` +
                `${data.columns.length} cols · ` +
                `${data.execution_time_ms}ms`;

        } catch (e) {
            statusBar.innerHTML = `<span class="error">Error: ${e.message}</span>`;
        } finally {
            if (runBtn) runBtn.disabled = false;
        }
    },

    _renderQueryResults(data) {
        const wrapper = document.getElementById('queryResultsWrapper');
        if (!wrapper) return;

        if (!data.rows.length) {
            wrapper.innerHTML = '<div class="query-empty-state">Query returned 0 rows</div>';
            return;
        }

        const schemaMap = {};
        if (data.schema) {
            data.schema.forEach(s => { schemaMap[s.name] = s.type; });
        }

        let html = '<table><thead><tr>';
        html += data.columns.map(col =>
            `<th>${col}<span class="col-type">${schemaMap[col] || ''}</span></th>`
        ).join('');
        html += '</tr></thead><tbody>';

        for (const row of data.rows) {
            html += '<tr>';
            for (const col of data.columns) {
                const val = row[col];
                if (val === null || val === undefined) {
                    html += '<td class="null">NULL</td>';
                } else if (typeof val === 'number') {
                    html += `<td>${Number.isInteger(val) ? val.toLocaleString() : val.toFixed(6)}</td>`;
                } else {
                    const escaped = String(val).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                    html += `<td title="${escaped}">${escaped}</td>`;
                }
            }
            html += '</tr>';
        }

        html += '</tbody></table>';
        wrapper.innerHTML = html;
    },

    _clearQueryResults() {
        this._queryState.lastResult = null;
        const wrapper = document.getElementById('queryResultsWrapper');
        if (wrapper) {
            wrapper.innerHTML = '<div class="query-empty-state">Run a query to see results</div>';
        }
        const statusBar = document.getElementById('queryStatusBar');
        if (statusBar) statusBar.innerHTML = 'Ready';
    },

    _exportQueryCSV() {
        const result = this._queryState.lastResult;
        if (!result || !result.rows.length) return;

        const { columns, rows } = result;
        let csv = columns.map(c => `"${c}"`).join(',') + '\n';

        for (const row of rows) {
            csv += columns.map(col => {
                const val = row[col];
                if (val === null || val === undefined) return '';
                const str = String(val).replace(/"/g, '""');
                return `"${str}"`;
            }).join(',') + '\n';
        }

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'query_results.csv';
        a.click();
        URL.revokeObjectURL(url);
    },

    // =========================================================================
    // Pop-out / Detach
    // =========================================================================

    _setupPopout() {
        const popoutBtn = document.getElementById('analysis-popout-btn');
        if (popoutBtn) {
            popoutBtn.addEventListener('click', () => this.popout());
        }
    },

    /**
     * Pop out the analysis panel into a separate window.
     * Clones stylesheets and the active tab content into the new window.
     */
    popout() {
        // If already popped out, focus the existing window
        if (this._popoutWindow && !this._popoutWindow.closed) {
            this._popoutWindow.focus();
            return;
        }

        // Open a blank window
        const win = window.open('', 'analysisPopout',
            'width=1000,height=500,menubar=no,toolbar=no,location=no,status=no');

        if (!win) {
            if (typeof Toast !== 'undefined') {
                Toast.show('Popup blocked — please allow popups for this site', 'error');
            }
            return;
        }

        this._popoutWindow = win;

        // Build the popout document
        const doc = win.document;
        doc.open();
        doc.write('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">');
        doc.write('<title>Analysis Panel — Graph Analyzer</title>');

        // Copy all stylesheets from the parent
        for (const link of document.querySelectorAll('link[rel="stylesheet"]')) {
            doc.write(`<link rel="stylesheet" href="${link.href}">`);
        }

        // Popout-specific styles
        doc.write(`<style>
            body { margin: 0; background: #1a1a1a; color: #e0e0e0; font-family: inherit; overflow: hidden; }
            .popout-panel { display: flex; flex-direction: column; height: 100vh; }
            .popout-panel .analysis-tab-bar { flex-shrink: 0; }
            .popout-panel .analysis-tab-contents { flex: 1; position: relative; overflow: hidden; min-height: 0; }
            .popout-panel .analysis-tab-content { position: absolute; inset: 0; overflow-y: auto; overflow-x: hidden; padding: 12px 16px; display: none; }
            .popout-panel .analysis-tab-content.active { display: block; }
            .popout-panel .analysis-tab-content[data-analysis-tab="data-explorer-tab"] { padding: 0 !important; }
            .popout-panel .analysis-tab-content[data-analysis-tab="query-tab"] { padding: 0 !important; }
            .popout-panel .analysis-tab-actions { display: none; }
            .popout-panel .analysis-tab-content .style-row { max-width: 400px; }
            .popout-panel .analysis-tab-content .option-row { max-width: 400px; }
            .popout-panel .analysis-tab-content select,
            .popout-panel .analysis-tab-content input[type="text"],
            .popout-panel .analysis-tab-content input[type="number"] { max-width: 300px; }
            .popout-panel .analysis-tab-content .button-group { max-width: 400px; }
            .popout-panel .analysis-tab-content .flow-endpoints { max-width: 400px; }
            .popout-panel .analysis-tab-content .flow-tabs { max-width: 400px; }
        </style>`);

        doc.write('</head><body>');
        doc.write('<div class="popout-panel">');

        // Clone the tab bar
        const tabBar = this.panel.querySelector('.analysis-tab-bar');
        if (tabBar) {
            doc.write(tabBar.outerHTML);
        }

        // Clone all tab contents
        const tabContentsEl = this.panel.querySelector('.analysis-tab-contents');
        if (tabContentsEl) {
            doc.write(tabContentsEl.outerHTML);
        }

        doc.write('</div>');

        // Add icons script for icon injection
        doc.write('<script src="/static/js/icons.js"><\/script>');
        doc.write(`<script>
            // Tab switching in popout
            document.querySelectorAll('.analysis-tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    const tabName = tab.dataset.analysisTab;
                    document.querySelectorAll('.analysis-tab').forEach(t =>
                        t.classList.toggle('active', t.dataset.analysisTab === tabName));
                    document.querySelectorAll('.analysis-tab-content').forEach(c =>
                        c.classList.toggle('active', c.dataset.analysisTab === tabName));

                    // Lazy-load data explorer iframe
                    if (tabName === 'data-explorer-tab') {
                        const iframe = document.getElementById('data-explorer-iframe');
                        if (iframe && !iframe.src.includes('${this.DATA_EXPLORER_URL}')) {
                            iframe.src = '${this.DATA_EXPLORER_URL}';
                        }
                    }

                    // Re-inject icons
                    if (typeof Icons !== 'undefined') {
                        setTimeout(() => Icons.inject(), 50);
                    }
                });
            });

            // Relay LOCATE_NODE messages from iframes to the main window
            window.addEventListener('message', (event) => {
                if (event.data && event.data.type === 'LOCATE_NODE' && window.opener) {
                    window.opener.postMessage(event.data, '*');
                }
            });

            // Inject icons after load
            window.addEventListener('load', () => {
                if (typeof Icons !== 'undefined') Icons.inject();
            });
        <\/script>`);

        doc.write('</body></html>');
        doc.close();

        // Listen for the popout window closing
        const checkClosed = setInterval(() => {
            if (win.closed) {
                clearInterval(checkClosed);
                this._popoutWindow = null;
                // Panel can be reopened inline
            }
        }, 500);

        // Close the inline panel
        this.close();
    },

    // =========================================================================
    // Snapshot Sub-tabs
    // =========================================================================

};
