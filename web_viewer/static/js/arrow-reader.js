/**
 * Arrow IPC Reader
 *
 * Fetches binary Arrow IPC streams from the backend and converts
 * them to typed arrays for direct use in cosmos.gl / Cytoscape.js.
 *
 * Requires Apache Arrow JS (loaded via CDN in index.html).
 */

const ArrowReader = {

    /**
     * Fetch an Arrow IPC endpoint and return an Arrow Table.
     * @param {string} url - API endpoint URL
     * @returns {Promise<Arrow.Table|null>} Arrow table or null on error
     */
    async fetchTable(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) return null;

            const buffer = await response.arrayBuffer();
            if (typeof Arrow === 'undefined') {
                console.warn('[ArrowReader] Apache Arrow JS not loaded, falling back');
                return null;
            }
            return Arrow.tableFromIPC(new Uint8Array(buffer));
        } catch (err) {
            console.warn('[ArrowReader] fetch failed:', err);
            return null;
        }
    },

    /**
     * Convert an Arrow table of nodes to arrays usable by renderers.
     *
     * @param {Arrow.Table} table - Arrow table with columns: id, x, y, ...metrics
     * @returns {{
     *   ids: string[],
     *   positions: Float32Array,
     *   metrics: Object<string, Float64Array>,
     *   nodeObjects: Object[]
     * }}
     */
    arrowToNodeArrays(table) {
        const n = table.numRows;
        const ids = [];
        const positions = new Float32Array(n * 2);
        const metrics = {};
        const nodeObjects = [];

        // Get column accessors
        const idCol = table.getChild('id');
        const xCol = table.getChild('x');
        const yCol = table.getChild('y');

        // Identify metric columns (everything except id, x, y)
        const metricNames = [];
        for (const field of table.schema.fields) {
            if (!['id', 'x', 'y'].includes(field.name)) {
                metricNames.push(field.name);
            }
        }

        // Pre-allocate metric arrays
        for (const name of metricNames) {
            const col = table.getChild(name);
            if (col && col.type && col.type.typeId !== undefined) {
                // Check if numeric type
                const typeId = col.type.typeId;
                if (typeId >= 2 && typeId <= 12) { // Int/Float types in Arrow
                    metrics[name] = new Float64Array(n);
                }
            }
        }

        for (let i = 0; i < n; i++) {
            const id = idCol ? String(idCol.get(i)) : String(i);
            const x = xCol ? Number(xCol.get(i)) : 0;
            const y = yCol ? Number(yCol.get(i)) : 0;

            ids.push(id);
            positions[i * 2] = x;
            positions[i * 2 + 1] = y;

            // Build node object for Cytoscape/cosmos setData()
            const obj = { id, x, y, _hasPosition: x !== 0 || y !== 0 };

            for (const name of metricNames) {
                const col = table.getChild(name);
                if (col) {
                    const val = col.get(i);
                    if (metrics[name]) {
                        metrics[name][i] = Number(val) || 0;
                    }
                    obj[name] = val;
                }
            }

            nodeObjects.push(obj);
        }

        return { ids, positions, metrics, nodeObjects };
    },

    /**
     * Convert an Arrow table of edges to arrays.
     *
     * @param {Arrow.Table} table - Arrow table with: source, target, [source_idx, target_idx]
     * @returns {{
     *   edges: {source: string, target: string}[],
     *   linkIndices: Int32Array|null
     * }}
     */
    arrowToEdgeArrays(table) {
        const n = table.numRows;
        const sourceCol = table.getChild('source');
        const targetCol = table.getChild('target');
        const srcIdxCol = table.getChild('source_idx');
        const tgtIdxCol = table.getChild('target_idx');

        const edges = [];
        let linkIndices = null;

        if (srcIdxCol && tgtIdxCol) {
            linkIndices = new Float32Array(n * 2);
        }

        for (let i = 0; i < n; i++) {
            const source = String(sourceCol.get(i));
            const target = String(targetCol.get(i));
            edges.push({ source, target, id: `${source}-${target}` });

            if (linkIndices) {
                linkIndices[i * 2] = Number(srcIdxCol.get(i));
                linkIndices[i * 2 + 1] = Number(tgtIdxCol.get(i));
            }
        }

        return { edges, linkIndices };
    },

    /**
     * Check if Arrow JS is loaded and available.
     */
    isAvailable() {
        return typeof Arrow !== 'undefined' && typeof Arrow.tableFromIPC === 'function';
    }
};
