/**
 * Cytoscape.js Layout Service
 * Computes graph layouts using Cytoscape's algorithms
 * Supports incremental layout via locked node positions.
 */

const express = require('express');
const cors = require('cors');
const cytoscape = require('cytoscape');

// Register additional layout algorithms
const coseBilkent = require('cytoscape-cose-bilkent');
const fcose = require('cytoscape-fcose');
const cola = require('cytoscape-cola');
const dagre = require('cytoscape-dagre');
const klay = require('cytoscape-klay');

cytoscape.use(coseBilkent);
cytoscape.use(fcose);
cytoscape.use(cola);
cytoscape.use(dagre);
cytoscape.use(klay);

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        service: 'cytoscape-layout-service',
        algorithms: [
            'cose', 'fcose', 'cose-bilkent', 'cola', 
            'dagre', 'klay', 'grid', 'circle', 
            'concentric', 'breadthfirst', 'random'
        ]
    });
});

/**
 * Compute layout for a graph
 */
app.post('/compute-layout', (req, res) => {
    try {
        const { nodes, edges, algorithm = 'fcose', options = {}, lockedPositions = {} } = req.body;
        
        if (!nodes || !Array.isArray(nodes)) {
            return res.status(400).json({ error: 'Invalid nodes data' });
        }
        
        const isIncremental = Object.keys(lockedPositions).length > 0;
        console.log(`[LAYOUT SERVICE] Computing ${algorithm} layout for ${nodes.length} nodes (${isIncremental ? 'Incremental' : 'Full'})`);
        
        // Create Cytoscape instance
        const startTime = Date.now();
        
        const cy = cytoscape({
            elements: {
                nodes: nodes.map(node => {
                    // Support both string IDs and object definitions with initial positions
                    if (typeof node === 'string') {
                        return { data: { id: node } };
                    }
                    // If node is an object { data: {id: 'x'}, position: {x: 1, y: 2} }
                    return node;
                }),
                edges: edges.map(edge => ({
                    data: { 
                        id: `${edge.source}-${edge.target}`,
                        source: edge.source, 
                        target: edge.target 
                    }
                }))
            },
            headless: true,
            styleEnabled: false
        });

        // Apply Locked Positions (Incremental Layout Strategy)
        if (isIncremental) {
            console.log(`[LAYOUT SERVICE] Locking ${Object.keys(lockedPositions).length} anchor nodes`);
            cy.nodes().forEach(node => {
                const id = node.id();
                if (lockedPositions[id]) {
                    node.position(lockedPositions[id]);
                    node.lock(); // Crucial: Lock the node so the layout doesn't move it
                }
            });
        }
        
        // Layout configurations matching Cytoscape Desktop
        const layoutConfigs = {
            // fCoSE - Best force-directed
            'fcose': {
                name: 'fcose',
                quality: 'proof',
                // Randomize only if we are doing a full layout. 
                // If incremental, we rely on the initial positions (centroids) passed from backend.
                randomize: !isIncremental, 
                animate: false,
                packComponents: false,
                numIter: isIncremental ? 500 : 100, // More iterations for incremental to settle nicely
                sampleSize: 100,
                idealEdgeLength: edge => 30,
                edgeElasticity: edge => 0.45,
                nodeRepulsion: node => 4500,
                gravity: 0.25,
                // Disable tiling for incremental to avoid forcing organic nodes into grid lines
                tile: !isIncremental, 
                fit: !isIncremental, // Don't fit/zoom on incremental updates to preserve coordinate system
                padding: 30,
                ...options
            },
            
            // Standard CoSE
            'cose': {
                name: 'cose',
                animate: false,
                randomize: !isIncremental,
                numIter: 100,
                idealEdgeLength: 30,
                nodeRepulsion: node => 4500,
                fit: !isIncremental,
                ...options
            },
            
            // CoSE-Bilkent
            'cose-bilkent': {
                name: 'cose-bilkent',
                quality: 'proof',
                animate: false,
                randomize: !isIncremental,
                fit: !isIncremental,
                tile: !isIncremental,
                nodeRepulsion: 4500,
                idealEdgeLength: 50,
                numIter: 2500,
                ...options
            },
            
            // Cola
            'cola': {
                name: 'cola',
                animate: false,
                fit: !isIncremental,
                randomize: !isIncremental,
                handleDisconnected: true,
                ...options
            },

            'preset': { name: 'preset' }
        };
        
        // Get layout configuration
        let layoutConfig = layoutConfigs[algorithm] || layoutConfigs['fcose'];
        
        // Run layout
        const layout = cy.layout(layoutConfig);
        layout.run();
        
        // Get positions
        const positions = {};
        cy.nodes().forEach(node => {
            const pos = node.position();
            positions[node.id()] = { 
                x: pos.x, 
                y: pos.y 
            };
        });
        
        const endTime = Date.now();
        const duration = (endTime - startTime) / 1000;
        
        console.log(`[LAYOUT SERVICE] Layout computed in ${duration.toFixed(2)}s`);
        
        res.json({ 
            positions,
            algorithm,
            duration,
            nodeCount: nodes.length,
            edgeCount: edges.length
        });
        
    } catch (error) {
        console.error('[LAYOUT SERVICE] Error:', error);
        res.status(500).json({ 
            error: 'Layout computation failed', 
            message: error.message 
        });
    }
});

/**
 * Get available layout algorithms
 */
app.get('/algorithms', (req, res) => {
    res.json({
        algorithms: [
            { id: 'fcose', name: 'fCoSE', description: 'Fast Compound Spring Embedder (Recommended)' },
            { id: 'cose', name: 'CoSE', description: 'Compound Spring Embedder' },
            { id: 'cose-bilkent', name: 'CoSE-Bilkent', description: 'High Quality CoSE' },
            { id: 'cola', name: 'Cola', description: 'Constraint-based' },
            { id: 'dagre', name: 'Dagre', description: 'Hierarchical (DAG)' },
            { id: 'klay', name: 'Klay', description: 'Layered Hierarchical' },
            { id: 'circle', name: 'Circle', description: 'Circular arrangement' },
            { id: 'grid', name: 'Grid', description: 'Grid arrangement' }
        ]
    });
});

// Start server
app.listen(PORT, () => {
    console.log('='.repeat(70));
    console.log('CYTOSCAPE.JS LAYOUT SERVICE');
    console.log('='.repeat(70));
    console.log(`Server running on http://localhost:${PORT}`);
});