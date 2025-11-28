/**
 * Layout Service
 * 
 * Express server that computes graph layouts using Cytoscape.js.
 * Used as fallback when Cytoscape Desktop is unavailable.
 */

const express = require('express');
const cytoscape = require('cytoscape');

const app = express();
app.use(express.json({ limit: '500mb' }));

const PORT = process.env.PORT || 3000;

/**
 * POST /layout
 * 
 * Compute layout for graph elements.
 * 
 * Request body:
 * {
 *   "elements": [...], // Cytoscape.js elements
 *   "options": {       // Optional layout options
 *     "name": "cose",
 *     "animate": false,
 *     ...
 *   }
 * }
 * 
 * Response:
 * {
 *   "positions": [
 *     { "id": "node1", "x": 100, "y": 200 },
 *     ...
 *   ]
 * }
 */
app.post('/layout', async (req, res) => {
    const startTime = Date.now();
    
    try {
        const { elements, options = {} } = req.body;
        
        if (!elements || !Array.isArray(elements)) {
            return res.status(400).json({ error: 'elements array required' });
        }
        
        console.log(`[LAYOUT] Processing ${elements.length} elements...`);
        
        // Create headless Cytoscape instance
        const cy = cytoscape({
            headless: true,
            styleEnabled: false,
            elements: elements
        });
        
        // Default layout options
        const layoutOptions = {
            name: options.name || 'cose',
            animate: false,
            fit: true,
            padding: 50,
            nodeRepulsion: 400000,
            idealEdgeLength: 100,
            edgeElasticity: 100,
            nestingFactor: 5,
            gravity: 80,
            numIter: 1000,
            initialTemp: 200,
            coolingFactor: 0.95,
            minTemp: 1.0,
            ...options
        };
        
        // Run layout
        const layout = cy.layout(layoutOptions);
        
        await new Promise((resolve) => {
            layout.on('layoutstop', resolve);
            layout.run();
        });
        
        // Extract positions
        const positions = [];
        cy.nodes().forEach(node => {
            const pos = node.position();
            positions.push({
                id: node.id(),
                x: pos.x,
                y: pos.y
            });
        });
        
        const elapsed = Date.now() - startTime;
        console.log(`[LAYOUT] Computed ${positions.length} positions in ${elapsed}ms`);
        
        // Clean up
        cy.destroy();
        
        res.json({ positions, elapsed });
        
    } catch (error) {
        console.error('[LAYOUT] Error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /health
 * 
 * Health check endpoint.
 */
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', service: 'layout-service' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Layout service listening on port ${PORT}`);
});