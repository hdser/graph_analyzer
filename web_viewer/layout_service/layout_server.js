/**
 * Cytoscape.js Layout Service
 * Computes graph layouts using Cytoscape's algorithms
 * Same algorithms as Cytoscape Desktop!
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
    console.log('[LAYOUT SERVICE] Received layout request');
    
    try {
        const { nodes, edges, algorithm = 'fcose', options = {} } = req.body;
        
        if (!nodes || !Array.isArray(nodes)) {
            return res.status(400).json({ error: 'Invalid nodes data' });
        }
        
        console.log(`[LAYOUT SERVICE] Computing ${algorithm} layout for ${nodes.length} nodes, ${edges.length} edges`);
        
        // Create Cytoscape instance
        const startTime = Date.now();
        
        const cy = cytoscape({
            elements: {
                nodes: nodes.map(nodeId => ({
                    data: { id: nodeId }
                })),
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
        
        // Layout configurations matching Cytoscape Desktop
        const layoutConfigs = {
            // fCoSE - Best force-directed, matches Prefuse Force Directed settings
            'fcose': {
                name: 'fcose',
                quality: 'proof', // Highest quality (matches your 700 iterations)
                randomize: false, // Start from scratch
                animate: false,
                packComponents: false, // Don't partition graph
                numIter: 100, // Match your iterations setting
                sampleSize: 100,
                
                // Edge settings matching your configuration
                idealEdgeLength: edge => {
                    const weight = edge.data('weight') || 0.5;
                    return 30; // Default Spring Length = 30
                },
                
                edgeElasticity: edge => {
                    const weight = edge.data('weight') || 0.5;
                    return 0.45 * weight;
                },
                
                // Node repulsion (Spring Coefficient = 1×10^-5)
                nodeRepulsion: node => 4500,
                
                // Node mass
                uniformNodeDimensions: true, // Default Node Mass = 1
                nodeDimensionsIncludeLabels: false,
                
                // Gravity
                gravity: 0.25,
                gravityRangeCompound: 1.5,
                gravityCompound: 1.0,
                gravityRange: 3.8,
                
                // Cooling
                initialEnergyOnIncremental: 0.3,
                
                // Tiling for deterministic results
                tile: true,
                tilingPaddingVertical: 10,
                tilingPaddingHorizontal: 10,
                
                // Layout bounds
                fit: true,
                padding: 30,
                
                // Compound improvements
                improveCompound: true,
                
                // Step mode
                step: 'all',
                
                ...options
            },
            
            // Standard CoSE
            'cose': {
                name: 'cose',
                animate: false,
                randomize: false,
                numIter: 100,
                idealEdgeLength: 30,
                nodeRepulsion: node => 4500,
                nodeOverlap: 20,
                refresh: 20,
                fit: true,
                padding: 30,
                componentSpacing: 100,
                edgeElasticity: edge => 0.45,
                nestingFactor: 1.2,
                gravity: 0.4,
                initialTemp: 1000,
                coolingFactor: 0.99,
                minTemp: 1.0,
                ...options
            },
            
            // CoSE-Bilkent (High quality, slower)
            'cose-bilkent': {
                name: 'cose-bilkent',
                quality: 'proof',
                animate: false,
                randomize: false,
                fit: true,
                padding: 30,
                nodeDimensionsIncludeLabels: false,
                uniformNodeDimensions: false,
                packComponents: true,
                nodeRepulsion: 4500,
                nodeOverlap: 10,
                idealEdgeLength: 50,
                edgeElasticity: 0.45,
                nestingFactor: 0.1,
                gravity: 0.25,
                gravityRange: 3.8,
                coolingFactor: 0.99,
                initialTemp: 200,
                minTemp: 1.0,
                numIter: 2500,
                tile: true,
                tilingPaddingVertical: 10,
                tilingPaddingHorizontal: 10,
                ...options
            },
            
            // Cola (Constraint-based)
            'cola': {
                name: 'cola',
                animate: false,
                refresh: 1,
                maxSimulationTime: 4000,
                ungrabifyWhileSimulating: false,
                fit: true,
                padding: 30,
                boundingBox: undefined,
                nodeDimensionsIncludeLabels: false,
                randomize: false,
                avoidOverlap: true,
                handleDisconnected: true,
                convergenceThreshold: 0.01,
                nodeSpacing: node => 10,
                flow: undefined,
                alignment: undefined,
                gapInequalities: undefined,
                edgeLength: undefined,
                edgeSymDiffLength: undefined,
                edgeJaccardLength: undefined,
                unconstrIter: undefined,
                userConstIter: undefined,
                allConstIter: undefined,
                ...options
            },
            
            // Dagre (Hierarchical)
            'dagre': {
                name: 'dagre',
                fit: true,
                padding: 30,
                rankDir: 'TB',
                rankSep: 50,
                nodeSep: 10,
                edgeSep: 10,
                ranker: 'network-simplex',
                acyclicer: undefined,
                ...options
            },
            
            // ELK (Multiple algorithms)
            'elk': {
                name: 'elk',
                elk: {
                    algorithm: 'layered',
                    'elk.direction': 'DOWN',
                    'elk.layered.spacing.nodeNodeBetweenLayers': 100,
                    'elk.spacing.nodeNode': 80,
                },
                fit: true,
                padding: 30,
                ...options
            },
            
            // Klay (Hierarchical)
            'klay': {
                name: 'klay',
                fit: true,
                padding: 30,
                klay: {
                    direction: 'DOWN',
                    layoutHierarchy: false,
                    intCoordinates: false,
                    edgeRouting: 'ORTHOGONAL',
                    edgeSpacingFactor: 0.5,
                    feedbackEdges: false,
                    fixedAlignment: 'NONE',
                    inLayerSpacingFactor: 1.0,
                    linearSegmentsDeflectionDampening: 0.3,
                    mergeEdges: false,
                    mergeHierarchyCrossingEdges: false,
                    nodeLayering: 'NETWORK_SIMPLEX',
                    nodePlacement: 'BRANDES_KOEPF',
                    randomizationSeed: 1,
                    routeSelfLoopInside: false,
                    separateConnectedComponents: true,
                    spacing: 20,
                    thoroughness: 7
                },
                ...options
            },
            
            // Euler (Fast force-directed)
            'euler': {
                name: 'euler',
                springLength: edge => 80,
                springCoeff: edge => 0.0008,
                mass: node => 4,
                gravity: -1.2,
                pull: 0.001,
                theta: 0.666,
                dragCoeff: 0.02,
                movementThreshold: 1,
                timeStep: 20,
                refresh: 10,
                animate: false,
                animationDuration: undefined,
                animationEasing: undefined,
                maxIterations: 1000,
                maxSimulationTime: 4000,
                ungrabifyWhileSimulating: false,
                fit: true,
                padding: 30,
                randomize: false,
                ...options
            },
            
            // Spread (Space-filling)
            'spread': {
                name: 'spread',
                animate: false,
                ready: undefined,
                stop: undefined,
                fit: true,
                minDist: 20,
                padding: 20,
                expandingFactor: -1.0,
                maxExpandIterations: 4,
                boundingBox: undefined,
                randomize: false,
                ...options
            },
            
            // AVSDF (Circular with edge crossing minimization)
            'avsdf': {
                name: 'avsdf',
                animate: false,
                fit: true,
                padding: 30,
                ungrabifyWhileSimulating: false,
                ...options
            },
            
            // CiSE (Circular clusters)
            'cise': {
                name: 'cise',
                animate: false,
                refresh: 10,
                fit: true,
                padding: 30,
                nodeSeparation: 12.5,
                idealInterClusterEdgeLengthCoefficient: 1.4,
                allowNodesInsideCircle: false,
                maxRatioOfNodesInsideCircle: 0.1,
                springCoeff: 0.45,
                nodeRepulsion: 4500,
                gravity: 0.25,
                gravityRange: 3.8,
                ...options
            },
            
            // Circle layout
            'circle': {
                name: 'circle',
                fit: true,
                padding: 30,
                radius: undefined,
                startAngle: 3 / 2 * Math.PI,
                sweep: undefined,
                clockwise: true,
                sort: undefined,
                animate: false,
                animationDuration: 500,
                animationEasing: undefined,
                ...options
            },
            
            // Concentric layout
            'concentric': {
                name: 'concentric',
                fit: true,
                padding: 30,
                startAngle: 3 / 2 * Math.PI,
                sweep: undefined,
                clockwise: true,
                equidistant: false,
                minNodeSpacing: 10,
                height: undefined,
                width: undefined,
                concentric: function(node) {
                    return node.degree();
                },
                levelWidth: function(nodes) {
                    return 2;
                },
                animate: false,
                animationDuration: 500,
                animationEasing: undefined,
                ...options
            },
            
            // Grid layout
            'grid': {
                name: 'grid',
                fit: true,
                padding: 30,
                position: function(node) { return null; },
                avoidOverlap: true,
                avoidOverlapPadding: 10,
                nodeDimensionsIncludeLabels: false,
                spacingFactor: undefined,
                condense: false,
                rows: undefined,
                cols: undefined,
                sort: undefined,
                animate: false,
                animationDuration: 500,
                animationEasing: undefined,
                ...options
            },
            
            // Breadthfirst layout
            'breadthfirst': {
                name: 'breadthfirst',
                fit: true,
                directed: false,
                padding: 30,
                circle: false,
                grid: false,
                spacingFactor: 1.75,
                boundingBox: undefined,
                avoidOverlap: true,
                nodeDimensionsIncludeLabels: false,
                roots: undefined,
                maximal: false,
                animate: false,
                animationDuration: 500,
                animationEasing: undefined,
                ...options
            },
            
            // Random layout
            'random': {
                name: 'random',
                fit: true,
                padding: 30,
                boundingBox: undefined,
                animate: false,
                animationDuration: 500,
                animationEasing: undefined,
                ...options
            },
            
            // Preset layout (use existing positions)
            'preset': {
                name: 'preset',
                fit: true,
                padding: 30,
                animate: false,
                animationDuration: 500,
                animationEasing: undefined,
                positions: undefined,
                zoom: undefined,
                pan: undefined,
                ...options
            }
        };
        
        // Get layout configuration
        const layoutConfig = layoutConfigs[algorithm] || layoutConfigs['fcose'];
        
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
            {
                id: 'fcose',
                name: 'fCoSE (Fast Compound Spring Embedder)',
                description: 'Fast, high-quality force-directed layout. Best overall choice.',
                speed: 'fast',
                quality: 'high'
            },
            {
                id: 'cose',
                name: 'CoSE (Compound Spring Embedder)',
                description: 'Standard force-directed layout. Same as Cytoscape Desktop.',
                speed: 'medium',
                quality: 'high'
            },
            {
                id: 'cose-bilkent',
                name: 'CoSE-Bilkent',
                description: 'High-quality force-directed layout with better edge crossing reduction.',
                speed: 'medium',
                quality: 'very high'
            },
            {
                id: 'cola',
                name: 'Cola',
                description: 'Constraint-based layout. Good for directed graphs.',
                speed: 'medium',
                quality: 'high'
            },
            {
                id: 'dagre',
                name: 'Dagre',
                description: 'Hierarchical layout for directed acyclic graphs.',
                speed: 'fast',
                quality: 'medium'
            },
            {
                id: 'klay',
                name: 'Klay',
                description: 'Layer-based layout. Better than dagre for hierarchies.',
                speed: 'fast',
                quality: 'high'
            },
            {
                id: 'circle',
                name: 'Circle',
                description: 'Nodes arranged in a circle.',
                speed: 'very fast',
                quality: 'low'
            },
            {
                id: 'concentric',
                name: 'Concentric',
                description: 'Nodes arranged in concentric circles.',
                speed: 'very fast',
                quality: 'medium'
            },
            {
                id: 'grid',
                name: 'Grid',
                description: 'Nodes arranged in a grid.',
                speed: 'very fast',
                quality: 'low'
            },
            {
                id: 'breadthfirst',
                name: 'Breadthfirst',
                description: 'Tree layout using breadth-first traversal.',
                speed: 'fast',
                quality: 'medium'
            },
            {
                id: 'random',
                name: 'Random',
                description: 'Random positions.',
                speed: 'instant',
                quality: 'none'
            }
        ]
    });
});

// Start server
app.listen(PORT, () => {
    console.log('='.repeat(70));
    console.log('CYTOSCAPE.JS LAYOUT SERVICE');
    console.log('='.repeat(70));
    console.log(`Server running on http://localhost:${PORT}`);
    console.log('Available endpoints:');
    console.log(`  GET  http://localhost:${PORT}/health`);
    console.log(`  GET  http://localhost:${PORT}/algorithms`);
    console.log(`  POST http://localhost:${PORT}/compute-layout`);
    console.log('='.repeat(70));
});