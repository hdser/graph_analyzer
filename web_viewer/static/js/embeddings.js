/**
 * Embeddings API Module
 * 
 * Client-side API for GIT-CD deep learning endpoints.
 */

const EmbeddingsAPI = (function() {
    'use strict';
    
    const API_BASE = '/api/embeddings';
    
    // State
    let _availabilityCache = null;
    let _currentModel = null;
    let _isTraining = false;
    
    /**
     * Fetch JSON with error handling.
     */
    async function fetchJSON(url, options = {}) {
        const response = await fetch(url, {
            headers: { 
                'Content-Type': 'application/json', 
                ...options.headers 
            },
            ...options
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        return response.json();
    }
    
    /**
     * Check if deep learning is available.
     */
    async function checkAvailability() {
        if (_availabilityCache) {
            return _availabilityCache;
        }
        
        try {
            const response = await fetchJSON(`${API_BASE}/info`);
            
            _availabilityCache = {
                available: response.deep_learning?.available || false,
                torch: response.deep_learning?.torch_available || false,
                cuda: response.deep_learning?.cuda_available || false,
                mps: response.deep_learning?.mps_available || false,
                pyg: response.deep_learning?.pyg_available || false,
                hasModel: response.has_model || false,
            };
            
            if (response.has_model) {
                _currentModel = true;
            }
            
            return _availabilityCache;
        } catch (error) {
            console.error('[Embeddings] Availability check failed:', error);
            _availabilityCache = {
                available: false,
                torch: false,
                cuda: false,
                pyg: false,
                hasModel: false,
                error: error.message
            };
            return _availabilityCache;
        }
    }
    
    /**
     * Check if a model is loaded.
     */
    function hasModel() {
        return _currentModel !== null;
    }
    
    /**
     * Check if training is in progress.
     */
    function isTraining() {
        return _isTraining;
    }
    
    /**
     * Start training a GIT-CD model (async).
     */
    async function trainModel(config = {}) {
        if (_isTraining) {
            throw new Error('Training already in progress');
        }
        
        _isTraining = true;
        
        try {
            const request = {
                graph_name: config.graph_name || null,  // IMPORTANT: which graph to train on
                model_name: config.model_name || config.modelName || null,  // null = auto-generate
                num_clusters: config.num_clusters || config.numClusters || 20,
                hidden_dim: config.hidden_dim || config.hiddenDim || 128,
                num_gnn_layers: config.num_gnn_layers || config.numGnnLayers || 1,
                num_transformer_layers: config.num_transformer_layers || config.numTransformerLayers || 2,
                dropout: config.dropout || 0.5,
                max_epochs: config.max_epochs || config.maxEpochs || 200,
                learning_rate: config.learning_rate || config.learningRate || 0.0003,
                weight_decay: config.weight_decay || config.weightDecay || 0.0005,
                patience: config.patience || 5,
            };
            
            console.log('[EmbeddingsAPI] Training request:', request);
            
            const result = await fetchJSON(`${API_BASE}/train`, {
                method: 'POST',
                body: JSON.stringify(request)
            });
            
            if (result.task_id) {
                console.log('[EmbeddingsAPI] Training started, task_id:', result.task_id);
            } else if (result.success) {
                _currentModel = result.model_name;
                _isTraining = false;
            }
            
            return result;
        } catch (error) {
            _isTraining = false;
            throw error;
        }
    }
    
    /**
     * Get training task status.
     */
    async function getTrainingStatus(taskId) {
        const result = await fetchJSON(`${API_BASE}/train/status/${taskId}`);
        
        if (result.status === 'completed') {
            _isTraining = false;
            if (result.result?.model_name) {
                _currentModel = result.result.model_name;
            }
        } else if (result.status === 'failed') {
            _isTraining = false;
        }
        
        return result;
    }
    
    /**
     * Get community assignments.
     */
    async function getCommunities(includeConfidence = true) {
        return fetchJSON(`${API_BASE}/communities?include_confidence=${includeConfidence}`);
    }
    
    /**
     * Find similar nodes.
     */
    async function findSimilar(queryNode, k = 10, metric = 'cosine') {
        return fetchJSON(`${API_BASE}/similar`, {
            method: 'POST',
            body: JSON.stringify({
                query_node: queryNode,
                k: k,
                metric: metric
            })
        });
    }
    
    /**
     * List available models.
     */
    async function listModels() {
        return fetchJSON(`${API_BASE}/models`);
    }
    
    /**
     * Load a saved model.
     */
    async function loadModel(modelName) {
        const result = await fetchJSON(`${API_BASE}/models/load`, {
            method: 'POST',
            body: JSON.stringify({ model_name: modelName })
        });
        
        if (result.success) {
            _currentModel = modelName;
        }
        
        return result;
    }
    
    /**
     * Clear cache.
     */
    async function clearCache() {
        _availabilityCache = null;
        return fetchJSON(`${API_BASE}/cache`, { method: 'DELETE' });
    }
    
    // Public API
    return {
        checkAvailability,
        hasModel,
        isTraining,
        trainModel,
        getTrainingStatus,
        getCommunities,
        findSimilar,
        listModels,
        loadModel,
        clearCache,
    };
})();

window.EmbeddingsAPI = EmbeddingsAPI;