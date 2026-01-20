/**
 * WebGL Capability Detector
 * 
 * Determines if cosmos.gl can run and at what performance tier.
 * cosmos.gl requires specific WebGL extensions that aren't universally available.
 */

const WebGLDetector = {
    // Cached capabilities (computed once)
    _capabilities: null,
    
    /**
     * Detect WebGL capabilities
     * @returns {Object} Capability report
     */
    detect() {
        // Return cached result if available
        if (this._capabilities) {
            return this._capabilities;
        }
        
        const canvas = document.createElement('canvas');
        
        // Try WebGL2 first, then WebGL1
        let gl = canvas.getContext('webgl2');
        const isWebGL2 = !!gl;
        
        if (!gl) {
            gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        }
        
        if (!gl) {
            this._capabilities = {
                supported: false,
                tier: 'none',
                reason: 'WebGL not available',
                recommendation: 'cytoscape',
                details: {
                    isWebGL2: false,
                    gpuVendor: 'Unknown',
                    gpuRenderer: 'Unknown',
                    isSoftwareRenderer: false,
                    isMobile: this.isMobile(),
                    extensions: {
                        oesTextureFloat: false,
                        extFloatBlend: false,
                        colorBufferFloat: false
                    },
                    limits: {
                        maxTextureSize: 0,
                        maxPointSize: 0,
                        maxVertexTextureUnits: 0
                    },
                    estimatedMaxNodes: 0
                }
            };
            return this._capabilities;
        }
        
        // Get GPU info for diagnostics
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        const gpuVendor = debugInfo 
            ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) 
            : 'Unknown';
        const gpuRenderer = debugInfo 
            ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) 
            : 'Unknown';
        
        // Check cosmos.gl required extensions
        // NOTE: In WebGL2, OES_texture_float is a CORE FEATURE (not an extension)
        // So we check for the extension only in WebGL1
        let hasFloatTextures;
        let hasFloatBlend;
        let hasColorBufferFloat;
        
        if (isWebGL2) {
            // WebGL2: Float textures are built-in
            hasFloatTextures = true;
            // EXT_color_buffer_float provides renderable float textures in WebGL2
            hasColorBufferFloat = !!gl.getExtension('EXT_color_buffer_float');
            // EXT_float_blend is still needed for blending with float buffers
            hasFloatBlend = !!gl.getExtension('EXT_float_blend');
        } else {
            // WebGL1: Need extensions
            hasFloatTextures = !!gl.getExtension('OES_texture_float');
            hasColorBufferFloat = !!gl.getExtension('WEBGL_color_buffer_float');
            hasFloatBlend = !!gl.getExtension('EXT_float_blend');
        }
        
        // Check for software renderer (SwiftShader, llvmpipe, etc.)
        const isSoftwareRenderer = this.isSoftwareRenderer(gpuRenderer);
        
        // Get hardware limits
        const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
        const maxPointSize = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
        const maxVertexTextureUnits = gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS);
        
        // Mobile detection
        const isMobile = this.isMobile();
        
        // Determine capability tier
        let tier, recommendation, reason;
        
        if (!hasFloatTextures) {
            tier = 'incompatible';
            reason = 'Float textures not available';
            recommendation = 'cytoscape';
        } else if (isSoftwareRenderer) {
            tier = 'software';
            reason = `Software renderer detected: ${gpuRenderer}`;
            recommendation = 'cytoscape'; // Software WebGL is slower than Canvas
        } else if (!hasFloatBlend) {
            tier = 'limited';
            reason = 'EXT_float_blend not available - force simulation may be limited';
            recommendation = 'cosmos_limited'; // Can work without full force simulation
        } else if (maxTextureSize < 4096) {
            tier = 'constrained';
            reason = `Small max texture size: ${maxTextureSize}`;
            recommendation = maxTextureSize >= 2048 ? 'cosmos' : 'cytoscape';
        } else if (isMobile) {
            tier = 'mobile';
            reason = 'Mobile device - reduced capacity';
            recommendation = 'cosmos'; // cosmos works on mobile but with lower limits
        } else {
            tier = 'full';
            reason = 'All capabilities available';
            recommendation = 'cosmos';
        }
        
        // Estimate max nodes based on tier
        const estimatedMaxNodes = this.estimateMaxNodes(tier, maxTextureSize, isMobile);
        
        this._capabilities = {
            supported: tier !== 'incompatible' && tier !== 'none',
            tier,
            reason,
            recommendation,
            details: {
                isWebGL2,
                gpuVendor,
                gpuRenderer,
                isSoftwareRenderer,
                isMobile,
                extensions: {
                    floatTextures: hasFloatTextures,
                    floatBlend: hasFloatBlend,
                    colorBufferFloat: hasColorBufferFloat
                },
                limits: {
                    maxTextureSize,
                    maxPointSize,
                    maxVertexTextureUnits
                },
                estimatedMaxNodes
            }
        };
        
        // Cleanup WebGL context
        const loseContext = gl.getExtension('WEBGL_lose_context');
        if (loseContext) {
            loseContext.loseContext();
        }
        
        console.log('[WebGLDetector] Capabilities:', this._capabilities);
        return this._capabilities;
    },
    
    /**
     * Check if running on mobile device
     */
    isMobile() {
        return /iPhone|iPad|Android|webOS|Mobile/i.test(navigator.userAgent);
    },
    
    /**
     * Check if using software renderer
     */
    isSoftwareRenderer(renderer) {
        const softwareIndicators = [
            'swiftshader',
            'llvmpipe',
            'softpipe',
            'software',
            'microsoft basic render',
            'vmware',
            'virtualbox',
            'parallels'
        ];
        const lowerRenderer = renderer.toLowerCase();
        return softwareIndicators.some(ind => lowerRenderer.includes(ind));
    },
    
    /**
     * Estimate maximum nodes based on capabilities
     */
    estimateMaxNodes(tier, maxTextureSize, isMobile) {
        // cosmos.gl stores positions in textures, so texture size limits node count
        // Each texture pixel = 1 node (RGBA float = x, y, vx, vy)
        const maxFromTexture = maxTextureSize * maxTextureSize;
        
        const tierLimits = {
            'full': Math.min(maxFromTexture, 1000000),
            'mobile': Math.min(maxFromTexture, 50000),
            'constrained': Math.min(maxFromTexture, 100000),
            'limited': Math.min(maxFromTexture, 200000),
            'software': 5000,
            'incompatible': 0,
            'none': 0
        };
        
        return tierLimits[tier] || 0;
    },
    
    /**
     * Get a summary suitable for UI display
     */
    getSummary() {
        const caps = this.detect();
        return {
            cosmosAvailable: caps.supported,
            tier: caps.tier,
            reason: caps.reason,
            recommendation: caps.recommendation,
            maxNodes: caps.details.estimatedMaxNodes,
            gpuInfo: caps.details.gpuRenderer,
            isMobile: caps.details.isMobile,
            isWebGL2: caps.details.isWebGL2
        };
    },
    
    /**
     * Check if cosmos.gl is available
     */
    isCosmosAvailable() {
        return this.detect().supported;
    },
    
    /**
     * Get recommended renderer for a given graph size
     * @param {number} nodeCount - Number of nodes
     * @param {string} userPreference - User preference ('auto', 'cosmos', 'cytoscape')
     * @returns {string} 'cosmos' or 'cytoscape'
     */
    getRecommendedRenderer(nodeCount, userPreference = 'auto') {
        const caps = this.detect();
        const settings = RendererSettings.getThresholds();
        
        // User explicitly chose cytoscape
        if (userPreference === 'cytoscape') {
            return 'cytoscape';
        }
        
        // User explicitly chose cosmos
        if (userPreference === 'cosmos') {
            if (!caps.supported) {
                console.warn('[WebGLDetector] cosmos.gl requested but not available:', caps.reason);
            }
            return caps.supported ? 'cosmos' : 'cytoscape';
        }
        
        // Auto selection
        
        // Small graphs: Cytoscape is fine and more feature-rich
        if (nodeCount < settings.cosmosMinNodes) {
            return 'cytoscape';
        }
        
        // Medium to large graphs: Use cosmos if available
        if (nodeCount >= settings.cosmosMinNodes) {
            if (!caps.supported) {
                console.warn(
                    `[WebGLDetector] Graph size (${nodeCount} nodes) would benefit from cosmos.gl ` +
                    `but it's unavailable: ${caps.reason}`
                );
                return 'cytoscape';
            }
            
            // Warn if graph exceeds estimated capacity
            if (nodeCount > caps.details.estimatedMaxNodes) {
                console.warn(
                    `[WebGLDetector] Graph size (${nodeCount}) may exceed estimated max ` +
                    `(${caps.details.estimatedMaxNodes}) for this device`
                );
            }
            
            return 'cosmos';
        }
        
        return caps.supported ? 'cosmos' : 'cytoscape';
    },
    
    /**
     * Clear cached capabilities (useful for testing)
     */
    reset() {
        this._capabilities = null;
    }
};

// Make available globally
window.WebGLDetector = WebGLDetector;