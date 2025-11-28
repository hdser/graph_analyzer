/**
 * Toast Module
 * Toast notification system
 */

const Toast = {
    /**
     * Show a toast notification
     */
    show(message, type = 'info', duration = 3000) {
        if (!DOMCache.toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerText = message;
        
        DOMCache.toastContainer.appendChild(toast);
        
        // Trigger reflow for animation
        void toast.offsetWidth;
        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (DOMCache.toastContainer.contains(toast)) {
                    DOMCache.toastContainer.removeChild(toast);
                }
            }, 300);
        }, duration);
    },

    /**
     * Show a loading toast with spinner
     * Returns a dismiss function
     */
    showLoading(message) {
        if (!DOMCache.toastContainer) return null;
        
        const toast = document.createElement('div');
        toast.className = 'toast loading';
        toast.innerHTML = `<div class="spinner"></div><span>${message}</span>`;
        
        DOMCache.toastContainer.appendChild(toast);
        void toast.offsetWidth;
        toast.classList.add('show');
        
        return () => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (DOMCache.toastContainer.contains(toast)) {
                    DOMCache.toastContainer.removeChild(toast);
                }
            }, 300);
        };
    },

    /**
     * Convenience methods
     */
    success(msg, duration) { this.show(msg, 'success', duration); },
    error(msg, duration) { this.show(msg, 'error', duration); },
    info(msg, duration) { this.show(msg, 'info', duration); },
    warning(msg, duration) { this.show(msg, 'warning', duration); }
};

/**
 * Legacy function for compatibility
 */
function showToast(message, type) {
    Toast.show(message, type);
}

/**
 * Update status - shows toast and logs errors
 */
function updateStatus(msg, type) {
    Toast.show(msg, type);
    if (type === 'error') {
        console.error(msg);
    }
}