"""
CSS and JavaScript for Extract SID page
"""

EXTRACT_SID_CSS = """
/* Extract SID Page Styles */
.extract-sid-page {
    padding: 40px 60px !important;
    background-color: #ffffff !important;
    min-height: 100vh !important;
}

/* Title (centered) */
.extract-sid-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #171717 !important;
    margin: 0 0 20px 0 !important;
    text-align: center !important;
}

/* Header row with loading status (left) and button (right) */
.extract-sid-header-row {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 30px !important;
    gap: 20px !important;
}


.extract-sid-header-row > * {
    flex-shrink: 0 !important;
}

/* Main container */
.extract-sid-main-container {
    display: flex !important;
    flex-direction: column !important;
    gap: 20px !important;
    margin-top: 10px !important;
}

/* Card Row - Grid Layout */
.extract-sid-card-row {
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 15px !important;
    align-items: start !important;
    margin-bottom: 20px !important;
}

/* The Card itself (Input + Output) */
.extract-sid-card {
    background-color: #f9f9f9 !important;
    padding: 20px !important;
    border-radius: 24px !important;
    border: 1px solid #e0e0e0 !important;
    display: grid !important;
    grid-template-columns: 1fr 1fr !important; /* Equal width columns */
    gap: 20px !important;
}


.extract-sid-card .form {
    border: none !important;
    box-shadow: none !important;
}

/* Input/Output Labels */
.extract-sid-label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #666666 !important;
    margin-bottom: 8px !important;
    padding-left: 2px !important;
    display: block !important;
}

/* Gradio Input Box Styling Override */
.extract-sid-input-box {
    height: 100% !important;
    background-color: #F9F9F9 !important;
    border: none !important;
    padding: 0 !important;
}

.extract-sid-input-box label {
    border: none !important;
}

.extract-sid-input-box label span {
    color: #666666 !important;
}

.extract-sid-input-box textarea {
    border-radius: 16px !important;
    border: 1px solid #d0d0d0 !important;
    padding: 12px !important;
    font-size: 14px !important;
    resize: vertical !important;
    min-height: 120px !important;
    background-color: #ffffff !important;
    font-family: inherit !important;
}

.extract-sid-input-box textarea:focus {
    border-color: #666666 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(102, 102, 102, 0.1) !important;
}

/* Custom HTML Output Box Styling */
.custom-output-wrapper {
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}

.custom-output-box {
    position: relative !important;
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 16px !important;
    padding: 12px !important;
    min-height: 120px !important;
    height: 100% !important;
    font-size: 13px !important;
    color: #333333 !important;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace !important;
    line-height: 1.6 !important;
    word-break: break-all !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
}

.custom-output-content {
    flex-grow: 1 !important;
    white-space: pre-wrap !important;
}

.custom-output-content.empty {
    color: #999999 !important;
    font-style: italic !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 100% !important;
}

/* Copy Button Styling (From User Request) */
.custom-output-box .copy-btn-container {
    position: absolute !important;
    top: 8px !important;
    right: 8px !important;
    display: flex !important;
    justify-content: flex-end !important;
    align-items: flex-end !important;
    margin: 0 !important;
    z-index: 10 !important;
}

.custom-output-box .copy-btn-container .copy-btn {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px !important;
    min-width: auto !important;
    width: auto !important;
    cursor: pointer !important;
    opacity: 0.6 !important;
    transition: all 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 8px !important;
}

.custom-output-box .copy-btn-container .copy-btn:hover {
    opacity: 1 !important;
    background: rgba(0,0,0,0.05) !important;
}

.custom-output-box .copy-btn-container .copy-btn svg {
    width: 20px !important;
    height: 20px !important;
    display: block !important;
}

.custom-output-box .copy-btn-container .copy-btn svg path {
    fill: #666666 !important;
}

/* Convert Button */
.extract-sid-convert-btn {
    border-radius: 20px !important;
    padding: 10px 30px !important;
    background-color: #52c41a !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.extract-sid-convert-btn:hover:not([disabled]) {
    background-color: #73d13d !important;
}

/* Loading Status Styles (Retained) */
.extract-sid-loading-status {
    flex-shrink: 0 !important;
    margin-right: 20px !important;
}

.loading-status {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 16px !important;
    border-radius: 20px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}

.loading-status.loading {
    background-color: #e6f7ff !important;
    color: #1890ff !important;
    border: 1px solid #91d5ff !important;
    display: inline-block !important;
    padding: 10px 16px !important;
    min-width: 200px !important;
}

.loading-status-content {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    margin-bottom: 6px !important;
}

.loading-progress-bar {
    width: 100% !important;
    height: 6px !important;
    background-color: #d1e9ff !important;
    border-radius: 3px !important;
    overflow: hidden !important;
}

.loading-progress-fill {
    height: 100% !important;
    background: linear-gradient(90deg, #1890ff, #69c0ff) !important;
    border-radius: 3px !important;
    transition: width 0.5s ease !important;
}

.loading-status.done {
    background-color: #f6ffed !important;
    color: #52c41a !important;
    border: 1px solid #b7eb8f !important;
}

.loading-status.error {
    background-color: #fff2f0 !important;
    color: #ff4d4f !important;
    border: 1px solid #ffccc7 !important;
}

.loading-status.idle {
    background-color: #f5f5f5 !important;
    color: #999999 !important;
    border: 1px solid #d9d9d9 !important;
}

.loading-status .status-icon {
    font-size: 14px !important;
    font-weight: bold !important;
}

.loading-spinner {
    width: 14px !important;
    height: 14px !important;
    border: 2px solid #91d5ff !important;
    border-top-color: #1890ff !important;
    border-radius: 50% !important;
    animation: spin 0.8s linear infinite !important;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

/* Expand/collapse text links for output */
.sid-expand-link, .sid-collapse-link {
    color: #1976d2 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    display: block !important;
    text-align: left !important;
    user-select: none !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    margin-top: 8px !important;
}

.sid-expand-link:hover, .sid-collapse-link:hover {
    text-decoration: underline !important;
}

/* Pre tag styling for output content */
.custom-output-content pre {
    margin: 0 !important;
    white-space: pre-wrap !important;
    word-break: break-all !important;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
"""


EXTRACT_SID_JS = """
function() {
    // Function to toggle text display between preview and full
    window.toggleSidText = function(contentId) {
        const previewDiv = document.getElementById(contentId + '_preview');
        const fullDiv = document.getElementById(contentId + '_full');

        if (!previewDiv || !fullDiv) {
            console.error('Text elements not found for:', contentId);
            return;
        }

        // Toggle visibility
        if (previewDiv.style.display === 'none') {
            // Currently showing full, switch to preview (collapse)
            previewDiv.style.display = 'block';
            fullDiv.style.display = 'none';
        } else {
            // Currently showing preview, switch to full (expand)
            previewDiv.style.display = 'none';
            fullDiv.style.display = 'block';
        }
    };

    // Global copy function
    window.copySidToClipboard = function(btn) {
        const content = btn.dataset.content;
        if (!content) {
            showNotification('没有可复制的内容', 'warning');
            return;
        }

        navigator.clipboard.writeText(content).then(function() {
            showNotification('复制成功！', 'success');
        }).catch(function(err) {
            console.error('Failed to copy:', err);
            showNotification('复制失败', 'error');
        });
    };

    // Show notification helper
    window.showNotification = function(message, type) {
        const existing = document.querySelector('.copy-notification-dynamic');
        if (existing) existing.remove();

        const notification = document.createElement('div');
        notification.className = 'copy-notification-dynamic';
        notification.textContent = message;

        const bgColors = {
            success: '#52c41a',
            warning: '#faad14',
            error: '#ff4d4f'
        };

        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            backgroundColor: bgColors[type] || bgColors.success,
            color: 'white',
            padding: '12px 24px',
            borderRadius: '8px',
            fontSize: '14px',
            fontWeight: '500',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            zIndex: '10000',
            animation: 'slideInRight 0.3s ease'
        });

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    };

    // Add keyframes if not present
    if (!document.getElementById('notification-keyframes')) {
        const style = document.createElement('style');
        style.id = 'notification-keyframes';
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}
"""
