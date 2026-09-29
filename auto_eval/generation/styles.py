"""
CSS styles for the generation page
"""

GENERATION_CSS = """
/* Generation Page Container */
.generation-container {
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

.generation-container .form {
    background-color: white !important;
    border-radius: 16px !important;
}

.generation-item {
    padding: 0 !important;
}

/* Model Path Input */
.model-path-input {
    border-radius: 16px !important;
}

.model-path-input textarea {
    font-family: monospace;
    font-size: 14px;
}

/* Think Switches Row */
.think-switches-row {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 20px;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 8px;
    border: 1px solid #eee;
}

.think-switch-container {
    display: flex;
    align-items: center;
}

.think-switch-item {
    display: flex;
    align-items: center;
    gap: 5px;
}

.think-switch-label {
    font-weight: 600;
    font-size: 14px;
    color: #333;
}

.tooltip-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background-color: #ddd;
    color: #666;
    font-size: 12px;
    cursor: help;
    position: relative;
}

.tooltip-icon:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background-color: #333;
    color: white;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    z-index: 1000;
    margin-bottom: 5px;
}

/* Parameters Section */
.params-container {
    margin-bottom: 20px;
    padding: 15px;
    border: 1px solid #eee;
    border-radius: 16px;
    background-color: white;
}

.params-container .form {
    padding: 15px;
    border: 1px solid #eee;
    border-radius: 8px;
    background-color: white;
    border-radius: 10px;
}

/* Prompt Input */

.prompt-input {
    border-radius: 16px !important;
}

.prompt-input textarea {
    font-family: monospace;
    font-size: 14px;
    min-height: 150px;
}

/* Template Shortcuts */
.template-shortcuts {
    display: flex;
    gap: 15px;
    margin-top: 5px;
    margin-bottom: 20px;
    font-size: 13px;
}

.template-link {
    color: #1976d2;
    cursor: pointer;
    text-decoration: none;
    border-bottom: 1px dashed #1976d2;
}

.template-link:hover {
    color: #1565c0;
    border-bottom: 1px solid #1565c0;
}

/* Start Button */
.start-btn {
    background-color: #28a745 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    padding: 12px !important;
    transition: background-color 0.3s !important;
}

.start-btn:hover {
    background-color: #218838 !important;
}

/* Result Container */
.result-container {
    margin-top: 30px;
    border-top: 1px solid #eee;
    padding-top: 20px;
}

/* Reuse Analyze Page Styles for Results */
.model-card {
    transition: all 0.3s ease;
}

.model-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Status Message */
.status-message {
    padding: 20px;
    background-color: #e3f2fd;
    border: 1px solid #90caf9;
    border-radius: 8px;
    color: #1565c0;
    font-size: 14px;
    line-height: 1.6;
}

.status-message .task-info {
    margin-bottom: 10px;
}

.status-message .task-label {
    font-weight: bold;
}

.status-message code {
    background-color: #bbdefb;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 13px;
}

.status-message .task-status {
    color: #0d47a1;
}

.status-message .task-status::before {
    content: "⏳ ";
}

/* Refresh Button */
.refresh-btn {
    background-color: #1976d2 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    margin-top: 10px !important;
    transition: background-color 0.3s !important;
}

.refresh-btn:hover {
    background-color: #1565c0 !important;
}

/* Error Message */
.error-message {
    padding: 20px;
    background-color: #ffebee;
    border: 1px solid #ef9a9a;
    border-radius: 8px;
    color: #c62828;
    font-size: 14px;
    line-height: 1.6;
}

.error-message::before {
    content: "❌ ";
}

/* Error Container (for detailed errors) */
.error-container {
    padding: 20px;
    background-color: #fff3e0;
    border: 1px solid #ffcc80;
    border-radius: 8px;
    margin-top: 10px;
}

.error-container h3 {
    margin-top: 0;
    color: #e65100;
}

.error-container h4 {
    color: #f57c00;
    margin-bottom: 10px;
}

.error-container pre {
    background-color: #fafafa;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 12px;
    max-height: 300px;
    overflow-y: auto;
}

/* Copy Icon Button (for results) */
.copy-icon-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 5px;
    color: #666;
    opacity: 0.6;
    transition: opacity 0.2s;
}

.copy-icon-btn:hover {
    opacity: 1;
    color: #1976d2;
}

/* Text Container */
.text-container {
    position: relative;
}

.text-container pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: monospace;
    font-size: 13px;
    line-height: 1.5;
}

/* ==================== */
/* Generation Comparison Styles */
/* ==================== */

/* Header with model filter */
.gen-comparison-header {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid #eee;
}

.gen-model-filter {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.gen-model-pill {
    display: flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    user-select: none;
}

.gen-model-pill:hover {
    opacity: 0.8;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.gen-model-pill.hidden-model {
    opacity: 0.4;
    background-color: #eee !important;
    text-decoration: line-through;
    border-left-color: #ccc !important;
}

/* Sample container (collapsible) */
.gen-sample-details {
    margin-bottom: 20px;
    border: 1px solid #ddd;
    border-radius: 12px;
    overflow: hidden;
    background-color: #fff;
}

.gen-sample-summary {
    padding: 12px 20px;
    background-color: #f5f5f5;
    cursor: pointer;
    font-weight: bold;
    font-size: 14px;
    color: #333;
    list-style: none;
    display: flex;
    align-items: center;
}

.gen-sample-summary::before {
    content: '▶';
    display: inline-block;
    margin-right: 10px;
    transition: transform 0.2s;
    font-size: 10px;
    color: #666;
}

.gen-sample-details[open] .gen-sample-summary::before {
    transform: rotate(90deg);
}

.gen-sample-summary::-webkit-details-marker {
    display: none;
}

.gen-sample-content {
    padding: 15px;
}

/* Model cards container */
.gen-model-cards {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}

/* Individual model card */
.gen-model-card {
    flex: 1;
    min-width: 300px;
    max-width: 600px;
    background-color: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    transition: box-shadow 0.2s;
}

.gen-model-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.gen-model-title {
    margin: 0 0 12px 0;
    font-size: 13px;
    font-weight: 600;
    word-break: break-word;
}

/* Section title (Prompt, Generation) */
.gen-section-title {
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
    color: #555;
    margin-bottom: 8px;
    padding: 4px 0;
}

/* Text container with copy button */
.gen-text-container {
    position: relative;
    background-color: #fff;
    border: 1px solid #eee;
    border-radius: 6px;
    padding: 10px;
    margin-top: 5px;
}

.gen-text-container pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'SF Mono', Monaco, 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #333;
}

/* Copy button */
.gen-copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 4px;
    opacity: 0.4;
    transition: opacity 0.2s;
    color: #666;
    z-index: 10;
}

.gen-copy-btn:hover {
    opacity: 1;
    color: #1976d2;
}

/* Expand/collapse links */
.gen-expand-link, .gen-collapse-link {
    color: #1976d2;
    font-size: 12px;
    cursor: pointer;
    display: inline-block;
    margin-top: 5px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

.gen-expand-link:hover, .gen-collapse-link:hover {
    text-decoration: underline;
}

/* Toast notification for copy feedback */
.gen-toast {
    position: fixed;
    top: 20px;
    right: 20px;
    background-color: #333;
    color: white;
    padding: 10px 20px;
    border-radius: 6px;
    z-index: 9999;
    opacity: 0;
    transform: translateY(-10px);
    transition: all 0.3s;
    font-size: 13px;
}

.gen-toast.show {
    opacity: 1;
    transform: translateY(0);
}
"""

GENERATION_JS = """
function() {
    console.log('=== GENERATION PAGE JS LOADED ===');

    // Toast notification
    window.showGenToast = function(message, duration = 2000) {
        let toast = document.getElementById('gen-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'gen-toast';
            toast.className = 'gen-toast';
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), duration);
    };

    // Function to append text to prompt
    window.appendTemplate = function(templateType) {
        const promptArea = document.querySelector('#prompt-input textarea');
        if (!promptArea) {
            console.error('Prompt textarea not found');
            return;
        }

        let textToAppend = '';
        if (templateType === 'think') {
            textToAppend = "<|im_start|>system\\\\nYou are a helpful assistant.<|im_end|>\\\\n<|im_start|>user\\\\n你好/think<|im_end|>\\\\n<|im_start|>assistant\\\\n";
        } else if (templateType === 'no_think') {
            textToAppend = "<|im_start|>system\\\\nYou are a helpful assistant.<|im_end|>\\\\n<|im_start|>user\\\\n你好/no_think<|im_end|>\\\\n<|im_start|>assistant\\\\n<think>\\\\n\\\\n</think>";
        } else if (templateType === 'sid') {
            textToAppend = "<|sid_begin|><s_a_0><s_b_0><s_c_0><|sid_end|>";
        }

        // Append text
        promptArea.value += textToAppend;

        // Trigger input event for Gradio to update state
        promptArea.dispatchEvent(new Event('input', { bubbles: true }));
    };

    // Toggle model visibility (for comparison view)
    window.toggleGenModel = function(modelId) {
        console.log('Toggling model:', modelId);

        // Toggle pill style
        const pill = document.getElementById('pill_' + modelId);
        if (pill) {
            pill.classList.toggle('hidden-model');
        }

        // Toggle all model cards with this model
        const cards = document.querySelectorAll(`.gen-model-card[data-model="${modelId}"]`);
        cards.forEach(card => {
            if (card.style.display === 'none') {
                card.style.display = '';
            } else {
                card.style.display = 'none';
            }
        });
    };

    // Toggle text expand/collapse
    window.toggleGenText = function(contentId) {
        const previewDiv = document.getElementById(contentId + '_preview');
        const fullDiv = document.getElementById(contentId + '_full');

        if (!previewDiv || !fullDiv) {
            console.error('Text elements not found:', contentId);
            return;
        }

        if (previewDiv.style.display === 'none') {
            previewDiv.style.display = 'block';
            fullDiv.style.display = 'none';
        } else {
            previewDiv.style.display = 'none';
            fullDiv.style.display = 'block';
        }
    };

    // Copy text content
    window.copyGenText = function(contentId) {
        const fullDiv = document.getElementById(contentId + '_full');
        const previewDiv = document.getElementById(contentId + '_preview');

        let textElement;
        if (fullDiv && fullDiv.style.display !== 'none') {
            textElement = fullDiv.querySelector('pre');
        } else if (previewDiv) {
            textElement = previewDiv.querySelector('pre');
        }

        if (!textElement) {
            console.error('Text element not found');
            window.showGenToast('复制失败');
            return;
        }

        const text = textElement.textContent;

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(() => {
                window.showGenToast('复制成功');
            }).catch(err => {
                console.error('Copy failed:', err);
                fallbackCopy(text);
            });
        } else {
            fallbackCopy(text);
        }
    };

    // Fallback copy for older browsers
    function fallbackCopy(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            window.showGenToast('复制成功');
        } catch(err) {
            window.showGenToast('复制失败');
        }
        document.body.removeChild(textarea);
    }

    // Legacy copy function (for old code)
    window.copyText = function(elementId) {
        window.copyGenText(elementId);
    };
}
"""
