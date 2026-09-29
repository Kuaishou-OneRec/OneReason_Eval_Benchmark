"""
Combined JavaScript for the main application
Includes navigation and checkbox functionality
"""

COMBINED_JS = r"""
function() {
    console.log('=== MAIN APP JS LOADED ===');

    // Navigation handling
    function setupNavigation() {
        console.log('[Navigation] Setting up navigation...');

        const navButtons = document.querySelectorAll('.nav-button');

        navButtons.forEach(function(button) {
            button.addEventListener('click', function() {
                // Remove active class from all buttons
                navButtons.forEach(function(btn) {
                    btn.classList.remove('active');
                });

                // Add active class to clicked button
                this.classList.add('active');

                console.log('[Navigation] Switched to:', this.textContent.trim());
            });
        });
    }

    // Initialize navigation
    setTimeout(setupNavigation, 500);
    setTimeout(setupNavigation, 1000);

    // ===== CHECKBOX FUNCTIONALITY FOR SUBMIT PAGE =====
    console.log('=== SUBMIT PAGE JS LOADED ===');

    // Add Del column checkbox functionality
    function setupDelColumnCheckboxes() {
        console.log('[setupDelColumnCheckboxes] Starting setup...');

        function addCheckboxesToCells() {
            const deletableDfs = document.querySelectorAll('.deletable-df');
            console.log('[addCheckboxesToCells] Found', deletableDfs.length, 'deletable dataframes');

            deletableDfs.forEach(function(df) {
                const firstColCells = df.querySelectorAll('tbody td:first-child');
                console.log('[addCheckboxesToCells] Processing', firstColCells.length, 'cells');

                firstColCells.forEach(function(cell, index) {
                    // Skip if checkbox already added
                    if (cell.querySelector('.del-checkbox')) {
                        return;
                    }

                    // Clear cell content
                    cell.innerHTML = '';

                    // Create checkbox container
                    const container = document.createElement('div');
                    container.style.cssText = `
                        width: 100%;
                        height: 100%;
                        min-height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    `;

                    // Create checkbox
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.className = 'del-checkbox';
                    checkbox.style.cssText = `
                        width: 18px;
                        height: 18px;
                        cursor: pointer;
                        margin: 0;
                        border-radius: 4px;
                        accent-color: #1976d2;
                    `;

                    // Checkbox change event
                    checkbox.addEventListener('change', function(e) {
                        console.log('[Checkbox] Row', index, ':', this.checked);
                    });

                    container.appendChild(checkbox);
                    cell.appendChild(container);
                });
            });
        }

        // Initial add checkboxes
        setTimeout(addCheckboxesToCells, 500);
        setTimeout(addCheckboxesToCells, 1000);
        setTimeout(addCheckboxesToCells, 2000);

        // Listen for data updates
        const observer = new MutationObserver(function(mutations) {
            let needsUpdate = false;
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    needsUpdate = true;
                }
            });
            if (needsUpdate) {
                setTimeout(addCheckboxesToCells, 100);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        console.log('[setupDelColumnCheckboxes] Setup complete');

        // Add global styles
        if (!document.getElementById('del-column-style')) {
            const style = document.createElement('style');
            style.id = 'del-column-style';
            style.textContent = `
                .deletable-df td:first-child {
                    user-select: none !important;
                }
                .deletable-df td:first-child:hover {
                    background-color: #f5f5f5 !important;
                }

                /* Beautify checkbox */
                .del-checkbox {
                    width: 18px !important;
                    height: 18px !important;
                    cursor: pointer !important;
                    border-radius: 4px !important;
                    border: 2px solid #999 !important;
                    background-color: white !important;
                    accent-color: #1976d2 !important;
                    appearance: auto !important;
                    -webkit-appearance: checkbox !important;
                    -moz-appearance: checkbox !important;
                }

                .del-checkbox:hover {
                    border-color: #1976d2 !important;
                }

                .del-checkbox:checked {
                    background-color: #1976d2 !important;
                    border-color: #1976d2 !important;
                }

                /* Hide edit button */
                .deletable-df td:first-child span.edit {
                    display: none !important;
                }
            `;
            document.head.appendChild(style);
        }
    }

    function fixColumnWidths() {
        const deletableDfs = document.querySelectorAll('.deletable-df');
        console.log('[fixColumnWidths] Found', deletableDfs.length, 'deletable dataframes');

        deletableDfs.forEach(function(df, index) {
            try {
                const tableWraps = df.querySelectorAll('.table-wrap');
                console.log('[fixColumnWidths] DF', index, 'has', tableWraps.length, 'table-wraps');

                tableWraps.forEach(function(tableWrap, twIndex) {
                    const totalWidth = tableWrap.offsetWidth || 880;

                    const col0Width = 40;
                    const col1Width = Math.floor(totalWidth * 0.45);
                    const col2Width = Math.floor(totalWidth * 0.20);
                    const col3Width = Math.floor(totalWidth * 0.30);

                    console.log('[fixColumnWidths] Setting widths:', col0Width, col1Width, col2Width, col3Width);

                    tableWrap.style.setProperty('--cell-width-0', col0Width + 'px', 'important');
                    tableWrap.style.setProperty('--cell-width-1', col1Width + 'px', 'important');
                    tableWrap.style.setProperty('--cell-width-2', col2Width + 'px', 'important');
                    tableWrap.style.setProperty('--cell-width-3', col3Width + 'px', 'important');
                });

                const allFirstHeaders = df.querySelectorAll('th:first-child');
                allFirstHeaders.forEach(function(th) {
                    th.style.setProperty('width', '40px', 'important');
                    th.style.setProperty('min-width', '40px', 'important');
                    th.style.setProperty('max-width', '40px', 'important');
                });

                const allFirstCells = df.querySelectorAll('td:first-child');
                allFirstCells.forEach(function(td) {
                    td.style.setProperty('width', '40px', 'important');
                    td.style.setProperty('min-width', '40px', 'important');
                    td.style.setProperty('max-width', '40px', 'important');
                });
            } catch (e) {
                console.error('Error fixing column widths:', e);
            }
        });
    }

    // Move getSelectedTaskIds logic to global function for Gradio to call
    window.getSelectedTaskIds = function() {
        const activeTab = document.querySelector('.tabs button[aria-selected="true"]');
        if (!activeTab) {
            console.log('[Delete] No active tab');
            return JSON.stringify({ taskJsons: [], queueType: '' });
        }

        const tabText = activeTab.textContent.trim();
        console.log('[Delete] Active tab:', tabText);

        let dfId, queueType;

        if (tabText.includes('等待中')) {
            dfId = 'pending-df';
            queueType = 'pending';
        } else if (tabText.includes('已完成')) {
            dfId = 'done-df';
            queueType = 'done';
        } else if (tabText.includes('已失败')) {
            dfId = 'failed-df';
            queueType = 'failed';
        } else {
            console.log('[Delete] Tab does not support deletion');
            return JSON.stringify({ taskJsons: [], queueType: '' });
        }

        const df = document.getElementById(dfId);
        if (!df) {
            console.log('[Delete] DataFrame not found:', dfId);
            return JSON.stringify({ taskJsons: [], queueType: '' });
        }

        const selectedTaskJsons = [];
        const rows = df.querySelectorAll('tbody tr');

        console.log('[Delete] Total rows:', rows.length);

        rows.forEach(function(row, index) {
            const checkbox = row.querySelector('td:first-child .del-checkbox');
            const cells = row.querySelectorAll('td');

            // Task JSON is at index 4 (columns: "", "Model Path", "Output Dir", "Log Path", "Task JSON", ...)
            const taskJsonCell = cells[4];

            if (checkbox && checkbox.checked && taskJsonCell) {
                let taskJson = taskJsonCell.textContent.trim();

                const span = taskJsonCell.querySelector('span');
                if (span) {
                    taskJson = span.textContent.trim() || taskJson;
                }

                if (!taskJson) {
                    taskJson = taskJsonCell.innerText.trim();
                }

                console.log('[Delete] Row', index, 'Task JSON:', taskJson);

                if (taskJson && taskJson !== 'N/A') {
                    selectedTaskJsons.push(taskJson);
                }
            }
        });

        console.log('[Delete] Selected task JSONs:', selectedTaskJsons);
        console.log('[Delete] Queue type:', queueType);

        const result = JSON.stringify({ taskJsons: selectedTaskJsons, queueType: queueType });
        console.log('[Delete] Returning:', result);
        return result;
    };

    function processUpdates() {
        fixColumnWidths();
    }

    setupDelColumnCheckboxes();

    setTimeout(processUpdates, 100);
    setTimeout(processUpdates, 500);
    setTimeout(processUpdates, 1000);

    try {
        const observer = new MutationObserver(function(mutations) {
            let needsUpdate = false;
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' &&
                    (mutation.attributeName === 'style' || mutation.attributeName === 'class')) {
                    needsUpdate = true;
                } else if (mutation.type === 'childList') {
                    needsUpdate = true;
                }
            });
            if (needsUpdate) {
                setTimeout(processUpdates, 100);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['style', 'class']
        });
    } catch (e) {
        console.error('Error setting up MutationObserver:', e);
    }

    try {
        const tabButtons = document.querySelectorAll('button[role="tab"]');
        tabButtons.forEach(function(btn) {
            btn.addEventListener('click', function() {
                setTimeout(processUpdates, 100);
                setTimeout(processUpdates, 300);
                setTimeout(processUpdates, 600);
            });
        });
    } catch (e) {
        console.error('Error setting up tab listeners:', e);
    }

    setInterval(fixColumnWidths, 2000);
}
"""
