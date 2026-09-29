"""
Business logic for PID to SID conversion
"""
import os
import re
import threading
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

# Set to False to skip loading pickle file (for testing)
ENABLE_PICKLE_LOADING = False

# Initialize pid_to_code mapping and loading state
pid_to_code = {}
_loading_state = {
    "status": "idle",  # idle, loading, done, error
    "message": "",
    "count": 0,
    "start_time": 0,
    "elapsed": 0
}
_loading_lock = threading.Lock()

# Expected loading time in seconds
EXPECTED_LOADING_TIME = 160

# Data file path
PICKLE_PATH = Path(os.environ.get("PID_TO_SID_PICKLE", "data/pid_to_code.pkl"))


def get_loading_status() -> dict:
    """Get current loading status.

    Also auto-corrects status if data is already loaded but status is stale
    (e.g., after hot reload or page refresh).
    """
    with _loading_lock:
        # Auto-correct: if data exists but status is not "done", fix it
        if pid_to_code and _loading_state["status"] in ("idle", "loading"):
            _loading_state["status"] = "done"
            _loading_state["count"] = len(pid_to_code)
            _loading_state["message"] = f"Loaded {len(pid_to_code)} mappings"
        return _loading_state.copy()


def is_data_ready() -> bool:
    """Check if data is loaded and ready for queries."""
    with _loading_lock:
        return _loading_state["status"] == "done"


def _update_elapsed_time():
    """Update elapsed time periodically while loading."""
    import time
    while True:
        with _loading_lock:
            if _loading_state["status"] != "loading":
                break
            _loading_state["elapsed"] = time.time() - _loading_state["start_time"]
        time.sleep(1)


def _load_data_background():
    """Background task to load pickle data."""
    import time
    global pid_to_code

    with _loading_lock:
        _loading_state["status"] = "loading"
        _loading_state["message"] = "Loading data..."
        _loading_state["start_time"] = time.time()
        _loading_state["elapsed"] = 0

    # Start timer thread to update elapsed time
    timer_thread = threading.Thread(target=_update_elapsed_time, daemon=True)
    timer_thread.start()

    try:
        if not PICKLE_PATH.exists():
            with _loading_lock:
                _loading_state["status"] = "error"
                _loading_state["message"] = f"Pickle file not found: {PICKLE_PATH}"
            print(f"[WARNING] Pickle file not found: {PICKLE_PATH}")
            return

        print(f"[INFO] Loading pickle from: {PICKLE_PATH}")
        with open(PICKLE_PATH, 'rb') as f:
            pid_to_code = pickle.load(f)

        with _loading_lock:
            _loading_state["status"] = "done"
            _loading_state["count"] = len(pid_to_code)
            _loading_state["elapsed"] = time.time() - _loading_state["start_time"]
            _loading_state["message"] = f"Loaded {len(pid_to_code)} mappings"

        print(f"[INFO] Loaded {len(pid_to_code)} PID to SID mappings in {_loading_state['elapsed']:.1f}s")

    except Exception as e:
        with _loading_lock:
            _loading_state["status"] = "error"
            _loading_state["message"] = str(e)
        print(f"[ERROR] Failed to load pickle file: {e}")


# Start background loading on server
if ENABLE_PICKLE_LOADING:
    print("[INFO] Starting background data loading...")
    _loader_thread = threading.Thread(target=_load_data_background, daemon=True)
    _loader_thread.start()
elif not ENABLE_PICKLE_LOADING:
    print("[INFO] Pickle loading disabled (ENABLE_PICKLE_LOADING=False)")
else:
    print("[INFO] Running in local mode, PID to SID conversion disabled")


def parse_pids(text: str) -> List[int]:
    """
    Parse PIDs from text input.

    Supports multiple formats:
    - Single PID: "12345"
    - Comma-separated: "12345, 67890, 11111"
    - Newline-separated: "12345\n67890\n11111"
    - Mixed: "12345, 67890\n11111"

    Args:
        text: Input text containing PIDs

    Returns:
        List of parsed PID integers
    """
    if not text or not text.strip():
        return []

    # Replace newlines and commas with spaces, then split
    # This handles all combinations of separators
    text = text.replace('\n', ' ').replace(',', ' ')

    # Extract all numbers
    pids = []
    for token in text.split():
        token = token.strip()
        if token.isdigit():
            pids.append(int(token))

    return pids


def find_sid_codes(pid: int) -> Optional[Tuple[int, int, int]]:
    """
    Look up SID codes for a given PID.

    Args:
        pid: Product ID to look up

    Returns:
        Tuple of (code1, code2, code3) if found, None otherwise
    """
    return pid_to_code.get(pid)


def convert_pid_to_sid(text: str) -> str:
    """
    Convert text containing PIDs to SID codes.

    Args:
        text: Input text containing PIDs (single or multiple)

    Returns:
        Formatted string of SID codes, one per line
        Format: "code1, code2, code3"
    """
    pids = parse_pids(text)
    if not pids:
        return ""

    results = []
    for pid in pids:
        codes = find_sid_codes(pid)
        if codes:
            # Format as "code1, code2, code3"
            results.append(f"{codes[0]}, {codes[1]}, {codes[2]}")
        else:
            # PID not found
            results.append(f"PID {pid}: 未找到")

    return "\n".join(results)


def batch_convert_pids(texts: List[str]) -> List[str]:
    """
    Convert multiple texts containing PIDs to SID codes.

    Args:
        texts: List of input texts

    Returns:
        List of formatted SID code strings
    """
    return [convert_pid_to_sid(text) for text in texts]
