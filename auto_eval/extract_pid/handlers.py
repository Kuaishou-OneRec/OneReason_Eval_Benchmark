"""
Business logic for SID to PID conversion
"""
import os
import re
import threading
import pickle
from pathlib import Path
from typing import List, Tuple

# Set to False to skip loading pickle file (for testing)
ENABLE_PICKLE_LOADING = False

# Initialize code_to_pid mapping and loading state
code_to_pid = {}
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

# Encoding constants for (code1, code2, code3) -> single int
# Each code is in range [0, 8192], needs 13 bits
CODE_MULTIPLIER_1 = 8192 * 8192  # 67108864
CODE_MULTIPLIER_2 = 8192

# Data file path
PICKLE_PATH = Path(os.environ.get("SID_TO_PID_PICKLE", "data/code_to_pid.pkl"))


def encode_sid(c1: int, c2: int, c3: int) -> int:
    """Encode (code1, code2, code3) into a single integer key."""
    return c1 * CODE_MULTIPLIER_1 + c2 * CODE_MULTIPLIER_2 + c3


def get_loading_status() -> dict:
    """Get current loading status.

    Also auto-corrects status if data is already loaded but status is stale
    (e.g., after hot reload or page refresh).
    """
    with _loading_lock:
        # Auto-correct: if data exists but status is not "done", fix it
        if code_to_pid and _loading_state["status"] in ("idle", "loading"):
            _loading_state["status"] = "done"
            _loading_state["count"] = len(code_to_pid)
            _loading_state["message"] = f"Loaded {len(code_to_pid)} mappings"
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
    global code_to_pid

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
            code_to_pid = pickle.load(f)

        with _loading_lock:
            _loading_state["status"] = "done"
            _loading_state["count"] = len(code_to_pid)
            _loading_state["elapsed"] = time.time() - _loading_state["start_time"]
            _loading_state["message"] = f"Loaded {len(code_to_pid)} mappings"

        print(f"[INFO] Loaded {len(code_to_pid)} SID to PID mappings in {_loading_state['elapsed']:.1f}s")

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
    print("[INFO] Running in local mode, SID to PID conversion disabled")


def extract_sids(text: str) -> List[Tuple[int, int, int]]:
    """
    Extract SID codes from text using regex pattern.
    
    Args:
        text: Input text containing SID patterns like <|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>
    
    Returns:
        List of tuples (a, b, c) representing extracted SID codes
    """
    pattern = r'<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><\|sid_end\|>'
    matches = re.findall(pattern, text)
    return [(int(a), int(b), int(c)) for a, b, c in matches]


def find_pids(text: str) -> List[int]:
    """
    Extract SIDs from text and convert them to PIDs.
    
    Args:
        text: Input text containing SID patterns
    
    Returns:
        List of PIDs (filters out 0 values when SID not found)
    """
    if not text or not text.strip():
        return []
    
    sids = extract_sids(text)
    if not sids:
        return []

    # Convert SIDs to PIDs using encoded key, filter out 0 values (not found)
    pids = [code_to_pid.get(encode_sid(*sid), 0) for sid in sids]
    pids = [pid for pid in pids if pid != 0]
    
    return pids


def convert_text_to_pids(text: str) -> str:
    """
    Convert text containing SIDs to comma-separated PID string.
    
    Args:
        text: Input text containing SID patterns
    
    Returns:
        Comma-separated string of PIDs, or empty string if no valid PIDs found
    """
    pids = find_pids(text)
    if not pids:
        return ""
    
    return ", ".join(map(str, pids))


def batch_convert_texts(texts: List[str]) -> List[str]:
    """
    Convert multiple texts containing SIDs to PIDs.
    
    Args:
        texts: List of input texts
    
    Returns:
        List of comma-separated PID strings
    """
    return [convert_text_to_pids(text) for text in texts]
