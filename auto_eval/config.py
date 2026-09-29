"""
Configuration file for OneReason Eval Benchmark Auto Evaluation System
"""
import os
import json
from pathlib import Path
from datetime import datetime


# Base directories
BASE_DIR = Path(__file__).parent
VERSION = os.environ.get("VERSION", "v3.1")
DATA_VERSION = os.environ.get("DATA_VERSION", "v3.1")

# Local directories; caller may override them through environment variables
DATA_RESULTS_BASE_DIR = Path(os.environ.get("BENCHMARK_WORK_DIR", str(BASE_DIR.parent / "workspace")))
RESULTS_BASE_DIR = DATA_RESULTS_BASE_DIR / "results"
DATA_DIR = Path(os.environ.get("BENCHMARK_DATA_DIR", str(BASE_DIR.parent.parent / "data")))

# Additional result roots (searched in order; first match wins)
RESULT_ROOTS = [
    RESULTS_BASE_DIR,
]

# Queue and log base directory on SSD
QUEUE_LOG_BASE_DIR = Path(os.environ.get("BENCHMARK_WORK_DIR", str(BASE_DIR.parent / "workspace")))

# HuggingFace model cache configuration
# Set cache directory for transformers/bert_score models
HF_CACHE_DIR = QUEUE_LOG_BASE_DIR / ".cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))

# Respect caller-selected online/offline behavior
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
QUEUE_BASE_DIR = QUEUE_LOG_BASE_DIR / "auto_eval" / VERSION
LOG_BASE_DIR = QUEUE_LOG_BASE_DIR / "auto_eval_logs"

# Multi-worker configuration
WORKER_ID = os.environ.get("WORKER_ID", None)  # Read from environment variable

# Queue directories for task submission
# GPU and CPU tasks use completely separate directories to avoid interference.
CPU_WORKER_ID = os.environ.get("CPU_WORKER_ID", None)

if WORKER_ID:
    # Multi-worker GPU mode
    QUEUE_DIRS = {
        "incoming": QUEUE_BASE_DIR / "incoming",
        "incoming_worker": QUEUE_BASE_DIR / f"incoming-{WORKER_ID}",
        "processing": QUEUE_BASE_DIR / f"processing-{WORKER_ID}",
        "done": QUEUE_BASE_DIR / "done",
        "failed": QUEUE_BASE_DIR / "failed",
    }
else:
    # Single-worker GPU mode
    QUEUE_DIRS = {
        "incoming": QUEUE_BASE_DIR / "incoming",
        "processing": QUEUE_BASE_DIR / "processing",
        "done": QUEUE_BASE_DIR / "done",
        "failed": QUEUE_BASE_DIR / "failed",
    }

# CPU queue directories (always available, completely separate from GPU)
CPU_QUEUE_DIRS = {
    "incoming": QUEUE_BASE_DIR / "incoming_cpu",
    "processing": QUEUE_BASE_DIR / f"processing_cpu_{CPU_WORKER_ID}" if CPU_WORKER_ID else QUEUE_BASE_DIR / "processing_cpu",
    "done": QUEUE_BASE_DIR / "done_cpu",
    "failed": QUEUE_BASE_DIR / "failed_cpu",
}

# Speed test task directories
# Completely separate from eval and generation queues
SPEEDTEST_BASE_DIR = QUEUE_LOG_BASE_DIR / "speedtest"
SPEEDTEST_QUEUE_DIRS = {
    "incoming": SPEEDTEST_BASE_DIR / "incoming",
    "processing": SPEEDTEST_BASE_DIR / "processing",
    "done": SPEEDTEST_BASE_DIR / "done",
    "failed": SPEEDTEST_BASE_DIR / "failed",
}
SPEEDTEST_OUTPUT_DIR = SPEEDTEST_BASE_DIR / "results"

# Generation task directories
# Separate from eval queue, stored in a dedicated location
GENERATION_BASE_DIR = QUEUE_LOG_BASE_DIR / "generation_case"

GENERATION_DIRS = {
    "incoming": GENERATION_BASE_DIR / "incoming",
    "processing": GENERATION_BASE_DIR / "processing",
    "done": GENERATION_BASE_DIR / "done",
    "failed": GENERATION_BASE_DIR / "failed"
}

# Generation polling interval for Web UI (seconds)
GENERATION_POLL_INTERVAL = 10

# Results analysis directories
RESULTS_DIR = RESULTS_BASE_DIR

# Extra result versions without a benchmark/tasks registry (dashboard path lookup only)
EXTRA_VERSIONS: list = []
EVAL_RESULTS_FILENAME = "eval_results.json"
EVAL_RESULTS_SIDMODEL_FILENAME = "eval_results_sidmodel.json"
TEST_GENERATED_FILENAME = "test_generated.json.debug"

# Task submission settings
MAX_RETRIES = 5
EMAIL_DOMAIN = os.environ.get("BENCHMARK_EMAIL_DOMAIN", "")

# UI settings
MAX_RECENT_TASKS = 20  # Maximum number of recent completed/failed tasks to display
AUTO_REFRESH_INTERVAL = 600  # Auto-refresh interval in seconds (10 minutes)

# Icon paths
ICON_DIR = BASE_DIR / "icon"
SUBMIT_ICON = ICON_DIR / "submit.svg"
ANALYZE_ICON = ICON_DIR / "analyze.svg"
TRANSFORM_ICON = ICON_DIR / "transform.svg"
REFRESH_ICON = ICON_DIR / "reflesh.svg"
DELETE_ICON = ICON_DIR / "delete.svg"
DELETE2_ICON = ICON_DIR / "delete2.svg"
COPY_ICON = ICON_DIR / "copy.svg"
QUERY_ICON = ICON_DIR / "query.svg"
GENERATION_ICON = ICON_DIR / "query.svg"  # Reusing query icon for generation
DASHBOARD_ICON = ICON_DIR / "dashboard.svg"  # Reusing analyze icon for dashboard
LEADERBOARD_ICON = ICON_DIR / "leaderboard.svg"
ADD_ICON = ICON_DIR / "add.svg"
EXCLUDE_ICON = ICON_DIR / "exclude.svg"

# Plot settings
PLOT_OUTPUT_DIR = QUEUE_BASE_DIR / "plots"
DASHBOARD_OUTPUT_DIR = QUEUE_BASE_DIR / "htmls"
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 10)

# Sample comparison settings
MAX_SAMPLES_TO_COMPARE = 10  # Maximum number of samples to compare in modal

# Worker management functions
def get_all_worker_dirs():
    """
    Get all worker-specific directory pairs (incoming, processing).

    Returns:
        Dict mapping worker_id to {'incoming': Path, 'processing': Path}
    """
    worker_dirs = {}

    # Find all incoming-* directories
    for incoming_dir in QUEUE_BASE_DIR.glob("incoming-*"):
        worker_id = incoming_dir.name.replace("incoming-", "")
        processing_dir = QUEUE_BASE_DIR / f"processing-{worker_id}"

        if processing_dir.exists():
            worker_dirs[worker_id] = {
                'incoming': incoming_dir,
                'processing': processing_dir
            }

    return worker_dirs


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    for queue_dir in QUEUE_DIRS.values():
        queue_dir.mkdir(parents=True, exist_ok=True)

    # Create generation directories
    for gen_dir in GENERATION_DIRS.values():
        gen_dir.mkdir(parents=True, exist_ok=True)

    # Create speed test directories
    for st_dir in SPEEDTEST_QUEUE_DIRS.values():
        st_dir.mkdir(parents=True, exist_ok=True)
    SPEEDTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for result_root in RESULT_ROOTS:
        try:
            result_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[WARNING] Could not create result root {result_root}: {e}")

    # Create HuggingFace cache directories
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (HF_CACHE_DIR / "hub").mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_directories()
