import os
import json
import time
from datetime import datetime

# download logs:  poetry run modal volume get nano-vllm-logs nanovllm_debug_*.log logs/

_log_file = None
_timestamp = None

def _is_debug() -> bool:
    """Check debug flag lazily (after env var may have been set)."""
    return os.getenv("NANOVLLM_DEBUG", "0") == "1"

def _get_timestamp() -> str:
    """Get timestamp for log filename. Uses env var if set, otherwise generates one."""
    global _timestamp
    if _timestamp is None:
        # Check if timestamp was set via env var (from live_bench.py)
        _timestamp = os.getenv("NANOVLLM_DEBUG_TIMESTAMP")
        if _timestamp is None:
            # Generate timestamp if not set
            _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _timestamp

def _get_log_file():
    global _log_file
    if _log_file is None and _is_debug():
        log_dir = "/vol/logs" if os.path.exists("/vol/logs") else "."
        timestamp = _get_timestamp()
        log_filename = f"stage0_debug_{timestamp}.log"
        _log_file = open(f"{log_dir}/{log_filename}", "a")
    return _log_file

def debug_log(msg: str | dict):
    """Log a debug message to the debug log file only (no stdout)."""
    if not _is_debug():
        return
    
    if isinstance(msg, dict):
        msg = {"ts": time.perf_counter(), **msg}
        line = "[nanovllm-debug] " + json.dumps(msg, sort_keys=True)
    else:
        line = msg
    
    log_file = _get_log_file()
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()

def is_debug_enabled() -> bool:
    return _is_debug()
