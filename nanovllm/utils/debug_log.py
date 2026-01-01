import os
import json

_log_file = None

def _is_debug() -> bool:
    """Check debug flag lazily (after env var may have been set)."""
    return os.getenv("NANOVLLM_DEBUG", "0") == "1"

def _get_log_file():
    global _log_file
    if _log_file is None and _is_debug():
        log_dir = "/vol/logs" if os.path.exists("/vol/logs") else "."
        _log_file = open(f"{log_dir}/stage0_debug.log", "a")
    return _log_file

def debug_log(msg: str | dict):
    """Log a debug message to the debug log file only (no stdout)."""
    if not _is_debug():
        return
    
    if isinstance(msg, dict):
        line = "[nanovllm-debug] " + json.dumps(msg, sort_keys=True)
    else:
        line = msg
    
    log_file = _get_log_file()
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()

def is_debug_enabled() -> bool:
    return _is_debug()
