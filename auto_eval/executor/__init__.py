"""
Task Executor Module

This module contains scripts for executing evaluation tasks:
- eval_script.py: Main evaluation script wrapper
- eval_script.sh: Shell script that runs actual evaluation commands
- sentinel_worker.py: Background worker that monitors task queues

These scripts are called by the sentinel worker process, not directly by the web interface.
"""

__all__ = []
