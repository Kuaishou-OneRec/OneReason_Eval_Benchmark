"""
Shared utility functions for OneReason Eval Benchmark Auto Evaluation System
"""
import re
import os
from pathlib import Path
from typing import Optional
import requests
import json


def validate_email(email: str) -> bool:
    """
    Validate email format

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def build_email_from_username(username: str, domain: str = None) -> str:
    """
    Build email from username

    Args:
        username: Username (with or without @domain)
        domain: Optional email domain; defaults to BENCHMARK_EMAIL_DOMAIN

    Returns:
        Complete email address
    """
    domain = domain or os.environ.get("BENCHMARK_EMAIL_DOMAIN", "")
    if '@' in username:
        return username
    return f"{username}@{domain}" if domain else username


def extract_directory_name(path: Path) -> str:
    """
    Extract clean directory name from path

    Args:
        path: Directory path

    Returns:
        Directory name without full path
    """
    return path.name


def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp for display

    Args:
        timestamp: ISO format timestamp string

    Returns:
        Formatted timestamp
    """
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def safe_json_load(file_path: Path) -> Optional[dict]:
    """
    Safely load JSON file with error handling

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data or None if error
    """
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            return json.loads(content)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None


def get_model_name_from_path(path: str) -> str:
    """
    Extract clean model name from file path

    Args:
        path: File or directory path

    Returns:
        Cleaned model name
    """
    path_obj = Path(path)
    name = path_obj.name

    # Remove 'results_' prefix if present
    if name.startswith('results_'):
        name = name[8:]

    return name


def send_message(robot_key, text):
    """Compatibility hook; notifications are disabled in the public benchmark."""
    return None


def load_svg_icon(icon_path: Path) -> str:
    """
    Load SVG icon as string

    Args:
        icon_path: Path to SVG file

    Returns:
        SVG content string or empty string if failed
    """
    try:
        with open(icon_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to load icon {icon_path}: {e}")
        return ""
