"""
Helper Utilities

Common utility functions used across the application.
"""

from typing import List, Dict, Any, Optional, Union
import json
import os
from datetime import datetime
from pathlib import Path


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Datetime object (defaults to now)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%H:%M:%S")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """
    Safely serialize data to JSON string.
    
    Args:
        data: Data to serialize
        indent: JSON indentation
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    except Exception as e:
        return f"JSON serialization error: {e}"


def safe_json_loads(json_str: str) -> Any:
    """
    Safely deserialize JSON string.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Deserialized data or None if error
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        return None


def ensure_directory(path: str) -> bool:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        return False


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from path.
    
    Args:
        file_path: File path
        
    Returns:
        File extension (with dot)
    """
    return Path(file_path).suffix.lower()


def is_text_file(file_path: str) -> bool:
    """
    Check if file is likely a text file based on extension.
    
    Args:
        file_path: File path
        
    Returns:
        True if likely text file, False otherwise
    """
    text_extensions = {
        '.txt', '.md', '.rst', '.py', '.js', '.html', '.css', '.json', '.xml',
        '.csv', '.tsv', '.log', '.conf', '.cfg', '.ini', '.yaml', '.yml'
    }
    return get_file_extension(file_path) in text_extensions


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    invalid_chars = '<>:"/\\|?*'
    cleaned = filename
    for char in invalid_chars:
        cleaned = cleaned.replace(char, '_')
    return cleaned


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def count_items_by_key(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    """
    Count items by a specific key.
    
    Args:
        items: List of dictionaries
        key: Key to count by
        
    Returns:
        Dictionary mapping key values to counts
    """
    counts = {}
    for item in items:
        value = item.get(key, "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def filter_items_by_key(items: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
    """
    Filter items by key-value pair.
    
    Args:
        items: List of dictionaries
        key: Key to filter by
        value: Value to match
        
    Returns:
        Filtered list of items
    """
    return [item for item in items if item.get(key) == value]


def group_items_by_key(items: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group items by a specific key.
    
    Args:
        items: List of dictionaries
        key: Key to group by
        
    Returns:
        Dictionary mapping key values to lists of items
    """
    groups = {}
    for item in items:
        value = item.get(key, "unknown")
        if value not in groups:
            groups[value] = []
        groups[value].append(item)
    return groups


def validate_file_path(file_path: str) -> Dict[str, Any]:
    """
    Validate file path and return information.
    
    Args:
        file_path: File path to validate
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "path": file_path,
        "exists": False,
        "is_file": False,
        "is_directory": False,
        "readable": False,
        "writable": False,
        "size_bytes": 0,
        "extension": "",
        "error": None
    }
    
    try:
        path = Path(file_path)
        result["exists"] = path.exists()
        
        if result["exists"]:
            result["is_file"] = path.is_file()
            result["is_directory"] = path.is_directory()
            result["readable"] = os.access(file_path, os.R_OK)
            result["writable"] = os.access(file_path, os.W_OK)
            
            if result["is_file"]:
                result["size_bytes"] = path.stat().st_size
                result["extension"] = path.suffix.lower()
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


def create_backup_filename(original_path: str, suffix: str = "backup") -> str:
    """
    Create backup filename for a file.
    
    Args:
        original_path: Original file path
        suffix: Backup suffix
        
    Returns:
        Backup file path
    """
    path = Path(original_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_{suffix}_{timestamp}{path.suffix}"
    return str(path.parent / backup_name)


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def deep_merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Deep merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dictionaries(result[key], value)
        else:
            result[key] = value
    
    return result


def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information.
    
    Returns:
        Dictionary with environment information
    """
    import platform
    import sys
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "OPENAI_API_KEY": "***" if os.environ.get("OPENAI_API_KEY") else "Not set"
        }
    }
