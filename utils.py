"""
Utility functions for document processing
"""

import os
from pathlib import Path
from typing import List, Optional


def is_supported_format(filename: str) -> bool:
    """Check if file format is supported for ingestion"""
    supported = {'.pdf', '.txt', '.md', '.pptx', '.ppt'}
    return Path(filename).suffix.lower() in supported


def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(filepath)


def format_file_size(bytes_size: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-').lower()
