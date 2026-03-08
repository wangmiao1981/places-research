# Copyright 2025 Miao Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Security utilities for safe file operations and path validation.
Prevents path traversal attacks and ensures files are written only to safe locations.
"""

import os
import re
from pathlib import Path
from typing import Optional, Union


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


class SafeFileOperations:
    """Secure file operations with path traversal protection."""

    def __init__(self, base_directory: str = "./output", max_file_size_mb: int = 100):
        """
        Initialize safe file operations.

        Args:
            base_directory: Base directory for file operations (default: ./output)
            max_file_size_mb: Maximum file size in megabytes (default: 100MB)
        """
        self.base_directory = Path(base_directory).resolve()
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Create base directory if it doesn't exist
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for file system operations

        Raises:
            SecurityError: If filename cannot be made safe
        """
        if not filename or not filename.strip():
            raise SecurityError("Filename cannot be empty")

        # Remove or replace dangerous characters
        # Allow only alphanumeric, spaces, dots, hyphens, underscores
        sanitized = re.sub(r'[^\w\s\-_.]', '_', filename.strip())

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')

        # Ensure filename is not empty after sanitization
        if not sanitized:
            raise SecurityError("Filename becomes empty after sanitization")

        # Prevent reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }

        base_name = sanitized.split('.')[0].upper()
        if base_name in reserved_names:
            sanitized = f"safe_{sanitized}"

        # Limit filename length (255 is common filesystem limit)
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            max_name_length = 255 - len(ext)
            sanitized = name[:max_name_length] + ext

        return sanitized

    def get_safe_path(self, filename: str, subdirectory: str = None) -> Path:
        """
        Get a safe file path within the base directory.

        Args:
            filename: Desired filename
            subdirectory: Optional subdirectory within base directory

        Returns:
            Safe absolute path within base directory

        Raises:
            SecurityError: If path would escape base directory
        """
        # Sanitize filename
        safe_filename = self.sanitize_filename(filename)

        # Construct target directory
        if subdirectory:
            # Sanitize subdirectory name
            safe_subdir = self.sanitize_filename(subdirectory)
            target_dir = self.base_directory / safe_subdir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.base_directory

        # Construct full path
        target_path = target_dir / safe_filename

        # Resolve path and ensure it's within base directory
        resolved_path = target_path.resolve()

        # Security check: ensure resolved path is within base directory
        try:
            resolved_path.relative_to(self.base_directory)
        except ValueError:
            raise SecurityError(
                f"Path traversal detected: {resolved_path} is outside {self.base_directory}"
            )

        return resolved_path

    def safe_write_text(self, content: str, filename: str, subdirectory: str = None) -> str:
        """
        Safely write text content to a file.

        Args:
            content: Text content to write
            filename: Desired filename
            subdirectory: Optional subdirectory within base directory

        Returns:
            Absolute path to written file

        Raises:
            SecurityError: If operation is unsafe
            OSError: If file operation fails
        """
        # Validate content size
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > self.max_file_size_bytes:
            raise SecurityError(
                f"Content size ({len(content_bytes)} bytes) exceeds maximum "
                f"({self.max_file_size_bytes} bytes)"
            )

        # Get safe path
        safe_path = self.get_safe_path(filename, subdirectory)

        # Write file atomically
        temp_path = safe_path.with_suffix(safe_path.suffix + '.tmp')

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Atomic rename
            temp_path.replace(safe_path)

            return str(safe_path)

        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to write file: {e}") from e

    def safe_read_text(self, file_path: Union[str, Path], validate_path: bool = True) -> str:
        """
        Safely read text content from a file.

        Args:
            file_path: Path to file to read
            validate_path: Whether to validate path is within safe directory

        Returns:
            File content as string

        Raises:
            SecurityError: If path validation fails
            OSError: If file operation fails
        """
        path = Path(file_path).resolve()

        # Validate path is within base directory (if requested)
        if validate_path:
            try:
                path.relative_to(self.base_directory)
            except ValueError:
                raise SecurityError(
                    f"Path {path} is outside safe directory {self.base_directory}"
                )

        # Validate file size before reading
        try:
            file_size = path.stat().st_size
            if file_size > self.max_file_size_bytes:
                raise SecurityError(
                    f"File size ({file_size} bytes) exceeds maximum "
                    f"({self.max_file_size_bytes} bytes)"
                )
        except OSError as e:
            raise OSError(f"Cannot access file {path}: {e}") from e

        # Read file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise OSError(f"Failed to read file {path}: {e}") from e

    def validate_input_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that an input file path is safe to read.

        Args:
            file_path: Path to validate

        Returns:
            Validated Path object

        Raises:
            SecurityError: If path is unsafe
            OSError: If file doesn't exist or is inaccessible
        """
        path = Path(file_path).resolve()

        # Check if file exists
        if not path.exists():
            raise OSError(f"File not found: {path}")

        # Check if it's actually a file
        if not path.is_file():
            raise SecurityError(f"Path is not a regular file: {path}")

        # Basic security checks
        # Prevent reading sensitive system files
        sensitive_paths = [
            '/etc/passwd', '/etc/shadow', '/etc/hosts',
            '/proc/version', '/sys', '/dev'
        ]

        path_str = str(path).lower()
        for sensitive in sensitive_paths:
            if sensitive in path_str:
                raise SecurityError(f"Access to sensitive path denied: {path}")

        # Check file size
        try:
            file_size = path.stat().st_size
            if file_size > self.max_file_size_bytes:
                raise SecurityError(
                    f"File size ({file_size} bytes) exceeds maximum "
                    f"({self.max_file_size_bytes} bytes)"
                )
        except OSError as e:
            raise OSError(f"Cannot access file {path}: {e}") from e

        return path


# Global instance for convenience
default_safe_ops = SafeFileOperations()


def safe_write_text(content: str, filename: str, subdirectory: str = None) -> str:
    """
    Convenience function for safe text writing using default instance.

    Args:
        content: Text content to write
        filename: Desired filename
        subdirectory: Optional subdirectory

    Returns:
        Path to written file
    """
    return default_safe_ops.safe_write_text(content, filename, subdirectory)


def safe_read_text(file_path: Union[str, Path], validate_path: bool = True) -> str:
    """
    Convenience function for safe text reading using default instance.

    Args:
        file_path: Path to file
        validate_path: Whether to validate path is in safe directory

    Returns:
        File content
    """
    return default_safe_ops.safe_read_text(file_path, validate_path)


def validate_input_path(file_path: Union[str, Path]) -> Path:
    """
    Convenience function for input path validation using default instance.

    Args:
        file_path: Path to validate

    Returns:
        Validated Path object
    """
    return default_safe_ops.validate_input_path(file_path)


def sanitize_filename(filename: str) -> str:
    """
    Convenience function for filename sanitization using default instance.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    return default_safe_ops.sanitize_filename(filename)