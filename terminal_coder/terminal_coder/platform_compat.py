#!/usr/bin/env python3
"""
Platform compatibility utilities for Terminal Coder
Handles cross-platform differences between Linux and Windows
"""

import sys
import platform
import locale
import os


def is_linux():
    """Check if running on Linux"""
    return platform.system() == 'Linux'


def is_windows():
    """Check if running on Windows"""
    return platform.system() == 'Windows'


def supports_unicode():
    """Check if terminal supports Unicode/emoji characters"""
    if is_linux():
        # Linux terminals generally support Unicode well
        return True

    if is_windows():
        # Check if we're in a modern terminal that supports Unicode
        terminal = os.environ.get('TERM', '').lower()
        wt_session = os.environ.get('WT_SESSION')  # Windows Terminal

        # Windows Terminal definitely supports Unicode
        if wt_session:
            return True

        # Check for actual xterm-like terminals (not just TERM variable)
        # Only trust xterm if we also have proper UTF encoding
        if 'xterm' in terminal and hasattr(sys.stdout, 'encoding'):
            stdout_encoding = sys.stdout.encoding.lower()
            if 'utf' in stdout_encoding:
                return True

        # Check if running in Windows Subsystem for Linux
        if 'microsoft' in os.environ.get('WSL_DISTRO_NAME', '').lower():
            return True

        # Check if we're actually in PowerShell (not just having PowerShell installed)
        # PowerShell sets specific environment variables when running
        if os.environ.get('PSEdition') or 'powershell' in os.environ.get('_', '').lower():
            return True

        # More restrictive check for Windows Command Prompt
        try:
            # Check stdout encoding specifically
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
                stdout_encoding = sys.stdout.encoding.lower()

                # CP1252, ASCII, and similar don't support emojis well
                if any(enc in stdout_encoding for enc in ['cp1252', 'ascii', 'latin-1', 'iso-8859']):
                    return False

                # UTF-8 and UTF-16 should support Unicode
                if 'utf' in stdout_encoding:
                    # Double check by trying to encode a test emoji
                    try:
                        test_emoji = '🐧'
                        test_emoji.encode(sys.stdout.encoding)
                        return True
                    except UnicodeEncodeError:
                        return False

            # Final fallback test
            try:
                encoding = locale.getpreferredencoding() or 'utf-8'
                if encoding.lower() in ['cp1252', 'ascii', 'latin-1']:
                    return False

                test_emoji = '🐧'
                test_emoji.encode(encoding)
                return True
            except (UnicodeEncodeError, LookupError, AttributeError):
                return False

        except Exception:
            # If all checks fail, assume no Unicode support for Windows
            return False

    return True  # Default to True for other platforms


def get_status_symbols():
    """Get appropriate status symbols for the platform"""
    if supports_unicode():
        return {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'rocket': '🚀',
            'penguin': '🐧',
            'robot': '🤖',
            'folder': '📂',
            'search': '🔍',
            'thinking': '💭',
            'party': '🎉'
        }
    else:
        return {
            'success': '[OK]',
            'error': '[FAIL]',
            'warning': '[WARN]',
            'info': '[INFO]',
            'rocket': '[READY]',
            'penguin': '[LINUX]',
            'robot': '[AI]',
            'folder': '[DIR]',
            'search': '[SCAN]',
            'thinking': '[INPUT]',
            'party': '[DONE]'
        }


def safe_print(message, **kwargs):
    """Safely print message handling encoding issues"""
    try:
        print(message, **kwargs)
    except UnicodeEncodeError:
        # First try with UTF-8 encoding
        try:
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout.buffer.write(message.encode('utf-8'))
                sys.stdout.buffer.write(b'\n')
                return
        except (AttributeError, UnicodeEncodeError):
            pass

        # Final fallback: replace problematic Unicode characters
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message, **kwargs)


# Global symbols that adapt to platform
SYMBOLS = get_status_symbols()


def format_title(title):
    """Format title with appropriate symbols"""
    return f"{SYMBOLS['penguin']} {title}"


def format_status(status_type, message):
    """Format status message with appropriate symbol"""
    symbol = SYMBOLS.get(status_type, SYMBOLS['info'])
    return f"{symbol} {message}"