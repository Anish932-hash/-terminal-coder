#!/usr/bin/env python3
"""
Test script to verify Terminal Coder works on Linux systems
"""

import sys
import os
import subprocess

def test_terminal_coder():
    """Test Terminal Coder functionality"""
    print("🐧 Testing Terminal Coder on Linux...")

    # Test 1: Import test
    try:
        sys.path.insert(0, '.')
        from terminal_coder.main import TerminalCoder
        print("✅ TerminalCoder import: SUCCESS")
    except ImportError as e:
        print(f"❌ TerminalCoder import: FAILED - {e}")
        return False

    # Test 2: CLI help
    try:
        result = subprocess.run([sys.executable, '-m', 'terminal_coder.cli', '--help'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CLI help command: SUCCESS")
        else:
            print(f"❌ CLI help command: FAILED - {result.stderr}")
    except Exception as e:
        print(f"❌ CLI help command: FAILED - {e}")

    # Test 3: Doctor command
    try:
        result = subprocess.run([sys.executable, '-m', 'terminal_coder.cli', 'doctor'],
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            print("✅ CLI doctor command: SUCCESS")
        else:
            print(f"❌ CLI doctor command: FAILED - {result.stderr}")
    except Exception as e:
        print(f"❌ CLI doctor command: FAILED - {e}")

    # Test 4: AI setup command
    try:
        # Test the AI setup command (will fail because it needs input, but should not crash)
        result = subprocess.run([sys.executable, '-m', 'terminal_coder.cli', 'ai', '--status'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ AI status command: SUCCESS")
        else:
            print(f"⚠️  AI status command: {result.stderr}")
    except Exception as e:
        print(f"❌ AI status command: FAILED - {e}")

    # Test 5: Unicode/emoji support
    try:
        # Test if emojis render properly (should work on Linux terminals)
        test_emojis = "🚀🐧✅❌⚠️🤖📂🔍💭"
        print(f"✅ Unicode/emoji test: {test_emojis}")
    except UnicodeEncodeError as e:
        print(f"❌ Unicode/emoji test: FAILED - {e}")

    print("\n🎉 Terminal Coder Linux testing completed!")
    return True

if __name__ == "__main__":
    test_terminal_coder()