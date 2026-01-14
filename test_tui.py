#!/usr/bin/env python3
##Script function and purpose: TUI Integration Test for The JENOVA Cognitive Architecture
##This script tests the Bubble Tea TUI with mock messages to verify IPC communication
##without requiring the full JENOVA backend to be running
"""
Simple test script for the Bubble Tea TUI.
Tests basic message passing without requiring the full JENOVA backend.
"""

import json
import subprocess
import time
import threading
import sys

##Function purpose: Test TUI process with mock JSON messages via stdin/stdout
def test_tui():
    """Test the TUI with mock messages."""
    tui_path = "./tui/jenova-tui"
    
    print("Starting TUI test...")
    
    # Start the TUI process
    process = subprocess.Popen(
        [tui_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    
    def read_output():
        """Read and display TUI output."""
        try:
            for line in iter(process.stdout.readline, b''):
                if not line:
                    break
                line_str = line.decode('utf-8').strip()
                if line_str:
                    print(f"TUI -> Python: {line_str}")
                    try:
                        msg = json.loads(line_str)
                        if msg.get('type') == 'exit':
                            print("Exit message received")
                            break
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error reading: {e}")
    
    # Start reading thread
    read_thread = threading.Thread(target=read_output, daemon=True)
    read_thread.start()
    
    # Give TUI time to start
    time.sleep(1)
    
    # Send test messages
    messages = [
        {"type": "banner", "content": "TEST BANNER", "data": {"attribution": "Test"}},
        {"type": "info", "content": "This is a test message"},
        {"type": "system_message", "content": "System message test"},
    ]
    
    print("\nSending test messages...")
    for msg in messages:
        json_str = json.dumps(msg)
        print(f"Python -> TUI: {json_str}")
        try:
            process.stdin.write((json_str + "\n").encode('utf-8'))
            process.stdin.flush()
            time.sleep(0.5)
        except Exception as e:
            print(f"Error sending: {e}")
            break
    
    # Wait a bit to see the messages
    print("\nWaiting for TUI to display messages...")
    time.sleep(3)
    
    # Cleanup
    print("\nTerminating TUI...")
    process.terminate()
    process.wait(timeout=5)
    
    print("Test complete!")

if __name__ == "__main__":
    test_tui()
