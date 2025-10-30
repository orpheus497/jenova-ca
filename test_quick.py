#!/usr/bin/env python3
"""Quick test to verify JENOVA responds to queries."""

import subprocess
import time

print("Starting JENOVA...")
proc = subprocess.Popen(
    ["python", "-m", "jenova.main"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Wait for startup
print("Waiting for startup...")
for line in iter(proc.stdout.readline, ''):
    print(line, end='')
    if "Initialized and Ready" in line or "orpheus497@JENOVA>" in line:
        break

print("\n--- Sending query ---")
proc.stdin.write("Hello\n")
proc.stdin.flush()

# Wait for response (max 4 minutes)
start_time = time.time()
response_started = False
for line in iter(proc.stdout.readline, ''):
    elapsed = time.time() - start_time
    if elapsed > 240:  # 4 minute timeout
        print(f"\n!!! TIMEOUT after {elapsed:.1f}s !!!")
        break

    print(line, end='')

    if "JENOVA:" in line:
        response_started = True

    if response_started and "orpheus497@JENOVA>" in line:
        print(f"\n✓ Response received in {elapsed:.1f}s")
        break

# Send exit
proc.stdin.write("exit\n")
proc.stdin.flush()
proc.wait(timeout=10)

print("\n✓ Test complete")
