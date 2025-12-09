#!/usr/bin/env python3
"""
Runtime fix for chromadb compatibility issues with Python 3.14 and Pydantic 2.12
This script patches chromadb's Settings class to handle missing/None values properly
"""

import os
import sys
import re

def fix_chromadb_config():
    """Fix chromadb config.py to handle Python 3.14/pydantic 2.12 compatibility"""
    # Find chromadb config.py
    chromadb_paths = []
    import site
    for site_packages in site.getsitepackages():
        config_path = os.path.join(site_packages, 'chromadb', 'config.py')
        if os.path.exists(config_path):
            chromadb_paths.append(config_path)
            break
    
    # Also check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        config_path = os.path.join(user_site, 'chromadb', 'config.py')
        if os.path.exists(config_path) and config_path not in chromadb_paths:
            chromadb_paths.append(config_path)
    
    if not chromadb_paths:
        print("⚠ Warning: Could not find chromadb config.py")
        return False
    
    config_path = chromadb_paths[0]
    print(f"Found chromadb config at: {config_path}")
    
    # Read the file
    with open(config_path, 'r') as f:
        content = f.read()
    
    original_content = content
    fixes_applied = []
    
    # Fix 1: Add type annotation for chroma_coordinator_host if missing
    if 'chroma_coordinator_host = ' in content and 'chroma_coordinator_host:' not in content:
        content = re.sub(
            r'(\s+)chroma_coordinator_host\s*=\s*"localhost"',
            r'\1chroma_coordinator_host: str = "localhost"',
            content
        )
        fixes_applied.append("chroma_coordinator_host type annotation")
    
    # Fix 2: Add type annotation for chroma_logservice_port if missing
    if 'chroma_logservice_port = ' in content and 'chroma_logservice_port:' not in content:
        content = re.sub(
            r'(\s+)chroma_logservice_port\s*=\s*(\d+)',
            r'\1chroma_logservice_port: int = \2',
            content
        )
        fixes_applied.append("chroma_logservice_port type annotation")
    
    # Fix 3: Fix chroma_server_nofile access in System.__init__
    if 'if settings["chroma_server_nofile"]' in content:
        # Replace with getattr approach
        old_pattern = r'if settings\["chroma_server_nofile"\] is not None:'
        new_pattern = '''# Note: chroma_server_nofile is commented out in Settings class
        # Use getattr with default None to handle missing attribute gracefully
        chroma_server_nofile = getattr(settings, "chroma_server_nofile", None)
        if chroma_server_nofile is not None:'''
        content = re.sub(old_pattern, new_pattern, content)
        
        # Also fix the reference to desired_soft
        if 'desired_soft = settings["chroma_server_nofile"]' in content:
            content = content.replace(
                'desired_soft = settings["chroma_server_nofile"]',
                'desired_soft = chroma_server_nofile'
            )
        fixes_applied.append("chroma_server_nofile access fix")
    
    # Fix 4: Fix port fields to be Optional[int] instead of Optional[str] if they're causing issues
    # Check if ports are defined as Optional[str] when they should be Optional[int]
    port_fields = ['chroma_server_http_port', 'chroma_server_grpc_port', 'clickhouse_port']
    for field in port_fields:
        # If it's Optional[str] but should be Optional[int], fix it
        pattern = rf'(\s+){field}:\s*Optional\[str\]\s*=\s*None'
        if re.search(pattern, content):
            content = re.sub(pattern, rf'\1{field}: Optional[int] = None', content)
            fixes_applied.append(f"{field} type fix (str -> int)")
    
    # Write back if changes were made
    if content != original_content:
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"✓ Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")
        return True
    else:
        print("✓ No fixes needed (already patched)")
        return True

if __name__ == '__main__':
    success = fix_chromadb_config()
    sys.exit(0 if success else 1)

