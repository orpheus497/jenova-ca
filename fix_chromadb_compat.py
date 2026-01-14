#!/usr/bin/env python3
##Script function and purpose: ChromaDB Compatibility Fix for The JENOVA Cognitive Architecture
##This script patches ChromaDB's Settings class to handle Python 3.14 and Pydantic 2.12
##compatibility issues, including missing type annotations and forward reference resolution
"""
Runtime fix for chromadb compatibility issues with Python 3.14 and Pydantic 2.12
This script patches chromadb's Settings class to handle missing/None values properly
"""

import os
import sys
import re

##Function purpose: Patch chromadb config.py for Pydantic 2.12 compatibility
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
    
    # Fix 0: Ensure Optional is imported from typing
    if 'from typing import' in content and 'Optional' not in content.split('from typing import')[1].split('\n')[0]:
        # Add Optional to existing typing import
        content = re.sub(
            r'from typing import (\w+)',
            lambda m: f'from typing import {m.group(1)}, Optional' if 'Optional' not in m.group(1) else m.group(0),
            content,
            count=1
        )
        if 'Optional' not in content.split('from typing import')[1].split('\n')[0]:
            # If Optional wasn't added, add a new import line
            content = content.replace('from typing import', 'from typing import Optional,', 1)
        fixes_applied.append("Optional import")
    elif 'from typing import' not in content:
        # Add typing import if it doesn't exist
        import_line = 'from typing import Optional\n'
        # Insert after other imports
        first_import = content.find('import ')
        if first_import != -1:
            line_end = content.find('\n', first_import)
            content = content[:line_end+1] + import_line + content[line_end+1:]
            fixes_applied.append("Optional import")
    
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
    
    # Fix 5: Add model_rebuild() call after Settings class definition if not present
    # This ensures forward references are resolved before Settings is instantiated
    if 'class Settings' in content and 'Settings.model_rebuild()' not in content:
        # Find the end of the Settings class by looking for the closing of nested Config class
        # Settings class contains a nested Config class, so we need to find where Config ends
        settings_start = content.find('class Settings')
        config_start = content.find('    class Config:', settings_start)
        if config_start != -1:
            # Find the end of Config class (next line at same or less indentation)
            lines = content[config_start:].split('\n')
            config_end_line = 1
            for i, line in enumerate(lines[1:], 1):
                # Config class ends when we hit a line with <= 4 spaces of indentation (same as Config)
                if line and len(line) - len(line.lstrip()) <= 4 and line.strip():
                    config_end_line = i
                    break
            
            # Find the actual end of Settings class (after Config closes)
            # Look for next line with <= 0 spaces (module level)
            settings_end = config_start
            for i in range(config_end_line, len(lines)):
                line = lines[i]
                if line.strip() and len(line) - len(line.lstrip()) == 0:
                    settings_end = config_start + len('\n'.join(lines[:i]))
                    break
            else:
                # If we didn't find module-level code, Settings class goes to end of file
                settings_end = len(content)
            
            # Remove any existing model_rebuild() call that might be inside the class
            before_end = content[:settings_end]
            if 'Settings.model_rebuild()' in before_end[settings_start:]:
                # Remove it first
                content = before_end.replace('Settings.model_rebuild()', '').replace('# Call model_rebuild() to resolve forward references (Pydantic 2.12 compatibility)', '').replace('\n\n\n', '\n\n') + content[settings_end:]
                settings_end = content.find('class Settings')  # Recalculate
                lines = content[settings_end:].split('\n')
                config_start = content.find('    class Config:', settings_end)
                if config_start != -1:
                    lines_after_config = content[config_start:].split('\n')
                    for i, line in enumerate(lines_after_config[1:], 1):
                        if line.strip() and len(line) - len(line.lstrip()) <= 4:
                            settings_end = config_start + len('\n'.join(lines_after_config[:i]))
                            break
            
            # Insert model_rebuild() call right after Settings class ends
            rebuild_call = '\n# Call model_rebuild() to resolve forward references (Pydantic 2.12 compatibility)\nSettings.model_rebuild()\n'
            content = content[:settings_end] + rebuild_call + content[settings_end:]
            fixes_applied.append("Settings.model_rebuild() call after class definition")
    
    # Write back if changes were made
    if content != original_content:
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"✓ Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")
        
        # Also patch chromadb's __init__.py to ensure model_rebuild() is called before Settings() instantiation
        fix_chromadb_init()
        return True
    else:
        print("✓ No fixes needed (already patched)")
        return True

##Function purpose: Patch chromadb __init__.py to call model_rebuild() before Settings instantiation
def fix_chromadb_init():
    """Fix chromadb __init__.py to call model_rebuild() before Settings instantiation"""
    import site
    init_path = None
    for site_packages in site.getsitepackages():
        path = os.path.join(site_packages, 'chromadb', '__init__.py')
        if os.path.exists(path):
            init_path = path
            break
    
    if not init_path:
        user_site = site.getusersitepackages()
        if user_site:
            path = os.path.join(user_site, 'chromadb', '__init__.py')
            if os.path.exists(path):
                init_path = path
    
    if not init_path:
        return
    
    with open(init_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Patch __settings = chromadb.config.Settings() to call model_rebuild() first
    if '__settings = chromadb.config.Settings()' in content and 'Settings.model_rebuild()' not in content:
        patched = '''# Ensure model_rebuild() is called before instantiation (Pydantic 2.12 compatibility)
try:
    chromadb.config.Settings.model_rebuild()
except Exception:
    pass
__settings = chromadb.config.Settings()'''
        content = content.replace('__settings = chromadb.config.Settings()', patched)
        
        with open(init_path, 'w') as f:
            f.write(content)
        print("✓ Patched chromadb __init__.py to call model_rebuild() before Settings instantiation")

if __name__ == '__main__':
    success = fix_chromadb_config()
    sys.exit(0 if success else 1)

