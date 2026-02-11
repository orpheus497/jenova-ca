#!/usr/bin/env python3
##Script function and purpose: Fix ChromaDB Python 3.14 compatibility by patching config.py
##Fix: Addresses Pydantic V1 type inference failure for chroma_server_nofile on Python 3.14+
##Date: 2026-02-11T02:02:44Z
##Note: This patch moves attribute declaration before validator to enable proper type inference

"""
ChromaDB Python 3.14 Compatibility Patcher
===========================================

ISSUE: ChromaDB 1.5.0 uses Pydantic V1 which has breaking changes in Python 3.14.
       The `chroma_server_nofile` attribute cannot be inferred because its validator
       is declared before the attribute itself.

FIX: Reorder the Settings class to declare attributes before validators.

SAFETY: This script creates a backup before patching and can be re-run safely.
"""

import re
import shutil
import sys
from pathlib import Path


def find_chromadb_config() -> Path | None:
    """Locate the ChromaDB config.py file in site-packages."""
    ##Action purpose: Search for chromadb config in common locations
    import site
    
    site_packages = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        site_packages.append(site.getusersitepackages())
    
    for sp in site_packages:
        config_path = Path(sp) / "chromadb" / "config.py"
        if config_path.exists():
            return config_path
    
    return None


def backup_file(filepath: Path) -> Path:
    """Create a backup of the file with .orig extension."""
    backup_path = filepath.with_suffix(filepath.suffix + '.orig')
    if not backup_path.exists():
        shutil.copy2(filepath, backup_path)
        print(f"✓ Created backup: {backup_path}")
    else:
        print(f"✓ Backup already exists: {backup_path}")
    return backup_path


def check_if_patched(content: str) -> bool:
    """Check if the file has already been patched."""
    ##Condition purpose: Look for our patch marker comment
    return "##Fix: Python 3.14 compatibility" in content


def apply_patch(filepath: Path) -> bool:
    """Apply the Python 3.14 compatibility patch to ChromaDB config.py."""
    
    ##Step purpose: Read current file content
    content = filepath.read_text()
    
    ##Condition purpose: Skip if already patched
    if check_if_patched(content):
        print("✓ File already patched. Skipping.")
        return True
    
    ##Step purpose: Create backup before modifying
    backup_file(filepath)
    
    ##Fix: Move attribute BEFORE validator (Python 3.14 requires this order)
    ##Pattern 1: Add Field import if not present
    import_pattern = r'(from pydantic\.v1 import BaseSettings)'
    import_replacement = r'\1, Field'
    
    new_content = re.sub(import_pattern, import_replacement, content)
    
    ##Pattern 2: Find the validator+attribute block and reorder them
    ##Note: The attribute must come BEFORE the validator for Python 3.14 type inference
    pattern = re.compile(
        r'(    )@validator\("chroma_server_nofile", pre=True, always=True, allow_reuse=True\)\n'
        r'(    )def empty_str_to_none\(cls, v: str\) -> Optional\[str\]:\n'
        r'(        )if type\(v\) is str and v\.strip\(\) == "":\n'
        r'(            )return None\n'
        r'(        )return v\n'
        r'\n'
        r'(    )chroma_server_nofile: Optional\[int\] = None',
        re.MULTILINE
    )
    
    ##Step purpose: Reorder: attribute FIRST, then validator
    replacement = (
        r'\1##Fix: Python 3.14 compatibility - attribute must be declared before validator (2026-02-11T02:02:44Z)\n'
        r'\1chroma_server_nofile: Optional[int] = Field(default=None)\n'
        r'\n'
        r'\1@validator("chroma_server_nofile", pre=True, always=True, allow_reuse=True)\n'
        r'\2def empty_str_to_none(cls, v: str) -> Optional[str]:\n'
        r'\3if type(v) is str and v.strip() == "":\n'
        r'\4return None\n'
        r'\5return v'
    )
    
    ##Action purpose: Apply the patch
    new_content, count = pattern.subn(replacement, new_content)
    
    if count == 0:
        print("✗ Pattern not found. ChromaDB structure may have changed.")
        return False
    
    ##Action purpose: Write patched content
    filepath.write_text(new_content)
    print(f"✓ Successfully patched {filepath}")
    print(f"  - Added Field import to Pydantic V1")
    print(f"  - Moved chroma_server_nofile declaration BEFORE validator")
    print(f"  - Used Field(default=None) for explicit type hint")
    print(f"  - {count} occurrence(s) fixed")
    
    return True


def main():
    """Main execution function."""
    print("ChromaDB Python 3.14 Compatibility Patcher")
    print("=" * 50)
    
    ##Step purpose: Locate the ChromaDB config file
    config_path = find_chromadb_config()
    
    if not config_path:
        print("✗ Could not locate chromadb/config.py in site-packages")
        print("  Make sure ChromaDB is installed in the current environment")
        sys.exit(1)
    
    print(f"✓ Found ChromaDB config: {config_path}")
    
    ##Step purpose: Apply the compatibility patch
    success = apply_patch(config_path)
    
    if success:
        print("\n" + "=" * 50)
        print("✓ PATCH APPLIED SUCCESSFULLY")
        print("  ChromaDB should now work with Python 3.14")
        print("\nTo restore original:")
        print(f"  cp {config_path}.orig {config_path}")
    else:
        print("\n" + "=" * 50)
        print("✗ PATCH FAILED")
        print("  Please check the error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()
