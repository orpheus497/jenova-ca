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

##Note: Targeted ChromaDB versions - 1.5.0, 1.5.1 (D3-2026-02-11T08:22:24Z)
"""

##Refactor: Alphabetized stdlib imports per PEP 8 (D3-2026-02-11T07:30:05Z)
import contextlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

##Note: Supported ChromaDB versions for this patch (D3-2026-02-11T08:22:24Z)
SUPPORTED_CHROMADB_VERSIONS = {"1.5.0", "1.5.1"}


##Function purpose: Locate the ChromaDB config.py file in site-packages
def find_chromadb_config() -> Path | None:
    """Locate the ChromaDB config.py file in site-packages."""
    ##Action purpose: Search for chromadb config in common locations
    import site
    import sysconfig

    ##Refactor: Add defensive guard for getsitepackages() (D3-2026-02-11T08:22:24Z)
    site_packages = []
    if hasattr(site, "getsitepackages"):
        site_packages = site.getsitepackages()
    else:
        ##Fallback purpose: Use sysconfig if getsitepackages unavailable
        purelib = sysconfig.get_path("purelib")
        if purelib:
            site_packages = [purelib]

    if hasattr(site, "getusersitepackages"):
        user_site = site.getusersitepackages()
        if user_site:
            site_packages.append(user_site)

    for sp in site_packages:
        config_path = Path(sp) / "chromadb" / "config.py"
        if config_path.exists():
            return config_path

    return None


##Function purpose: Create a backup of the file with .orig extension
def backup_file(filepath: Path) -> Path:
    """Create a backup of the file with .orig extension."""
    backup_path = filepath.with_suffix(filepath.suffix + ".orig")
    if not backup_path.exists():
        ##Refactor: Added error handling for filesystem operations (D3-2026-02-11T08:22:24Z)
        try:
            shutil.copy2(filepath, backup_path)
            print(f"✓ Created backup: {backup_path}")
        except (OSError, PermissionError) as e:
            print(
                "✗ ERROR: Cannot create backup — check file permissions or run with elevated privileges"
            )
            print(f"  File: {filepath}")
            print(f"  Backup: {backup_path}")
            print(f"  Error: {e}")
            raise
    else:
        print(f"✓ Backup already exists: {backup_path}")
    return backup_path


##Function purpose: Check if the file has already been patched
def check_if_patched(content: str) -> bool:
    """Check if the file has already been patched."""
    ##Condition purpose: Look for our patch marker comment
    return "##Fix: Python 3.14 compatibility" in content


##Function purpose: Apply the Python 3.14 compatibility patch to ChromaDB config.py
def apply_patch(filepath: Path) -> bool:
    """Apply the Python 3.14 compatibility patch to ChromaDB config.py."""

    ##Step purpose: Read current file content with explicit UTF-8 encoding
    ##Refactor: Added UTF-8 encoding specification (D3-2026-02-11T07:03:00Z)
    content = filepath.read_text(encoding="utf-8")

    ##Condition purpose: Skip if already patched
    if check_if_patched(content):
        print("✓ File already patched. Skipping.")
        return True

    ##Step purpose: Create backup before modifying
    backup_file(filepath)

    ##Fix: Move attribute BEFORE validator (Python 3.14 requires this order)
    ##Pattern 1: Add Field import if not present (check for duplicates first)
    ##Refactor: Added duplicate import check (D3-2026-02-11T07:03:00Z)
    import_pattern = r"(from pydantic\.v1 import BaseSettings)"
    import_replacement = r"\1, Field"

    ##Step purpose: Check if Field is already imported to avoid duplicates
    field_already_imported = re.search(r"from pydantic\.v1 import.*\bField\b", content)

    if not field_already_imported:
        new_content = re.sub(import_pattern, import_replacement, content)
    else:
        new_content = content
        print("✓ Field import already present, skipping...")

    ##Pattern 2: Find the validator+attribute block and reorder them
    ##Note: The attribute must come BEFORE the validator for Python 3.14 type inference
    ##Refactor: Relaxed regex for flexible whitespace, added diagnostics (D3-2026-02-11T07:03:00Z)
    pattern = re.compile(
        r'@validator\("chroma_server_nofile",\s+pre=True,\s+always=True,\s+allow_reuse=True\)\s+'
        r"def\s+empty_str_to_none\([^)]+\)[^:]+:\s+"
        r".*?return\s+v\s+"
        r"chroma_server_nofile:\s*Optional\[int\]\s*=\s*None",
        re.DOTALL | re.MULTILINE,
    )

    ##Step purpose: Reorder: attribute FIRST, then validator
    replacement = (
        r"##Fix: Python 3.14 compatibility - attribute must be declared before validator (2026-02-11T02:02:44Z)\n"
        r"    chroma_server_nofile: Optional[int] = Field(default=None)\n"
        r"\n"
        r'    @validator("chroma_server_nofile", pre=True, always=True, allow_reuse=True)\n'
        r"    def empty_str_to_none(cls, v: str) -> Optional[str]:\n"
        r'        if type(v) is str and v.strip() == "":\n'
        r"            return None\n"
        r"        return v"
    )

    ##Refactor: Version check before applying substitution (D3-2026-02-11T08:22:24Z)
    ##Step purpose: Get ChromaDB version to verify patch applicability
    try:
        import importlib.metadata as md

        chromadb_version = md.version("chromadb")
    except (ModuleNotFoundError, md.PackageNotFoundError):
        chromadb_version = "unknown"

    ##Condition purpose: Only apply patch to supported ChromaDB versions
    if chromadb_version not in SUPPORTED_CHROMADB_VERSIONS and chromadb_version != "unknown":
        print(
            f"⚠ Warning: ChromaDB version {chromadb_version} is not in supported versions: {SUPPORTED_CHROMADB_VERSIONS}"
        )
        print("   Skipping patch to avoid potential issues.")
        print("   This patch is designed for ChromaDB 1.5.0 and 1.5.1.")
        return False

    ##Step purpose: Apply substitution and check if pattern matched
    new_content, num_subs = pattern.subn(replacement, new_content)

    ##Condition purpose: Log diagnostic if pattern not found (upstream changes)
    if num_subs == 0:
        print("⚠ Warning: Pattern not matched in config.py")
        print(f"   ChromaDB version: {chromadb_version}")
        print("   This may indicate upstream changes in ChromaDB.")
        print("   The patch may need to be updated.")
        return False

    ##Step purpose: Write patched content to file with atomic write pattern
    ##Refactor: Implemented atomic write with temp file and os.replace (D3-2026-02-11T07:03:00Z)
    ##Refactor: Initialize tmp_path before try to avoid fragile locals() check (D3-2026-02-11T08:22:24Z)
    tmp_path = None
    try:
        ##Step purpose: Create temp file in same directory for atomic replacement
        with tempfile.NamedTemporaryFile(
            dir=filepath.parent,
            delete=False,
            mode="w",
            encoding="utf-8",
            prefix=".tmp_chromadb_patch_",
            suffix=".py",
        ) as tmp_file:
            tmp_file.write(new_content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = tmp_file.name

        ##Step purpose: Atomically replace original file with patched version
        os.replace(tmp_path, filepath)
        print(f"✓ Patched: {filepath}")
        print("  - Added Field import (if not present)")
        print("  - Moved chroma_server_nofile declaration BEFORE validator")
        print(f"  - {num_subs} occurrence(s) fixed")
        return True

    ##Refactor: Narrowed to OSError for file I/O, removed dead code (D3-2026-02-11T07:30:05Z)
    except OSError as e:
        ##Error purpose: Clean up temp file on failure
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
        raise OSError(f"Failed to write patched file: {e}") from e


##Function purpose: Main entry point for ChromaDB Python 3.14 compatibility patcher
def main() -> None:
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
