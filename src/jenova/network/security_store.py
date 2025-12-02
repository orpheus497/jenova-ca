# The JENOVA Cognitive Architecture - Secure Credential Storage
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Secure storage for cryptographic credentials and keys.

This module provides encrypted storage for private keys, certificates,
and other sensitive credentials using password-based encryption.
"""

import getpass
import hashlib
import os
import secrets
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend


class SecureCredentialStore:
    """
    Manages secure storage of cryptographic credentials.

    Features:
    - Password-based key derivation (Scrypt)
    - Encrypted private key storage
    - Secure master password management
    - Salt-based key derivation
    """

    def __init__(self, cert_dir: Path, file_logger):
        """
        Initialize secure credential store.

        Args:
            cert_dir: Directory for storing credentials
            file_logger: Logger for file output
        """
        self.cert_dir = Path(cert_dir).expanduser()
        self.file_logger = file_logger

        # Ensure directory exists with restrictive permissions
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.cert_dir, 0o700)  # Owner read/write/execute only

        # Password cache (in memory only, never written to disk)
        self._password_cache: Optional[bytes] = None

    def get_master_password(self, prompt: bool = False) -> bytes:
        """
        Get or create master password for key encryption.

        Args:
            prompt: Whether to prompt user for password

        Returns:
            Master password as bytes
        """
        if self._password_cache is not None:
            return self._password_cache

        password_file = self.cert_dir / ".password_hash"

        if password_file.exists():
            # Load existing password hash
            if prompt:
                password = getpass.getpass("Enter JENOVA security password: ")
                password_bytes = password.encode("utf-8")

                # Verify password
                stored_hash = password_file.read_bytes()
                computed_hash = self._hash_password(password_bytes)

                if computed_hash != stored_hash:
                    raise ValueError("Incorrect password")

                self._password_cache = password_bytes
                return password_bytes
            else:
                # Non-interactive mode: use a default derived password
                # This is less secure but necessary for automated deployments
                self.file_logger.log_warning(
                    "Using default password for key encryption. "
                    "For production, set a master password with --set-password"
                )
                default_password = self._get_default_password()
                self._password_cache = default_password
                return default_password
        else:
            # Create new password
            if prompt:
                password = getpass.getpass("Create JENOVA security password: ")
                confirm = getpass.getpass("Confirm password: ")

                if password != confirm:
                    raise ValueError("Passwords do not match")

                password_bytes = password.encode("utf-8")
            else:
                # Generate random password for non-interactive setup
                password_bytes = self._get_default_password()
                self.file_logger.log_info(
                    "Generated default security password. "
                    "Use --set-password to change it."
                )

            # Store password hash
            password_hash = self._hash_password(password_bytes)
            password_file.write_bytes(password_hash)
            os.chmod(password_file, 0o600)  # Owner read/write only

            self._password_cache = password_bytes
            return password_bytes

    def _get_default_password(self) -> bytes:
        """
        Generate a default password based on system characteristics.

        This provides basic security for automated deployments where
        interactive password entry isn't possible.

        Returns:
            Default password bytes
        """
        # Use machine ID if available
        machine_id_file = Path("/etc/machine-id")
        if machine_id_file.exists():
            machine_id = machine_id_file.read_text().strip()
        else:
            # Fallback to hostname
            import socket

            machine_id = socket.gethostname()

        # Combine with a constant salt
        salt = b"JENOVA-DEFAULT-PASSWORD-SALT-V1"
        combined = f"{machine_id}:{salt.decode('utf-8')}".encode("utf-8")

        # Hash to create password
        return hashlib.sha256(combined).digest()

    def _hash_password(self, password: bytes) -> bytes:
        """
        Hash password for storage verification using Argon2id.

        SECURITY FIX (Phase 21): Replaced SHA-256 with Argon2id password hashing.
        SHA-256 is too fast for password hashing and susceptible to GPU-accelerated
        brute-force attacks. Argon2id is the OWASP 2024 recommended algorithm for
        password hashing, providing resistance to both side-channel and GPU attacks.

        Args:
            password: Password to hash

        Returns:
            Password hash in Argon2 format (includes salt, parameters, and hash)
        """
        try:
            from argon2 import PasswordHasher, Type
            from argon2.exceptions import HashingError

            # Use Argon2id with OWASP 2024 recommended parameters
            # time_cost=3, memory_cost=65536 (64 MB), parallelism=4
            ph = PasswordHasher(
                time_cost=3,          # Number of iterations
                memory_cost=65536,    # 64 MB memory usage
                parallelism=4,        # Number of parallel threads
                hash_len=32,          # 256-bit hash output
                salt_len=16,          # 128-bit salt
                type=Type.ID          # Argon2id (hybrid mode)
            )

            # Hash password - returns string in Argon2 format
            # Format: $argon2id$v=19$m=65536,t=3,p=4$<salt>$<hash>
            hash_str = ph.hash(password)

            # Convert to bytes for storage
            return hash_str.encode('utf-8')

        except ImportError:
            # Fallback to PBKDF2 if argon2-cffi not available
            # This maintains security while ensuring graceful degradation
            import os
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend

            salt = os.urandom(32)  # 256-bit salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=600000,  # OWASP 2024 recommendation
                backend=default_backend(),
            )
            hash_value = kdf.derive(password)

            # Return salt + hash for verification
            return salt + hash_value

        except HashingError as e:
            # Log error and raise
            if hasattr(self, 'file_logger') and self.file_logger:
                self.file_logger.log_error(f"Password hashing failed: {e}")
            raise RuntimeError(f"Password hashing failed: {e}") from e

    def derive_encryption_key(
        self, password: Optional[bytes] = None, salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using Scrypt.

        Args:
            password: Password to derive from (uses master if None)
            salt: Salt for derivation (generates random if None)

        Returns:
            Tuple of (encryption_key, salt)
        """
        if password is None:
            password = self.get_master_password()

        if salt is None:
            salt = secrets.token_bytes(32)

        # Scrypt parameters (balanced for security and performance)
        kdf = Scrypt(
            salt=salt,
            length=32,  # 256-bit key
            n=2**14,  # CPU/memory cost (16384)
            r=8,  # Block size
            p=1,  # Parallelization
            backend=default_backend(),
        )

        key = kdf.derive(password)
        return key, salt

    def get_encryption_algorithm(self, password: Optional[bytes] = None):
        """
        Get encryption algorithm for private key serialization.

        Args:
            password: Password to use (uses master if None)

        Returns:
            Encryption algorithm instance
        """
        if password is None:
            password = self.get_master_password()

        return serialization.BestAvailableEncryption(password)

    def save_private_key(
        self, private_key, filename: str, password: Optional[bytes] = None
    ):
        """
        Save private key with encryption.

        Args:
            private_key: Private key object
            filename: Filename to save to
            password: Password for encryption (uses master if None)
        """
        filepath = self.cert_dir / filename

        encryption = self.get_encryption_algorithm(password)

        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )

        filepath.write_bytes(pem)
        os.chmod(filepath, 0o600)  # Owner read/write only

        self.file_logger.log_info(f"Saved encrypted private key to {filepath}")

    def load_private_key(self, filename: str, password: Optional[bytes] = None):
        """
        Load encrypted private key.

        Args:
            filename: Filename to load from
            password: Password for decryption (uses master if None)

        Returns:
            Private key object
        """
        filepath = self.cert_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Private key not found: {filepath}")

        if password is None:
            password = self.get_master_password()

        pem = filepath.read_bytes()

        private_key = serialization.load_pem_private_key(
            pem, password=password, backend=default_backend()
        )

        self.file_logger.log_info(f"Loaded encrypted private key from {filepath}")
        return private_key

    def clear_password_cache(self):
        """Clear cached master password from memory."""
        if self._password_cache is not None:
            # Overwrite memory before deleting
            self._password_cache = b"\x00" * len(self._password_cache)
            self._password_cache = None
            self.file_logger.log_info("Cleared password cache")
