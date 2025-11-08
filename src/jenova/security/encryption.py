"""
JENOVA Encryption Module - Encryption at rest and secure secret management.

This module provides:
- Encryption for ChromaDB vector storage
- Secure secret storage using OS keyrings
- User password-derived encryption keys
- Automatic migration from plaintext

Fixes: VULN-H3 (High Severity) - JWT secrets in plaintext
Implements: FEATURE-A4 - Encrypted secrets management
Implements: FEATURE-C1 - Encryption at rest

Copyright (c) 2024-2025, orpheus497. All rights reserved.
Licensed under the MIT License.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend

try:
    import keyring

    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False
    logging.warning("keyring not available - using encrypted file fallback")

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Manages encryption at rest for memory systems and sensitive data.

    Uses Fernet (symmetric encryption) with user-derived keys via PBKDF2.
    """

    # PBKDF2 iterations (higher = more secure but slower)
    PBKDF2_ITERATIONS = 600000  # OWASP recommendation for 2024

    # Salt size in bytes
    SALT_SIZE = 32

    def __init__(self, password: Optional[str] = None):
        """
        Initialize the EncryptionManager.

        Args:
            password: User password for key derivation (None = auto-generate)
        """
        self.password = password
        self._fernet: Optional[Fernet] = None

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: User password
            salt: Random salt

        Returns:
            32-byte encryption key
        """
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.PBKDF2_ITERATIONS,
            backend=default_backend(),
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))

    def initialize_encryption(
        self,
        password: str,
        salt_file: Path,
    ) -> Fernet:
        """
        Initialize Fernet encryption with password-derived key.

        Args:
            password: User password
            salt_file: Path to file storing salt

        Returns:
            Configured Fernet instance
        """
        # Load or generate salt
        if salt_file.exists():
            with open(salt_file, "rb") as f:
                salt = f.read()
            logger.debug(f"Loaded existing salt from {salt_file}")
        else:
            salt = os.urandom(self.SALT_SIZE)
            salt_file.parent.mkdir(parents=True, exist_ok=True)
            with open(salt_file, "wb") as f:
                f.write(salt)
            os.chmod(salt_file, 0o600)  # Owner read/write only
            logger.info(f"Generated new salt at {salt_file}")

        # Derive encryption key
        key = self._derive_key(password, salt)
        self._fernet = Fernet(key)

        return self._fernet

    def encrypt(self, plaintext: Union[str, bytes]) -> bytes:
        """
        Encrypt plaintext data.

        Args:
            plaintext: Data to encrypt

        Returns:
            Encrypted data

        Raises:
            RuntimeError: If encryption not initialized
        """
        if self._fernet is None:
            raise RuntimeError(
                "Encryption not initialized. Call initialize_encryption() first."
            )

        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        return self._fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt encrypted data.

        Args:
            ciphertext: Encrypted data

        Returns:
            Decrypted plaintext

        Raises:
            RuntimeError: If encryption not initialized
            cryptography.fernet.InvalidToken: If decryption fails
        """
        if self._fernet is None:
            raise RuntimeError(
                "Encryption not initialized. Call initialize_encryption() first."
            )

        return self._fernet.decrypt(ciphertext)

    def decrypt_to_string(self, ciphertext: bytes) -> str:
        """
        Decrypt and decode to UTF-8 string.

        Args:
            ciphertext: Encrypted data

        Returns:
            Decrypted string
        """
        plaintext_bytes = self.decrypt(ciphertext)
        return plaintext_bytes.decode("utf-8")


class SecureSecretManager:
    """
    Manages secrets using OS keyring with encrypted file fallback.

    Fixes VULN-H3: JWT secrets no longer stored in plaintext.

    Priority order:
    1. OS keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service)
    2. Encrypted file with user password
    """

    KEYRING_SERVICE_NAME = "jenova-ai"

    def __init__(
        self,
        fallback_dir: Optional[Path] = None,
        encryption_manager: Optional[EncryptionManager] = None,
    ):
        """
        Initialize the SecureSecretManager.

        Args:
            fallback_dir: Directory for encrypted fallback storage
            encryption_manager: Encryption manager for fallback mode
        """
        self.use_keyring = HAS_KEYRING
        self.fallback_dir = fallback_dir
        self.encryption_manager = encryption_manager

        if self.use_keyring:
            logger.info("Using OS keyring for secret storage")
        else:
            logger.warning("OS keyring not available - using encrypted file fallback")
            if not self.fallback_dir:
                raise ValueError("fallback_dir required when keyring not available")
            if not self.encryption_manager:
                raise ValueError(
                    "encryption_manager required when keyring not available"
                )

    def store_secret(self, key: str, value: str) -> None:
        """
        Store a secret securely.

        Args:
            key: Secret key/name
            value: Secret value

        Raises:
            RuntimeError: If storage fails
        """
        if self.use_keyring:
            try:
                keyring.set_password(self.KEYRING_SERVICE_NAME, key, value)
                logger.debug(f"Stored secret '{key}' in OS keyring")
                return
            except Exception as e:
                logger.error(f"Keyring storage failed: {e}")
                # Fall through to encrypted file fallback

        # Encrypted file fallback
        if not self.encryption_manager:
            raise RuntimeError("No encryption manager available for fallback storage")

        try:
            encrypted = self.encryption_manager.encrypt(value)
            secret_file = self.fallback_dir / f"{key}.enc"
            secret_file.parent.mkdir(parents=True, exist_ok=True)

            with open(secret_file, "wb") as f:
                f.write(encrypted)

            os.chmod(secret_file, 0o600)  # Owner read/write only
            logger.debug(f"Stored secret '{key}' in encrypted file {secret_file}")

        except Exception as e:
            logger.error(f"Encrypted file storage failed: {e}")
            raise RuntimeError(f"Failed to store secret '{key}': {e}")

    def retrieve_secret(self, key: str) -> Optional[str]:
        """
        Retrieve a secret.

        Args:
            key: Secret key/name

        Returns:
            Secret value or None if not found

        Raises:
            RuntimeError: If retrieval fails
        """
        if self.use_keyring:
            try:
                value = keyring.get_password(self.KEYRING_SERVICE_NAME, key)
                if value is not None:
                    logger.debug(f"Retrieved secret '{key}' from OS keyring")
                    return value
            except Exception as e:
                logger.error(f"Keyring retrieval failed: {e}")
                # Fall through to encrypted file fallback

        # Encrypted file fallback
        if not self.encryption_manager:
            return None

        try:
            secret_file = self.fallback_dir / f"{key}.enc"
            if not secret_file.exists():
                logger.debug(f"Secret '{key}' not found")
                return None

            with open(secret_file, "rb") as f:
                encrypted = f.read()

            value = self.encryption_manager.decrypt_to_string(encrypted)
            logger.debug(f"Retrieved secret '{key}' from encrypted file")
            return value

        except Exception as e:
            logger.error(f"Encrypted file retrieval failed: {e}")
            raise RuntimeError(f"Failed to retrieve secret '{key}': {e}")

    def delete_secret(self, key: str) -> None:
        """
        Delete a secret.

        Args:
            key: Secret key/name
        """
        if self.use_keyring:
            try:
                keyring.delete_password(self.KEYRING_SERVICE_NAME, key)
                logger.debug(f"Deleted secret '{key}' from OS keyring")
                return
            except Exception as e:
                logger.warning(f"Keyring deletion failed: {e}")

        # Encrypted file fallback
        if self.encryption_manager and self.fallback_dir:
            try:
                secret_file = self.fallback_dir / f"{key}.enc"
                if secret_file.exists():
                    secret_file.unlink()
                    logger.debug(f"Deleted secret '{key}' from encrypted file")
            except Exception as e:
                logger.error(f"Encrypted file deletion failed: {e}")

    def migrate_from_plaintext(
        self,
        old_file: Path,
        secret_key: str,
    ) -> None:
        """
        Migrate a plaintext secret file to secure storage.

        Args:
            old_file: Path to plaintext secret file
            secret_key: Key to store secret under

        Raises:
            RuntimeError: If migration fails
        """
        if not old_file.exists():
            logger.debug(f"No plaintext file to migrate: {old_file}")
            return

        try:
            # Read plaintext secret
            with open(old_file, "r") as f:
                plaintext_secret = f.read().strip()

            # Store securely
            self.store_secret(secret_key, plaintext_secret)

            # Delete plaintext file
            old_file.unlink()
            logger.info(f"Migrated secret from plaintext {old_file} to secure storage")

        except Exception as e:
            logger.error(f"Secret migration failed: {e}")
            raise RuntimeError(f"Failed to migrate secret from {old_file}: {e}")
