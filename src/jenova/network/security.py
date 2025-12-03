# The JENOVA Cognitive Architecture - Network Security
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Security layer for distributed JENOVA.

Provides certificate-based peer authentication and encrypted communication
channels for secure LAN networking.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import jwt
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from jenova.network.security_store import SecureCredentialStore


class SecurityManager:
    """
    Manages security for distributed JENOVA communication.

    Features:
    - Self-signed certificate generation for peers
    - JWT token-based authentication
    - Certificate validation
    - Encrypted channel establishment
    """

    def __init__(self, config: dict, file_logger, cert_dir: Optional[str] = None):
        """
        Initialize security manager.

        Args:
            config: JENOVA configuration
            file_logger: Logger for file output
            cert_dir: Directory for certificate storage (default: ~/.jenova-ai/certs)
        """
        self.config = config
        self.file_logger = file_logger

        # Certificate directory
        if cert_dir is None:
            cert_dir = os.path.join(os.path.expanduser("~"), ".jenova-ai", "certs")
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.private_key_path = self.cert_dir / "jenova.key"
        self.certificate_path = self.cert_dir / "jenova.crt"

        # Secure credential store for encrypted key management
        self.credential_store = SecureCredentialStore(self.cert_dir, file_logger)

        # JWT secret (generated on first run)
        self.jwt_secret_path = self.cert_dir / "jwt_secret"
        self.jwt_secret = self._load_or_generate_jwt_secret()

        # Security settings
        security_config = config.get("network", {}).get("security", {})
        self.security_enabled = security_config.get("enabled", True)
        self.require_auth = security_config.get("require_auth", True)
        self.cert_validity_days = 365

        # Trusted peer certificates (for certificate pinning)
        self.trusted_peers: dict = {}  # peer_id -> certificate_fingerprint

    def ensure_certificates(self, instance_name: str) -> Tuple[str, str]:
        """
        Ensure SSL certificates exist, generating them if necessary.

        Args:
            instance_name: Name for the certificate

        Returns:
            Tuple of (private_key_path, certificate_path)
        """
        try:
            if self.private_key_path.exists() and self.certificate_path.exists():
                self.file_logger.log_info("Using existing SSL certificates")
                return str(self.private_key_path), str(self.certificate_path)

            self.file_logger.log_info("Generating new SSL certificates...")
            self._generate_self_signed_cert(instance_name)
            self.file_logger.log_info("SSL certificates generated successfully")

            return str(self.private_key_path), str(self.certificate_path)

        except Exception as e:
            self.file_logger.log_error(f"Failed to ensure certificates: {e}")
            raise

    def _generate_self_signed_cert(self, instance_name: str):
        """
        Generate a self-signed certificate for this instance.

        Args:
            instance_name: Name to include in certificate
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        # Save private key with encryption using secure credential store
        self.credential_store.save_private_key(private_key, filename="jenova.key")

        # Generate certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "AI"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "LAN"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "JENOVA"),
                x509.NameAttribute(NameOID.COMMON_NAME, instance_name),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(
                datetime.now(timezone.utc) + timedelta(days=self.cert_validity_days)
            )
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.DNSName("*.local"),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256(), default_backend())
        )

        # Write certificate to file
        with open(self.certificate_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        self.file_logger.log_info(
            f"Generated self-signed certificate valid for {self.cert_validity_days} days"
        )

    def _load_or_generate_jwt_secret(self) -> str:
        """Load or generate JWT secret key."""
        if self.jwt_secret_path.exists():
            with open(self.jwt_secret_path, "r") as f:
                return f.read().strip()
        else:
            # Generate new secret
            import secrets

            secret = secrets.token_urlsafe(32)
            with open(self.jwt_secret_path, "w") as f:
                f.write(secret)
            os.chmod(self.jwt_secret_path, 0o600)
            return secret

    def create_auth_token(
        self, instance_id: str, instance_name: str, validity_seconds: int = 3600
    ) -> str:
        """
        Create a JWT authentication token.

        Args:
            instance_id: Unique instance ID
            instance_name: Instance name
            validity_seconds: Token validity duration

        Returns:
            JWT token string
        """
        if not self.require_auth:
            return ""

        payload = {
            "instance_id": instance_id,
            "instance_name": instance_name,
            "issued_at": int(time.time()),
            "expires_at": int(time.time()) + validity_seconds,
            "version": "5.0.0",
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token

    def verify_auth_token(self, token: str) -> Optional[dict]:
        """
        Verify a JWT authentication token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded payload if valid, None otherwise
        """
        if not self.require_auth:
            return {"instance_id": "anonymous", "instance_name": "anonymous"}

        if not token:
            self.file_logger.log_warning(
                "Authentication required but no token provided"
            )
            return None

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            # Check expiration
            if payload.get("expires_at", 0) < int(time.time()):
                self.file_logger.log_warning("Token expired")
                return None

            return payload

        except jwt.InvalidTokenError as e:
            self.file_logger.log_warning(f"Invalid token: {e}")
            return None

    def validate_peer_certificate(self, cert_path: str) -> bool:
        """
        Validate a peer's certificate.

        Args:
            cert_path: Path to peer's certificate

        Returns:
            True if valid, False otherwise
        """
        if not self.security_enabled:
            return True

        try:
            with open(cert_path, "rb") as f:
                cert_data = f.read()

            cert = x509.load_pem_x509_certificate(cert_data, default_backend())

            # Check if certificate is expired
            now = datetime.now(timezone.utc)
            if now < cert.not_valid_before or now > cert.not_valid_after:
                self.file_logger.log_warning(
                    f"Certificate expired or not yet valid: {cert_path}"
                )
                return False

            # Additional validation could be added here
            # (e.g., certificate chain validation, CRL checking)

            return True

        except Exception as e:
            self.file_logger.log_error(f"Failed to validate certificate: {e}")
            return False

    def get_ssl_credentials(self):
        """
        Get SSL credentials for gRPC.

        Returns:
            Tuple of (private_key, certificate_chain) as bytes
        """
        try:
            # Read the encrypted private key
            with open(self.private_key_path, "rb") as f:
                encrypted_key_pem = f.read()

            # Get the password and decrypt the private key
            password = self.credential_store.get_master_password(prompt=False)
            
            # Load and decrypt the private key
            private_key_obj = serialization.load_pem_private_key(
                encrypted_key_pem,
                password=password,
                backend=default_backend()
            )
            
            # Serialize the private key without encryption for gRPC
            private_key = private_key_obj.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )

            with open(self.certificate_path, "rb") as f:
                certificate_chain = f.read()

            return private_key, certificate_chain

        except Exception as e:
            self.file_logger.log_error(f"Failed to load SSL credentials: {e}")
            raise

    def is_security_enabled(self) -> bool:
        """Check if security is enabled."""
        return self.security_enabled

    def is_auth_required(self) -> bool:
        """Check if authentication is required."""
        return self.require_auth

    def get_status(self) -> dict:
        """Get security manager status."""
        return {
            "security_enabled": self.security_enabled,
            "auth_required": self.require_auth,
            "cert_dir": str(self.cert_dir),
            "certificates_exist": (
                self.private_key_path.exists() and self.certificate_path.exists()
            ),
            "cert_validity_days": self.cert_validity_days,
            "trusted_peers_count": len(self.trusted_peers),
        }

    def get_certificate_fingerprint(self, cert_path: str) -> Optional[str]:
        """
        Get SHA256 fingerprint of a certificate.

        Args:
            cert_path: Path to certificate file

        Returns:
            Hex-encoded fingerprint or None on error
        """
        try:
            with open(cert_path, "rb") as f:
                cert_data = f.read()

            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            fingerprint = cert.fingerprint(hashes.SHA256())
            return fingerprint.hex()

        except Exception as e:
            self.file_logger.log_error(f"Failed to get certificate fingerprint: {e}")
            return None

    def pin_peer_certificate(self, peer_id: str, cert_path: str) -> bool:
        """
        Pin a peer's certificate for future validation (certificate pinning).

        Args:
            peer_id: Unique peer identifier
            cert_path: Path to peer's certificate

        Returns:
            True if pinning succeeded, False otherwise
        """
        fingerprint = self.get_certificate_fingerprint(cert_path)
        if fingerprint:
            self.trusted_peers[peer_id] = fingerprint
            self.file_logger.log_info(
                f"Pinned certificate for peer {peer_id}: {fingerprint[:16]}..."
            )
            return True
        return False

    def validate_pinned_peer(self, peer_id: str, cert_path: str) -> bool:
        """
        Validate a peer's certificate against pinned fingerprint.

        Args:
            peer_id: Unique peer identifier
            cert_path: Path to peer's certificate

        Returns:
            True if certificate matches pinned fingerprint, False otherwise
        """
        if peer_id not in self.trusted_peers:
            self.file_logger.log_warning(
                f"No pinned certificate for peer {peer_id}, trusting on first use"
            )
            # Trust on first use (TOFU)
            return self.pin_peer_certificate(peer_id, cert_path)

        current_fingerprint = self.get_certificate_fingerprint(cert_path)
        if current_fingerprint is None:
            return False

        pinned_fingerprint = self.trusted_peers[peer_id]

        # SECURITY FIX (Phase 21): Use constant-time comparison to prevent timing attacks
        # The previous implementation used '==' which is vulnerable to timing attacks.
        # An attacker could measure comparison time to deduce information about the
        # expected fingerprint. hmac.compare_digest() performs constant-time comparison
        # to prevent this attack vector.
        import hmac
        if hmac.compare_digest(current_fingerprint, pinned_fingerprint):
            return True
        else:
            self.file_logger.log_error(
                f"Certificate fingerprint mismatch for peer {peer_id}! "
                f"Expected {pinned_fingerprint[:16]}..., "
                f"got {current_fingerprint[:16]}..."
            )
            return False

    def unpin_peer_certificate(self, peer_id: str):
        """
        Remove pinned certificate for a peer.

        Args:
            peer_id: Unique peer identifier
        """
        if peer_id in self.trusted_peers:
            del self.trusted_peers[peer_id]
            self.file_logger.log_info(f"Unpinned certificate for peer {peer_id}")

    def get_ssl_context(self):
        """
        Get an SSL context configured with the server's certificates.
        
        Returns:
            ssl.SSLContext configured for gRPC server use, or None if security is disabled
        """
        if not self.security_enabled:
            return None
        
        import ssl
        
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            # Get the master password for encrypted private key
            password = self.credential_store.get_master_password(prompt=False)
            ssl_context.load_cert_chain(
                certfile=str(self.certificate_path),
                keyfile=str(self.private_key_path),
                password=password.decode('utf-8') if isinstance(password, bytes) else password,
            )
            return ssl_context
        except Exception as e:
            self.file_logger.log_error(f"Failed to create SSL context: {e}")
            return None

    def close(self):
        """Clean up security resources."""
        # Clear password cache on shutdown
        self.credential_store.clear_password_cache()
