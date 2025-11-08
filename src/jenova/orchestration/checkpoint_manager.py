"""
JENOVA Cognitive Architecture - Checkpoint Manager Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides checkpoint save/restore functionality with file locking.
"""

import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from filelock import FileLock, Timeout


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for conversation and task state.

    Features:
    - Atomic save operations with file locking
    - Multiple checkpoint versions
    - Automatic backup rotation
    - Compression support
    - Metadata tracking
    """

    def __init__(
        self,
        checkpoint_dir: str = ".jenova/checkpoints",
        max_checkpoints: int = 10,
        use_compression: bool = False
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            use_compression: Whether to compress checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.use_compression = use_compression

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Lock file for atomic operations
        self.lock_file = self.checkpoint_dir / ".checkpoint.lock"

    def save(
        self,
        data: Dict[str, Any],
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.

        Args:
            data: Data to save
            checkpoint_id: Optional checkpoint ID (auto-generated if not provided)
            metadata: Optional metadata to include

        Returns:
            Checkpoint ID

        Raises:
            IOError: If save fails
        """
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        lock = FileLock(str(self.lock_file), timeout=10)

        try:
            with lock:
                # Prepare checkpoint data
                checkpoint_data = {
                    "id": checkpoint_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": data,
                    "metadata": metadata or {}
                }

                # Save to temporary file first
                temp_path = checkpoint_path.with_suffix('.tmp')

                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, default=str)

                # Atomic rename
                shutil.move(str(temp_path), str(checkpoint_path))

                logger.info(f"Saved checkpoint: {checkpoint_id}")

                # Clean up old checkpoints
                self._cleanup_old_checkpoints()

                return checkpoint_id

        except Timeout:
            logger.error(f"Timeout acquiring lock for checkpoint {checkpoint_id}")
            raise IOError(f"Failed to acquire lock for checkpoint {checkpoint_id}")
        except Exception as e:
            logger.error(f"Error saving checkpoint {checkpoint_id}: {e}")
            raise IOError(f"Failed to save checkpoint: {e}")

    def load(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to load

        Returns:
            Checkpoint data

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            IOError: If load fails
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        lock = FileLock(str(self.lock_file), timeout=10)

        try:
            with lock:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

                logger.info(f"Loaded checkpoint: {checkpoint_id}")
                return checkpoint_data["data"]

        except Timeout:
            logger.error(f"Timeout acquiring lock for checkpoint {checkpoint_id}")
            raise IOError(f"Failed to acquire lock for checkpoint {checkpoint_id}")
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            raise IOError(f"Failed to load checkpoint: {e}")

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
            Checkpoint data or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.info("No checkpoints found")
            return None

        latest = checkpoints[0]
        return self.load(latest["id"])

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint metadata dictionaries sorted by timestamp (newest first)
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                checkpoints.append({
                    "id": data["id"],
                    "timestamp": data["timestamp"],
                    "metadata": data.get("metadata", {}),
                    "size": checkpoint_file.stat().st_size
                })
            except Exception as e:
                logger.warning(f"Error reading checkpoint {checkpoint_file}: {e}")

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

        return checkpoints

    def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            return False

        lock = FileLock(str(self.lock_file), timeout=10)

        try:
            with lock:
                checkpoint_path.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return True
        except Timeout:
            logger.error(f"Timeout acquiring lock for deleting {checkpoint_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            # Delete oldest checkpoints
            for checkpoint in checkpoints[self.max_checkpoints:]:
                self.delete(checkpoint["id"])
                logger.info(f"Cleaned up old checkpoint: {checkpoint['id']}")

    def export_checkpoint(self, checkpoint_id: str, export_path: str) -> bool:
        """
        Export a checkpoint to a different location.

        Args:
            checkpoint_id: Checkpoint ID to export
            export_path: Path to export to

        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.load(checkpoint_id)
            export_path_obj = Path(export_path)
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting checkpoint {checkpoint_id}: {e}")
            return False

    def import_checkpoint(
        self,
        import_path: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Import a checkpoint from a file.

        Args:
            import_path: Path to import from
            checkpoint_id: Optional checkpoint ID (auto-generated if not provided)

        Returns:
            Checkpoint ID if successful, None otherwise
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            checkpoint_id = self.save(data, checkpoint_id=checkpoint_id)
            logger.info(f"Imported checkpoint from {import_path} as {checkpoint_id}")
            return checkpoint_id
        except Exception as e:
            logger.error(f"Error importing checkpoint from {import_path}: {e}")
            return None

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint without loading it.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint info or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                "id": data["id"],
                "timestamp": data["timestamp"],
                "metadata": data.get("metadata", {}),
                "size": checkpoint_path.stat().st_size
            }
        except Exception as e:
            logger.error(f"Error reading checkpoint info {checkpoint_id}: {e}")
            return None

    def clear_all(self) -> int:
        """
        Delete all checkpoints.

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()
        count = 0

        for checkpoint in checkpoints:
            if self.delete(checkpoint["id"]):
                count += 1

        logger.info(f"Cleared {count} checkpoints")
        return count

    def create_backup(self, checkpoint_id: str) -> Optional[str]:
        """
        Create a backup of a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to backup

        Returns:
            Backup checkpoint ID or None if failed
        """
        try:
            data = self.load(checkpoint_id)
            backup_id = f"{checkpoint_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return self.save(data, checkpoint_id=backup_id, metadata={"backup_of": checkpoint_id})
        except Exception as e:
            logger.error(f"Error creating backup of {checkpoint_id}: {e}")
            return None
