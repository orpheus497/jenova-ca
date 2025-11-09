# The JENOVA Cognitive Architecture - Access Control
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 30: Access Control - Permission and access management.

Manages user permissions, role-based access control, and resource
access policies for collaborative sessions.
"""

import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from jenova.collaboration.user_session import UserRole


class Permission(str, Enum):
    """Permission types."""
    # Session management
    SESSION_CREATE = "session_create"
    SESSION_DELETE = "session_delete"
    SESSION_CONFIGURE = "session_configure"

    # User management
    USER_INVITE = "user_invite"
    USER_REMOVE = "user_remove"
    USER_PROMOTE = "user_promote"
    USER_DEMOTE = "user_demote"

    # Content
    CONTENT_ADD = "content_add"
    CONTENT_EDIT = "content_edit"
    CONTENT_DELETE = "content_delete"
    CONTENT_VIEW = "content_view"

    # Collaboration
    TURN_REQUEST = "turn_request"
    TURN_RELEASE = "turn_release"

    # Moderation
    CONTRIBUTION_APPROVE = "contribution_approve"
    CONTRIBUTION_REJECT = "contribution_reject"

    # State
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"


# Default permissions per role
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.OWNER: {
        # All permissions
        Permission.SESSION_CREATE,
        Permission.SESSION_DELETE,
        Permission.SESSION_CONFIGURE,
        Permission.USER_INVITE,
        Permission.USER_REMOVE,
        Permission.USER_PROMOTE,
        Permission.USER_DEMOTE,
        Permission.CONTENT_ADD,
        Permission.CONTENT_EDIT,
        Permission.CONTENT_DELETE,
        Permission.CONTENT_VIEW,
        Permission.TURN_REQUEST,
        Permission.TURN_RELEASE,
        Permission.CONTRIBUTION_APPROVE,
        Permission.CONTRIBUTION_REJECT,
        Permission.STATE_READ,
        Permission.STATE_WRITE,
    },
    UserRole.ADMIN: {
        Permission.SESSION_CONFIGURE,
        Permission.USER_INVITE,
        Permission.USER_REMOVE,
        Permission.CONTENT_ADD,
        Permission.CONTENT_EDIT,
        Permission.CONTENT_DELETE,
        Permission.CONTENT_VIEW,
        Permission.TURN_REQUEST,
        Permission.TURN_RELEASE,
        Permission.CONTRIBUTION_APPROVE,
        Permission.CONTRIBUTION_REJECT,
        Permission.STATE_READ,
        Permission.STATE_WRITE,
    },
    UserRole.CONTRIBUTOR: {
        Permission.CONTENT_ADD,
        Permission.CONTENT_VIEW,
        Permission.TURN_REQUEST,
        Permission.TURN_RELEASE,
        Permission.STATE_READ,
    },
    UserRole.VIEWER: {
        Permission.CONTENT_VIEW,
        Permission.STATE_READ,
    },
}


@dataclass
class AccessPolicy:
    """
    Access policy for resource.

    Attributes:
        policy_id: Unique policy identifier
        resource_id: Resource this policy applies to
        allowed_roles: Roles with access
        denied_users: Specific users denied access
        allowed_users: Specific users allowed access (overrides role)
        metadata: Additional policy metadata
    """
    policy_id: str
    resource_id: str
    allowed_roles: Set[UserRole] = field(default_factory=set)
    denied_users: Set[str] = field(default_factory=set)
    allowed_users: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccessController:
    """
    Manage access control and permissions.

    Implements role-based access control with fine-grained permissions
    and resource-specific policies.

    Example:
        >>> controller = AccessController()
        >>> controller.check_permission(user_role, Permission.CONTENT_ADD)
        >>> controller.grant_access(policy_id, user_id)
    """

    def __init__(self):
        """Initialize access controller."""
        # Resource policies (resource_id -> AccessPolicy)
        self.policies: Dict[str, AccessPolicy] = {}

        # Custom role permissions (role -> set of permissions)
        self.custom_permissions: Dict[UserRole, Set[Permission]] = {}

    def check_permission(
        self,
        user_role: UserRole,
        permission: Permission
    ) -> bool:
        """
        Check if role has permission.

        Args:
            user_role: User's role
            permission: Permission to check

        Returns:
            True if permission granted

        Example:
            >>> has_perm = controller.check_permission(UserRole.CONTRIBUTOR, Permission.CONTENT_ADD)
        """
        # Check custom permissions first
        if user_role in self.custom_permissions:
            return permission in self.custom_permissions[user_role]

        # Check default permissions
        return permission in ROLE_PERMISSIONS.get(user_role, set())

    def check_access(
        self,
        resource_id: str,
        user_id: str,
        user_role: UserRole
    ) -> bool:
        """
        Check if user has access to resource.

        Args:
            resource_id: Resource identifier
            user_id: User identifier
            user_role: User's role

        Returns:
            True if access granted

        Example:
            >>> has_access = controller.check_access("session_123", "alice", UserRole.CONTRIBUTOR)
        """
        policy = self.policies.get(resource_id)
        if not policy:
            # No policy = default role-based access
            return True

        # Check if user is explicitly denied
        if user_id in policy.denied_users:
            return False

        # Check if user is explicitly allowed
        if user_id in policy.allowed_users:
            return True

        # Check role-based access
        return user_role in policy.allowed_roles

    def create_policy(
        self,
        resource_id: str,
        allowed_roles: Optional[Set[UserRole]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AccessPolicy:
        """
        Create access policy for resource.

        Args:
            resource_id: Resource identifier
            allowed_roles: Roles with access
            metadata: Additional metadata

        Returns:
            Created AccessPolicy

        Example:
            >>> policy = controller.create_policy("session_123", {UserRole.OWNER, UserRole.ADMIN})
        """
        policy_id = f"policy_{resource_id}"

        policy = AccessPolicy(
            policy_id=policy_id,
            resource_id=resource_id,
            allowed_roles=allowed_roles or set(),
            metadata=metadata or {}
        )

        self.policies[resource_id] = policy
        return policy

    def grant_access(
        self,
        resource_id: str,
        user_id: str
    ) -> bool:
        """
        Grant user access to resource.

        Args:
            resource_id: Resource identifier
            user_id: User identifier

        Returns:
            True if granted

        Example:
            >>> controller.grant_access("session_123", "alice")
        """
        policy = self.policies.get(resource_id)
        if not policy:
            # Create policy if doesn't exist
            policy = self.create_policy(resource_id)

        policy.allowed_users.add(user_id)
        return True

    def revoke_access(
        self,
        resource_id: str,
        user_id: str
    ) -> bool:
        """
        Revoke user access to resource.

        Args:
            resource_id: Resource identifier
            user_id: User identifier

        Returns:
            True if revoked

        Example:
            >>> controller.revoke_access("session_123", "bob")
        """
        policy = self.policies.get(resource_id)
        if not policy:
            return False

        policy.allowed_users.discard(user_id)
        policy.denied_users.add(user_id)
        return True

    def add_role_to_policy(
        self,
        resource_id: str,
        role: UserRole
    ) -> bool:
        """
        Add role to resource policy.

        Args:
            resource_id: Resource identifier
            role: User role

        Returns:
            True if added

        Example:
            >>> controller.add_role_to_policy("session_123", UserRole.CONTRIBUTOR)
        """
        policy = self.policies.get(resource_id)
        if not policy:
            policy = self.create_policy(resource_id)

        policy.allowed_roles.add(role)
        return True

    def remove_role_from_policy(
        self,
        resource_id: str,
        role: UserRole
    ) -> bool:
        """
        Remove role from resource policy.

        Args:
            resource_id: Resource identifier
            role: User role

        Returns:
            True if removed

        Example:
            >>> controller.remove_role_from_policy("session_123", UserRole.VIEWER)
        """
        policy = self.policies.get(resource_id)
        if not policy:
            return False

        policy.allowed_roles.discard(role)
        return True

    def grant_permission_to_role(
        self,
        role: UserRole,
        permission: Permission
    ) -> None:
        """
        Grant custom permission to role.

        Args:
            role: User role
            permission: Permission to grant

        Example:
            >>> controller.grant_permission_to_role(UserRole.VIEWER, Permission.CONTENT_ADD)
        """
        if role not in self.custom_permissions:
            # Start with default permissions
            self.custom_permissions[role] = ROLE_PERMISSIONS.get(role, set()).copy()

        self.custom_permissions[role].add(permission)

    def revoke_permission_from_role(
        self,
        role: UserRole,
        permission: Permission
    ) -> None:
        """
        Revoke permission from role.

        Args:
            role: User role
            permission: Permission to revoke

        Example:
            >>> controller.revoke_permission_from_role(UserRole.CONTRIBUTOR, Permission.CONTENT_DELETE)
        """
        if role not in self.custom_permissions:
            self.custom_permissions[role] = ROLE_PERMISSIONS.get(role, set()).copy()

        self.custom_permissions[role].discard(permission)

    def get_role_permissions(self, role: UserRole) -> Set[Permission]:
        """
        Get all permissions for role.

        Args:
            role: User role

        Returns:
            Set of permissions

        Example:
            >>> perms = controller.get_role_permissions(UserRole.ADMIN)
        """
        if role in self.custom_permissions:
            return self.custom_permissions[role].copy()
        return ROLE_PERMISSIONS.get(role, set()).copy()

    def get_policy(self, resource_id: str) -> Optional[AccessPolicy]:
        """
        Get access policy for resource.

        Args:
            resource_id: Resource identifier

        Returns:
            AccessPolicy or None

        Example:
            >>> policy = controller.get_policy("session_123")
        """
        return self.policies.get(resource_id)

    def delete_policy(self, resource_id: str) -> bool:
        """
        Delete access policy.

        Args:
            resource_id: Resource identifier

        Returns:
            True if deleted

        Example:
            >>> controller.delete_policy("session_123")
        """
        if resource_id in self.policies:
            del self.policies[resource_id]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get access control statistics.

        Returns:
            Statistics dictionary

        Example:
            >>> stats = controller.get_stats()
        """
        total_allowed_users = sum(
            len(p.allowed_users) for p in self.policies.values()
        )
        total_denied_users = sum(
            len(p.denied_users) for p in self.policies.values()
        )

        return {
            "total_policies": len(self.policies),
            "total_allowed_users": total_allowed_users,
            "total_denied_users": total_denied_users,
            "custom_role_permissions": len(self.custom_permissions),
        }
