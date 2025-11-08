# The JENOVA Cognitive Architecture - User Module
# Copyright (c) 2024, orpheus497. All rights reserved.

"""
Phase 10: User recognition, profiling, and personalization.

This module provides:
- User profile management
- Preference learning
- Interaction history
- Personalization engine
"""

from jenova.user.profile import UserProfile, UserProfileManager
from jenova.user.personalization import PersonalizationEngine

__all__ = ['UserProfile', 'UserProfileManager', 'PersonalizationEngine']
