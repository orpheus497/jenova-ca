# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2025-09-28

### Changed
- Refactored the insight and memory systems to be user-specific, storing data in user-dedicated directories.

### Fixed
- Fixed a bug that caused user input to be repeated in the UI.
- The AI now correctly recognizes and can use the current user's username in conversation.
- Corrected a crash in `UILogger` by replacing `error` method calls with `system_message`.
- Removed fixed-width constraint on UI output panels to prevent text truncation.
- The cognitive functions context wheel now displays a static "Thinking..." message.
- Increased the number of search results retrieved from each memory type to improve utilization.
- Fixed a bug where the application would not correctly recognize the current user, leading to impersonal and incorrect responses.
- Fixed a `TypeError` in the `/memory-insight` command by passing the required `username` argument to the `MemorySearch.search_all()` method.

## [1.1.0] - 2025-09-28

### Added
- `CHANGELOG.md` to track project changes.
- Command-based system for on-demand insight generation and reflection:
  - `/insight`: Develop insights from the current conversation.
  - `/reflect`: Reflect on all existing insights to create meta-insights.
  - `/memory-insight`: Generate insights from a broad search of long-term memory.
- `get_all_insights` method to `InsightManager` to retrieve all stored insights.

### Changed
- The insight generation system is now more proactive, triggering after every conversational turn instead of on a fixed interval.
- Refactored `CognitiveEngine` to remove periodic reflection and introduce public methods for command-driven insight generation.
- Updated `TerminalUI` to parse and handle the new command system.
- Updated `README.md` to document the new active insight engine and the available user commands.
- The insight generation system is now more reflective, triggering every 5 turns to reduce noise.
- Improved the insight generation system to be more robust by providing more conversational context and a more detailed prompt.
- Updated the `README.md` to accurately describe the reflective insight engine.
- Improved the intelligence of the insight commands (`/insight`, `/reflect`, `/memory-insight`) by providing more detailed and structured prompts.

## [1.0.0] - 2025-09-28

### Added
- Initial release of the Jenova Cognitive Architecture.
- Multi-layered memory system (Episodic, Semantic, Procedural).
- Dynamic Insight Engine for learning.
- Terminal UI.
