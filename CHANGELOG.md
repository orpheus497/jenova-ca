# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-10-01

### Added
- **Superior Intelligence**: Enhanced the Cortex to provide a more organized and developed cognitive graph.
- **Insight Development**: Added a `/develop_insight <node_id>` command to generate a more detailed and developed version of an existing insight.
- **Psychological Memory**: The Cortex now analyzes the sentiment of new nodes and adds it as metadata, providing a psychological dimension to the cognitive graph.
- **In-app Fine-tuning**: Added a `/finetune` command to trigger the fine-tuning process from within the application.
- **Cortex Architecture**: A new central hub for the AI's cognitive architecture that manages a graph of interconnected cognitive nodes (insights, memories, assumptions).
- **Deep Reflection**: The Cortex can perform deep reflection on the cognitive graph to find patterns, infer relationships, and generate meta-insights.
- **Proactive Engine**: A new engine that analyzes the cognitive graph to generate proactive suggestions and questions for the user.
- New Assumption System to generate, store, and verify assumptions about the user.
- Assumptions are categorized as `verified`, `unverified`, `true`, or `false`.
- New `/verify` command to initiate the assumption verification process.
- Re-introduced Meta-Insight Generation with the `/meta` command.

### Changed
- **Fleshed-out Proactivity**: The `ProactiveEngine` is now more sophisticated, analyzing clusters of insights to generate more meaningful and engaging suggestions.
- The `InsightManager` and `AssumptionManager` now use the `Cortex` to create and link insights and assumptions.
- The `CognitiveEngine` now orchestrates the `Cortex` and `ProactiveEngine`.
- Overhauled the insight system to be more organized and proactive.
- Insights are now organized by "concerns" or "topics".
- New insights are grouped with existing concerns to avoid duplication.
- The reflection system now reorganizes and interlinks insights.
- Memory insights are now integrated into the new concern-based system.

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
