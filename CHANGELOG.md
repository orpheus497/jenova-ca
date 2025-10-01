# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2025-10-02

### Added
- **RAG System:** A new Retrieval-Augmented Generation (RAG) system is now a core component of the cognitive architecture.
- **Document Processor:** A new system that allows the AI to scan and process documents in the `docs` folder to generate new cognitive nodes.

### Changed
- **Cognitive Engine:** The engine now prioritizes the AI's own insights and memories over its general knowledge.
- **Insight System:** The system is now more comprehensive, with insights being interlinked with other cognitive nodes in the Cortex.
- **Reflect System:** The reflection process is now more sophisticated, using graph analysis to find patterns, and is triggered automatically during the cognitive cycle.
- **Memory System:** The memory system is now more comprehensive and includes an emotional component for more intelligent responses.
- **Assumption System:** The assumption system is more robust and intelligent, using the LLM to resolve assumptions proactively during conversation.
- **Proactive Engine:** The proactive engine is more sophisticated, using graph analysis to find underdeveloped areas of the cognitive graph and trigger suggestions more frequently.
- **RAG.md Dependency:** Removed the hardcoded dependency on the `RAG.md` file, as this is now handled by the core RAG system.

### Fixed
- **Commands:** The `/develop_insight` and `/finetune` commands are now working properly and are more robust.
- **Error Handling:** Improved error handling across the application to prevent crashes and provide better feedback to the user.

## [1.3.0] - 2025-10-02

### Fixed
- **Command System:**
  - `/meta`: The command now provides feedback to the user when a new meta-insight is generated.
  - `/develop_insight`: The command now correctly parses the `node_id` and provides a usage message if it's missing.
  - `/verify`: The command is now fully interactive, allowing users to confirm or deny assumptions in real-time.
  - `/finetune`: The command now triggers a real fine-tuning process, including data preparation and a `llama.cpp`-based fine-tuning command.
- **Insight Organization:** The `/reflect` command now properly cleans up old, empty topic folders after reorganizing insights.

### Changed
- **Fine-tuning:** The fine-tuning data preparation script (`finetune/prepare_data.py`) now creates a more structured and effective training file.
- **Configuration:** The `main_config.yaml` has been updated to support the new fine-tuning process.

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
