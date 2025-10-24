# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GGUF model support via llama-cpp-python for flexible model selection and better performance
- Virtualenv-based local installation system for better isolation and user control
- Models directory structure with README for GGUF model downloads and configuration
- GPU detection in install script with automatic CUDA support build for llama-cpp-python
- Document canonical storage as knowledge nodes without automatic insight generation
- Comprehensive fine-tuning data generation from complete cognitive architecture (insights, memories, assumptions, documents)

### Changed
- Dependencies: Replaced `transformers`, `accelerate`, `peft`, `bitsandbytes` with `llama-cpp-python`; kept `sentence-transformers`, `chromadb`, `torch`, `rich`, `prompt-toolkit`, `pyyaml`, `numpy`, `selenium`, `webdriver-manager`, `requests`
- Configuration: Added GGUF-specific options in `main_config.yaml`: `model_path`, `threads`, `gpu_layers`, `mlock`, `n_batch`; removed automatic context_size detection
- Installation: Virtualenv-based local installation without sudo; no automatic model downloads; users provide GGUF models; includes GPU driver guidance
- LLM Interface: Complete rewrite using llama-cpp-python `Llama` class; removed tokenizer dependencies; simplified generation API with `create_completion()`
- Document Processing: Documents stored as 'document' and 'document_chunk' nodes in cognitive graph; no automatic insight generation; documents chunked for RAG retrieval; full content preserved as canonical knowledge
- Cortex: Removed `grammar` parameter from LLM generation calls (not supported by llama-cpp-python); enhanced JSON extraction with fallback parsing
- Fine-Tuning: Updated to generate comprehensive training data from all cognitive sources (insights, episodic/semantic/procedural memory, assumptions, documents); creates .jsonl for use with external tools

### Removed
- System-wide installation: No longer installs globally or creates `/usr/local/share/jenova-ai/`
- Automatic model downloads: Install script no longer downloads models from HuggingFace
- HuggingFace dependencies: Removed transformers, accelerate packages
- Grammar system: Removed grammar-constrained generation (incompatible with llama-cpp-python)
- Auto-insight generation: Documents no longer automatically analyzed for insights during processing

### Fixed
- Model loading: Clear error messages when GGUF model file is missing
- GPU support: Proper detection and configuration for NVIDIA GPU acceleration
- Resource cleanup: Improved LLM resource cleanup on shutdown

### Security
- Local-only operation: No external API calls or automatic downloads during installation
- User control: Users explicitly provide and control their GGUF models
- Virtualenv isolation: Better dependency isolation and security through virtualenv

---

## [3.1.1] - 2025-10-19

### Added
- **Uninstall Script:** Added a comprehensive `uninstall.sh` script to allow administrators to easily remove the application, user data, and downloaded models from the system.

### Fixed
- **AI Behavior:** Overhauled prompt engineering and increased the repetition penalty to improve the quality, coherence, and reduce repetitiveness of the AI's responses.
- **Context Window:** Fixed an `OverflowError` during tokenization and now correctly cap the context size to a reasonable value to prevent crashes and improve performance.
- **Stability:** Resolved several startup and runtime errors, including `SyntaxError`, `TypeError`, and deprecated argument usage.
- **Code Quality:** Addressed multiple issues reported by `pylint`, removing unused code, arguments, and fixing warnings throughout the codebase.
## [3.1.0] - 2025-10-18

### Changed
- **Complete Architecture Migration:** Migrated the core architecture from `llama-cpp-python` (GGUF models) to the HuggingFace `transformers` library.
- **Model Management:** The installation process (`install.sh`) now automatically downloads the `TinyLlama-1.1B` model to a system-wide `/usr/local/share/jenova-ai/models` directory.
- **Dependencies:** Replaced `llama-cpp-python` with `transformers` and added `accelerate` for better GPU support.
- **Configuration:** Simplified hardware configuration in `main_config.yaml`, removing GGUF-specific settings (`threads`, `gpu_layers`, `mlock`).
- **Hardware Utilization:** The model loader now automatically detects and utilizes CUDA GPUs when available, with a fallback to CPU.
- **Grammar System:** Removed the dependency on `llama.cpp`'s JSON grammar system.

### Fixed
- **Critical Regression from Threading Fix:** Fixed multiple critical issues introduced by the v3.0.3 threading fix that broke core functionality, including message queue initialization, missing UI spinners, and broken RAG system logger references.
- **Memory Stability:** Fixed a `TypeError` in all memory modules that caused crashes when NLP analysis failed to extract metadata. A new `sanitize_metadata` utility now prevents `None` values from being passed to the database.
- **Resource Management:** Improved GPU memory cleanup by calling `torch.cuda.empty_cache()`.

## [3.0.3] - 2025-10-17

### Fixed
- **Threading Deadlock:** Resolved a critical deadlock in the cognitive cycle caused by threading conflicts between background cognitive processes and UI updates. Implemented a thread-safe `queue.Queue` message bus in `TerminalUI` for asynchronous UI updates. Modified `UILogger` to be non-blocking by queuing status updates and log messages instead of directly manipulating the UI. Refactored `TerminalUI` to process queued messages on the main thread while cognitive operations run in background threads, eliminating circular wait conditions.
- **Missing PyTorch Dependency:** Added `torch` to both `requirements.txt` and `pyproject.toml` dependencies to fix import errors in `model_loader.py` which uses PyTorch for GPU detection when loading embedding models.

## [3.0.2] - 2025-10-15

### Fixed
- **Critical Race Condition:** Fixed a critical race condition on multi-core systems where background cognitive tasks and the main UI loop competed for console control, causing the "Only one live display may be active at once" error. Implemented thread-safe console locking using `threading.Lock` in the `UILogger` class to ensure exclusive access to console operations.
- **UI Thread Safety:** Refactored all console access points in `UILogger` to use exclusive locking, including `banner()`, `info()`, `system_message()`, `help_message()`, `reflection()`, `cognitive_process()`, `thinking_process()`, `user_query()`, and `jenova_response()`.
- **Spinner Thread Safety:** Updated the `TerminalUI` spinner to respect the console lock, preventing conflicts with Rich's live display system during concurrent operations.

### Changed
- **Branding Update:** Updated all references throughout the codebase and documentation to refer to the AI as "JENOVA" (instead of "Jenova AI") and the engine as "The JENOVA Cognitive Architecture" for consistent branding and identity.
  - Updated `persona.yaml` identity configuration
  - Updated `setup.py` description and version
  - Updated `README.md` throughout
  - Updated `terminal.py` prompts and messages
  - Updated `logger.py` panel titles and display names
  - Updated `main.py` docstrings and messages
  - Updated `finetune/README.md`
- **Enhanced /help Command:** Completely redesigned the `/help` command display to be more visually appealing and user-friendly with:
  - Structured sections with decorative borders (Cognitive Commands, Learning Commands, System Commands, Innate Capabilities)
  - Clear command syntax highlighting in bright yellow
  - Detailed descriptions in lavender with usage examples
  - Italic dim text for additional context and explanations
  - Visual separators between sections
  - Helpful tips section at the bottom

## [3.0.1] - 2025-10-11

### Fixed
- **Startup Crash (Model Load):** Fixed a critical `TypeError` that occurred during startup when reading model metadata by correctly instantiating the `Llama` object without a `with` statement.
- **Startup Crash (No Models Found):** Fixed an `AttributeError` that occurred if no GGUF models were found by using the correct logger method.
- **Shutdown Crash:** Fixed an `AttributeError` that occurred on exit by removing an incorrect call to `self.model.close()`.
- **Database Incompatibility:** Resolved a `sqlite3.OperationalError` by removing outdated ChromaDB database files after a schema change in the library.
- **Installation Conflict:** The `install.sh` script now uses `pip install --ignore-installed` to prevent conflicts with system-managed packages.
- **Module Not Found Error:** Fixed a `ModuleNotFoundError` for the `jenova.assumptions` package by adding a missing `__init__.py` file.

## [3.0.0] - 2025-10-11

### Security
- **Remote Code Execution:** Patched a critical RCE vulnerability in the tool handler by replacing the unsafe `eval()` with `ast.literal_eval()` for parsing tool arguments.
- **Shell Injection:** Hardened the `SystemTools` and fine-tuning commands against shell injection vulnerabilities.
- **Path Traversal:** Corrected a path traversal vulnerability in `FileTools` by fixing the sandbox validation logic.

### Changed
- **System-Wide Installation:** Overhauled the installation process for multi-user, system-wide deployment. The `install.sh` script now installs the package globally, making the `jenova` command available to all system users.
- **Project Cleanup:** Performed a major cleanup of the repository, removing the large `llama.cpp/` source directory and other development artifacts in favor of the `llama-cpp-python` dependency.
- **Fine-tuning Process:** Redesigned the fine-tuning workflow into a single, modular `finetune/train.py` script that generates a training dataset from user insights.
- **Tool System:** Removed dysfunctional tools (`web_search`, `weather_search`) and refactored the tool handling system to be more modular and extensible under a central `ToolHandler`.
- **Web Search:** Replaced the previous web search library with a more powerful, `selenium`-based implementation. The AI can now access and process the full content of web pages for richer information gathering.
- **Documentation:** The `README.md` has been completely rewritten to reflect the new system-wide installation model and current features.
- **Performance:** Optimized the memory pre-loading process by using threads to load collections in parallel.
- **UI:** The terminal UI has been updated to reflect the new command system. The `/finetune` and `/search` commands have been removed, and a new `/train` command has been added. The `/help` command is updated to reflect the current tools and commands.
- **`.gitignore`:** The `.gitignore` file has been updated to be more comprehensive.

### Fixed
- **Critical Shutdown Error:** Fixed a `TypeError: 'NoneType' object is not callable` on exit by ensuring all `llama-cpp-python` model resources are explicitly closed.
- **UI/Engine Stability:** Implemented multi-layered defenses against the persistent `TypeError: string indices must be integers, not 'str'`, hardening the UI, cognitive engine, and memory systems.
- **Startup Crash:** Fixed a `TypeError` during `SemanticMemory` initialization related to `chromadb` embedding functions and implemented a self-healing mechanism to handle collection conflicts and prevent data loss.
- **Cognitive Degradation:** Overhauled the `Cortex` and `CognitiveEngine` to fix a cascading failure in the AI's intelligence, reinforcing the AI's persona, stabilizing memory generation, and making cognition more reliable with `gbnf` grammars.
- **Stability & Correctness:** Fixed numerous critical bugs across the application, including `NameError` in `ProactiveEngine` and `AssumptionManager`, `UnboundLocalError` in the `think` method, `AttributeError` in the web search tool, and other `TypeError` issues to improve overall stability.
- **Tool Handling:** Fixed a critical bug in the `CognitiveEngine` where it was attempting to call a non-existent `tool_handler` object.


## [2.1.0] - 2025-10-06

### Added
- **Weather Tool:** Added a new `WeatherTool` that can fetch real-time weather information for a given location using the free `wttr.in` service, removing the need for an API key.
- **File Sandbox:** Implemented a secure file sandbox for the `FileTools`. All file operations are now restricted to a configurable directory (`~/jenova_files` by default).
- **Memory Pre-loading:** Added a `preload_memories` option to `main_config.yaml` to allow pre-loading all memories into RAM at startup for faster response times.
- **Dependency:** Added `selenium` and `webdriver-manager` to `requirements.txt` to support the new browser-based web search functionality.
- **Weather Tool:** Added a new `WeatherTool` that can fetch real-time weather information for a given location using the OpenWeatherMap API. The AI can now be asked about the weather and will use this tool to provide an answer.
- **Configuration:** Added a new `apis` section to `main_config.yaml` to store API keys for external services, starting with `openweathermap_api_key`.
- **Dependency:** Added `selenium` and `webdriver-manager` to `requirements.txt` to support the new browser-based web search functionality.
- **Insight System Logging:** Added detailed logging to the insight management and memory search systems to improve observability and aid in debugging the AI's cognitive functions.
- **UI:** A `/help` command has been added to the terminal UI to provide users with a clear, on-demand list of all available commands and their functions.
- **Web Search Capability:** Jenova can now search the web for up-to-date information. This can be triggered manually with the `/search <query>` command, or autonomously by the AI when it determines its own knowledge is insufficient. Search results are stored in the cognitive graph.
- **Document Reading and Insight Generation:** Implemented a new system for reading documents from the `src/jenova/docs` directory. The system processes new or modified documents, chunks their content, generates summaries and individual insights, and links them within the cognitive graph. This allows the AI to learn from external documents, expanding its knowledge base.
- **Document Processor:** Added a sample `example.md` file to the `src/jenova/docs` directory.
- **UI Enhancements:** Implemented a visual spinner in the `TerminalUI` to indicate when long-running cognitive processes are occurring, improving user experience.
- A new end-to-end finetuning script (`finetune/run_finetune.py`) that automates the entire process, including downloading and building `llama.cpp`, preparing data, and running the finetuning process.

### Changed
- **Web Search:** Replaced the `duckduckgo-search` library with a more powerful, `selenium`-based implementation. The AI now uses a headless browser to perform web searches, allowing it to access and extract the full content of web pages, leading to much richer and more accurate information gathering.
- **Enhanced Web Search Comprehension:** The web search result processing has been significantly enhanced. The AI now processes the full content of web pages in chunks, extracting a summary, key takeaways, and potential questions from each chunk, and stores this structured information in the Cortex.
- **`/search` Command:** The `/search` command now provides the same conversational web search experience as the inline `(search: <query>)` syntax, ensuring a consistent user experience.
- **Cognitive Engine:** The `_plan` method has been enhanced to make the AI aware of the new `WeatherTool` and the enhanced `FileTools`, allowing it to intelligently decide when to use these tools to fulfill user requests.
- **`.gitignore`:** Updated the `.gitignore` file to exclude LLM models and the content of the documentation directory.
- **FileTools:** The `FileTools` class has been completely overhauled to use a secure, configurable sandbox directory (`~/jenova_files` by default). All file operations are now restricted to this directory, and the AI is prevented from accessing hidden files, significantly improving security and user control.
- **Cognitive Engine:** The `_plan` method has been enhanced to make the AI aware of the new `WeatherTool` and the enhanced `FileTools`, allowing it to intelligently decide when to use these tools to fulfill user requests.
- **Web Search:** Replaced the `duckduckgo-search` library with a more powerful, `selenium`-based implementation. The AI now uses a headless browser to perform web searches, allowing it to access and extract the full content of web pages, leading to much richer and more accurate information gathering.
- **`/search` Command:** The `/search` command now provides the same conversational web search experience as the inline `(search: <query>)` syntax, ensuring a consistent user experience.
- **Conversational Web Search:** The web search functionality is now fully conversational. When a search is performed, the AI presents a summary of the findings and asks for further instructions, allowing for a more collaborative exploration of information.
- **Enhanced Document Comprehension:** The document processing system has been significantly enhanced. It now performs a much deeper analysis of documents, extracting not just insights, but also key takeaways and a list of questions the document can answer. This information is then stored in the cognitive graph as a rich structure of interconnected nodes, allowing the AI to have a much deeper understanding of the documents it reads.
- **README Update:** The `README.md` has been significantly updated to accurately reflect the current state of the program, including the new cognitive architecture, conversational web search, and other enhancements.
- **Conversational Web Search:** The web search functionality has been made more conversational and interactive. The AI now presents a summary of search results and asks for further instructions, such as performing a deeper search.
- **Spinner Consistency:** Removed conflicting spinners from the `Cortex` to ensure a consistent and smooth user experience during long-running operations like `/reflect`.
- **UI Help Command:** The `/help` command in the `TerminalUI` has been significantly enhanced to provide detailed, comprehensive descriptions for each command, explaining its purpose, impact on the AI, and usage, with improved visual styling including highlighted commands (bright yellow) and subdued descriptions (bright lavender).
- **Proactive Engine:** The `ProactiveEngine` has been enhanced to be more "hyper-aware" and proactive. It now considers underdeveloped insights (low centrality nodes) and high-potential insights (high centrality nodes) within the cognitive graph, in addition to unverified assumptions, when generating proactive suggestions for the user.
- **Interactive Procedure Learning:** The `/learn_procedure` command has been refactored to provide an interactive, guided experience. The AI now prompts the user for the procedure's name, individual steps, and expected outcome, ensuring structured and comprehensive intake of procedural knowledge.
- **Web Search Tool:** Renamed the `google_web_search` function to `web_search` in `src/jenova/default_api.py` and updated all references in `main.py` and `engine.py` to accurately reflect its use of the `duckduckgo-search` library.
- **Cognitive Architecture:** The core reflection process has been significantly improved. The old, redundant `reorganize_insights` task has been removed from the cognitive cycle. The `/reflect` command now correctly triggers the powerful, unified `Cortex.reflect` method. This method now uses a more robust graph traversal algorithm to find clusters for meta-insight generation, leading to deeper and more relevant high-level insights.
- **Document Processing:** The document processor is no longer triggered automatically at startup. Document processing is now an on-demand action initiated via the `/develop_insight` command, improving startup time and giving the user more control.
- **Hardware Optimization:** The default configuration in `main_config.yaml` has been updated for better performance. `gpu_layers` is now set to -1 to maximize GPU offloading, and `mlock` is enabled by default to keep the model locked in RAM.
- **Enhanced Intelligence and Learning:** The AI's cognitive processes have been significantly enhanced for deeper understanding and more robust learning.
  - **Smarter Web Search:** The autonomous web search now uses more advanced heuristics to decide when to search for up-to-date information.
  - **Deeper Semantic Comprehension:** When processing documents and web search results, the AI now extracts structured data including key entities, topics, and sentiment, leading to a richer understanding of the information.
  - **Advanced Knowledge Interlinking:** The reflection process is now more sophisticated. It not only links insights to external data but also finds relationships between different external sources (document-to-document, web-result-to-web-result) and identifies and creates insights about contradictions it discovers.
- **Web Search:** Implemented a more natural web search syntax `(search: <query>)` that can be used directly in the conversation. The AI can also use this syntax autonomously.
- **Proactive Engine:** The proactive engine is now more context-aware, using conversation history to generate more relevant and diverse suggestions.
- **Memory System:** The memory system's metadata extraction is now more robust, avoiding default values when the LLM fails to extract information.
- **Enhanced Reflection:** The reflection process has been improved to create links between insights and external information sources like documents and web search results, creating a more interconnected knowledge graph.
- **`/develop_insight` Command:** The `/develop_insight` command has been enhanced. When used without a `node_id`, it now triggers the new document reading and insight generation process. The existing functionality of developing a specific insight by providing a `node_id` is preserved.
- **Document Processor:** The document processor now runs at startup, processing all documents in the `docs` directory.
- **Document Processing:** Improved the document processing system to provide better feedback and search capabilities.
  - The system now prints a message to the console when it starts reading a document.
  - The document's title (filename) is now included with the content when processing, allowing the AI to better understand the context and relevance of the information.
- **AssumptionManager:** Improved assumption system robustness by modifying `AssumptionManager.add_assumption` to prevent re-adding assumptions that have already been verified (confirmed or false), and to return the ID of an existing unverified assumption if found.
- **ProactiveEngine:** Enhanced thought generation by modifying `ProactiveEngine.get_suggestion` to prioritize unverified assumptions and avoid repeating recent suggestions, making the process more cognitive.
- **UI Enhancements:** Improved line spacing in `TerminalUI` for better readability of user input, system messages, and AI output.
- **Cortex Stability and Intelligence:** Overhauled the Cortex system to be more robust, intelligent, and less prone to degradation (i.e., "brain rot").
  - **Emotion Analysis:** Replaced simplistic sentiment analysis with a more sophisticated emotion analysis, providing a richer psychological dimension to the cognitive graph.
  - **Weighted Centrality:** Implemented a weighted centrality calculation for more accurate node importance, leading to better meta-insight generation.
  - **Graph Pruning:** Introduced an automated graph pruning mechanism to remove old, irrelevant nodes, keeping the cognitive graph healthy and efficient.
  - **Reliable Linking:** Hardened the node linking process (`_link_orphans`) with more robust JSON parsing and error logging to prevent graph fragmentation.
  - **High-Quality Meta-Insights:** Improved the meta-insight generation process to prevent duplicates and produce more novel, higher-level insights. The selection of cluster centers for meta-insight generation is now more dynamic, using a centrality threshold instead of a fixed number of nodes.
- **Configuration:** The Cortex is now more configurable via `main_config.yaml`, allowing for tuning of relationship weights and pruning settings.
- **Cognitive Cycle:** Replaced the rigid, hardcoded cognitive cycle with a flexible and configurable `CognitiveScheduler`.
- **Memory Search:** Made the number of results for each memory type configurable.
- **Command Handling:** Refactored the command handling in the `TerminalUI` to be more concise and extensible.
- **AI Recognition:** The AI now recognizes the user and can communicate about its insights and assumptions.
- **Commands:** The `/reflect`, `/meta`, and `/verify` commands are now working correctly.
- **Logging:** Added more detailed logging to the cognitive functions to give the user a better idea of what's happening behind the scenes.
- **.gitignore:** Updated the `.gitignore` file to protect user data and the virtual environment.
- **Data Integrity:**
  - Prevented the addition of duplicate assumptions.
  - Made the insight reorganization process safer and more efficient to prevent data loss.
- **Circular Dependency:** Removed the circular dependency between `InsightManager` and `MemorySearch`.
- **Code Quality:**
  - Removed redundant code in `LLMInterface`.
  - Improved the reliability of JSON parsing across the application.
  - Made the `FileLogger`'s `log_file_path` a public attribute.
- The `finetune/prepare_data.py` script has been refactored to be more modular and robust. The `prepare_history_data` function now supports a structured JSONL format.
- The /finetune command now checks for the existence of the required 'llama.cpp' executables before running.
- The application now automatically discovers and loads a model from the `models/` directory if the `model_path` in the configuration is not set.
- Upgraded the fine-tuning process to a perfected, two-stage workflow. The `/finetune` command now first creates a LoRA adapter and then automatically merges it with the base model to produce a new, fully fine-tuned `.gguf` model, ready for use.
- Enhanced the fine-tuning data preparation script (`finetune/prepare_data.py`) to create more advanced, context-aware training examples in a conversational format, leading to higher quality learning.

### Fixed
- **Cognitive Degradation:** Overhauled the `Cortex` and `CognitiveEngine` to fix a cascading failure in the AI's intelligence. This includes:
  - **Robust Persona:** Reinforced the AI's identity ("Jenova", created by "The Architect") in all cognitive prompts to ensure a consistent persona and proper user recognition.
  - **Stable Memory:** Corrected a critical bug in meta-insight generation that prevented the AI from deepening its understanding of topics over time.
  - **Reliable Cognition:** Hardened the AI's cognitive functions by replacing fragile JSON parsing with robust `gbnf` grammars, preventing errors and ensuring the reliable creation and linking of cognitive nodes.
  - **Structured Emotion:** Improved the emotion analysis system to use a fixed list of emotions, providing more consistent and useful data for understanding context.
- **Memory Management:** Changed the default `mlock` setting to `false` to encourage the operating system to utilize SWAP memory for the model, freeing up RAM for other tasks.
- **Startup Crash:** Fixed a `TypeError: 'str' object is not callable` error that occurred during the initialization of `SemanticMemory`. This was caused by an issue with how the custom embedding function was being handled by `chromadb`. The fix ensures that the application starts up reliably by using a custom embedding function with a `name` method.
- **Startup Crash:** Implemented a self-healing mechanism in `SemanticMemory` to handle `chromadb` embedding function conflicts. If a conflict is detected, the old collection is backed up, deleted, and recreated with the new embedding function, and the data is migrated to the new collection, preventing data loss.
- **AssumptionManager:** Fixed a `NameError` in `add_assumption` caused by undefined variables (`assumption_data`, `content`). The duplicate checking logic has been rewritten to be more robust.
- **Cortex:** Fixed a `SyntaxError` in `develop_insights_from_docs` caused by improper f-string quoting.
- **Cognitive Engine:** Fixed an issue in the `_execute` method where the `WeatherTool` was not being called correctly.
- **Startup Crash:** Fixed a `TypeError: 'str' object is not callable` error that occurred during the initialization of `SemanticMemory`. This was caused by an issue with how the custom embedding function was being handled by `chromadb`. The fix ensures that the application starts up reliably.
- **AssumptionManager:** Fixed a `NameError` in `add_assumption` caused by undefined variables (`assumption_data`, `content`). The duplicate checking logic has been rewritten to be more robust.
- **Cortex:** Fixed a `SyntaxError` in `develop_insights_from_docs` caused by improper f-string quoting.
- **Bug Fix:** Fixed a bug where the application would crash due to incorrect error logging calls (`file_logger.error` instead of `file_logger.log_error`).
- **SyntaxError:** Resolved a `SyntaxError` in `src/jenova/cortex/cortex.py` caused by incorrect f-string syntax in the `develop_insights_from_docs` method.
- **`/reflect` Command:** The `/reflect` command now correctly triggers the deep reflection process in the `Cortex`, ensuring that the AI's most powerful cognitive function is accessible on-demand.
- **Tool Security:** Enhanced the security of the local `FileTools` by adding path traversal checks to the `read_file` and `list_directory` methods. Access is now restricted to the user's home directory and the application's designated output directory.
- **Startup Crash:** Fixed a `ModuleNotFoundError` that prevented the application from starting.
- **Document Processor:** The document processor now correctly persists its state, preventing it from reprocessing all documents on every startup.
- **Memory System:** Improved the robustness of the memory system by enabling error logging and using UUIDs for unique document IDs in semantic memory.
- **Cognitive Engine:** Fixed a `NameError` in `generate_assumption_from_history` by renaming a variable.
- **Document Processor:** Fixed the document processor by creating the `src/jenova/docs` directory, which was missing.
- **Document Processor:** The document processor now provides feedback to the user when processing documents.
- **ConcernManager:** Resolved a persistent `SyntaxError` in `src/jenova/insights/concerns.py` by rewriting the file and clearing `__pycache__`, ensuring correct parsing.
- **TerminalUI:** Fixed `TypeError: 'NoneType' object is not iterable` for the `/memory-insight` command by modifying `TerminalUI._handle_command` to robustly handle the return type of `develop_insights_from_memory`.
- **CognitiveScheduler:** Fixed `TypeError: Cortex.reflect() got an unexpected keyword argument 'username'` by correcting `CognitiveScheduler` to pass the `user` argument instead of `username` to `Cortex.reflect`.
- **TerminalUI:** Improved `/verify` command feedback by modifying `TerminalUI._verify_assumption` to always provide feedback to the user, even when no unverified assumptions are found.
- **CognitiveEngine:** Corrected calls to `InsightManager.get_latest_insight_id()` by passing the `username` argument, resolving a `TypeError`.
- **ConcernManager:** Removed the `rich` spinner context manager from `reorganize_insights` to ensure the intended "three yellow dot" loading indicator is displayed for the `/reflect` command.
- **InsightManager:** Resolved `AttributeError: 'InsightManager' object has no attribute 'get_latest_insight_id'` by implementing the missing method to retrieve the latest insight's ID.
- **InsightManager:** Fixed `NoneType` object has no attribute 'append' error in `reorganize_insights` by ensuring the method always returns a list and correcting a typo.
- **CognitiveEngine:** Addressed `NoneType` object is not iterable error in `develop_insights_from_memory` by adding a defensive check for `context` being `None`.
- **CognitiveEngine:** Ensured the `/meta` command provides user feedback even when no new meta-insight is generated.
- **ConcernManager:** Resolved `AttributeError: 'ConcernManager' object has no attribute 'get_all_concerns'` by implementing the missing method to retrieve all existing concern topics.
- **UI Bug:** Fixed an issue in `TerminalUI` where empty `jenova_response` calls were creating unintended empty boxes, improving the visual presentation of AI output.
- **UI Bug:** Resolved the repetitive cluttering of the custom spinner by ensuring messages are returned from cognitive engine methods and printed only after the spinner has stopped, providing a clean and consistent processing indicator across commands.
- **Bug Fix:** Corrected `develop_insights_from_memory` to properly retrieve and display assumption IDs, resolving the `AttributeError: 'AssumptionManager' object has no attribute 'get_latest_assumption_id'`.
- **UI Bug:** Fixed the `TerminalUI` processing spinner to display yellow spinning dots only during long-running commands, ensuring it clears correctly and does not interfere with the prompt or AI output.
- **SyntaxError:** Fixed a `SyntaxError` in `jenova/ui/terminal.py` and `jenova-ai/src/jenova/ui/terminal.py` caused by incorrect syntax in the `_handle_command` method.
- **KeyError:** Fixed a `KeyError` in `jenova/assumptions/manager.py` and `jenova-ai/src/jenova/assumptions/manager.py` by ensuring `cortex_id` is present when loading and accessing assumption objects.
- **Error Handling:** Added robust error handling to all file I/O operations, LLM calls, and other critical parts of the codebase to prevent crashes.
- **NameError:** Fixed a `NameError` in `jenova/insights/concerns.py` caused by a missing `import os` statement.
- **NameError:** Fixed a `NameError` in `jenova/cognitive_engine/engine.py` caused by a missing `from jenova.cortex.proactive_engine import ProactiveEngine` statement.
- **TypeError:** Fixed a `TypeError` in `jenova/main.py` caused by a missing `config` argument in the `RAGSystem` constructor.
- **KeyError:** Fixed a `KeyError` in `jenova/assumptions/manager.py` caused by a missing `cortex_id` in the assumption object.
- **Security:**
  - Fixed a path traversal vulnerability in `FileTools`.
  - Fixed a shell injection vulnerability in `SystemTools`.
- The finetuning process is no longer a manual, multi-step process but a single, executable script.
- A bug in the /finetune command that caused a crash due to a missing 'os' import.
- A bug in the `finetune/prepare_data.py` script that caused incorrect parsing of conversation history.
- The application no longer crashes if the model path is not configured. It now provides clear instructions to the user.
- A bug in the `/finetune` command that caused a crash due to incorrect handling of shell command results.
- Resolved a bug where the fine-tuning process would fail due to a missing `model_path` in the configuration. The application will now correctly prompt the user to set the path.
- Hardened the JSON parsing logic across the application when processing responses from the LLM. This prevents crashes caused by malformed JSON, such as the one occurring during insight reorganization (`/reflect`).
- Cleaned up the UI to no longer display raw RAG debug information in the chat output, providing a cleaner user experience.
- Fixed a bug in the `/verify` command that caused a crash due to a mismatch in the expected return value from the cognitive engine.
- Fixed a bug where the model path was not being read from the configuration, causing the wrong model to be loaded if multiple models were present.

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
