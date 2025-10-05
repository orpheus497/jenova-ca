import json
import os
import re
from jenova.cortex.proactive_engine import ProactiveEngine
from jenova.cognitive_engine.scheduler import CognitiveScheduler

class CognitiveEngine:
    """The Perfected Cognitive Engine. Manages the refined cognitive cycle."""
    def __init__(self, llm, memory_search, file_tools, insight_manager, assumption_manager, config, ui_logger, file_logger, cortex, system_tools, rag_system, web_search, weather_tool):
        self.llm = llm
        self.memory_search = memory_search
        self.file_tools = file_tools
        self.insight_manager = insight_manager
        self.assumption_manager = assumption_manager
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.cortex = cortex
        self.system_tools = system_tools
        self.proactive_engine = ProactiveEngine(cortex, llm, ui_logger)
        self.rag_system = rag_system
        self.web_search = web_search
        self.weather_tool = weather_tool
        self.scheduler = CognitiveScheduler(config, cortex, insight_manager)
        self.history = []
        self.turn_count = 0
        self.MAX_HISTORY_TURNS = 10 # Keep the last 10 conversation turns
        self.pending_assumption = None
        self.pending_search_results = None
        self.original_user_input = None

    def think(self, user_input: str, username: str) -> str:
        """Runs the full cognitive cycle: Retrieve, Plan, Execute, and Reflect."""
        with self.ui_logger.cognitive_process("Thinking..."):
            self.file_logger.log_info(f"New query received from {username}: {user_input}")
            self.turn_count += 1

            # Check for search follow-up
            if self.pending_search_results:
                # The user is responding to the search results
                plan = f"User has reviewed the search results for '{self.original_user_input}' and has provided further instructions: '{user_input}'. I will now use the search results and the new instructions to formulate a final response."
                context = self.memory_search.search_all(self.original_user_input, username)
                if context is None:
                    context = []
                
                response = self._execute(self.original_user_input, context, plan, username, search_results=self.pending_search_results)

                # Reset pending search state
                self.pending_search_results = None
                self.original_user_input = None

                self.history.append(f"{username}: {user_input}")
                self.history.append(f"Jenova: {response}")
                if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                    self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]
                self.memory_search.episodic_memory.add_episode(f"{username}: {user_input}\nJenova: {response}", username)
                
                return response

            # Direct Web Search from user input
            search_match = re.search(r'\(search:\s*(.*?)\)', user_input)
            if search_match:
                query = search_match.group(1)
                search_results = self.search_web(query, username)
                
                if search_results:
                    self.pending_search_results = search_results
                    self.original_user_input = user_input
                    response = "I found the following information:\n\n"
                    for result in search_results:
                        response += f"- **{result['title']}**: {result['summary']}\n"
                    response += "\nWhat would you like to do next? For example, you can ask me to summarize the findings, answer a specific question based on them, or perform a deeper search."
                else:
                    response = "I couldn't find any information on that topic."

                self.history.append(f"{username}: {user_input}")
                self.history.append(f"Jenova: {response}")
                if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                    self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]
                self.memory_search.episodic_memory.add_episode(f"{username}: {user_input}\nJenova: {response}", username)
                
                return response

            if self.pending_assumption:
                self.assumption_manager.resolve_assumption(self.pending_assumption, user_input, username)
                self.pending_assumption = None

            # Get and execute cognitive tasks from the scheduler
            cognitive_tasks = self.scheduler.get_cognitive_tasks(self.turn_count, user_input, username)
            for task_name, task_args in cognitive_tasks:
                try:
                    if hasattr(self, task_name):
                        getattr(self, task_name)(**task_args)
                    elif hasattr(self.cortex, task_name):
                        getattr(self.cortex, task_name)(**task_args)
                    elif hasattr(self.insight_manager, task_name):
                        getattr(self.insight_manager, task_name)(**task_args)
                except Exception as e:
                    self.ui_logger.system_message(f"Error during cognitive task '{task_name}': {e}")
                    self.file_logger.log_error(f"Error during cognitive task '{task_name}': {e}")

            # Proactive suggestion
            if self.turn_count % 5 == 0:
                suggestion = self.proactive_engine.get_suggestion(username, self.history)
                if suggestion:
                    self.ui_logger.system_message(f"Jenova has a thought: {suggestion}")

            # If the user_input is a command, it should not be processed as conversational input.
            if user_input.startswith('/'):
                return ""

            context = self.memory_search.search_all(user_input, username)
            if context is None:
                context = []
            plan = self._plan(user_input, context, username)

            # Autonomous Web Search
            if "[SEARCH:" in plan:
                query = plan.split("[SEARCH:")[-1].split("]")[0]
                search_results = self.search_web(query, username)
                
                if search_results:
                    self.pending_search_results = search_results
                    self.original_user_input = user_input
                    response = "My research has led me to the following information:\n\n"
                    for result in search_results:
                        response += f"- **{result['title']}**: {result['summary']}\n"
                    response += "\nI will now use this to answer your question. If you'd like me to do something else with this information, please let me know."
                    
                    self.history.append(f"{username}: {user_input}")
                    self.history.append(f"Jenova: {response}")
                    if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                        self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]
                    self.memory_search.episodic_memory.add_episode(f"{username}: {user_input}\nJenova: {response}", username)

                    return response
                else:
                    # No search results, continue with normal execution
                    response = self._execute(user_input, context, plan, username)

            else:
                response = self._execute(user_input, context, plan, username)

            # Only add to history if it's not a command
            if not user_input.startswith('/'):
                self.history.append(f"{username}: {user_input}")
                self.history.append(f"Jenova: {response}")
                if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                    self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]

                # Add to episodic memory
                self.memory_search.episodic_memory.add_episode(f"{username}: {user_input}\nJenova: {response}", username)

        return response

    def _plan(self, user_input: str, context: list[str], username: str) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        
        # Load persona from config
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        directives = persona_config.get('directives', [])
        
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')

        # Check if the current user is the creator
        user_title = "User"
        if username == creator_name:
            user_title = creator_alias

        prompt = f'''You are {ai_name}, a self-aware AI. Your creator is {creator_name}, whom you know as {creator_alias}.
Your core directives are: {directives}

== CONTEXT ==
{context_str if context else "No context available."}

== TASK ==
Based on your identity, directives, the user's query, and the provided context, create a short, step-by-step plan to formulate a response. The plan can include generating a new insight if a novel conclusion is reached.

**Available Tools:**
- **Web Search:** Use the syntax `[SEARCH: <query>]` to search the web for up-to-date information.
- **Get Date and Time:** Use the syntax `<TOOL:GET_CURRENT_DATETIME()>`
- **Get Weather:** Use the syntax `<TOOL:GET_WEATHER(location="<location>")>`
- **File System:** Read, write, and list files in a sandboxed environment:
    - `<TOOL:READ_FILE(path="<file_path>")>`
    - `<TOOL:WRITE_FILE(path="<file_path>", content="<file_content>")>`
    - `<TOOL:LIST_DIRECTORY(path="<directory_path>")>`

{user_title} ({username}): "{user_input}"

Plan:'''
        with self.ui_logger.thinking_process("Formulating plan..."):
            plan = self.llm.generate(prompt, temperature=0.1, stop=["\n\n"])
        self.file_logger.log_info(f"Generated Plan: {plan}")
        return plan

    def _execute(self, user_input: str, context: list[str], plan: str, username: str, search_results: str = None) -> str:
        with self.ui_logger.thinking_process("Executing plan..."):
            response = self.rag_system.generate_response(user_input, username, self.history, plan, search_results=search_results)
        self.file_logger.log_info(f"Generated Response: {response}")
        response = self.file_tools.handle_tool_request(response)
        response = self.system_tools.handle_tool_request(response)
        response = self.weather_tool.handle_tool_request(response)
        return response

    def search_web(self, query: str, username: str) -> list[dict]:
        """
        Searches the web using the web_search tool, processes the full content of the pages,
        and stores a detailed analysis in the Cortex.
        """
        self.ui_logger.system_message(f"Searching the web for: {query}...")
        
        try:
            search_results_raw = self.web_search(query=query, max_results=3)
            
            if not search_results_raw or not search_results_raw.get('results'):
                self.ui_logger.system_message("No search results found.")
                return []

            search_node_id = self.cortex.add_node('web_search', f"Web search for '{query}'", username)
            processed_results = []

            for result in search_results_raw.get('results', []):
                title = result.get('title')
                link = result.get('link')
                content = result.get('content')
                
                if not all([title, link, content]):
                    continue

                self.ui_logger.system_message(f"Analyzing content from: {link}")

                # 1. Generate an overall summary for the page
                summary_prompt = f'''Summarize the following web page content in 3-4 sentences.

Title: {title}
Content: {content[:8000]}

Summary:'''
                overall_summary = self.llm.generate(summary_prompt, temperature=0.3)

                # 2. Create a main web_search_result node
                main_result_node_id = self.cortex.add_node(
                    'web_search_result', 
                    f"Summary of '{title}'", 
                    username, 
                    metadata={'source_link': link, 'summary': overall_summary}
                )
                self.cortex.add_link(main_result_node_id, search_node_id, 'search_result_for')
                
                processed_results.append({'title': title, 'link': link, 'summary': overall_summary})

                # 3. Process content in chunks for detailed analysis
                chunks = self.cortex._chunk_text(content)
                for i, chunk in enumerate(chunks):
                    self.ui_logger.system_message(f"Analyzing chunk {i+1}/{len(chunks)} of {title}...")
                    
                    prompt = f'''Analyze the following text from a web page. Your task is to perform a comprehensive analysis and extract the following information:
1.  A concise summary of the text chunk.
2.  A list of key takeaways or main points (as a list of strings).
3.  A list of any questions that this text can answer (as a list of strings).
4.  A list of key entities (people, places, organizations).
5.  The overall sentiment of the text.

Respond with a single, valid JSON object containing the keys: 'summary', 'takeaways', 'questions', 'entities', and 'sentiment'.

Text: """{chunk}"""

JSON Response:'''
                    
                    analysis_data = None
                    for attempt in range(2):
                        analysis_json_str = self.llm.generate(prompt, temperature=0.2)
                        try:
                            analysis_data = extract_json(analysis_json_str)
                            break
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            self.file_logger.log_error(f"Attempt {attempt + 1}: Failed to process analysis from web page chunk. Invalid JSON: {analysis_json_str}. Error: {e}")
                            prompt = f'''The previous attempt to generate JSON failed. Please try again.

Text: """{chunk}"""

JSON Response (must be a valid JSON object):'''

                    if not analysis_data:
                        continue

                    chunk_summary = analysis_data.get('summary')
                    if not chunk_summary:
                        continue

                    # Create insight node for the chunk's summary
                    chunk_metadata = {
                        'entities': analysis_data.get('entities'),
                        'sentiment': analysis_data.get('sentiment'),
                        'source_chunk': i,
                        'source_link': link
                    }
                    chunk_summary_id = self.cortex.add_node('insight', chunk_summary, username, metadata=chunk_metadata)
                    self.cortex.add_link(chunk_summary_id, main_result_node_id, 'part_of_web_result')

                    # Create insight nodes for takeaways
                    for takeaway in analysis_data.get('takeaways', []):
                        takeaway_id = self.cortex.add_node('insight', takeaway, username, metadata={'source_chunk': i, 'source_link': link})
                        self.cortex.add_link(takeaway_id, chunk_summary_id, 'elaborates_on')

                    # Create question nodes
                    for question in analysis_data.get('questions', []):
                        question_id = self.cortex.add_node('question', question, username, metadata={'source_chunk': i, 'source_link': link})
                        self.cortex.add_link(question_id, chunk_summary_id, 'answered_by')

            self.ui_logger.system_message("Web search and analysis complete. Insights stored in Cortex.")
            return processed_results

        except Exception as e:
            self.file_logger.log_error(f"An error occurred during web search: {e}")
            self.ui_logger.system_message(f"An error occurred during the web search: {e}")
            return []

    def generate_insight_from_history(self, username: str):
        """Analyzes recent conversation history to generate and save a new, high-quality insight."""
        self.ui_logger.info("Analyzing conversation history to generate a new insight...")
        if len(self.history) < 2:
            return

        conversation_segment = "\n".join(self.history[-8:])
        
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing a conversation with {user_title} ({username}). Your goal is to identify a single, high-quality insight that represents a new understanding, a significant user preference, a correction of a previous assumption, or a key takeaway that should be remembered for future interactions.

Format the output as a valid JSON object with "topic" and "insight" keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        with self.ui_logger.thinking_process("Generating insight from recent conversation..."):
            insight_json_str = self.llm.generate(prompt, temperature=0.2, grammar=self.cortex.json_grammar)
        
        try:
            data = json.loads(insight_json_str)
            topic = data.get('topic')
            insight = data.get('insight')
            if topic and insight:
                self.insight_manager.save_insight(insight, username, topic=topic)
        except (json.JSONDecodeError, ValueError):
            self.ui_logger.system_message(f"Failed to decode insight from LLM response: {insight_json_str}")

    def generate_assumption_from_history(self, username: str):
        """Analyzes recent conversation history to generate a new assumption about the user."""
        self.ui_logger.info("Analyzing conversation history to generate a new assumption...")
        if len(self.history) < 4:
            return

        conversation_segment = "\n".join(self.history[-8:])

        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""You are {ai_name}. You are analyzing a conversation with {user_title} ({username}). Your goal is to identify a single, non-trivial assumption about the user (their preferences, goals, knowledge level, etc.) that is implied but not explicitly stated. This assumption will be verified later.

Format the output as a valid JSON object with an "assumption" key.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        with self.ui_logger.thinking_process("Forming new assumption..."):
            assumption_json_str = self.llm.generate(prompt, temperature=0.3, grammar=self.cortex.json_grammar)

        try:
            assumption_data = json.loads(assumption_json_str)
            assumption = assumption_data.get('assumption')
            if assumption:
                self.assumption_manager.add_assumption(assumption, username)
        except (json.JSONDecodeError, ValueError):
            self.ui_logger.system_message(f"Failed to decode assumption from LLM response: {assumption_json_str}")

    def develop_insights_from_conversation(self, username: str) -> list[str]:
        """Command to develop insights from the full current conversation history."""
        messages = []
        
        conversation_segment = "\n".join(self.history)
        
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing a conversation with {user_title} ({username}). Identify up to three high-quality insights.

Format the output as a valid JSON array of objects, where each object has 'topic' and 'insight' keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        insights_json_str = self.llm.generate(prompt, temperature=0.3, grammar=self.cortex.json_grammar)
        
        try:
            insights = json.loads(insights_json_str)
            for insight_data in insights:
                topic = insight_data.get('topic')
                insight = insight_data.get('insight')
                if topic and insight:
                    insight_id = self.insight_manager.save_insight(insight, username, topic=topic)
                    messages.append(f"New insight node created: {insight_id}")
        except (json.JSONDecodeError, ValueError):
            messages.append(f"Failed to decode insights from LLM response: {insights_json_str}")
        return messages

    def reflect_on_insights(self, username: str) -> list[str]:
        """Command to trigger a deep reflection on the cognitive graph."""
        return self.cortex.reflect(user=username)

    def generate_meta_insight(self, username: str) -> list[str]:
        """Command to generate a new, meta-insight from existing insights."""
        messages = []
        
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            messages.append("No existing insights to reflect on.")
            return messages

        self.file_logger.log_info(f"All insights: {all_insights}")

        insights_str = "\n".join([f"- {i['topic']}: {i['content']}" for i in all_insights])
        
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing a collection of insights for {user_title} ({username}). Your goal is to identify a new, higher-level 'meta-insight'. This is a conclusion drawn from analyzing existing insights, finding connections, patterns, or underlying themes.

[EXISTING INSIGHTS]
{insights_str}

[JSON_OUTPUT]
"""
        self.file_logger.log_info(f"Meta-insight prompt: {prompt}")

        meta_insight_json_str = self.llm.generate(prompt, temperature=0.4, grammar=self.cortex.json_grammar)
        
        self.file_logger.log_info(f"Meta-insight response: {meta_insight_json_str}")

        try:
            data = json.loads(meta_insight_json_str)
            topic = data.get('topic', 'meta')
            insight = data.get('insight')
            if insight:
                self.insight_manager.save_insight(insight, username, topic=topic)
                messages.append(f"New meta-insight generated under topic '{topic}': {insight}")
            else:
                messages.append("No new meta-insight could be generated from the existing insights.")
        except (json.JSONDecodeError, ValueError):
            messages.append(f"Failed to decode meta-insight from LLM response: {meta_insight_json_str}")
        return messages

    def develop_insights_from_memory(self, username: str) -> list[str]:
        messages = []
        """Command to develop new insights or assumptions from a broad search of long-term memory."""
        messages.append("Developing new insights from long-term memory...")
        
        context = self.memory_search.search_all("general knowledge and past experiences", username)
        if context is None:
            context = []
        context_str = "\n".join(f"- {c}" for c in context)

        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing context from your long-term memory for {user_title} ({username}). Your goal is to extract a single, high-quality piece of information. Determine if it is a concrete insight or an unverified assumption about the user.

Format the output as a valid JSON object with one of two structures:
1. For an insight: {{"type": "insight", "topic": "<topic>", "content": "<insight_content>"}}
2. For an assumption: {{"type": "assumption", "content": "<assumption_content>"}}

[MEMORY CONTEXT]
{context_str}

[JSON_OUTPUT]
"""
        insight_json_str = self.llm.generate(prompt, temperature=0.3, grammar=self.cortex.json_grammar)
        
        try:
            data = json.loads(insight_json_str)
            if data.get('type') == 'insight':
                topic = data.get('topic')
                insight = data.get('content')
                if topic and insight:
                    insight_id = self.insight_manager.save_insight(insight, username, topic=topic)
                    messages.append(f"New insight node created: {insight_id}")
            elif data.get('type') == 'assumption':
                assumption = data.get('content')
                if assumption:
                    assumption_id = self.assumption_manager.add_assumption(assumption, username)
                    if assumption_id != "Assumption already exists.": # Check if a new assumption was actually added
                        messages.append(f"New assumption node created: {assumption_id}")
                    else:
                        messages.append(assumption_id) # Append the "Assumption already exists." message
        except (json.JSONDecodeError, ValueError):
            messages.append(f"Failed to decode insight from LLM response: {insight_json_str}")
        return messages

    def proactively_verify_assumption(self, username: str):
        """Proactively verifies an unverified assumption with the user."""
        assumption, question = self.assumption_manager.get_assumption_to_verify(username)
        if assumption and question:
            self.pending_assumption = assumption
            return question
        return None

    def verify_assumptions(self, username: str):
        """Command to verify unverified assumptions with the user."""
        question = self.proactively_verify_assumption(username)
        if question:
            return self.pending_assumption, question
        else:
            return None, "No unverified assumptions to check."

    def finetune(self, include_history: bool = False) -> list[str]:
        messages = []
        """Command to trigger the perfected, two-stage fine-tuning process."""
        messages.append("Initiating perfected fine-tuning process...")
        
        # Check for llama.cpp executables
        finetune_exec = "./llama.cpp/finetune"
        export_lora_exec = "./llama.cpp/export-lora"
        if not os.path.exists(finetune_exec) or not os.path.exists(export_lora_exec):
            messages.append(f"Error: Fine-tuning executables not found. Please ensure 'llama.cpp' is cloned and built in the project root directory.")
            messages.append(f"Missing: {' '.join([p for p in [finetune_exec, export_lora_exec] if not os.path.exists(p)])}")
            return messages

        finetune_config = self.config.get('finetuning', {})
        insights_dir = self.insight_manager.insights_root
        training_file = finetune_config.get('training_file', 'finetune_train.jsonl')
        history_file = self.file_logger.log_file_path if include_history else None

        # Step 1: Prepare the data with the advanced script
        messages.append("Step 1: Preparing advanced training data...")
        prepare_command = f"python finetune/prepare_data.py --insights-dir \"{insights_dir}\" --output-file \"{training_file}\""
        if history_file:
            prepare_command += f" --include-history \"{history_file}\""
        
        result = self.system_tools.execute_shell_command(prepare_command, "Preparing fine-tuning data...")
        if result.get('error'):
            messages.append(f"Error during data preparation: {result.get('stderr')}")
            return messages

        # Step 2: Run the fine-tuning to create a LoRA adapter
        messages.append("Step 2: Creating LoRA adapter...")
        base_model_path = self.config.get('model', {}).get('model_path')
        lora_output = finetune_config.get('lora_output_file', 'models/lora-jenova-adapter.bin')
        threads = self.config.get('hardware', {}).get('threads', 4)
        gpu_layers = self.config.get('hardware', {}).get('gpu_layers', 0)

        if not base_model_path or not os.path.exists(base_model_path):
            messages.append(f"Error: model_path '{base_model_path}' not found in config or does not exist. Please specify the base model path.")
            return messages

        finetune_command = f"""
{finetune_exec} --model-base {base_model_path} \
--train-data \"{training_file}\" \
--lora-out {lora_output} \
--threads {threads} --gpu-layers {gpu_layers} \
--batch-size {finetune_config.get('batch_size', 4)} --epochs {finetune_config.get('epochs', 3)} \
--use-flash-attn"""

        messages.append("Executing fine-tuning command. This may take a while...")
        result = self.system_tools.execute_shell_command(finetune_command, "Running fine-tuning...")
        if result.get('error'):
            messages.append(f"Error during LoRA creation: {result.get('stderr')}")
            return messages
        messages.append(f"LoRA adapter created successfully at {lora_output}")

        # Step 3: Merge the LoRA adapter to create a new GGUF model
        messages.append("Step 3: Merging LoRA adapter into a new GGUF model...")
        finetuned_model_path = finetune_config.get('finetuned_model_output', 'models/jenova-finetuned.gguf')

        export_command = f"""
{export_lora_exec} --model-base {base_model_path} \
--lora-in {lora_output} \
--lora-out {finetuned_model_path}"""

        messages.append("Executing model merge command...")
        result = self.system_tools.execute_shell_command(export_command, "Merging model...")
        if result.get('error'):
            messages.append(f"Error during model merge: {result.get('stderr')}")
            return messages

        messages.append(f"Perfected fine-tuning process completed. The new, updated model is available at {finetuned_model_path}")
        messages.append("Please update your main_config.yaml to point to this new model to use it.")
        return messages

    def learn_procedure(self, procedure_data: dict, username: str) -> list[str]:
        """Command to learn a new procedure interactively."""
        messages = []
        procedure_name = procedure_data.get('name')
        steps = procedure_data.get('steps', [])
        outcome = procedure_data.get('outcome')

        if not procedure_name or not steps or not outcome:
            messages.append("Error: Incomplete procedure data provided.")
            return messages

        # Generate context for the procedure using LLM
        context_prompt = f"""Given the following procedure:
Name: {procedure_name}
Steps: {steps}
Outcome: {outcome}

Generate a concise, 1-2 sentence context for this procedure. What is its general purpose or domain?

Context:"""
        with self.ui_logger.thinking_process("Generating procedure context..."):
            context = self.llm.generate(context_prompt, temperature=0.3)

        # Combine name and steps for the main procedure document
        procedure_doc = f"Procedure: {procedure_name}\nSteps: {'; '.join(steps)}\nOutcome: {outcome}"

        self.memory_search.procedural_memory.add_procedure(
            procedure=procedure_doc,
            username=username,
            goal=outcome,
            steps=steps,
            context=context
        )
        messages.append(f"Procedure '{procedure_name}' learned successfully.")
        return messages