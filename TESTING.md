# JENOVA Cognitive Architecture - Testing Instructions

Comprehensive testing procedures to verify your JENOVA installation and validate all cognitive functions.

**Author:** Documentation generated for the JENOVA Cognitive Architecture, designed and developed by orpheus497.

---

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Pre-Installation Tests](#pre-installation-tests)
3. [Installation Verification](#installation-verification)
4. [Component Testing](#component-testing)
5. [Cognitive Function Testing](#cognitive-function-testing)
6. [Performance Testing](#performance-testing)
7. [Integration Testing](#integration-testing)
8. [Regression Testing](#regression-testing)
9. [Troubleshooting Test Failures](#troubleshooting-test-failures)

---

## Testing Overview

### Test Categories

| Category | Purpose | Duration | Frequency |
|----------|---------|----------|-----------|
| **Pre-Installation** | Verify system compatibility | 2 min | Before install |
| **Installation** | Confirm successful installation | 5 min | After install |
| **Component** | Test individual modules | 10 min | After install |
| **Cognitive** | Validate AI functions | 15 min | After install |
| **Performance** | Measure speed and resource usage | 10 min | Optional |
| **Integration** | Test end-to-end workflows | 20 min | Optional |
| **Regression** | Ensure updates don't break features | 15 min | After updates |

### Testing Philosophy

JENOVA's tests follow these principles:
- **Functional**: Verify features work as documented
- **Observable**: Each test produces clear pass/fail results
- **Reproducible**: Tests can be run multiple times with same results
- **Comprehensive**: Cover all major components and workflows

---

## Pre-Installation Tests

Run these tests BEFORE installing JENOVA to ensure your system is ready.

### Test 1.1: Python Version Check

**Purpose:** Verify Python 3.10+ is installed

```bash
python3 --version
```

**Expected Output:**
```
Python 3.10.x or Python 3.11.x or Python 3.12.x
```

**Pass Criteria:** Version is 3.10.0 or higher
**Fail Action:** Install Python 3.10+

---

### Test 1.2: Python Venv Module

**Purpose:** Verify virtualenv capability

```bash
python3 -m venv --help
```

**Expected Output:**
```
usage: venv [-h] [--system-site-packages] [--symlinks | --copies] ...
```

**Pass Criteria:** Help text displays without errors
**Fail Action:** Install python3-venv package

---

### Test 1.3: Git Availability

**Purpose:** Verify git is installed

```bash
git --version
```

**Expected Output:**
```
git version 2.x.x
```

**Pass Criteria:** Version 2.0 or higher
**Fail Action:** Install git

---

### Test 1.4: Disk Space Check

**Purpose:** Verify sufficient disk space

```bash
df -h ~
```

**Expected Output:**
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       100G   40G   60G  40% /home
```

**Pass Criteria:** At least 10 GB available
**Recommended:** 50+ GB for multiple models
**Fail Action:** Free up disk space

---

### Test 1.5: RAM Check

**Purpose:** Verify sufficient memory

```bash
free -h
```

**Expected Output:**
```
              total        used        free      shared  buff/cache   available
Mem:           15Gi       2.0Gi        10Gi       100Mi       3.0Gi        13Gi
```

**Pass Criteria:**
- Minimum: 4 GB total
- Recommended: 16+ GB for larger models

**Fail Action:** Consider using smaller models or upgrading RAM

---

### Test 1.6: GPU Detection (Optional)

**Purpose:** Check for NVIDIA GPU availability

```bash
nvidia-smi
```

**Expected Output (if GPU present):**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.x.x    Driver Version: 525.x.x    CUDA Version: 12.x      |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P0    25W / 250W |   1234MiB /  8192MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**Pass Criteria:**
- GPU detected with available VRAM
- Driver version displayed

**If no GPU:** JENOVA works fine on CPU, just slower

---

### Test 1.7: CUDA Toolkit (Optional, for GPU users)

**Purpose:** Verify CUDA is installed for GPU acceleration

```bash
nvcc --version
```

**Expected Output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 11.x or 12.x
```

**Pass Criteria:** CUDA 11.x or 12.x detected
**If not installed:** GPU will not be used for acceleration

---

## Installation Verification

Run these tests AFTER running `./install.sh` to verify successful installation.

### Test 2.1: Virtual Environment Creation

**Purpose:** Verify venv was created

```bash
ls -ld venv/
```

**Expected Output:**
```
drwxr-xr-x. 1 user user 100 Oct 29 10:00 venv/
```

**Pass Criteria:** Directory exists with appropriate permissions
**Fail Action:** Re-run ./install.sh

---

### Test 2.2: Virtual Environment Activation

**Purpose:** Verify venv can be activated

```bash
source venv/bin/activate
echo $VIRTUAL_ENV
```

**Expected Output:**
```
/home/user/jenova-ai/venv
```

**Pass Criteria:** VIRTUAL_ENV variable points to venv directory
**Fail Action:** Re-run ./install.sh

---

### Test 2.3: Python Package Installation

**Purpose:** Verify all required packages are installed

```bash
source venv/bin/activate
pip list | grep -E "llama-cpp-python|chromadb|sentence-transformers|rich|torch"
```

**Expected Output:**
```
chromadb                      0.x.x
llama-cpp-python              0.x.x
rich                          13.x.x
sentence-transformers         2.x.x
torch                         2.x.x
```

**Pass Criteria:** All packages present
**Fail Action:**
```bash
pip install -r requirements.txt
```

---

### Test 2.4: JENOVA Package Installation

**Purpose:** Verify JENOVA is installed as package

```bash
source venv/bin/activate
pip show jenova-ai
```

**Expected Output:**
```
Name: jenova-ai
Version: 4.0.0
Summary: The JENOVA Cognitive Architecture...
Author: orpheus497
License: MIT
Location: /home/user/jenova-ai/src
```

**Pass Criteria:** Package is installed in editable mode
**Fail Action:**
```bash
pip install -e .
```

---

### Test 2.5: llama-cpp-python CUDA Support (GPU users only)

**Purpose:** Verify llama-cpp-python was built with CUDA

```bash
source venv/bin/activate
python -c "from llama_cpp import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

**Expected Output (GPU build):**
```
GPU offload: True
```

**Expected Output (CPU build):**
```
GPU offload: False
```

**Pass Criteria (GPU users):** True
**Fail Action (if False but GPU available):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall
```

---

### Test 2.6: PyTorch CUDA Support (GPU users only)

**Purpose:** Verify PyTorch can access GPU

```bash
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output (GPU system):**
```
CUDA: True
GPU: NVIDIA GeForce RTX 3080
```

**Pass Criteria (GPU users):** CUDA: True
**If False:** Embeddings will run on CPU (acceptable)

---

### Test 2.7: Configuration Files

**Purpose:** Verify configuration files exist

```bash
ls -l src/jenova/config/*.yaml
```

**Expected Output:**
```
-rw-r--r--. 1 user user 2134 Oct 29 10:00 src/jenova/config/main_config.yaml
-rw-r--r--. 1 user user  587 Oct 29 10:00 src/jenova/config/persona.yaml
```

**Pass Criteria:** Both YAML files exist
**Fail Action:** Check repository integrity, re-clone if needed

---

### Test 2.8: Model Directory

**Purpose:** Verify model directory exists

```bash
ls -ld models/ || ls -ld /usr/local/share/models/
```

**Expected Output:**
```
drwxr-xr-x. 1 user user 10 Oct 29 10:00 models/
```

**Pass Criteria:** At least one model directory exists
**Fail Action:**
```bash
mkdir -p models
```

---

## Component Testing

Test individual JENOVA components to ensure they function correctly.

### Test 3.1: Configuration Loading

**Purpose:** Verify YAML configuration can be loaded

```bash
source venv/bin/activate
python -c "
from jenova.config import load_configuration
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger
import queue
import os

msg_queue = queue.Queue()
ui_logger = UILogger(msg_queue)
file_logger = FileLogger('/tmp')

config = load_configuration(ui_logger, file_logger)
print('✓ Configuration loaded successfully')
print(f'  Model path: {config[\"model\"][\"model_path\"]}')
print(f'  GPU layers: {config[\"model\"][\"gpu_layers\"]}')
print(f'  Context size: {config[\"model\"][\"context_size\"]}')
"
```

**Expected Output:**
```
✓ Configuration loaded successfully
  Model path: /usr/local/share/models/model.gguf
  GPU layers: -1
  Context size: 8192
```

**Pass Criteria:** No errors, configuration values displayed
**Fail Action:** Check YAML syntax in config files

---

### Test 3.2: Embedding Model Loading

**Purpose:** Verify sentence-transformers can load

```bash
source venv/bin/activate
python -c "
from jenova.utils.model_loader import load_embedding_model

model = load_embedding_model('all-MiniLM-L6-v2', device='cpu')
print(f'✓ Embedding model loaded: {type(model).__name__}')

# Test embedding generation
embeddings = model.encode(['test sentence'])
print(f'✓ Embedding generated, dimension: {len(embeddings[0])}')
"
```

**Expected Output:**
```
✓ Embedding model loaded: SentenceTransformer
✓ Embedding generated, dimension: 384
```

**Pass Criteria:** Model loads and generates 384-dimensional embeddings
**Fail Action:** Check sentence-transformers installation

---

### Test 3.3: ChromaDB Database Creation

**Purpose:** Verify ChromaDB can create and query databases

```bash
source venv/bin/activate
python -c "
import chromadb
from chromadb.config import Settings
import tempfile
import os

# Create temporary database
temp_dir = tempfile.mkdtemp()
client = chromadb.PersistentClient(path=temp_dir)
collection = client.create_collection('test_collection')

# Add test data
collection.add(
    documents=['This is a test document'],
    ids=['test1']
)

# Query
results = collection.query(query_texts=['test'], n_results=1)
print(f'✓ ChromaDB working, found {len(results[\"documents\"][0])} results')

# Cleanup
import shutil
shutil.rmtree(temp_dir)
"
```

**Expected Output:**
```
✓ ChromaDB working, found 1 results
```

**Pass Criteria:** ChromaDB creates collection and queries successfully
**Fail Action:** Reinstall chromadb

---

### Test 3.4: LLM Model Loading (requires GGUF model)

**Purpose:** Verify GGUF model can be loaded

**Prerequisites:** Download a model first (see Step 3 of Quickstart)

```bash
source venv/bin/activate
python -c "
from jenova.config import load_configuration
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger
from jenova.utils.model_loader import load_llm_model
import queue

msg_queue = queue.Queue()
ui_logger = UILogger(msg_queue)
file_logger = FileLogger('/tmp')

config = load_configuration(ui_logger, file_logger)
llm = load_llm_model(config)

if llm:
    print('✓ LLM model loaded successfully')
    print(f'  Context size: {llm.n_ctx()}')
else:
    print('✗ Failed to load LLM model')
"
```

**Expected Output:**
```
✓ LLM model loaded successfully
  Context size: 8192
```

**Pass Criteria:** Model loads without errors
**Fail Action:**
- Verify model file exists
- Check model file is valid GGUF format
- Try downloading model again

---

### Test 3.5: Memory System Initialization

**Purpose:** Verify memory systems can be created

```bash
source venv/bin/activate
python -c "
from jenova.config import load_configuration
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger
from jenova.utils.model_loader import load_embedding_model, load_llm_model
from jenova.llm_interface import LLMInterface
from jenova.memory.semantic import SemanticMemory
import tempfile
import queue

msg_queue = queue.Queue()
ui_logger = UILogger(msg_queue)
file_logger = FileLogger('/tmp')

config = load_configuration(ui_logger, file_logger)
llm = load_llm_model(config)
embedding_model = load_embedding_model('all-MiniLM-L6-v2', device='cpu')
llm_interface = LLMInterface(config, ui_logger, file_logger, llm)

temp_dir = tempfile.mkdtemp()
semantic_memory = SemanticMemory(config, ui_logger, file_logger, temp_dir, llm_interface, embedding_model)

print('✓ Semantic memory initialized')

# Test adding a fact
semantic_memory.add('The capital of France is Paris.')
print('✓ Fact added to semantic memory')

# Test searching
results = semantic_memory.search('capital France', n_results=1)
print(f'✓ Memory search returned {len(results)} results')

# Cleanup
import shutil
shutil.rmtree(temp_dir)
del llm
"
```

**Expected Output:**
```
✓ Semantic memory initialized
✓ Fact added to semantic memory
✓ Memory search returned 1 results
```

**Pass Criteria:** Memory system creates, adds, and searches without errors
**Fail Action:** Check ChromaDB and embedding model installations

---

### Test 3.6: Cortex Graph Initialization

**Purpose:** Verify cognitive graph can be created

```bash
source venv/bin/activate
python -c "
from jenova.config import load_configuration
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger
from jenova.utils.model_loader import load_llm_model
from jenova.llm_interface import LLMInterface
from jenova.cortex.cortex import Cortex
import tempfile
import queue

msg_queue = queue.Queue()
ui_logger = UILogger(msg_queue)
file_logger = FileLogger('/tmp')

config = load_configuration(ui_logger, file_logger)
llm = load_llm_model(config)
llm_interface = LLMInterface(config, ui_logger, file_logger, llm)

temp_dir = tempfile.mkdtemp()
cortex = Cortex(config, ui_logger, file_logger, llm_interface, temp_dir)

print('✓ Cortex initialized')

# Add test node
node_id = cortex.add_node('insight', 'Test insight content')
print(f'✓ Node added: {node_id}')

# Cleanup
import shutil
shutil.rmtree(temp_dir)
del llm
"
```

**Expected Output:**
```
✓ Cortex initialized
✓ Node added: insight_xxxxxxxx
```

**Pass Criteria:** Cortex creates and adds nodes successfully
**Fail Action:** Check cortex module installation

---

## Cognitive Function Testing

Test JENOVA's core AI capabilities end-to-end.

### Test 4.1: Complete Startup

**Purpose:** Verify JENOVA starts without errors

```bash
source venv/bin/activate
timeout 30 python -m jenova.main <<EOF
exit
EOF
```

**Expected Output:**
```
>> Initializing Intelligence Matrix...
>> Loading configuration...
...
>> Cognitive Engine: Online.
╔══════════════════════════════════════════════════════════════╗
║                  ✨ JENOVA ✨                                ║
...
JENOVA shutting down.
```

**Pass Criteria:**
- No error messages during startup
- "Cognitive Engine: Online" appears
- Clean shutdown on exit

**Fail Action:** Check error messages, review logs

---

### Test 4.2: Basic Conversation

**Purpose:** Verify JENOVA can generate responses

**Test Procedure:**
1. Start JENOVA: `python -m jenova.main`
2. Type: `Hello, can you hear me?`
3. Observe response
4. Type: `exit`

**Expected Behavior:**
- JENOVA responds with greeting
- Response is coherent and relevant
- No errors during generation

**Pass Criteria:** JENOVA generates appropriate response
**Fail Action:**
- Check model loading
- Verify LLM interface
- Review temperature settings

---

### Test 4.3: Memory Storage

**Purpose:** Verify episodic memory stores conversations

**Test Procedure:**
1. Start JENOVA
2. Have a 3-turn conversation about a specific topic (e.g., Python)
3. Exit JENOVA
4. Check if memory was stored:
   ```bash
   ls -lh ~/.jenova-ai/users/$USER/memory/episodic/
   ```

**Expected Output:**
```
chroma.sqlite3  (should exist and be > 0 bytes)
```

**Pass Criteria:** Database file exists with data
**Fail Action:** Check episodic memory initialization

---

### Test 4.4: Memory Retrieval

**Purpose:** Verify memory can be retrieved across sessions

**Test Procedure:**
1. Session 1:
   ```
   You: My favorite color is blue.
   JENOVA: [responds]
   You: exit
   ```

2. Session 2 (restart JENOVA):
   ```
   You: What's my favorite color?
   JENOVA: [should mention blue based on previous session]
   ```

**Pass Criteria:** JENOVA recalls information from previous session
**Fail Action:** Check memory search and retrieval logic

---

### Test 4.5: Insight Generation

**Purpose:** Verify /insight command works

**Test Procedure:**
1. Start JENOVA
2. Have a meaningful conversation (5+ turns)
3. Type: `/insight`
4. Observe output
5. Verify file created:
   ```bash
   find ~/.jenova-ai/users/$USER/insights/ -name "*.json" -mtime -1
   ```

**Expected Behavior:**
- Processing indicator appears
- Insight is generated and displayed
- JSON file created in insights directory

**Pass Criteria:** Insight file created with valid JSON
**Fail Action:**
- Check LLM generation
- Verify insight manager initialization
- Check file permissions

---

### Test 4.6: Reflection Process

**Purpose:** Verify /reflect command works

**Test Procedure:**
1. Generate 3-5 insights first (repeat Test 4.5)
2. Type: `/reflect`
3. Observe processing
4. Check for meta-insights

**Expected Behavior:**
- Reflection process runs
- Orphan linking occurs
- Meta-insights may be generated (if enough insights)

**Pass Criteria:** Reflection completes without errors
**Fail Action:**
- Ensure sufficient insights exist
- Check Cortex reflection logic
- Review LLM generation

---

### Test 4.7: Assumption System

**Purpose:** Verify assumption generation and verification

**Test Procedure:**
1. Have 7+ conversational turns (triggers assumption generation)
2. Check if assumption was created:
   ```bash
   cat ~/.jenova-ai/users/$USER/assumptions.json
   ```
3. Use `/verify` command
4. Answer question to verify/reject assumption

**Expected Behavior:**
- Assumption JSON file created
- Assumptions listed with status
- /verify presents assumption for confirmation

**Pass Criteria:** Assumptions stored and verifiable
**Fail Action:** Check assumption manager and scheduler

---

### Test 4.8: Document Processing

**Purpose:** Verify document ingestion works

**Test Procedure:**
1. Create test document:
   ```bash
   cat > src/jenova/docs/test_document.md << 'EOF'
   # Test Document

   This is a test document for JENOVA processing.

   ## Key Points
   - Point 1: JENOVA can read documents
   - Point 2: Knowledge is extracted and indexed
   - Point 3: Documents become queryable
   EOF
   ```

2. In JENOVA, type: `/develop_insight`
3. Observe processing
4. Query document knowledge:
   ```
   You: What did you learn from the test document?
   ```

**Expected Behavior:**
- Document is found and processed
- Chunks are created
- Knowledge is integrated into Cortex
- JENOVA can recall document content

**Pass Criteria:** Document successfully processed and queryable
**Fail Action:** Check document processor and Cortex integration

---

### Test 4.9: Procedure Learning

**Purpose:** Verify /learn_procedure command works

**Test Procedure:**
1. Type: `/learn_procedure`
2. Follow prompts:
   - Name: "Making coffee"
   - Goal: "Brew a cup of coffee"
   - Steps: "1. Boil water\n2. Add coffee grounds\n3. Pour water\n4. Wait 4 minutes\n5. Serve"
3. Verify stored:
   ```bash
   # Check procedural memory database
   ls -lh ~/.jenova-ai/users/$USER/memory/procedural/
   ```

**Expected Behavior:**
- Interactive prompts appear
- Procedure is saved
- Confirmation message displayed

**Pass Criteria:** Procedure stored in procedural memory
**Fail Action:** Check procedural memory and interactive UI

---

### Test 4.10: Fine-Tuning Data Generation

**Purpose:** Verify /train command generates JSONL

**Prerequisites:** Have at least 10 insights and some conversation history

**Test Procedure:**
1. Type: `/train`
2. Wait for processing
3. Check output file:
   ```bash
   ls -lh finetune_train.jsonl
   head -5 finetune_train.jsonl
   ```

**Expected Output:**
```
-rw-r--r--. 1 user user 45K Oct 29 12:00 finetune_train.jsonl

{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "insights", ...}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "episodic", ...}
...
```

**Pass Criteria:**
- JSONL file created
- Valid JSON on each line
- Multiple sources represented

**Fail Action:** Check train.py script and data sources

---

## Performance Testing

### Test 5.1: Token Generation Speed

**Purpose:** Measure inference performance

**Test Procedure:**
1. Start JENOVA
2. Type a question: `Explain quantum computing in detail.`
3. Observe time from Enter to response completion
4. Note approximate tokens/second

**Benchmarks:**
- **GPU (RTX 3080)**: 60-80 tokens/sec for 7B models
- **CPU (8-core)**: 8-15 tokens/sec for 1-3B models
- **CPU (16-core)**: 15-25 tokens/sec for 1-3B models

**Pass Criteria:** Performance within expected range for hardware
**Optimization:** If slow, check:
- Model size (use smaller model)
- GPU offload settings
- Thread configuration

---

### Test 5.2: Memory Search Performance

**Purpose:** Measure memory retrieval speed

**Test Procedure:**
1. Add 100+ facts to semantic memory (run conversations)
2. Time a memory search:
   ```bash
   time python -c "
   from jenova.memory.semantic import SemanticMemory
   # ... initialize ...
   results = semantic_memory.search('test query', n_results=5)
   "
   ```

**Expected Time:**
- First query: 0.5-2 seconds (model loading)
- Subsequent queries: 0.1-0.5 seconds

**Pass Criteria:** Search completes in reasonable time
**Optimization:** Ensure GPU is used for embeddings if available

---

### Test 5.3: Resource Usage

**Purpose:** Monitor RAM and VRAM usage

**Test Procedure:**
```bash
# Terminal 1: Start monitoring
watch -n 1 'free -h && echo "---" && nvidia-smi'

# Terminal 2: Run JENOVA
python -m jenova.main
```

**Monitor:**
- RAM usage should stabilize after model loading
- VRAM usage (if GPU) should be consistent
- No memory leaks over time

**Expected Usage:**
- **TinyLlama 1B (CPU)**: 2-3 GB RAM
- **Mistral 7B (GPU)**: 6-8 GB VRAM, 4-6 GB RAM
- **Embedding model**: ~500 MB (RAM or VRAM)

**Pass Criteria:** Memory usage stable, no continuous growth
**Fail Action:** Check for memory leaks, review configuration

---

## Integration Testing

### Test 6.1: End-to-End Learning Workflow

**Purpose:** Test complete learning cycle

**Test Workflow:**
1. Initial conversation (3-5 turns)
2. Generate insight (`/insight`)
3. Continue conversation (5 more turns)
4. Generate another insight
5. Reflect (`/reflect`)
6. Verify assumptions (`/verify`)
7. Exit and restart
8. Verify memory persistence

**Pass Criteria:** All steps complete without errors, memory persists
**Duration:** 10-15 minutes

---

### Test 6.2: Document to Knowledge Workflow

**Purpose:** Test document processing pipeline

**Test Workflow:**
1. Add 2-3 documents to `src/jenova/docs/`
2. Process with `/develop_insight`
3. Use `/reflect` to synthesize
4. Query document knowledge in conversation
5. Verify nodes in cognitive graph

**Pass Criteria:** Documents fully integrated and queryable
**Duration:** 5-10 minutes

---

### Test 6.3: Multi-Session Cognitive Growth

**Purpose:** Test knowledge accumulation over time

**Test Workflow:**
Day 1:
- 10 conversational turns on Topic A
- Generate 2 insights
- Exit

Day 2:
- Verify Topic A is remembered
- 10 turns on Topic B
- Generate 2 insights
- Reflect

Day 3:
- Verify both topics remembered
- Ask about relationships between A and B
- Generate meta-insight

**Pass Criteria:** All memories retained, relationships identified
**Duration:** 30 minutes total across sessions

---

## Regression Testing

Run these after updating JENOVA to ensure no features broke.

### Test 7.1: Post-Update Functionality

**Checklist:**
- [ ] JENOVA starts without errors
- [ ] All commands listed in `/help` work
- [ ] Memory persists from pre-update
- [ ] Configuration still valid
- [ ] Models still load
- [ ] Existing insights readable

**Procedure:** Run Tests 4.1 through 4.10

**Pass Criteria:** All previous functionality still works
**Fail Action:** Check CHANGELOG for breaking changes

---

### Test 7.2: Data Migration

**Purpose:** Verify user data survives updates

**Procedure:**
1. Before update: Count insights
   ```bash
   find ~/.jenova-ai/users/$USER/insights/ -name "*.json" | wc -l
   ```

2. Update JENOVA:
   ```bash
   git pull origin main
   source venv/bin/activate
   pip install -r requirements.txt --upgrade
   ```

3. After update: Count insights again

4. Start JENOVA and query old memory

**Pass Criteria:**
- Insight count unchanged
- Old memories still accessible
- No data corruption

---

## Troubleshooting Test Failures

### Common Issues and Solutions

**Issue: Model loading fails**
```
Error: No GGUF model found
```
**Solution:**
```bash
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
```

---

**Issue: ChromaDB errors**
```
sqlite3.OperationalError: no such table
```
**Solution:**
```bash
# Delete and recreate databases
rm -rf ~/.jenova-ai/users/$USER/memory/*/chroma.sqlite3
# Restart JENOVA (will recreate)
```

---

**Issue: Out of memory**
```
CUDA error: out of memory
```
**Solution:**
```yaml
# Edit src/jenova/config/main_config.yaml
gpu_layers: 20  # Reduce from -1
context_size: 2048  # Reduce from 8192
```

---

**Issue: Slow performance**
```
Responses take 30+ seconds
```
**Solution:**
- Use smaller model (TinyLlama)
- Enable GPU if available
- Reduce max_tokens in config

---

## Test Result Reporting

### Creating a Test Report

After running tests, create a report:

```bash
cat > test_report_$(date +%Y%m%d).txt << EOF
JENOVA Test Report
Generated: $(date)
System: $(uname -a)
Python: $(python3 --version)

Pre-Installation Tests:
[✓] Test 1.1: Python Version
[✓] Test 1.2: Venv Module
...

Installation Tests:
[✓] Test 2.1: Virtual Environment
...

Component Tests:
[✓] Test 3.1: Configuration Loading
...

Cognitive Tests:
[✓] Test 4.1: Startup
[✓] Test 4.2: Conversation
[✓] Test 4.5: Insight Generation
...

Performance Results:
- Token generation: 45 tokens/sec (GPU)
- Memory search: 0.3 sec average
- RAM usage: 6.2 GB stable

Overall Status: PASS
Notes: All critical tests passed. Ready for production use.
EOF
```

---

## Continuous Testing

### Regular Testing Schedule

**Weekly:**
- Test 4.2: Basic Conversation
- Test 4.5: Insight Generation
- Test 5.3: Resource Usage

**Monthly:**
- Complete Component Testing (Tests 3.1-3.6)
- Complete Cognitive Testing (Tests 4.1-4.10)

**After Updates:**
- Complete Regression Testing (Tests 7.1-7.2)

---

## Automated Testing (Advanced)

### Creating a Test Script

```bash
cat > run_tests.sh << 'EOF'
#!/bin/bash

echo "JENOVA Automated Test Suite"
echo "==========================="

source venv/bin/activate

# Test 1: Configuration Loading
python -c "
from jenova.config import load_configuration
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger
import queue

msg_queue = queue.Queue()
ui_logger = UILogger(msg_queue)
file_logger = FileLogger('/tmp')
config = load_configuration(ui_logger, file_logger)
print('[PASS] Configuration Loading')
" || echo "[FAIL] Configuration Loading"

# Test 2: Embedding Model
python -c "
from jenova.utils.model_loader import load_embedding_model
model = load_embedding_model('all-MiniLM-L6-v2', device='cpu')
print('[PASS] Embedding Model Loading')
" || echo "[FAIL] Embedding Model Loading"

# Add more tests...

echo "==========================="
echo "Test suite complete"
EOF

chmod +x run_tests.sh
```

**Usage:**
```bash
./run_tests.sh
```

---

## Summary

This comprehensive testing guide ensures your JENOVA installation is:
- ✅ Correctly installed
- ✅ Fully functional
- ✅ Performing optimally
- ✅ Ready for production use

**Minimum Required Tests:**
- All Pre-Installation Tests (1.1-1.7)
- All Installation Tests (2.1-2.8)
- Critical Cognitive Tests (4.1, 4.2, 4.5)

**Recommended Tests:**
- All Component Tests (3.1-3.6)
- All Cognitive Tests (4.1-4.10)

**Optional Tests:**
- Performance Tests (5.1-5.3)
- Integration Tests (6.1-6.3)

For issues not covered here, consult:
- README.md - Architecture details
- DEPLOYMENT.md - Configuration options
- Logs at `~/.jenova-ai/users/$USER/jenova.log`

---

**End of Testing Instructions**

The JENOVA Cognitive Architecture - Designed and developed by orpheus497

**Related Documentation:**
- QUICKSTART.md - Getting started guide
- USAGE_EXAMPLES.md - Usage examples
- DEPLOYMENT.md - Deployment scenarios
- README.md - Complete architecture documentation
