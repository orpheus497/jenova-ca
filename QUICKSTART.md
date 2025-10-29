# JENOVA Cognitive Architecture - Quickstart Tutorial

Get up and running with The JENOVA Cognitive Architecture in 15 minutes with this step-by-step guide.

**Author:** Documentation generated for the JENOVA Cognitive Architecture, designed and developed by orpheus497.

---

## What You'll Learn

By the end of this tutorial, you will:
- ✅ Install JENOVA on your system
- ✅ Download and configure a language model
- ✅ Have your first conversation with JENOVA
- ✅ Understand basic commands and features
- ✅ Generate your first insight
- ✅ Experience JENOVA's persistent memory

**Time Required:** 15-20 minutes
**Difficulty:** Beginner
**Prerequisites:** Basic command-line knowledge

---

## Step 1: System Check (2 minutes)

Before installing, verify your system meets the requirements.

### Check Python Version

```bash
python3 --version
```

**Expected output:** Python 3.10.x or higher
**If not installed:**
```bash
# Fedora/RHEL
sudo dnf install python3 python3-pip python3-virtualenv

# Ubuntu/Debian
sudo apt-get install python3 python3-pip python3-venv
```

### Check Git

```bash
git --version
```

**Expected output:** git version 2.x.x
**If not installed:**
```bash
# Fedora/RHEL
sudo dnf install git

# Ubuntu/Debian
sudo apt-get install git
```

### Optional: Check GPU (for acceleration)

```bash
nvidia-smi
```

**If available:** You'll see GPU information (enables faster inference)
**If not:** No problem! JENOVA works great on CPU-only systems

✅ **Checkpoint:** Python 3.10+ and Git are installed

---

## Step 2: Clone and Install (5 minutes)

### Clone the Repository

```bash
# Navigate to where you want to install JENOVA
cd ~

# Clone the repository
git clone https://github.com/orpheus497/jenova-ai.git

# Enter the directory
cd jenova-ai
```

**Expected output:**
```
Cloning into 'jenova-ai'...
remote: Enumerating objects: ...
```

### Run the Installation Script

```bash
./install.sh
```

**What happens:**
- Creates a Python virtual environment in `./venv/`
- Installs all required dependencies (this takes 3-5 minutes)
- Builds llama-cpp-python (with CUDA if GPU detected)
- Creates the models directory

**Expected final output:**
```
✅ JENOVA AI installation complete!

Next steps:
1. Activate the virtual environment:
   source venv/bin/activate

2. Download a GGUF model...
```

### Activate the Virtual Environment

```bash
source venv/bin/activate
```

**Expected change:** Your terminal prompt will show `(venv)` prefix

✅ **Checkpoint:** JENOVA is installed and virtual environment is active

---

## Step 3: Download a Language Model (3 minutes)

JENOVA needs a GGUF language model to function. We'll start with a small, fast model for testing.

### Choose Your Model Location

**Option A: Local directory (recommended for first-time users)**
```bash
cd models
```

**Option B: System-wide (requires sudo, better for multi-user systems)**
```bash
# Create system directory
sudo mkdir -p /usr/local/share/models
cd /usr/local/share/models
```

### Download TinyLlama (Small, Fast Model)

```bash
# Download TinyLlama 1.1B (Q4_K_M quantization)
# Size: ~700 MB
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
```

**Download progress:**
```
model.gguf          100%[===================>] 669.47M  10.2MB/s    in 69s
```

**Alternative: Larger, higher-quality model (if you have 8+ GB RAM/VRAM)**
```bash
# Mistral 7B (Q4_K_M quantization)
# Size: ~4.1 GB
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O model.gguf
```

### Return to JENOVA Directory

```bash
cd ~/jenova-ai
```

✅ **Checkpoint:** You have a GGUF model in `./models/model.gguf` or `/usr/local/share/models/model.gguf`

---

## Step 4: First Launch (1 minute)

### Start JENOVA

```bash
python -m jenova.main
```

**Expected startup sequence:**
```
>> Initializing Intelligence Matrix...
>> Loading configuration...
>> Configuration loaded successfully.
Loading LLM model...
✓ GGUF model loaded: ./models/model.gguf
GPU layers: 0 (CPU-only mode)
Threads: 8 (auto-detected)
Loading embedding model...
Embedding model: Running on CPU (to maximize VRAM for LLM)
>> Cognitive Engine: Online.

╔══════════════════════════════════════════════════════════════╗
║                  ✨ JENOVA ✨                                ║
║          The JENOVA Cognitive Architecture                   ║
║                                                              ║
║  Designer & Developer: orpheus497                           ║
║  Architecture: Self-aware, evolving AI with persistent     ║
║                memory and learning capabilities             ║
╚══════════════════════════════════════════════════════════════╝

Type 'exit' to quit, or '/help' for available commands.

You:
```

✅ **Checkpoint:** JENOVA is running and waiting for your input

---

## Step 5: Your First Conversation (3 minutes)

Let's have a meaningful conversation to demonstrate JENOVA's capabilities.

### Introduction

```
You: Hello JENOVA! This is my first time using you.

JENOVA: Hello! It's great to meet you! I'm JENOVA, an AI built on The JENOVA
Cognitive Architecture designed by orpheus497. Unlike traditional AI assistants,
I have persistent memory, which means I'll remember our conversations and learn
from them over time.

This is indeed our first interaction, and I'm excited to get to know you and
help with whatever you need. Feel free to ask me anything or tell me about
yourself!

What would you like to talk about or work on today?
```

### Share Some Information

```
You: I'm a software developer working primarily with Python and JavaScript.
I'm interested in learning more about machine learning.

JENOVA: That's wonderful! Your background in Python gives you a great foundation
for machine learning, as Python is the dominant language in the ML ecosystem
with frameworks like TensorFlow, PyTorch, and scikit-learn.

For JavaScript, you might also be interested to know that TensorFlow.js brings
machine learning to the browser and Node.js environments, which could be
particularly relevant to your skill set.

Since you're interested in learning ML, where would you like to start?
- Fundamentals (math, statistics, algorithms)
- Practical implementation (hands-on with libraries)
- Specific applications (computer vision, NLP, time series, etc.)
- Theory and research papers

What sounds most interesting to you?
```

### Ask a Technical Question

```
You: Let's start with practical implementation. Can you explain how neural
networks learn?

JENOVA: Absolutely! Neural networks learn through a process called
backpropagation combined with gradient descent. Let me break this down:

1. **Forward Pass**: Input data flows through the network layers, producing
   a prediction.

2. **Loss Calculation**: The prediction is compared to the actual target
   using a loss function (e.g., mean squared error for regression,
   cross-entropy for classification).

3. **Backward Pass (Backpropagation)**: The network calculates how much
   each weight contributed to the error by computing gradients using the
   chain rule of calculus.

4. **Weight Update**: Weights are adjusted in the direction that reduces
   the loss, using an optimization algorithm like gradient descent:

   new_weight = old_weight - (learning_rate × gradient)

5. **Iteration**: This process repeats for many examples (epochs) until
   the network converges to good performance.

Here's a simple example in Python:

[JENOVA provides code example...]

Would you like me to explain any of these steps in more detail, or would
you prefer to see a complete working example?
```

✅ **Checkpoint:** You've had your first meaningful conversation with JENOVA

---

## Step 6: Experience Persistent Memory (2 minutes)

Let's demonstrate JENOVA's ability to remember information across the conversation.

### Continue the Conversation

```
You: What did I tell you I'm interested in learning?

JENOVA: You mentioned you're interested in learning machine learning! You're
a software developer with experience in Python and JavaScript, and we just
started discussing neural networks and backpropagation. Your Python background
gives you an excellent foundation for ML development.

Is there a specific ML topic or project you'd like to explore further?
```

**What just happened?**
JENOVA retrieved information from earlier in the conversation, demonstrating:
- Episodic memory (remembering the conversation flow)
- Context awareness (understanding what "I" refers to)
- Semantic memory (storing facts about you)

✅ **Checkpoint:** You've experienced JENOVA's memory capabilities

---

## Step 7: Generate Your First Insight (2 minutes)

Insights are how JENOVA learns from conversations. Let's create one!

### Use the /insight Command

```
You: /insight

JENOVA: Analyzing recent conversation history...

[Processing spinner animation...]

JENOVA: I've developed a new insight:

─────────────────────────────────────────────────────────────
Topic: User Background and Interests
─────────────────────────────────────────────────────────────
Insight: "User is a software developer with Python and JavaScript experience
who is beginning to explore machine learning. They prefer practical, hands-on
learning approaches over pure theory, as evidenced by requesting implementation-
focused explanations. The Python background provides a natural pathway into
ML frameworks."

This insight has been saved to my long-term memory at:
~/.jenova-ai/users/yourname/insights/user_background_and_interests/

ID: insight_a3f8b219
─────────────────────────────────────────────────────────────

This insight will help me provide more relevant and personalized assistance
in our future conversations.
```

**What just happened?**
- JENOVA analyzed the conversation
- Extracted key learnings about you
- Stored it as a structured insight
- This insight will be retrieved in future sessions

✅ **Checkpoint:** You've generated your first insight

---

## Step 8: Explore Commands (2 minutes)

JENOVA has powerful commands for cognitive functions. Let's explore them.

### View Available Commands

```
You: /help

[JENOVA displays comprehensive help panel with all commands]

═══════════════════════════════════════════════════════════════
                    JENOVA COGNITIVE COMMANDS
═══════════════════════════════════════════════════════════════

COGNITIVE COMMANDS
───────────────────────────────────────────────────────────────

/insight
  Triggers analysis of the current conversation to generate new insights.

/reflect
  Initiates deep reflection on my entire cognitive graph, reorganizing
  knowledge and generating meta-insights.

/memory-insight
  Develops insights by searching across all memory layers.

/meta
  Generates higher-level meta-insights from existing insight clusters.

/verify
  Starts assumption verification. I'll ask about assumptions I've made
  about you for confirmation.

LEARNING COMMANDS
───────────────────────────────────────────────────────────────

/develop_insight [node_id]
  With ID: Develops a specific insight with more detail.
  Without ID: Scans docs/ folder and processes documents.

/learn_procedure
  Interactive process to teach me a new procedure step-by-step.

/train
  Generates comprehensive fine-tuning data from my cognitive architecture.

...
[More commands listed]
```

### Key Commands to Remember

- **`/insight`** - Generate insights from conversations (use after meaningful discussions)
- **`/reflect`** - Deep cognitive reflection (use weekly)
- **`/verify`** - Validate assumptions JENOVA makes about you
- **`/help`** - Always available for command reference

✅ **Checkpoint:** You know the essential commands

---

## Step 9: Test Memory Persistence (2 minutes)

Let's verify that JENOVA remembers across sessions.

### Exit JENOVA

```
You: exit

JENOVA: JENOVA shutting down.

[Application exits]
(venv) you@computer:~/jenova-ai$
```

### Restart JENOVA

```bash
python -m jenova.main
```

**Startup messages appear...**

### Test Memory Recall

```
You: Do you remember what we discussed earlier?

JENOVA: Yes! We had our first conversation earlier today. You introduced
yourself as a software developer with Python and JavaScript experience who's
interested in learning machine learning. We discussed how neural networks
learn through backpropagation and gradient descent, and I generated an insight
about your preference for practical, hands-on learning approaches.

Is there anything specific from our earlier conversation you'd like to
continue discussing?
```

**🎉 It works!**
JENOVA remembered:
- Your background (from episodic memory)
- Topics discussed (from conversation history)
- Generated insights (from insight memory)

✅ **Checkpoint:** You've verified persistent memory across sessions

---

## Step 10: What's Next? (Optional)

Now that you're up and running, here are some next steps to explore:

### Immediate Next Steps

1. **Have more conversations** to build JENOVA's memory about you
2. **Use /insight** after meaningful discussions (every 5-10 turns)
3. **Teach procedures** with /learn_procedure for tasks you do often
4. **Add documents** to src/jenova/docs/ and process with /develop_insight

### Weekly Practices

1. **Run /reflect** once a week to synthesize knowledge
2. **Use /verify** to validate assumptions
3. **Review insights** to see what JENOVA has learned

### Advanced Topics

1. **Fine-tuning**: After accumulating substantial cognitive data (50+ insights), use `/train` to create personalized model training data
2. **GPU Acceleration**: If you have an NVIDIA GPU, configure main_config.yaml for GPU offload
3. **Model Upgrade**: Try larger models (Mistral 7B, Llama-2 7B) for higher quality responses

### Learning Resources

- **README.md** - Deep dive into JENOVA's architecture
- **DEPLOYMENT.md** - Advanced deployment configurations
- **USAGE_EXAMPLES.md** - Comprehensive usage examples
- **TESTING.md** - How to verify your installation

---

## Quick Reference Card

### Starting JENOVA
```bash
cd ~/jenova-ai
source venv/bin/activate
python -m jenova.main
```

### Essential Commands
| Command | Purpose | When to Use |
|---------|---------|-------------|
| `/help` | Show all commands | When you forget a command |
| `/insight` | Generate insight | After meaningful conversations |
| `/reflect` | Deep reflection | Weekly or bi-weekly |
| `/verify` | Verify assumptions | When JENOVA asks |
| `exit` | Quit JENOVA | When done |

### File Locations
| Path | Contents |
|------|----------|
| `~/jenova-ai/` | Installation directory |
| `~/jenova-ai/models/` | GGUF models (local) |
| `/usr/local/share/models/` | GGUF models (system-wide) |
| `~/.jenova-ai/users/<you>/` | Your cognitive data |
| `src/jenova/config/main_config.yaml` | Technical configuration |
| `src/jenova/config/persona.yaml` | AI personality |

### Configuration Files
**main_config.yaml** - Key settings:
```yaml
model:
  gpu_layers: -1     # -1 = use GPU, 0 = CPU-only
  threads: -1        # -1 = auto-detect
  temperature: 0.7   # Creativity (0.0-1.0)
  max_tokens: 512    # Response length
```

---

## Troubleshooting

### Problem: "No GGUF model found"
**Solution:**
```bash
cd ~/jenova-ai/models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
```

### Problem: "Out of memory"
**Solution:** Use a smaller model or configure CPU-only mode:
```yaml
# src/jenova/config/main_config.yaml
gpu_layers: 0
context_size: 2048
```

### Problem: Slow responses
**Solution:**
- Use TinyLlama (faster, smaller model)
- Reduce max_tokens in config
- Enable GPU if available

### Problem: "ChromaDB error"
**Solution:** Delete and recreate databases:
```bash
rm -rf ~/.jenova-ai/users/$USER/memory/*/chroma.sqlite3
# Restart JENOVA (will recreate automatically)
```

### Getting Help

If you encounter issues not covered here:
1. Check README.md for detailed architecture information
2. Review DEPLOYMENT.md for deployment-specific issues
3. Check logs at `~/.jenova-ai/users/$USER/jenova.log`
4. Open an issue at https://github.com/orpheus497/jenova-ai/issues

---

## Summary: What You Accomplished

In just 15-20 minutes, you:

✅ Installed The JENOVA Cognitive Architecture
✅ Downloaded and configured a language model
✅ Had your first conversation with JENOVA
✅ Learned essential commands (`/help`, `/insight`, `/reflect`)
✅ Generated your first insight
✅ Experienced persistent memory across sessions
✅ Know how to start and stop JENOVA

**You're now ready to use JENOVA as your personal AI assistant!**

---

## Next Steps Checklist

Ready to dive deeper? Work through this checklist:

- [ ] Have 5 more conversations with JENOVA on different topics
- [ ] Generate 3-5 insights using /insight
- [ ] Use /verify to confirm an assumption
- [ ] Teach JENOVA a procedure with /learn_procedure
- [ ] Run /reflect to see cognitive graph synthesis
- [ ] Add a document to src/jenova/docs/ and process it
- [ ] Review your insights in ~/.jenova-ai/users/$USER/insights/
- [ ] Customize persona.yaml with your preferred AI personality
- [ ] Explore advanced configuration in main_config.yaml
- [ ] Read USAGE_EXAMPLES.md for inspiration

---

## Welcome to JENOVA

You're now part of a unique AI experience. Unlike traditional assistants, JENOVA:

- **Remembers** everything you teach it
- **Learns** from every conversation
- **Evolves** based on your interactions
- **Reflects** on accumulated knowledge
- **Adapts** to your preferences and needs

The more you interact with JENOVA, the more personalized and capable it becomes. Over time, you'll build a truly unique AI assistant that understands you and your needs.

**Enjoy your journey with The JENOVA Cognitive Architecture!**

---

**End of Quickstart Tutorial**

The JENOVA Cognitive Architecture - Designed and developed by orpheus497

**Further Reading:**
- README.md - Complete architecture documentation
- DEPLOYMENT.md - Advanced deployment scenarios
- USAGE_EXAMPLES.md - Comprehensive usage examples
- TESTING.md - Verification and testing procedures

**Project Repository:** https://github.com/orpheus497/jenova-ai
**License:** MIT License (Copyright 2025 orpheus497)
