# JENOVA Cognitive Architecture - Multi-Session Code Review Plan

## Purpose
This document contains the comprehensive plan for reviewing every file in the repository and adding inline comments according to the NON-NEGOTIABLE RULES.

---

## Commenting Standard

### Format Specification
All code must follow this commenting format:

```python
##Script function and purpose: [Explanation of what the entire script does]
##This script handles [detailed explanation]

import statements

##Function purpose: [Explanation of what the function does]
def function_name():
    """Optional docstring for documentation tools."""
    
    ##Block purpose: [Explanation of what this block does]
    code_block_here
    
    ##Block purpose: [Explanation of next block]
    next_code_block
```

### Rules
1. **Script Header:** Every file starts with `##Script function and purpose:` comment
2. **Function Comments:** Every function has `##Function purpose:` comment before `def`
3. **Block Comments:** Logical code blocks have `##Block purpose:` comments
4. **No Exceptions:** This applies to ALL code files
5. **Preserve Existing:** Don't remove existing comments, enhance them if needed

### Go File Commenting (tui/main.go)
```go
// Script function and purpose: [Explanation]

// Function purpose: [Explanation]
func functionName() {
    // Block purpose: [Explanation]
    codeBlock
}
```

---

## File Inventory

### Total Files to Review: 51

(Based on inventory: 10 Core Engine + 6 Cortex + 4 Memory + 4 UI + 10 Utils + 3 Insights + 2 Assumptions + 1 Config + 1 TUI + 6 Tests + 4 Root Scripts)

| Category | Count | Lines (Est.) | Priority |
|----------|-------|--------------|----------|
| Core Engine | 10 | 2,000 | HIGH |
| Cortex | 6 | 1,200 | HIGH |
| Memory | 4 | 800 | HIGH |
| UI | 4 | 600 | MEDIUM |
| Utils | 10 | 800 | MEDIUM |
| Insights | 3 | 400 | MEDIUM |
| Assumptions | 2 | 300 | MEDIUM |
| Config | 1 | 100 | LOW |
| TUI (Go) | 1 | 260 | MEDIUM |
| Tests | 6 | 500 | LOW |
| Root Scripts | 4 | 300 | LOW |

---

## Session-by-Session Plan

### Session 1: Initialization (CURRENT)
**Status:** IN PROGRESS
**Goals:**
- [x] Create .devdocs/ directory structure
- [x] Create all 8 documentation files
- [x] Document commenting standards
- [ ] Complete initial briefing
- [ ] Request permission for Phase 2

**Time Estimate:** 1 session

---

### Session 2: Core Engine - Entry Points
**Status:** PENDING
**Files:**
1. `src/jenova/main.py` (~130 lines) - PARTIALLY COMMENTED
   - Enhance existing comments
   - Add missing block comments
   
2. `src/jenova/main_bubbletea.py` (~140 lines)
   - Add script header
   - Add function comments
   - Add block comments

3. `src/jenova/llm_interface.py` (~200 lines)
   - Add script header
   - Add function comments
   - Add block comments

**Expected Output:** 3 files fully commented
**Time Estimate:** 1 session

---

### Session 3: Core Engine - Cognitive Processing
**Status:** PENDING
**Files:**
1. `src/jenova/cognitive_engine/engine.py` (~580 lines) - PARTIALLY COMMENTED
   - Enhance existing comments
   - Add missing block comments

2. `src/jenova/cognitive_engine/rag_system.py` (~200 lines)
   - Add script header
   - Add function comments
   - Add block comments

3. `src/jenova/cognitive_engine/memory_search.py` (~300 lines)
   - Add script header
   - Add function comments
   - Add block comments

**Expected Output:** 3 files fully commented
**Time Estimate:** 1 session

---

### Session 4: Core Engine - Query & Scheduling
**Status:** PENDING
**Files:**
1. `src/jenova/cognitive_engine/query_analyzer.py`
2. `src/jenova/cognitive_engine/scheduler.py`
3. `src/jenova/cognitive_engine/context_scorer.py`
4. `src/jenova/cognitive_engine/context_organizer.py`

**Expected Output:** 4 files fully commented
**Time Estimate:** 1 session

---

### Session 5: Cortex - Part 1
**Status:** PENDING
**Files:**
1. `src/jenova/cortex/cortex.py` (~700 lines) - PARTIALLY COMMENTED
   - Enhance existing comments
   - Add missing block comments

2. `src/jenova/cortex/clustering.py` (~200 lines)
3. `src/jenova/cortex/graph_metrics.py` (~200 lines)

**Expected Output:** 3 files fully commented
**Time Estimate:** 1 session

---

### Session 6: Cortex - Part 2
**Status:** PENDING
**Files:**
1. `src/jenova/cortex/graph_components.py`
2. `src/jenova/cortex/proactive_engine.py`
3. `src/jenova/cortex/__init__.py`

**Expected Output:** 3 files fully commented
**Time Estimate:** 1 session

---

### Session 7: Memory Systems
**Status:** PENDING
**Files:**
1. `src/jenova/memory/episodic.py`
2. `src/jenova/memory/semantic.py`
3. `src/jenova/memory/procedural.py`
4. `src/jenova/memory/__init__.py`

**Expected Output:** 4 files fully commented
**Time Estimate:** 1 session

---

### Session 8: UI Systems
**Status:** PENDING
**Files:**
1. `src/jenova/ui/terminal.py`
2. `src/jenova/ui/bubbletea.py`
3. `src/jenova/ui/logger.py`
4. `src/jenova/ui/__init__.py`
5. `tui/main.go` (Go commenting format)

**Expected Output:** 5 files fully commented
**Time Estimate:** 1 session

---

### Session 9: Utilities - Part 1
**Status:** PENDING
**Files:**
1. `src/jenova/utils/cache.py`
2. `src/jenova/utils/embedding.py`
3. `src/jenova/utils/file_logger.py`
4. `src/jenova/utils/json_parser.py`
5. `src/jenova/utils/__init__.py`

**Expected Output:** 5 files fully commented
**Time Estimate:** 1 session

---

### Session 10: Utilities - Part 2
**Status:** PENDING
**Files:**
1. `src/jenova/utils/model_loader.py`
2. `src/jenova/utils/performance_monitor.py`
3. `src/jenova/utils/pydantic_compat.py`
4. `src/jenova/utils/telemetry_fix.py`

**Expected Output:** 4 files fully commented
**Time Estimate:** 1 session

---

### Session 11: Supporting Modules
**Status:** PENDING
**Files:**
1. `src/jenova/insights/manager.py`
2. `src/jenova/insights/concerns.py`
3. `src/jenova/insights/__init__.py`
4. `src/jenova/assumptions/manager.py`
5. `src/jenova/assumptions/__init__.py`
6. `src/jenova/default_api.py`
7. `src/jenova/tools.py`

**Expected Output:** 7 files fully commented
**Time Estimate:** 1 session

---

### Session 12: Integration & Config
**Status:** PENDING
**Files:**
1. `src/jenova/cognitive_engine/integration_layer.py`
2. `src/jenova/cognitive_engine/document_processor.py`
3. `src/jenova/cognitive_engine/__init__.py`
4. `src/jenova/config/__init__.py`
5. `src/jenova/__init__.py`
6. `src/jenova/docs/__init__.py`

**Expected Output:** 6 files fully commented
**Time Estimate:** 1 session

---

### Session 13: Tests
**Status:** PENDING
**Files:**
1. `tests/test_basic.py`
2. `tests/test_cognitive_engine.py`
3. `tests/test_cortex.py`
4. `tests/test_memory.py`
5. `tests/conftest.py`
6. `tests/__init__.py`

**Expected Output:** 6 files fully commented
**Time Estimate:** 1 session

---

### Session 14: Root Scripts & Finetune
**Status:** PENDING
**Files:**
1. `setup.py`
2. `fix_chromadb_compat.py`
3. `test_tui.py`
4. `demo_ui.py`
5. `finetune/train.py`

**Expected Output:** 5 files fully commented
**Time Estimate:** 1 session

---

### Session 15: Final Review & Perfection
**Status:** PENDING
**Goals:**
1. Verify all 51+ files have 100% comment coverage
2. Review documentation consistency
3. Update all .devdocs/ files
4. Create final summary
5. Mark system as complete

**Time Estimate:** 1 session

---

## Quality Checklist

For each file reviewed, verify:
- [ ] Script has header comment (`##Script function and purpose:`)
- [ ] All functions have `##Function purpose:` comments
- [ ] All logical blocks have `##Block purpose:` comments
- [ ] Comments are clear and descriptive
- [ ] No code functionality was changed
- [ ] Existing comments were preserved/enhanced

---

## Workflow Per Session

1. **Read .devdocs/ documentation**
2. **Provide briefing to user**
3. **Request and receive permission**
4. **For each file in session:**
   a. View current file content
   b. Identify missing comments
   c. Add comments following standard
   d. Verify no code changes
   e. Mark file complete in PROGRESS.md
5. **Update all .devdocs/ files**
6. **Commit changes**
7. **Report to user**

---

## Notes

- Files marked "PARTIALLY COMMENTED" already have some comments in the standard format
- Preserve all existing functionality (NON-NEGOTIABLE)
- If unsure about a code block's purpose, mark for clarification
- Update this plan as sessions progress
