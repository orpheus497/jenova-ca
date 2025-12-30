# JENOVA Cognitive Architecture - Decisions Log

## Purpose
This document records all architectural and process decisions made during development. Each decision includes the context, rationale, alternatives considered, and outcome.

---

## Decision Log

### DEC-001: Documentation Folder Structure
**Date:** [Current Date]
**Session:** 1
**Author:** Copilot Coding Agent

**Context:**
The problem statement specifies NON-NEGOTIABLE RULES for documentation organization. A .devdocs/ folder structure was required.

**Decision:**
Create .devdocs/ folder with 8 required documentation files:
- BRIEFING.md - Current project status
- PROGRESS.md - Progress tracking
- SESSION_HANDOFF.md - Session-to-session continuity
- DECISIONS_LOG.md - Architectural decisions (this file)
- TODOS.md - Task lists
- PLANS.md - Planning documents
- TESTS.md - Test documentation
- SUMMARIES.md - Session summaries

**Rationale:**
- Follows NON-NEGOTIABLE RULES from problem statement
- Enables clear separation between AI/developer process docs and product code
- Provides structured handoff between sessions

**Alternatives Considered:**
- Using root-level documentation files (rejected: violates rules)
- Minimal documentation (rejected: violates rules)

**Outcome:**
Structure implemented as specified.

---

### DEC-002: Code Commenting Format Standard
**Date:** [Current Date]
**Session:** 1
**Author:** Copilot Coding Agent

**Context:**
The problem statement requires ALL code to have inline comments. A consistent format was needed.

**Decision:**
Adopt the format already used in main.py, engine.py, and cortex.py:
```python
##Script function and purpose: [Explanation]

##Function purpose: [Explanation]
def function_name():
    ##Block purpose: [Explanation]
    code
```

**Rationale:**
- Format already exists in codebase (consistency)
- `##` prefix distinguishes from regular Python `#` comments
- Clear hierarchy: Script > Function > Block
- Enables developers and AI agents to understand code immediately

**Alternatives Considered:**
- Docstrings only (rejected: doesn't cover blocks, not visible inline)
- Standard `#` comments (rejected: less distinctive)
- Google-style docstrings (rejected: existing pattern preferred)

**Outcome:**
Standard adopted and documented in PLANS.md.

---

### DEC-003: Multi-Session Review Approach
**Date:** [Current Date]
**Session:** 1
**Author:** Copilot Coding Agent

**Context:**
Repository contains 57 Python files (~6,117 lines). Complete review in single session not feasible.

**Decision:**
Divide review into 10-15 sessions with:
- 3-5 files per session
- Priority order: Core Engine → Cortex → Memory → UI → Utils → Tests
- Each session updates all documentation files
- Explicit permission required before each phase

**Rationale:**
- Manageable scope per session
- Priority order reflects component importance
- Documentation updates ensure continuity
- Permission checkpoints follow NON-NEGOTIABLE RULES

**Alternatives Considered:**
- Single massive session (rejected: not feasible)
- Arbitrary file order (rejected: no logical progression)
- Automatic continuation (rejected: violates permission rules)

**Outcome:**
Plan documented in PLANS.md.

---

### DEC-004: Existing Comments Preservation
**Date:** [Current Date]
**Session:** 1
**Author:** Copilot Coding Agent

**Context:**
Some files (main.py, engine.py, cortex.py) already have partial commenting following the standard format.

**Decision:**
- Preserve all existing comments
- Add missing comments to achieve 100% coverage
- Do not modify working code unless necessary for commenting

**Rationale:**
- Follows NON-NEGOTIABLE RULE: ALL features must be preserved
- Minimizes risk of breaking changes
- Respects existing developer work

**Alternatives Considered:**
- Rewriting all comments (rejected: unnecessary, risky)
- Ignoring partially commented files (rejected: incomplete coverage)

**Outcome:**
Approach documented in PLANS.md.

---

## Pending Decisions

### PENDING-001: Go File Commenting Convention
**Status:** Requires Discussion
**Context:** tui/main.go uses Go commenting conventions. Need to determine if Go style or adapted format should be used.

**Proposed Options:**
1. Use Go standard `//` comments with similar purpose structure
2. Adapt `##` format to `// ##Block purpose:`
3. Leave as-is with minimal additions

**Awaiting:** User input/permission

---

## Decision Template

```markdown
### DEC-XXX: [Title]
**Date:** [Date]
**Session:** [Session Number]
**Author:** [Author]

**Context:**
[Background and problem being addressed]

**Decision:**
[What was decided]

**Rationale:**
[Why this decision was made]

**Alternatives Considered:**
[Other options and why they were rejected]

**Outcome:**
[Result of implementing the decision]
```
