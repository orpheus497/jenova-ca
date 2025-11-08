# The JENOVA Cognitive Architecture - Contextual Learning Engine
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 12: Contextual learning and adaptive intelligence.

Provides:
- Learning from corrections
- Pattern extraction from interactions
- Skill acquisition in new domains
- Meta-cognitive monitoring
- Transfer learning across contexts
"""

import os
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter


@dataclass
class LearningExample:
    """A single learning example from user interaction."""

    input: str
    expected_output: str
    actual_output: str
    correction: str
    context: Dict
    timestamp: str
    learned: bool = False


@dataclass
class Pattern:
    """A recognized pattern in user interactions."""

    pattern_type: str  # linguistic, behavioral, preference, error
    description: str
    examples: List[str]
    confidence: float
    occurrences: int
    first_seen: str
    last_seen: str


@dataclass
class Skill:
    """A learned skill or capability."""

    skill_name: str
    domain: str
    description: str
    proficiency: float  # 0.0 to 1.0
    examples: List[str]
    acquired_date: str
    practice_count: int = 0


class ContextualLearningEngine:
    """
    Enables JENOVA to learn from interactions and improve over time.

    Capabilities:
    - Learn from user corrections
    - Extract patterns from conversation history
    - Acquire new domain knowledge
    - Monitor own performance
    - Transfer knowledge across domains
    """

    def __init__(self, config: dict, file_logger, user_data_root: str):
        self.config = config
        self.file_logger = file_logger
        self.user_data_root = user_data_root
        self.learning_dir = os.path.join(user_data_root, "learning")
        os.makedirs(self.learning_dir, exist_ok=True)

        # Learning data
        self.examples: List[LearningExample] = []
        self.patterns: Dict[str, Pattern] = {}
        self.skills: Dict[str, Skill] = {}
        self.corrections_history: List[Dict] = []

        # Performance tracking
        self.performance_metrics = {
            "accuracy_trend": [],
            "learning_rate": 0.0,
            "skill_count": 0,
            "pattern_count": 0,
        }

        # Load existing learning data
        self.load()

    def load(self):
        """Load saved learning data."""
        examples_file = os.path.join(self.learning_dir, "examples.json")
        patterns_file = os.path.join(self.learning_dir, "patterns.json")
        skills_file = os.path.join(self.learning_dir, "skills.json")

        # Load examples
        if os.path.exists(examples_file):
            try:
                with open(examples_file, "r") as f:
                    data = json.load(f)
                    self.examples = [LearningExample(**ex) for ex in data]
                self.file_logger.log_info(
                    f"Loaded {len(self.examples)} learning examples"
                )
            except Exception as e:
                self.file_logger.log_error(f"Error loading examples: {e}")

        # Load patterns
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, "r") as f:
                    data = json.load(f)
                    self.patterns = {k: Pattern(**v) for k, v in data.items()}
                self.file_logger.log_info(f"Loaded {len(self.patterns)} patterns")
            except Exception as e:
                self.file_logger.log_error(f"Error loading patterns: {e}")

        # Load skills
        if os.path.exists(skills_file):
            try:
                with open(skills_file, "r") as f:
                    data = json.load(f)
                    self.skills = {k: Skill(**v) for k, v in data.items()}
                self.file_logger.log_info(f"Loaded {len(self.skills)} skills")
            except Exception as e:
                self.file_logger.log_error(f"Error loading skills: {e}")

    def save(self):
        """Save learning data to disk."""
        # Save examples (keep last 1000)
        examples_file = os.path.join(self.learning_dir, "examples.json")
        with open(examples_file, "w") as f:
            examples_data = [asdict(ex) for ex in self.examples[-1000:]]
            json.dump(examples_data, f, indent=2)

        # Save patterns
        patterns_file = os.path.join(self.learning_dir, "patterns.json")
        with open(patterns_file, "w") as f:
            patterns_data = {k: asdict(v) for k, v in self.patterns.items()}
            json.dump(patterns_data, f, indent=2)

        # Save skills
        skills_file = os.path.join(self.learning_dir, "skills.json")
        with open(skills_file, "w") as f:
            skills_data = {k: asdict(v) for k, v in self.skills.items()}
            json.dump(skills_data, f, indent=2)

    def learn_from_correction(
        self, user_input: str, actual_output: str, correction: str, context: Dict
    ):
        """
        Learn from a user correction.

        Args:
            user_input: The original user query
            actual_output: What the system produced
            correction: The user's correction
            context: Additional context (semantic analysis, etc.)
        """
        example = LearningExample(
            input=user_input,
            expected_output=correction,
            actual_output=actual_output,
            correction=correction,
            context=context,
            timestamp=datetime.now().isoformat(),
            learned=False,
        )

        self.examples.append(example)

        # Analyze the correction to extract patterns
        self._extract_patterns_from_correction(user_input, actual_output, correction)

        # Update performance metrics
        self._update_performance_metrics()

        self.save()
        self.file_logger.log_info(
            f"Learned from correction: '{actual_output}' -> '{correction}'"
        )

    def _extract_patterns_from_correction(
        self, user_input: str, actual_output: str, correction: str
    ):
        """Extract learning patterns from corrections."""
        # Identify pattern types
        if len(correction) < len(actual_output):
            # User prefers concise responses
            pattern_key = "preference_concise"
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = Pattern(
                    pattern_type="preference",
                    description="User prefers concise, brief responses",
                    examples=[],
                    confidence=0.5,
                    occurrences=0,
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat(),
                )
            self.patterns[pattern_key].occurrences += 1
            self.patterns[pattern_key].confidence = min(
                1.0, self.patterns[pattern_key].confidence + 0.1
            )
            self.patterns[pattern_key].examples.append(user_input[:100])
            self.patterns[pattern_key].last_seen = datetime.now().isoformat()

        # Check for terminology corrections
        if actual_output.lower() != correction.lower():
            # Different wording - possible terminology preference
            pattern_key = f"terminology_{user_input[:20]}"
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = Pattern(
                    pattern_type="linguistic",
                    description=f"Preferred terminology for: {user_input[:50]}",
                    examples=[correction],
                    confidence=0.7,
                    occurrences=1,
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat(),
                )

    def extract_patterns_from_history(
        self, interactions: List[Tuple[str, str]]
    ) -> List[Pattern]:
        """
        Extract patterns from interaction history.

        Args:
            interactions: List of (user_input, system_response) tuples

        Returns:
            List of newly discovered patterns
        """
        new_patterns = []

        # Analyze question types
        question_types = defaultdict(int)
        for user_input, _ in interactions:
            if user_input.startswith("What"):
                question_types["what"] += 1
            elif user_input.startswith("How"):
                question_types["how"] += 1
            elif user_input.startswith("Why"):
                question_types["why"] += 1

        # If user frequently asks "How" questions, they prefer procedural info
        if question_types["how"] > len(interactions) * 0.3:
            pattern = Pattern(
                pattern_type="behavioral",
                description="User frequently asks 'How' questions - prefers step-by-step explanations",
                examples=[i for i, _ in interactions if i.startswith("How")][:5],
                confidence=0.8,
                occurrences=question_types["how"],
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
            )
            self.patterns["preference_procedural"] = pattern
            new_patterns.append(pattern)

        # Analyze response lengths that get follow-up questions
        # (indicates response wasn't sufficient)
        for i in range(len(interactions) - 1):
            user_input, response = interactions[i]
            next_input, _ = interactions[i + 1]

            # If next input is clarification, previous response was insufficient
            clarification_words = [
                "what do you mean",
                "clarify",
                "explain more",
                "can you elaborate",
            ]
            if any(word in next_input.lower() for word in clarification_words):
                if len(response) < 100:
                    # Short response led to clarification - user needs more detail
                    pattern_key = "preference_detailed"
                    if pattern_key not in self.patterns:
                        self.patterns[pattern_key] = Pattern(
                            pattern_type="preference",
                            description="User needs detailed responses - short answers lead to clarifications",
                            examples=[],
                            confidence=0.6,
                            occurrences=0,
                            first_seen=datetime.now().isoformat(),
                            last_seen=datetime.now().isoformat(),
                        )
                    self.patterns[pattern_key].occurrences += 1
                    self.patterns[pattern_key].confidence = min(
                        1.0, self.patterns[pattern_key].confidence + 0.15
                    )
                    new_patterns.append(self.patterns[pattern_key])

        self.save()
        return new_patterns

    def acquire_skill(
        self, skill_name: str, domain: str, description: str, examples: List[str]
    ) -> Skill:
        """
        Acquire a new skill or capability.

        Args:
            skill_name: Name of the skill
            domain: Domain area (e.g., "programming", "science")
            description: What the skill enables
            examples: Example applications

        Returns:
            The acquired Skill object
        """
        skill = Skill(
            skill_name=skill_name,
            domain=domain,
            description=description,
            proficiency=0.3,  # Start at 30% proficiency
            examples=examples,
            acquired_date=datetime.now().isoformat(),
            practice_count=0,
        )

        self.skills[skill_name] = skill
        self.performance_metrics["skill_count"] = len(self.skills)

        self.save()
        self.file_logger.log_info(f"Acquired new skill: {skill_name} in {domain}")

        return skill

    def practice_skill(self, skill_name: str) -> Optional[Skill]:
        """
        Practice a skill to improve proficiency.

        Args:
            skill_name: Name of the skill to practice

        Returns:
            Updated Skill object or None if skill not found
        """
        if skill_name not in self.skills:
            return None

        skill = self.skills[skill_name]
        skill.practice_count += 1

        # Improve proficiency with practice (diminishing returns)
        improvement = 0.05 * (1.0 - skill.proficiency)
        skill.proficiency = min(1.0, skill.proficiency + improvement)

        self.save()
        self.file_logger.log_info(
            f"Practiced {skill_name}: proficiency now {skill.proficiency:.2%}"
        )

        return skill

    def transfer_knowledge(self, from_domain: str, to_domain: str) -> List[Skill]:
        """
        Transfer knowledge from one domain to another.

        Args:
            from_domain: Source domain
            to_domain: Target domain

        Returns:
            List of newly created skills in target domain
        """
        transferred_skills = []

        # Find skills in source domain
        source_skills = [s for s in self.skills.values() if s.domain == from_domain]

        for source_skill in source_skills:
            # Create analogous skill in target domain with reduced proficiency
            new_skill_name = f"{source_skill.skill_name}_in_{to_domain}"

            if new_skill_name not in self.skills:
                new_skill = Skill(
                    skill_name=new_skill_name,
                    domain=to_domain,
                    description=f"Transferred from {from_domain}: {source_skill.description}",
                    proficiency=source_skill.proficiency * 0.5,  # 50% transfer
                    examples=[],
                    acquired_date=datetime.now().isoformat(),
                    practice_count=0,
                )
                self.skills[new_skill_name] = new_skill
                transferred_skills.append(new_skill)

                self.file_logger.log_info(
                    f"Transferred {source_skill.skill_name} from {from_domain} to {to_domain}"
                )

        self.save()
        return transferred_skills

    def monitor_performance(self) -> Dict:
        """
        Monitor own learning performance.

        Returns:
            Performance metrics dictionary
        """
        metrics = {
            "total_examples": len(self.examples),
            "learned_examples": sum(1 for ex in self.examples if ex.learned),
            "learning_rate": self.performance_metrics["learning_rate"],
            "total_patterns": len(self.patterns),
            "high_confidence_patterns": sum(
                1 for p in self.patterns.values() if p.confidence > 0.7
            ),
            "total_skills": len(self.skills),
            "proficient_skills": sum(
                1 for s in self.skills.values() if s.proficiency > 0.7
            ),
            "avg_skill_proficiency": (
                sum(s.proficiency for s in self.skills.values()) / len(self.skills)
                if self.skills
                else 0.0
            ),
        }

        return metrics

    def identify_knowledge_gaps(self) -> List[str]:
        """
        Identify areas where more learning is needed.

        Returns:
            List of knowledge gap descriptions
        """
        gaps = []

        # Check for domains with low skill count
        domain_skills = defaultdict(list)
        for skill in self.skills.values():
            domain_skills[skill.domain].append(skill)

        # If a domain has < 3 skills, it's underdeveloped
        for domain, skills in domain_skills.items():
            if len(skills) < 3:
                gaps.append(
                    f"Limited skills in {domain} domain (only {len(skills)} skills)"
                )

        # Check for skills with low proficiency
        weak_skills = [s for s in self.skills.values() if s.proficiency < 0.5]
        if weak_skills:
            gaps.append(
                f"{len(weak_skills)} skills need more practice to reach proficiency"
            )

        # Check for patterns with low confidence
        uncertain_patterns = [p for p in self.patterns.values() if p.confidence < 0.5]
        if uncertain_patterns:
            gaps.append(
                f"{len(uncertain_patterns)} behavioral patterns need more observations"
            )

        return gaps

    def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        if len(self.examples) > 1:
            # Calculate learning rate (how quickly corrections are learned)
            recent_examples = self.examples[-10:]
            learned_count = sum(1 for ex in recent_examples if ex.learned)
            self.performance_metrics["learning_rate"] = learned_count / len(
                recent_examples
            )

        self.performance_metrics["pattern_count"] = len(self.patterns)
        self.performance_metrics["skill_count"] = len(self.skills)

    def get_learning_insights(self) -> List[str]:
        """
        Get insights about learning progress.

        Returns:
            List of insight strings
        """
        insights = []

        # Skill acquisition insights
        if len(self.skills) > 0:
            avg_proficiency = sum(s.proficiency for s in self.skills.values()) / len(
                self.skills
            )
            insights.append(
                f"Acquired {len(self.skills)} skills with average proficiency of {avg_proficiency:.1%}"
            )

        # Pattern recognition insights
        high_conf_patterns = [p for p in self.patterns.values() if p.confidence > 0.8]
        if high_conf_patterns:
            insights.append(
                f"Identified {len(high_conf_patterns)} strong behavioral patterns"
            )

        # Learning rate insight
        if self.performance_metrics["learning_rate"] > 0.7:
            insights.append(
                f"High learning rate ({self.performance_metrics['learning_rate']:.1%}) - effectively integrating feedback"
            )
        elif self.performance_metrics["learning_rate"] < 0.3:
            insights.append(
                f"Learning rate could be improved ({self.performance_metrics['learning_rate']:.1%}) - more practice needed"
            )

        # Domain expertise insights
        domain_skills = defaultdict(list)
        for skill in self.skills.values():
            domain_skills[skill.domain].append(skill)

        for domain, skills in domain_skills.items():
            avg_prof = sum(s.proficiency for s in skills) / len(skills)
            if avg_prof > 0.7:
                insights.append(
                    f"Strong expertise in {domain} domain ({len(skills)} skills, {avg_prof:.1%} proficiency)"
                )

        return insights
