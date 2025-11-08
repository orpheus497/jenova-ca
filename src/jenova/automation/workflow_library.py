"""
JENOVA Cognitive Architecture - Workflow Library Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides a library of common workflow patterns and templates.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of workflows."""
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""

    id: str
    name: str
    description: str
    action: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    timeout: Optional[int] = None
    retry_on_failure: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Represents a complete workflow."""

    id: str
    name: str
    description: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps.append(step)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.workflow_type.value,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "description": step.description,
                    "dependencies": step.dependencies,
                    "optional": step.optional,
                    "timeout": step.timeout,
                    "retry_on_failure": step.retry_on_failure,
                    "metadata": step.metadata
                }
                for step in self.steps
            ],
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class WorkflowLibrary:
    """
    Library of common workflow patterns and templates.

    Provides pre-defined workflows for common tasks like:
    - Code review
    - Testing
    - Deployment
    - Refactoring
    - Documentation generation
    """

    def __init__(self):
        """Initialize the workflow library."""
        self.workflows: Dict[str, Workflow] = {}
        self._initialize_default_workflows()

    def _initialize_default_workflows(self) -> None:
        """Initialize default workflow templates."""
        self._create_code_review_workflow()
        self._create_testing_workflow()
        self._create_deployment_workflow()
        self._create_refactoring_workflow()
        self._create_documentation_workflow()
        self._create_analysis_workflow()

    def _create_code_review_workflow(self) -> None:
        """Create code review workflow."""
        workflow = Workflow(
            id="code_review",
            name="Code Review",
            description="Comprehensive code review workflow",
            workflow_type=WorkflowType.CODE_REVIEW
        )

        workflow.add_step(WorkflowStep(
            id="analyze_changes",
            name="Analyze Changes",
            description="Analyze code changes and identify modified files"
        ))

        workflow.add_step(WorkflowStep(
            id="check_style",
            name="Check Code Style",
            description="Verify code follows style guidelines",
            dependencies=["analyze_changes"]
        ))

        workflow.add_step(WorkflowStep(
            id="security_scan",
            name="Security Scan",
            description="Scan for security vulnerabilities",
            dependencies=["analyze_changes"]
        ))

        workflow.add_step(WorkflowStep(
            id="complexity_check",
            name="Complexity Check",
            description="Check code complexity metrics",
            dependencies=["analyze_changes"]
        ))

        workflow.add_step(WorkflowStep(
            id="generate_summary",
            name="Generate Review Summary",
            description="Generate code review summary",
            dependencies=["check_style", "security_scan", "complexity_check"]
        ))

        self.workflows[workflow.id] = workflow

    def _create_testing_workflow(self) -> None:
        """Create testing workflow."""
        workflow = Workflow(
            id="testing",
            name="Testing",
            description="Comprehensive testing workflow",
            workflow_type=WorkflowType.TESTING
        )

        workflow.add_step(WorkflowStep(
            id="setup_environment",
            name="Setup Test Environment",
            description="Setup testing environment and dependencies"
        ))

        workflow.add_step(WorkflowStep(
            id="run_unit_tests",
            name="Run Unit Tests",
            description="Execute unit tests",
            dependencies=["setup_environment"],
            retry_on_failure=True
        ))

        workflow.add_step(WorkflowStep(
            id="run_integration_tests",
            name="Run Integration Tests",
            description="Execute integration tests",
            dependencies=["setup_environment"],
            optional=True
        ))

        workflow.add_step(WorkflowStep(
            id="coverage_analysis",
            name="Coverage Analysis",
            description="Analyze test coverage",
            dependencies=["run_unit_tests", "run_integration_tests"]
        ))

        workflow.add_step(WorkflowStep(
            id="generate_report",
            name="Generate Test Report",
            description="Generate comprehensive test report",
            dependencies=["coverage_analysis"]
        ))

        self.workflows[workflow.id] = workflow

    def _create_deployment_workflow(self) -> None:
        """Create deployment workflow."""
        workflow = Workflow(
            id="deployment",
            name="Deployment",
            description="Application deployment workflow",
            workflow_type=WorkflowType.DEPLOYMENT
        )

        workflow.add_step(WorkflowStep(
            id="pre_deployment_checks",
            name="Pre-Deployment Checks",
            description="Run pre-deployment validation"
        ))

        workflow.add_step(WorkflowStep(
            id="build_application",
            name="Build Application",
            description="Build application artifacts",
            dependencies=["pre_deployment_checks"]
        ))

        workflow.add_step(WorkflowStep(
            id="run_tests",
            name="Run Tests",
            description="Execute test suite",
            dependencies=["build_application"]
        ))

        workflow.add_step(WorkflowStep(
            id="deploy_staging",
            name="Deploy to Staging",
            description="Deploy to staging environment",
            dependencies=["run_tests"]
        ))

        workflow.add_step(WorkflowStep(
            id="verify_staging",
            name="Verify Staging Deployment",
            description="Verify staging deployment",
            dependencies=["deploy_staging"]
        ))

        workflow.add_step(WorkflowStep(
            id="deploy_production",
            name="Deploy to Production",
            description="Deploy to production environment",
            dependencies=["verify_staging"],
            optional=True
        ))

        self.workflows[workflow.id] = workflow

    def _create_refactoring_workflow(self) -> None:
        """Create refactoring workflow."""
        workflow = Workflow(
            id="refactoring",
            name="Refactoring",
            description="Code refactoring workflow",
            workflow_type=WorkflowType.REFACTORING
        )

        workflow.add_step(WorkflowStep(
            id="analyze_code",
            name="Analyze Code Structure",
            description="Analyze current code structure and identify issues"
        ))

        workflow.add_step(WorkflowStep(
            id="identify_opportunities",
            name="Identify Refactoring Opportunities",
            description="Identify code that needs refactoring",
            dependencies=["analyze_code"]
        ))

        workflow.add_step(WorkflowStep(
            id="apply_changes",
            name="Apply Refactoring Changes",
            description="Apply refactoring transformations",
            dependencies=["identify_opportunities"]
        ))

        workflow.add_step(WorkflowStep(
            id="update_tests",
            name="Update Tests",
            description="Update tests to reflect changes",
            dependencies=["apply_changes"]
        ))

        workflow.add_step(WorkflowStep(
            id="verify_functionality",
            name="Verify Functionality",
            description="Verify that functionality is preserved",
            dependencies=["update_tests"]
        ))

        self.workflows[workflow.id] = workflow

    def _create_documentation_workflow(self) -> None:
        """Create documentation workflow."""
        workflow = Workflow(
            id="documentation",
            name="Documentation",
            description="Documentation generation workflow",
            workflow_type=WorkflowType.DOCUMENTATION
        )

        workflow.add_step(WorkflowStep(
            id="scan_codebase",
            name="Scan Codebase",
            description="Scan codebase for documented elements"
        ))

        workflow.add_step(WorkflowStep(
            id="extract_docstrings",
            name="Extract Docstrings",
            description="Extract docstrings and comments",
            dependencies=["scan_codebase"]
        ))

        workflow.add_step(WorkflowStep(
            id="generate_api_docs",
            name="Generate API Documentation",
            description="Generate API documentation",
            dependencies=["extract_docstrings"]
        ))

        workflow.add_step(WorkflowStep(
            id="create_guides",
            name="Create User Guides",
            description="Create user and developer guides",
            dependencies=["extract_docstrings"],
            optional=True
        ))

        workflow.add_step(WorkflowStep(
            id="build_docs",
            name="Build Documentation",
            description="Build final documentation artifacts",
            dependencies=["generate_api_docs", "create_guides"]
        ))

        self.workflows[workflow.id] = workflow

    def _create_analysis_workflow(self) -> None:
        """Create code analysis workflow."""
        workflow = Workflow(
            id="analysis",
            name="Code Analysis",
            description="Comprehensive code analysis workflow",
            workflow_type=WorkflowType.ANALYSIS
        )

        workflow.add_step(WorkflowStep(
            id="static_analysis",
            name="Static Analysis",
            description="Perform static code analysis"
        ))

        workflow.add_step(WorkflowStep(
            id="complexity_metrics",
            name="Complexity Metrics",
            description="Calculate code complexity metrics",
            dependencies=["static_analysis"]
        ))

        workflow.add_step(WorkflowStep(
            id="security_audit",
            name="Security Audit",
            description="Perform security audit",
            dependencies=["static_analysis"]
        ))

        workflow.add_step(WorkflowStep(
            id="dependency_analysis",
            name="Dependency Analysis",
            description="Analyze dependencies and detect issues",
            dependencies=["static_analysis"]
        ))

        workflow.add_step(WorkflowStep(
            id="generate_report",
            name="Generate Analysis Report",
            description="Generate comprehensive analysis report",
            dependencies=["complexity_metrics", "security_audit", "dependency_analysis"]
        ))

        self.workflows[workflow.id] = workflow

    def get(self, workflow_id: str) -> Optional[Workflow]:
        """
        Get a workflow by ID.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow object or None if not found
        """
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all available workflows.

        Returns:
            List of workflow metadata
        """
        return [
            {
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "type": wf.workflow_type.value,
                "steps_count": len(wf.steps)
            }
            for wf in self.workflows.values()
        ]

    def create_custom_workflow(
        self,
        name: str,
        description: str,
        steps: List[WorkflowStep]
    ) -> str:
        """
        Create a custom workflow.

        Args:
            name: Workflow name
            description: Workflow description
            steps: List of workflow steps

        Returns:
            Workflow ID
        """
        workflow_id = f"custom_{name.lower().replace(' ', '_')}"

        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            workflow_type=WorkflowType.CUSTOM,
            steps=steps
        )

        self.workflows[workflow_id] = workflow
        logger.info(f"Created custom workflow: {workflow_id}")

        return workflow_id

    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            True if deleted, False if not found
        """
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
        return False

    def clone_workflow(self, workflow_id: str, new_name: str) -> Optional[str]:
        """
        Clone an existing workflow.

        Args:
            workflow_id: Workflow ID to clone
            new_name: Name for the cloned workflow

        Returns:
            New workflow ID or None if source not found
        """
        source = self.workflows.get(workflow_id)
        if not source:
            return None

        new_id = f"clone_{new_name.lower().replace(' ', '_')}"

        # Create deep copy
        cloned = Workflow(
            id=new_id,
            name=new_name,
            description=f"Clone of {source.name}",
            workflow_type=source.workflow_type,
            steps=[
                WorkflowStep(
                    id=step.id,
                    name=step.name,
                    description=step.description,
                    dependencies=step.dependencies.copy(),
                    optional=step.optional,
                    timeout=step.timeout,
                    retry_on_failure=step.retry_on_failure,
                    metadata=step.metadata.copy()
                )
                for step in source.steps
            ],
            variables=source.variables.copy(),
            metadata=source.metadata.copy()
        )

        self.workflows[new_id] = cloned
        logger.info(f"Cloned workflow {workflow_id} as {new_id}")

        return new_id
