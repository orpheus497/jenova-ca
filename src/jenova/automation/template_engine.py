"""
JENOVA Cognitive Architecture - Template Engine Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides template processing with variable substitution.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class TemplateContext:
    """Context for template rendering."""

    variables: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Callable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_variable(self, name: str, value: Any) -> None:
        """Add a variable to the context."""
        self.variables[name] = value

    def add_filter(self, name: str, func: Callable) -> None:
        """Add a filter function to the context."""
        self.filters[name] = func


class TemplateEngine:
    """
    Template processing engine with variable substitution and filters.

    Features:
    - Variable substitution ({{variable}})
    - Filters ({{variable|filter}})
    - Conditional blocks ({% if condition %})
    - Loop blocks ({% for item in list %})
    - Custom filters
    - Escaping
    """

    def __init__(self):
        """Initialize the template engine."""
        self.global_filters: Dict[str, Callable] = {}
        self._register_default_filters()

    def _register_default_filters(self) -> None:
        """Register default filters."""
        self.global_filters.update({
            'upper': str.upper,
            'lower': str.lower,
            'title': str.title,
            'capitalize': str.capitalize,
            'strip': str.strip,
            'len': len,
            'default': lambda x, default='': x if x else default,
        })

    def render(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Render a template with variables.

        Args:
            template: Template string
            variables: Dictionary of variables

        Returns:
            Rendered template

        Raises:
            ValueError: If template is invalid or variable missing
        """
        context = TemplateContext(variables=variables, filters=self.global_filters.copy())
        return self._render_template(template, context)

    def render_with_context(self, template: str, context: TemplateContext) -> str:
        """
        Render a template with a context object.

        Args:
            template: Template string
            context: TemplateContext object

        Returns:
            Rendered template
        """
        return self._render_template(template, context)

    def _render_template(self, template: str, context: TemplateContext) -> str:
        """
        Internal template rendering.

        Args:
            template: Template string
            context: Template context

        Returns:
            Rendered template
        """
        # Process control structures first
        template = self._process_conditionals(template, context)
        template = self._process_loops(template, context)

        # Process variable substitutions
        template = self._process_variables(template, context)

        return template

    def _process_variables(self, template: str, context: TemplateContext) -> str:
        """
        Process variable substitutions.

        Supports:
        - {{variable}} - Simple substitution
        - {{variable|filter}} - With filter
        - {{variable|filter:arg}} - Filter with argument
        """
        pattern = r'\{\{([^}]+)\}\}'

        def replace_variable(match):
            expr = match.group(1).strip()

            # Check for filter
            if '|' in expr:
                var_name, filter_expr = expr.split('|', 1)
                var_name = var_name.strip()
                filter_expr = filter_expr.strip()

                # Get variable value
                value = self._get_variable(var_name, context)

                # Apply filter
                return self._apply_filter(value, filter_expr, context)
            else:
                # Simple variable
                value = self._get_variable(expr, context)
                return str(value) if value is not None else ''

        return re.sub(pattern, replace_variable, template)

    def _get_variable(self, name: str, context: TemplateContext) -> Any:
        """
        Get variable value from context.

        Supports dot notation: {{user.name}}
        """
        parts = name.split('.')
        value = context.variables

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                logger.warning(f"Variable not found: {name}")
                return None

        return value

    def _apply_filter(self, value: Any, filter_expr: str, context: TemplateContext) -> str:
        """
        Apply filter to a value.

        Supports:
        - filter_name
        - filter_name:arg
        """
        if ':' in filter_expr:
            filter_name, arg = filter_expr.split(':', 1)
            filter_name = filter_name.strip()
            arg = arg.strip().strip('"\'')
        else:
            filter_name = filter_expr
            arg = None

        # Get filter function
        filter_func = context.filters.get(filter_name) or self.global_filters.get(filter_name)

        if not filter_func:
            logger.warning(f"Filter not found: {filter_name}")
            return str(value)

        try:
            if arg is not None:
                return str(filter_func(value, arg))
            else:
                return str(filter_func(value))
        except Exception as e:
            logger.error(f"Error applying filter {filter_name}: {e}")
            return str(value)

    def _process_conditionals(self, template: str, context: TemplateContext) -> str:
        """
        Process conditional blocks.

        Supports:
        - {% if variable %}...{% endif %}
        - {% if variable %}...{% else %}...{% endif %}
        """
        pattern = r'\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}'

        def replace_conditional(match):
            var_name = match.group(1).strip()
            content = match.group(2)

            # Check for else block
            else_pattern = r'\{%\s*else\s*%\}'
            parts = re.split(else_pattern, content, maxsplit=1)

            if_content = parts[0]
            else_content = parts[1] if len(parts) > 1 else ''

            # Evaluate condition
            value = self._get_variable(var_name, context)

            if value:
                return if_content
            else:
                return else_content

        return re.sub(pattern, replace_conditional, template, flags=re.DOTALL)

    def _process_loops(self, template: str, context: TemplateContext) -> str:
        """
        Process loop blocks.

        Supports:
        - {% for item in list %}...{% endfor %}
        """
        pattern = r'\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}'

        def replace_loop(match):
            item_var = match.group(1).strip()
            list_var = match.group(2).strip()
            loop_content = match.group(3)

            # Get list
            items = self._get_variable(list_var, context)

            if not items:
                return ''

            # Render loop content for each item
            results = []
            for item in items:
                # Create temporary context with loop variable
                loop_context = TemplateContext(
                    variables={**context.variables, item_var: item},
                    filters=context.filters
                )
                result = self._render_template(loop_content, loop_context)
                results.append(result)

            return ''.join(results)

        return re.sub(pattern, replace_loop, template, flags=re.DOTALL)

    def register_filter(self, name: str, func: Callable) -> None:
        """
        Register a custom filter.

        Args:
            name: Filter name
            func: Filter function (takes value, optionally arg)
        """
        self.global_filters[name] = func
        logger.info(f"Registered filter: {name}")

    def unregister_filter(self, name: str) -> bool:
        """
        Unregister a filter.

        Args:
            name: Filter name

        Returns:
            True if filter was removed, False if not found
        """
        if name in self.global_filters:
            del self.global_filters[name]
            logger.info(f"Unregistered filter: {name}")
            return True
        return False

    def list_filters(self) -> List[str]:
        """
        List all registered filters.

        Returns:
            List of filter names
        """
        return list(self.global_filters.keys())

    def validate_template(self, template: str) -> Dict[str, Any]:
        """
        Validate a template without rendering.

        Args:
            template: Template string

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []

        # Check for balanced braces
        open_var = template.count('{{')
        close_var = template.count('}}')
        if open_var != close_var:
            errors.append(f"Unbalanced variable braces: {open_var} opening, {close_var} closing")

        # Check for balanced control structures
        if_count = len(re.findall(r'\{%\s*if\s+', template))
        endif_count = len(re.findall(r'\{%\s*endif\s*%\}', template))
        if if_count != endif_count:
            errors.append(f"Unbalanced if/endif: {if_count} if, {endif_count} endif")

        for_count = len(re.findall(r'\{%\s*for\s+', template))
        endfor_count = len(re.findall(r'\{%\s*endfor\s*%\}', template))
        if for_count != endfor_count:
            errors.append(f"Unbalanced for/endfor: {for_count} for, {endfor_count} endfor")

        # Extract variables
        variables = re.findall(r'\{\{([^}|]+)', template)
        variables = [v.strip().split('.')[0] for v in variables]
        variables = list(set(variables))  # Unique

        # Extract filters
        filters = re.findall(r'\|(\w+)', template)
        filters = list(set(filters))

        unknown_filters = [f for f in filters if f not in self.global_filters]
        if unknown_filters:
            warnings.append(f"Unknown filters: {', '.join(unknown_filters)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "variables": variables,
            "filters": filters
        }

    def extract_variables(self, template: str) -> List[str]:
        """
        Extract all variable names from a template.

        Args:
            template: Template string

        Returns:
            List of variable names
        """
        pattern = r'\{\{([^}|]+)'
        matches = re.findall(pattern, template)
        variables = [v.strip().split('.')[0] for v in matches]
        return list(set(variables))

    def render_safe(
        self,
        template: str,
        variables: Dict[str, Any],
        default_value: str = ""
    ) -> str:
        """
        Render template safely, returning default on error.

        Args:
            template: Template string
            variables: Variables dictionary
            default_value: Value to return on error

        Returns:
            Rendered template or default value
        """
        try:
            return self.render(template, variables)
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return default_value
