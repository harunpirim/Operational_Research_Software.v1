"""
Utility functions for OR Assistant
"""

import json
import os
from typing import Any, Dict
import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("or_assistant")


def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r") as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with appropriate decimal places.

    Args:
        value: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{value:,.{decimals}f}"


def validate_problem_data(data: Dict[str, Any]) -> tuple[bool, list]:
    """
    Validate problem data structure.

    Args:
        data: Problem data dictionary

    Returns:
        Tuple of (is_valid, errors_list)
    """
    errors = []

    required_fields = ["problem_type"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    return len(errors) == 0, errors


def pretty_print_solution(solution: Dict[str, Any]) -> str:
    """
    Create a human-readable string representation of solution.

    Args:
        solution: Solution dictionary

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SOLUTION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Status: {solution.get('status', 'Unknown')}")
    lines.append(f"Objective: {solution.get('objective_value', 'N/A')}")
    lines.append(f"Solve Time: {solution.get('solve_time', 'N/A')}s")
    lines.append("")

    if solution.get("variables"):
        lines.append("DECISION VARIABLES:")
        lines.append("-" * 60)
        for var, info in solution["variables"].items():
            value = info.get("value", 0)
            lines.append(f"  {var:30s} = {format_number(value)}")

    lines.append("=" * 60)

    return "\n".join(lines)


__all__ = [
    "setup_logging",
    "load_config",
    "save_results",
    "format_number",
    "validate_problem_data",
    "pretty_print_solution",
]
