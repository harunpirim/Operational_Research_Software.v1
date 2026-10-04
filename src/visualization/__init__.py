"""
Visualization module for OR Assistant.

This module provides chart generation capabilities for different types of
optimization problems including transportation networks, sensitivity analysis,
and scheduling Gantt charts.
"""

from .chart_generator import ChartGenerator, create_chart

__all__ = ["ChartGenerator", "create_chart"]
