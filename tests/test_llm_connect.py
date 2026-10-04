"""
Offline-safe tests for LLM to problem-definition model generation.
"""

from unittest.mock import Mock

import pulp
import pytest

from src.modeling.model_generator import ModelGenerator


def _build_api_mock(content: str) -> Mock:
    api = Mock()
    api.create_message.return_value = {
        "content": content,
        "model": "mock-model",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    return api


def _widget_problem_data() -> dict:
    return {
        "problem_type": "integer_programming",
        "objective": "maximize",
        "objective_description": "profit from producing widgets",
        "decision_variables": ["number of widgets to produce"],
        "constraints": [
            "2 labor hours per widget, at most 100 available",
            "3 material kg per widget, at most 120 available",
        ],
        "parameters": {
            "profit_per_widget": 50,
            "labor_hours_per_widget": 2,
            "material_kg_per_widget": 3,
            "available_labor_hours": 100,
            "available_material_kg": 120,
        },
        "confidence": 0.95,
        "notes": "Widget counts must be integer",
    }


def test_non_template_problem_uses_ai_generation_and_solves():
    generator = ModelGenerator()
    generator.api_client = _build_api_mock("""
import pulp

def create_model():
    prob = pulp.LpProblem("Widget_Problem", pulp.LpMaximize)
    widgets = pulp.LpVariable("widgets", lowBound=0, cat="Integer")
    prob += 50 * widgets, "Profit"
    prob += 2 * widgets <= 100, "Labor"
    prob += 3 * widgets <= 120, "Material"
    return prob
        """)

    model = generator.generate(_widget_problem_data())
    validation = generator.validate_model(model)

    assert validation["valid"] is True
    assert validation["num_variables"] == 1
    assert validation["num_constraints"] == 2

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[model.status] == "Optimal"
    assert pulp.value(model.objective) == 2000
    assert model.variables()[0].varValue == 40

    generator.api_client.create_message.assert_called_once()
    prompt = generator.api_client.create_message.call_args.kwargs["messages"][0][
        "content"
    ]
    assert "Problem Type: integer_programming" in prompt
    assert "profit from producing widgets" in prompt


def test_non_template_problem_requires_ai_api():
    generator = ModelGenerator()
    generator.api_client = None

    with pytest.raises(
        ValueError, match="AI API not configured for dynamic model generation"
    ):
        generator.generate({"problem_type": "linear_programming"})


def test_transportation_uses_template_and_skips_ai():
    generator = ModelGenerator()
    generator.api_client = Mock()

    model = generator.generate(
        {
            "problem_type": "transportation",
            "parameters": {
                "supply": [100, 150],
                "demand": [80, 120],
                "costs": [[5, 8], [6, 7]],
            },
        }
    )

    validation = generator.validate_model(model)

    assert validation["valid"] is True
    assert validation["num_variables"] == 4
    assert validation["num_constraints"] == 4
    generator.api_client.create_message.assert_not_called()


def test_ai_generation_handles_markdown_fences():
    generator = ModelGenerator()
    generator.api_client = _build_api_mock("""```python
import pulp

def create_model():
    prob = pulp.LpProblem("Fenced", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    prob += x, "Obj"
    prob += x >= 1, "C1"
    return prob
```""")

    model = generator.generate(_widget_problem_data())
    validation = generator.validate_model(model)

    assert validation["valid"] is True
    assert validation["num_variables"] == 1
    assert validation["num_constraints"] == 1
