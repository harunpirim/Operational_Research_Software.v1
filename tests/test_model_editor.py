"""
Tests for the Model Playground's LaTeX parsing and editor-state round trip.
"""

import pytest

from src.modeling.model_generator import ModelGenerator
from src.solvers.solver_interface import SolverInterface
from src.utils.model_editor import (
    editor_state_to_pulp,
    latex_expr_to_coefficients,
    pulp_to_editor_state,
)

NAMES = ["x_{chairs}", "x_{tables}"]


@pytest.mark.parametrize(
    "latex, expected",
    [
        (r"50x_{chairs}+30x_{tables}", {"x_{chairs}": 50.0, "x_{tables}": 30.0}),
        (r"x_{chairs}-x_{tables}", {"x_{chairs}": 1.0, "x_{tables}": -1.0}),
        (
            r"10\cdot x_{chairs}+5\cdot x_{tables}",
            {"x_{chairs}": 10.0, "x_{tables}": 5.0},
        ),
        (r"10\times x_{chairs}", {"x_{chairs}": 10.0, "x_{tables}": 0.0}),
        (r"-x_{chairs}+\frac{1}{2}x_{tables}", {"x_{chairs}": -1.0, "x_{tables}": 0.5}),
    ],
)
def test_latex_expr_to_coefficients(latex, expected):
    assert latex_expr_to_coefficients(latex, NAMES) == pytest.approx(expected)


def test_editor_state_round_trip_preserves_optimum():
    generator = ModelGenerator()
    generator.api_client = None
    model = generator.generate(
        {
            "problem_type": "transportation",
            "confidence": 0.9,
            "parameters": {
                "supply": [100, 150],
                "demand": [80, 120],
                "costs": [[5, 8], [6, 7]],
            },
        }
    )
    original = SolverInterface("pulp").solve(model)

    rebuilt, errors = editor_state_to_pulp(pulp_to_editor_state(model))
    round_tripped = SolverInterface("pulp").solve(rebuilt)

    assert errors == []
    assert round_tripped["objective_value"] == pytest.approx(
        original["objective_value"]
    )
