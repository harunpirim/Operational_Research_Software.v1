"""Tests for the MPS parser's objective-sense handling."""

import pulp
import pytest

from src.ingestion.file_parser import FileParser
from src.modeling.model_generator import ModelGenerator
from src.solvers.solver_interface import SolverInterface


def _two_var_problem(sense):
    prob = pulp.LpProblem("sense_test", sense)
    x = pulp.LpVariable("x", lowBound=0, upBound=10)
    y = pulp.LpVariable("y", lowBound=0, upBound=10)
    prob += 3 * x + 2 * y, "obj"
    prob += x + y <= 12, "cap"
    prob += x >= 2, "floor"
    return prob


@pytest.mark.parametrize(
    "sense, expected_sense, expected_obj",
    [(pulp.LpMaximize, "maximize", 34.0), (pulp.LpMinimize, "minimize", 6.0)],
)
def test_pulp_written_mps_keeps_objective_sense(
    tmp_path, monkeypatch, sense, expected_sense, expected_obj
):
    """PuLP records the sense as a '*SENSE:' comment; the parser must honour it."""
    monkeypatch.chdir(tmp_path)  # keep the solution cache out of the repo
    path = tmp_path / "model.mps"
    _two_var_problem(sense).writeMPS(str(path))
    assert "*SENSE:" in path.read_text()

    parsed = FileParser().parse(str(path), filename="model.mps")
    assert parsed["objective_sense"] == expected_sense

    generator = ModelGenerator()
    generator.api_client = None
    model, problem_data = generator.generate_from_mps(parsed)
    solution = SolverInterface("pulp_cbc", problem_data=problem_data).solve(model)
    assert solution["objective_value"] == pytest.approx(expected_obj)


def test_objsense_section_still_wins_over_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "model.mps"
    text = """NAME          T
OBJSENSE
    MAX
ROWS
 N  obj
 L  cap
COLUMNS
    x         obj            3   cap            1
    y         obj            2   cap            1
RHS
    RHS       cap           12
BOUNDS
 UP BND       x             10
 UP BND       y             10
ENDATA
"""
    path.write_text(text)
    parsed = FileParser().parse(str(path), filename="model.mps")
    assert parsed["objective_sense"] == "maximize"
