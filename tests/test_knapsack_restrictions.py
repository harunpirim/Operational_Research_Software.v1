"""Tests for mandatory/forbidden item handling in knapsack model generation."""

import pytest

from src.modeling.model_generator import InfeasibleModelError, ModelGenerator
from src.solvers.solver_interface import default_pulp_solver


def _problem(**params):
    base = {
        "problem_type": "knapsack",
        "objective": "maximize",
        "confidence": 0.95,
        "parameters": {
            "capacity": 300,
            "weights": [40, 15, 45, 50, 60, 100, 120],
            "values": [15.0, 40.0, 13.33, 12.0, 10.0, 6.0, 5.0],
            "item_names": ["mapua", "t1", "t2", "t3", "t4", "t5", "t6"],
        },
    }
    base["parameters"].update(params)
    return base


def _solve(model):
    model.solve(default_pulp_solver(msg=0))
    return {v.name: round(v.value()) for v in model.variables()}


@pytest.fixture
def generator():
    return ModelGenerator()


def test_mandatory_item_is_forced_into_solution(generator):
    model = generator.generate(_problem(mandatory_items=["mapua"]))

    # One capacity row plus one forcing row; the old template emitted only the
    # capacity row, which is how 'must include' used to get dropped.
    assert len(model.constraints) == 2

    assert _solve(model)["mapua"] == 1


def test_forbidden_item_is_excluded(generator):
    model = generator.generate(_problem(forbidden_items=["t1"]))
    assert _solve(model)["t1"] == 0


def test_reserved_capacity_shrinks_the_budget(generator):
    model = generator.generate(_problem(reserved_capacity=260))
    selection = _solve(model)
    weights = dict(
        zip(_problem()["parameters"]["item_names"], _problem()["parameters"]["weights"])
    )
    used = sum(weights[name] for name, taken in selection.items() if taken)
    assert used <= 300 - 260


def test_all_items_mandatory_reports_diagnosis_instead_of_solving(generator):
    with pytest.raises(InfeasibleModelError) as excinfo:
        generator.generate(
            _problem(mandatory_items=["mapua", "t1", "t2", "t3", "t4", "t5", "t6"])
        )

    diagnosis = excinfo.value.diagnosis
    assert diagnosis["reason"] == "mandatory_exceeds_capacity"
    assert diagnosis["required"] == 430
    assert diagnosis["capacity"] == 300
    assert diagnosis["overage"] == 130
    assert "over by 130" in str(excinfo.value)


def test_conflicting_restrictions_are_rejected(generator):
    with pytest.raises(InfeasibleModelError) as excinfo:
        generator.generate(_problem(mandatory_items=["t1"], forbidden_items=["t1"]))
    assert excinfo.value.diagnosis["reason"] == "conflicting_restrictions"


def test_unmatched_item_reference_fails_loudly(generator):
    with pytest.raises(ValueError, match="Could not match"):
        generator.generate(_problem(mandatory_items=["does_not_exist"]))


def test_absent_restrictions_preserve_old_behaviour(generator):
    model = generator.generate(_problem())
    assert len(model.constraints) == 1
    assert _solve(model)["t1"] == 1
