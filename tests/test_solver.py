"""
Tests for Solver Interface
"""

import pytest
import pulp
from src.solvers.solver_interface import (
    SolverInterface,
    SolverStatus,
    SensitivityAnalyzer,
)


def test_solver_initialization():
    """Test solver initialization."""
    solver = SolverInterface("pulp")
    assert solver.solver_type == "pulp"
    assert solver.solver is not None


def test_list_available_solvers():
    """Test listing available solvers."""
    solvers = SolverInterface.list_available_solvers()
    assert isinstance(solvers, dict)
    assert "pulp_cbc" in solvers
    assert solvers["pulp_cbc"]  # PuLP should always be available


def test_solve_simple_lp():
    """Test solving a simple LP problem."""
    # Create a simple problem
    prob = pulp.LpProblem("Test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)

    prob += 2 * x + 3 * y, "Cost"
    prob += x + y >= 10, "Constraint1"
    prob += 2 * x + y >= 15, "Constraint2"

    # Solve
    solver = SolverInterface("pulp")
    solution = solver.solve(prob)

    # Check solution
    assert solution["status"] == SolverStatus.OPTIMAL.value
    assert solution["objective_value"] is not None
    assert solution["solve_time"] > 0
    assert len(solution["variables"]) == 2


def test_solve_infeasible_problem():
    """Test handling of infeasible problems."""
    prob = pulp.LpProblem("Infeasible", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, upBound=5)

    prob += x, "Obj"
    prob += x >= 10, "Impossible"

    solver = SolverInterface("pulp")
    solution = solver.solve(prob)

    assert solution["status"] == SolverStatus.INFEASIBLE.value


def test_solver_info():
    """Test getting solver information."""
    solver = SolverInterface("pulp")
    info = solver.get_solver_info()

    assert "solver_type" in info
    assert info["solver_type"] == "pulp"
    assert "available" in info


def test_sensitivity_analyzer():
    """Test sensitivity analysis."""
    # Create and solve a problem
    prob = pulp.LpProblem("Test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)

    prob += 2 * x + 3 * y
    prob += x + y >= 10

    solver = SolverInterface()
    solution = solver.solve(prob)

    # Analyze
    analyzer = SensitivityAnalyzer(solution)
    shadow_prices = analyzer.analyze_shadow_prices()

    assert isinstance(shadow_prices, dict)


def test_transportation_problem_solve():
    """Test solving a transportation problem."""
    # Simple 2x2 transportation
    prob = pulp.LpProblem("Transport", pulp.LpMinimize)

    routes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    x = pulp.LpVariable.dicts("route", routes, lowBound=0)

    costs = {(0, 0): 5, (0, 1): 8, (1, 0): 6, (1, 1): 7}
    supply = [100, 150]
    demand = [80, 120]

    # Objective
    prob += pulp.lpSum([costs[r] * x[r] for r in routes])

    # Supply constraints
    prob += pulp.lpSum([x[0, j] for j in [0, 1]]) <= supply[0]
    prob += pulp.lpSum([x[1, j] for j in [0, 1]]) <= supply[1]

    # Demand constraints
    prob += pulp.lpSum([x[i, 0] for i in [0, 1]]) >= demand[0]
    prob += pulp.lpSum([x[i, 1] for i in [0, 1]]) >= demand[1]

    solver = SolverInterface()
    solution = solver.solve(prob)

    assert solution["status"] == SolverStatus.OPTIMAL.value
    assert solution["objective_value"] is not None


def _bounded_problem(cat="Continuous"):
    """max y - z + 4 with y in [0, 3.5], z in [2, 5]: optimum 5.5 (5 if y integer)."""
    prob = pulp.LpProblem("Bounded", pulp.LpMaximize)
    y = pulp.LpVariable("y", lowBound=0, upBound=3.5, cat=cat)
    z = pulp.LpVariable("z", lowBound=2, upBound=5)
    prob += y - z + 4
    prob += y + z <= 100, "Loose"
    return prob


@pytest.mark.parametrize("solver_key", ["cvxpy_osqp", "cvxpy"])
def test_cvxpy_conversion_keeps_bounds_and_constant(solver_key):
    """PuLP -> CVXPY conversion must keep variable bounds and the objective constant."""
    solution = SolverInterface(solver_key).solve(_bounded_problem())

    assert solution["is_optimal"]
    assert solution["objective_value"] == pytest.approx(5.5, abs=1e-4)
    assert solution["variables"]["y"] == pytest.approx(3.5, abs=1e-4)
    assert solution["variables"]["z"] == pytest.approx(2.0, abs=1e-4)
    assert solution["num_constraints"] == 1


def test_cvxpy_falls_back_when_solver_cannot_handle_integers():
    """OSQP can't solve MIPs; the CVXPY path should fall back rather than error."""
    solution = SolverInterface("cvxpy_osqp").solve(_bounded_problem("Integer"))

    assert solution["is_optimal"]
    assert solution["objective_value"] == pytest.approx(5.0, abs=1e-4)


def test_osqp_not_used_for_linear_objective():
    """OSQP is skipped for LPs (wrong tool, and its Linux wheel can segfault)."""
    solution = SolverInterface("cvxpy_osqp").solve(_bounded_problem())

    assert solution["is_optimal"]
    assert "OSQP" not in solution["solver_name"]
    assert any("OSQP skipped" in w for w in solution["warnings"])


def test_osqp_still_used_for_quadratic_objective():
    """A genuine QP (e.g. portfolio risk) still goes to OSQP."""
    import cvxpy as cp

    x = cp.Variable(2, name="x")
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [cp.sum(x) == 1, x >= 0])
    solution = SolverInterface("cvxpy_osqp").solve(
        {"type": "cvxpy", "problem": prob, "variables": {"x": x}}
    )

    assert solution["status"].startswith("Optimal")
    assert "OSQP" in solution["solver_name"]
    assert solution["objective_value"] == pytest.approx(0.5, abs=1e-3)
