"""
Solver Utilities — verify installed solvers across PuLP, CVXPY, etc.
"""

from typing import Dict, Any, List


def verify_cvxpy_solvers() -> Dict[str, Any]:
    """
    Check whether CVXPY is importable, which solvers it sees, and
    confirm at least one works by solving a trivial problem.

    Returns a dict with keys:
        available (bool)  — True if CVXPY is installed and functional
        solvers   (list)  — list of solver names CVXPY reports
        default   (str)   — the solver CVXPY chose for the test problem
        error     (str)   — non-empty only when available is False
    """
    try:
        import cvxpy as cp
    except ImportError:
        return {
            "available": False,
            "solvers": [],
            "default": "",
            "error": "cvxpy is not installed. Run: pip install cvxpy",
        }

    solvers: List[str] = []
    try:
        solvers = cp.installed_solvers()
    except Exception:
        pass

    try:
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 2])
        prob.solve()
        default_solver = prob.solver_stats.solver_name if prob.solver_stats else ""
    except Exception as exc:
        return {
            "available": False,
            "solvers": solvers,
            "default": "",
            "error": f"CVXPY imported but test solve failed: {exc}",
        }

    return {
        "available": True,
        "solvers": solvers,
        "default": default_solver,
        "error": "",
    }
