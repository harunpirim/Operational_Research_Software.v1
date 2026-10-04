"""
Solver Router Module

This module handles all solver selection logic, including auto-detection,
availability checking, and fallback chains. It separates the routing logic
from the SolverInterface which only needs to know how to call a specific solver.
"""

from typing import Dict, Any, Tuple, List


def resolve_solver(solver_key: str, problem_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Resolve the solver to use, handling auto-detection and fallbacks.

    Args:
        solver_key: The requested solver key (e.g., 'pulp_cbc', 'cvxpy_scip', 'auto')
        problem_data: Problem classification data containing type, size, etc.

    Returns:
        Tuple of (resolved_solver_key, explanation_string)
    """
    if solver_key == "auto":
        return auto_select_solver(problem_data)

    # Check availability and apply fallback chain
    available_solvers = _get_available_solvers()
    fallback_warnings = []

    # Define fallback chains for each solver
    fallback_chains = {
        "cvxpy_scip": ["cvxpy_scip", "cvxpy_glpk", "cvxpy_osqp", "pulp_cbc"],
        "cvxpy_glpk": ["cvxpy_glpk", "cvxpy_osqp", "pulp_cbc"],
        "cvxpy_osqp": ["cvxpy_osqp", "pulp_cbc"],
        "pulp_cbc": ["pulp_cbc"],  # No fallback for PuLP
        "auto": [],  # Already handled above
    }

    # Get the fallback chain for the requested solver
    chain = fallback_chains.get(solver_key, [solver_key])

    # Find the first available solver in the chain
    resolved_key = None
    for candidate in chain:
        if candidate == "pulp_cbc":
            # PuLP is always available
            resolved_key = candidate
            break
        elif candidate in available_solvers:
            resolved_key = candidate
            break
        else:
            display_name = get_solver_display_name(candidate)
            fallback_warnings.append(f"{display_name} not available")

    # Build explanation
    if resolved_key == solver_key:
        explanation = f"Using {get_solver_display_name(resolved_key)} as requested"
    else:
        original_name = get_solver_display_name(solver_key)
        resolved_name = get_solver_display_name(resolved_key)
        warnings_str = ", ".join(fallback_warnings)
        explanation = (
            f"Falling back from {original_name} to {resolved_name} ({warnings_str})"
        )

    return resolved_key, explanation


def auto_select_solver(problem_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Automatically select the best solver based on problem characteristics.

    Args:
        problem_data: Problem classification data

    Returns:
        Tuple of (solver_key, explanation)
    """
    # Get number of variables
    num_vars = problem_data.get("num_variables", 0)
    if num_vars == 0:
        # Try to count from decision_variables list
        decision_vars = problem_data.get("decision_variables", [])
        num_vars = len(decision_vars) if isinstance(decision_vars, list) else 0

    problem_type = problem_data.get("problem_type", "unknown")
    source = problem_data.get("source", "")

    # Get variable type counts
    var_types = problem_data.get("variable_types", {})
    num_binary = var_types.get("binary", 0)
    num_integer = var_types.get("integer", 0)

    # If no variable_types dict, count from decision_variables
    if not var_types and "decision_variables" in problem_data:
        decision_vars = problem_data.get("decision_variables", [])
        num_binary = sum(1 for v in decision_vars if v.get("type") == "binary")
        num_integer = sum(1 for v in decision_vars if v.get("type") == "integer")

    has_integers = num_binary > 0 or num_integer > 0

    # Check available solvers
    available_solvers = _get_available_solvers()

    # Special case: portfolio optimization
    if problem_type == "portfolio_optimization":
        if "cvxpy_osqp" in available_solvers:
            return "cvxpy_osqp", "OSQP is optimal for quadratic portfolio problems"
        else:
            return "pulp_cbc", "Using PuLP for portfolio (OSQP not available)"

    # Selection rules based on problem size and type
    if num_vars < 200 or problem_type == "linear_programming":
        return (
            "pulp_cbc",
            f"PuLP CBC is efficient for small problems ({num_vars} variables)",
        )

    if num_vars < 500 and not has_integers:
        return "pulp_cbc", f"PuLP CBC handles {num_vars} continuous variables well"

    if 500 <= num_vars <= 2000:
        if not has_integers:
            if "cvxpy_osqp" in available_solvers:
                return (
                    "cvxpy_osqp",
                    (
                        f"OSQP excels at medium-sized continuous problems ({num_vars}"
                        " variables)"
                    ),
                )
            else:
                return (
                    "pulp_cbc",
                    f"Using PuLP for {num_vars} variables (OSQP not available)",
                )
        else:
            # Has integer variables
            if "cvxpy_scip" in available_solvers:
                return (
                    "cvxpy_scip",
                    (
                        f"SCIP is best for medium MIP problems ({num_vars} variables,"
                        f" {num_binary} binary)"
                    ),
                )
            elif "cvxpy_glpk" in available_solvers:
                return (
                    "cvxpy_glpk",
                    f"GLPK for medium MIP ({num_vars} variables, SCIP not available)",
                )
            else:
                return (
                    "pulp_cbc",
                    f"PuLP CBC for {num_vars} variables (better solvers not available)",
                )

    # Large problems (>2000 vars) or MIPLIB source
    if num_vars > 2000 or source == "miplib":
        size_desc = f"{num_vars:,} variables" if num_vars > 2000 else "MIPLIB instance"
        if "cvxpy_scip" in available_solvers:
            return "cvxpy_scip", f"SCIP handles large problems best ({size_desc})"
        elif "cvxpy_glpk" in available_solvers:
            warning = (
                f"⚠️ GLPK may struggle with {size_desc}. Consider installing SCIP."
            )
            return "cvxpy_glpk", warning
        else:
            warning = (
                f"⚠️ PuLP may be slow for {size_desc}. Install CVXPY+SCIP for better"
                " performance."
            )
            return "pulp_cbc", warning

    # Default fallback
    return "pulp_cbc", "Using PuLP CBC as default solver"


def get_solver_display_name(solver_key: str) -> str:
    """
    Get a short display name for a solver key.

    Args:
        solver_key: The internal solver key

    Returns:
        str: Display name like "SCIP via CVXPY"
    """
    display_names = {
        "pulp_cbc": "PuLP CBC",
        "cvxpy_osqp": "OSQP via CVXPY",
        "cvxpy_scip": "SCIP via CVXPY",
        "cvxpy_glpk": "GLPK via CVXPY",
        "auto": "Auto-detect",
    }
    return display_names.get(solver_key, solver_key.upper())


def _get_available_solvers() -> List[str]:
    """
    Check which solvers are available on the system.

    Returns:
        List of available solver keys
    """
    available = ["pulp_cbc"]  # PuLP is always available

    try:
        import cvxpy as cp

        cvxpy_solvers = cp.installed_solvers()

        # Map CVXPY solver names to our keys
        solver_map = {
            "OSQP": "cvxpy_osqp",
            "GLPK_MI": "cvxpy_glpk",
            "SCIP": "cvxpy_scip",
        }

        for cvxpy_name, our_key in solver_map.items():
            if cvxpy_name in cvxpy_solvers:
                available.append(our_key)

        # Also check for pyscipopt separately
        if "cvxpy_scip" not in available:
            try:
                import pyscipopt  # noqa: F401  (availability check)

                if "SCIP" in cvxpy_solvers:
                    available.append("cvxpy_scip")
            except ImportError:
                pass

    except ImportError:
        # CVXPY not available
        pass

    return available
