"""
Solver Progress UI Module

This module handles all UI components related to solver execution, including
pre-solve information, progress updates, and result display. It provides a
clean interface for app.py to display solver-related information without
managing the formatting details.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional


def show_pre_solve_info(
    solver_key: str,
    solver_explanation: str,
    num_variables: int,
    num_constraints: int,
    time_limit: int,
    problem_data: Optional[Dict[str, Any]] = None,
) -> st.empty:
    """
    Display pre-solve information in a formatted info box.

    Args:
        solver_key: The internal solver key (e.g., 'cvxpy_scip')
        solver_explanation: Human-readable solver description
        num_variables: Number of variables in the problem
        num_constraints: Number of constraints in the problem
        time_limit: Time limit in seconds

    Returns:
        st.empty: A placeholder for the live timer updates
    """
    # Format solver name for display
    if solver_key == "auto":
        solver_display = "Auto-detect"
    else:
        solver_display = solver_key.replace("_", " / ").upper()

    # Build the info message
    info_parts = [
        f"Solving with {solver_display}",
        f"{num_variables:,} variables",
        f"{num_constraints:,} constraints",
    ]

    # Add time limit if applicable
    if solver_key in ["cvxpy_scip", "auto"]:
        info_parts.append(f"time limit {time_limit} seconds")

    info_message = " — ".join(info_parts) + "."

    st.info(info_message)

    # Special warning for MIPLIB problems with SCIP
    if (
        problem_data
        and problem_data.get("source") == "miplib"
        and "scip" in solver_key.lower()
    ):
        st.warning(
            "⏳ **MIPLIB problems can take time to solve.** SCIP is now processing your problem. "
            f"This may take up to {time_limit} seconds. The solver is working in the background — "
            "please wait for completion. Progress will be shown when finished."
        )

    # Return placeholder for timer updates
    return st.empty()


def update_solve_timer(
    placeholder: st.empty, elapsed_seconds: int, time_limit: Optional[int] = None
):
    """
    Update the timer placeholder with current solve progress.

    Args:
        placeholder: The st.empty() placeholder to update
        elapsed_seconds: Number of seconds elapsed
        time_limit: Optional time limit to show progress against
    """
    if time_limit:
        timer_text = f"⏱️ Solving... {elapsed_seconds}s / {time_limit}s"
    else:
        timer_text = f"⏱️ Solving... {elapsed_seconds}s"

    placeholder.text(timer_text)


def show_solve_result(solution: Dict[str, Any]):
    """
    Display the solve result with appropriate formatting based on status.

    Args:
        solution: The solution dictionary from SolverInterface
    """
    status = solution.get("status", "Unknown")

    # Handle different statuses
    if status == "Optimal":
        st.success(
            f"✅ **Optimal solution found!** "
            f"Objective value: {solution.get('objective_value', 'N/A'):.4f}"
        )

    elif "Feasible" in status and "time limit" in status:
        # Extract MIP gap if available
        mip_gap = solution.get("mip_gap")
        gap_text = f" (MIP gap: {mip_gap:.2%})" if mip_gap else ""

        st.warning(
            f"⚠️ **Feasible solution found** (time limit reached){gap_text}. "
            f"Objective value: {solution.get('objective_value', 'N/A'):.4f}. "
            f"This may not be the optimal solution."
        )

    elif status == "Infeasible":
        st.error(
            "❌ **Problem is infeasible!** No solution exists that satisfies "
            "all constraints. Check your problem formulation."
        )

    elif "cache" in status.lower():
        # Extract cache details from status or solution
        cache_time = solution.get("cache_timestamp", "Unknown time")
        original_solve_time = solution.get(
            "original_solve_time", solution.get("solve_time", "N/A")
        )

        st.success(
            f"✅ **Solution loaded from cache!** "
            f"Originally solved at {cache_time} in {original_solve_time:.2f}s. "
            f"Objective value: {solution.get('objective_value', 'N/A'):.4f}"
        )

    else:
        # Generic status display
        st.info(f"Solver status: {status}")

    # Show solver details
    solver_name = solution.get("solver_name", "Unknown solver")
    solver_caption_parts = [f"Solved using {solver_name}"]

    # Add performance suggestion if relevant
    warnings = solution.get("warnings", [])
    for warning in warnings:
        if "solver" in warning.lower() or "performance" in warning.lower():
            solver_caption_parts.append(warning)
            break

    # Show solver caption
    st.caption(" • ".join(solver_caption_parts))


def show_cached_result_banner(solution: Dict[str, Any]):
    """
    Display a prominent banner for cached results.

    This replaces the entire Step 3 progress flow when a cache hit occurs.

    Args:
        solution: The cached solution dictionary
    """
    # Extract cache metadata
    cache_meta = solution.get("cache_metadata", {})
    cache_timestamp = cache_meta.get("timestamp", "Unknown time")
    original_solve_time = cache_meta.get(
        "original_solve_time", solution.get("solve_time", 0)
    )
    solver_used = cache_meta.get("solver_used", "Unknown solver")
    objective_value = solution.get("objective_value", "N/A")

    # Format timestamp
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(cache_timestamp)
        cache_time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        cache_time_str = str(cache_timestamp)

    # Format objective value
    if isinstance(objective_value, (int, float)):
        obj_str = f"{objective_value:.6f}"
    else:
        obj_str = str(objective_value)

    # Display the banner
    st.success(
        f"🚀 **Solution loaded from cache!**\n\n"
        f"This problem was previously solved on **{cache_time_str}** "
        f"using **{solver_used}** in **{original_solve_time:.2f} seconds**.\n\n"
        f"**Objective value:** {obj_str}\n\n"
        f"*No computation needed — instant results!*"
    )


def show_variable_values(solution: Dict[str, Any], max_display: int = 50):
    """
    Display non-zero variable values from the solution.

    Args:
        solution: The solution dictionary
        max_display: Maximum number of variables to display
    """
    variables = solution.get("variables", {})
    nonzero = {k: v for k, v in variables.items() if v is not None and abs(v) > 1e-8}

    if nonzero:
        num_nonzero = len(nonzero)
        st.markdown(f"**Variable values** ({num_nonzero} non-zero):")

        # Limit display if too many
        display_items = list(nonzero.items())[:max_display]

        for vname, vval in display_items:
            st.markdown(f"- `{vname}` = **{vval:.6g}**")

        if num_nonzero > max_display:
            st.caption(
                f"*Showing first {max_display} of {num_nonzero} non-zero variables*"
            )
