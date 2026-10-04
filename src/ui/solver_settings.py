"""
Solver Settings UI Module

This module handles all solver-related UI components including solver selection,
time limits, and associated help text. It provides a clean interface for app.py
to render solver settings without managing the details.
"""

import streamlit as st

# Solver options with display labels, size guidance, and hint captions
SOLVER_OPTIONS = {
    "pulp_cbc": {
        "label": "PuLP / CBC — best for small problems (under 500 variables)",
        "size_guidance": "under 500 variables",
        "hint": (
            "Uses the built-in CBC solver. No additional installation needed. Best for"
            " small to medium linear and integer programming problems."
        ),
    },
    "cvxpy_osqp": {
        "label": "CVXPY / OSQP — best for medium problems (500–2,000 variables)",
        "size_guidance": "500–2,000 variables",
        "hint": (
            "Requires CVXPY with OSQP solver. Excellent for quadratic programming and"
            " medium-sized linear problems. Install with: pip install cvxpy"
        ),
    },
    "cvxpy_scip": {
        "label": "CVXPY / SCIP — best for large MIP problems (2,000–10,000 variables)",
        "size_guidance": "2,000–10,000 variables",
        "hint": (
            "Requires CVXPY with SCIP solver. State-of-the-art for large mixed-integer"
            " problems. Install with: pip install cvxpy pyscipopt"
        ),
    },
    "cvxpy_glpk": {
        "label": "CVXPY / GLPK — alternative for medium problems (500–3,000 variables)",
        "size_guidance": "500–3,000 variables",
        "hint": (
            "Requires CVXPY with GLPK solver. Good open-source alternative for medium"
            " problems. Install with: pip install cvxpy"
        ),
    },
    "auto": {
        "label": "Auto-detect — let the app choose based on problem size",
        "size_guidance": "any size",
        "hint": (
            "Automatically selects the best available solver based on problem type and"
            " size. Recommended for most users."
        ),
    },
}

# Time limit captions for different ranges
TIME_LIMIT_CAPTIONS = {
    (10, 30): {
        "mode": "Quick mode",
        "description": (
            "Quick mode: Finds good solutions fast, may not reach optimality"
        ),
    },
    (31, 120): {
        "mode": "Balanced mode",
        "description": (
            "Balanced mode: Good trade-off between solution quality and time"
        ),
    },
    (121, 300): {
        "mode": "Quality mode",
        "description": "Quality mode: Takes time to find high-quality solutions",
    },
    (301, 600): {
        "mode": "Research mode",
        "description": "Research mode: Maximum effort to prove optimality",
    },
}


def render_solver_settings():
    """
    Renders the complete solver settings UI block.

    This function should be called from within a sidebar context in app.py.
    It manages all solver-related UI components and state.
    """
    st.markdown("### ⚙️ Solver Settings")

    # Initialize session state if needed
    if "solver_type" not in st.session_state:
        st.session_state.solver_type = "auto"
    if "max_time" not in st.session_state:
        # Adaptive default: larger problems get more time on the slider
        _pdata = st.session_state.get("problem_data", {})
        _nv = _pdata.get("num_variables", 0) or 0
        if _nv > 20_000:
            st.session_state.max_time = 300
        elif _nv > 5_000:
            st.session_state.max_time = 180
        elif _nv > 500:
            st.session_state.max_time = 120
        else:
            st.session_state.max_time = 60

    # Create list of labels for the selectbox
    solver_labels = [opt["label"] for opt in SOLVER_OPTIONS.values()]

    # Find current selection index
    current_key = st.session_state.solver_type
    current_index = (
        list(SOLVER_OPTIONS.keys()).index(current_key)
        if current_key in SOLVER_OPTIONS
        else 0
    )

    # Solver selection dropdown
    selected_index = st.selectbox(
        "Optimization solver:",
        options=range(len(solver_labels)),
        format_func=lambda x: solver_labels[x],
        index=current_index,
        key="solver_selectbox",
        help="Choose a solver based on your problem size and type",
    )

    # Update session state with the selected solver key
    selected_key = list(SOLVER_OPTIONS.keys())[selected_index]
    st.session_state.solver_type = selected_key

    # Show hint caption for selected solver
    st.caption(SOLVER_OPTIONS[selected_key]["hint"])

    # Time limit slider
    st.markdown("---")

    # Check if solver supports time limits
    supports_time_limit = selected_key in ["cvxpy_scip", "auto"]

    if supports_time_limit:
        # Fully editable slider
        max_time = st.slider(
            "Time limit (seconds):",
            min_value=10,
            max_value=600,
            value=st.session_state.max_time,
            step=10,
            key="max_time_slider",
        )
        st.session_state.max_time = max_time

        # Find and show appropriate caption
        for (min_val, max_val), caption_info in TIME_LIMIT_CAPTIONS.items():
            if min_val <= max_time <= max_val:
                st.caption(caption_info["description"])
                break
    else:
        # Disabled slider
        st.slider(
            "Time limit (seconds):",
            min_value=10,
            max_value=600,
            value=60,
            step=10,
            disabled=True,
            help="Time limits are only available for SCIP and Auto-detect solvers",
        )
        st.caption(
            "⚠️ Time limits not supported by this solver. It will run until completion."
        )

    # Explanation about feasible vs optimal
    st.caption(
        "💡 Solvers may return feasible (good) solutions even if they can't prove"
        " optimality within the time limit."
    )


def get_selected_solver_key() -> str:
    """
    Returns the currently selected solver key from session state.

    Returns:
        str: The internal solver key (e.g., 'pulp_cbc', 'cvxpy_osqp', etc.)
    """
    return st.session_state.get("solver_type", "auto")


def get_selected_time_limit() -> int:
    """
    Returns the currently selected time limit from session state.

    Returns:
        int: The time limit in seconds
    """
    return st.session_state.get("max_time", 60)


def get_solver_explanation(solver_key: str) -> str:
    """
    Returns a brief explanation of the selected solver.

    Args:
        solver_key: The internal solver key

    Returns:
        str: Brief explanation of the solver's capabilities
    """
    if solver_key in SOLVER_OPTIONS:
        return SOLVER_OPTIONS[solver_key]["hint"]
    return "Unknown solver"
