"""
Solver Interface - Unified interface for different OR solvers
"""

from typing import Dict, Any, Optional
import functools
import pulp
import time
from enum import Enum
from .solver_router import get_solver_display_name


@functools.lru_cache(maxsize=1)
def _cbc_runs() -> bool:
    """
    Check that PuLP's bundled CBC binary can actually execute.

    ``PULP_CBC_CMD.available()`` only checks that the file exists; on Apple
    Silicon without Rosetta the bundled x86_64 binary exists but fails with
    "Bad CPU type in executable", so we probe with a one-variable problem.
    """
    try:
        probe = pulp.LpProblem("cbc_probe", pulp.LpMinimize)
        x = pulp.LpVariable("x", lowBound=0)
        probe += x
        probe += x >= 1
        probe.solve(pulp.PULP_CBC_CMD(msg=0))
        return probe.status == 1
    except Exception:
        return False


def default_pulp_solver(msg: int = 0):
    """
    Return the default PuLP solver: CBC when it runs on this machine,
    otherwise HiGHS (native wheel via ``highspy``).
    """
    if _cbc_runs():
        return pulp.PULP_CBC_CMD(msg=msg)
    highs = pulp.HiGHS(msg=bool(msg))
    if highs.available():
        return highs
    # Nothing better available; let the CBC error surface at solve time.
    return pulp.PULP_CBC_CMD(msg=msg)


class SolverStatus(Enum):
    """Solver status codes"""

    OPTIMAL = "Optimal"
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    NOT_SOLVED = "Not Solved"
    TIME_LIMIT = "Time Limit Reached"
    ERROR = "Error"


class SolverInterface:
    """
    Unified interface for various optimization solvers.
    Currently supports PuLP, with extensibility for OR-Tools, CVXPY, etc.
    """

    def __init__(self, solver_type: str = "pulp", problem_data: Dict[str, Any] = None):
        """
        Initialize solver interface.

        Args:
            solver_type: Type of solver. Accepted values (case-insensitive):
                'pulp', 'gurobi', 'cplex', 'glpk', 'auto-detect',
                'or-tools', 'ortools', 'cvxpy'.
            problem_data: Problem classification data (optional, used for routing)
        """
        self.original_solver_type = solver_type.lower().strip()
        self.problem_data = problem_data if problem_data is not None else {}
        self.solver_type = self.original_solver_type
        self.solver = self._initialize_solver()

    @property
    def will_use_cache(self) -> bool:
        """Check if this solve will use cached results."""
        if self.problem_data.get("source") == "miplib":
            instance_name = self.problem_data.get("instance_name")
            if instance_name:
                from src.storage.miplib_cache import MIPLIBCache

                cache = MIPLIBCache()
                return cache.get(instance_name) is not None
        return False

    def _initialize_solver(self):
        """Initialize the appropriate solver."""
        st = self.solver_type

        # Handle new solver key format
        if st == "pulp_cbc" or st in ("pulp", "cbc"):
            return default_pulp_solver()

        if st.startswith("cvxpy_") or st == "cvxpy":
            return None  # no PuLP solver needed; _solve_with_cvxpy handles it

        if st == "gurobi":
            try:
                return pulp.GUROBI_CMD(msg=0)
            except (ImportError, Exception):
                raise RuntimeError(
                    "Gurobi not available. Install and configure license."
                )

        if st == "cplex":
            try:
                return pulp.CPLEX_CMD(msg=0)
            except (ImportError, Exception):
                raise RuntimeError(
                    "CPLEX not available. Install and configure license."
                )

        if st == "glpk":
            try:
                return pulp.GLPK_CMD(msg=0)
            except (ImportError, Exception):
                raise RuntimeError("GLPK not available.")

        if st == "auto-detect" or st == "auto":
            # Auto-detect is now handled by solver_router
            return default_pulp_solver()

        if st in ("or-tools", "ortools"):
            self._ui_info(
                "OR-Tools selected — using PuLP CBC (OR-Tools integration coming soon)."
            )
            self.solver_type = "pulp"
            return default_pulp_solver()

        return default_pulp_solver()

    def _auto_detect_solver(self):
        """
        DEPRECATED: Auto-detection is now handled by solver_router.
        This method is kept for backward compatibility only.
        """
        for name, factory in (
            ("gurobi", pulp.GUROBI_CMD),
            ("cplex", pulp.CPLEX_CMD),
            ("glpk", pulp.GLPK_CMD),
        ):
            try:
                solver = factory(msg=0)
                if solver.available():
                    self.solver_type = name
                    print(f"INFO: Auto-detected solver: {name}")
                    return solver
            except (ImportError, Exception):
                continue
        self.solver_type = "pulp"
        print("INFO: Auto-detect — using default PuLP CBC solver.")
        return default_pulp_solver()

    @staticmethod
    def _ui_info(message: str) -> None:
        """Show an info message via Streamlit if running in a UI, else print."""
        try:
            import streamlit as st

            st.info(message)
        except (ImportError, Exception):
            print(f"INFO: {message}")

    _STATUS_LABELS = {
        1: "Optimal",
        0: "Not Solved",
        -1: "Infeasible",
        -2: "Unbounded",
        -3: "Undefined",
    }

    _STATUS_WARNINGS = {
        "Infeasible": (
            "Problem has no feasible solution — constraints may contradict each other."
        ),
        "Unbounded": "Problem is unbounded — add upper bounds to variables.",
        "Not Solved": "Solver did not find a solution within the time limit.",
        "Undefined": "Solver returned an undefined status.",
    }

    def _empty_result(self, **overrides) -> Dict[str, Any]:
        """Return a guaranteed-shape result dict, with optional overrides."""
        # Get display name for the resolved solver
        solver_name = self.solver_type
        if hasattr(self, "resolved_solver_key"):
            solver_name = get_solver_display_name(self.resolved_solver_key)

        base: Dict[str, Any] = {
            "status": "Error",
            "is_optimal": False,
            "objective_value": None,
            "variables": {},
            "num_variables": 0,
            "num_constraints": 0,
            "solve_time": 0.0,
            "solver_name": solver_name,
            "warnings": [],
            "error_message": "",
        }
        base.update(overrides)
        return base

    def _maybe_cache_and_return(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Cache MIPLIB solutions before returning."""
        # Only cache successful MIPLIB solutions
        if (
            self.problem_data.get("source") == "miplib"
            and result.get("is_optimal")
            and "cache_metadata" not in result
        ):  # Don't re-cache cached results

            instance_name = self.problem_data.get("instance_name")
            if instance_name:
                from src.storage.miplib_cache import MIPLIBCache

                cache = MIPLIBCache()
                cache.set(instance_name, result)

        return result

    def _get_adaptive_timeout(
        self,
        user_timeout: Optional[int],
        num_variables: int = 0,
        num_constraints: int = 0,
    ) -> int:
        """Return timeout in seconds. Respects user override; otherwise scales with problem size."""
        if user_timeout is not None and user_timeout > 0:
            return user_timeout
        if num_variables > 20_000 or num_constraints > 20_000:
            return 300
        elif num_variables > 5_000 or num_constraints > 5_000:
            return 180
        else:
            return 120

    def solve(
        self,
        model,
        max_time: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Solve the optimization model.  **Never raises** — always returns
        a dict with a consistent set of keys.

        Args:
            model: PuLP LpProblem object, or a dict for CVXPY.
            max_time: Maximum solution time in seconds.
            verbose: Print solver output.

        Returns:
            Dictionary with keys: status, is_optimal, objective_value,
            variables, num_variables, num_constraints, solve_time,
            solver_name, warnings, error_message.
        """
        # Check MIPLIB cache first
        from src.storage.miplib_cache import MIPLIBCache

        if self.problem_data.get("source") == "miplib":
            instance_name = self.problem_data.get("instance_name")
            if instance_name:
                cache = MIPLIBCache()
                cached_solution = cache.get(instance_name)
                if cached_solution:
                    # Only use cache if we're solving with the same solver type
                    # This prevents errors when switching between PuLP and CVXPY
                    cached_solver = cached_solution.get("solver_name", "").lower()
                    current_is_cvxpy = self.original_solver_type.startswith("cvxpy")
                    cached_is_cvxpy = "cvxpy" in cached_solver

                    if current_is_cvxpy == cached_is_cvxpy:
                        return cached_solution
                    # Otherwise, solve fresh with the new solver

        # The solver has already been resolved by app.py
        resolved_key = self.original_solver_type
        self.resolved_solver_key = resolved_key

        # Update solver type
        self.solver_type = resolved_key
        self.resolved_explanation = ""  # No longer needed since app.py handles this

        # Adaptive timeout: respect user override, otherwise scale with problem size
        _n_vars = self.problem_data.get("num_variables", 0) or 0
        _n_cons = self.problem_data.get("num_constraints", 0) or 0
        _timeout = self._get_adaptive_timeout(max_time, _n_vars, _n_cons)

        # Route to appropriate solver
        # Fast-path: SCIP selected + we have an MPS file → use PySCIPOpt directly.
        # This avoids the expensive PuLP → CVXPY conversion for large MIPLIB instances.
        if resolved_key == "cvxpy_scip":
            mps_path = self.problem_data.get("mps_file_path")
            if mps_path:
                direct_result = self._solve_with_pyscipopt(mps_path, max_time=_timeout)
                if direct_result is not None:
                    return direct_result

        if resolved_key.startswith("cvxpy_") or self.solver_type == "cvxpy":
            return self._solve_with_cvxpy(model, max_time=_timeout)

        if isinstance(model, dict) and model.get("type") == "cvxpy":
            return self._solve_with_cvxpy(model, max_time=_timeout)

        if model is None or not isinstance(model, pulp.LpProblem):
            return self._empty_result(
                error_message="No valid model to solve.",
            )

        start_time = time.time()

        try:
            if _timeout and hasattr(self.solver, "timeLimit"):
                self.solver.timeLimit = _timeout

            model.solve(self.solver)
            solve_time = round(time.time() - start_time, 4)

            status_code = model.status
            status_label = self._STATUS_LABELS.get(status_code, "Undefined")
            is_optimal = status_code == 1

            # --- variable extraction (individually guarded) --------------
            variables: Dict[str, Any] = {}
            try:
                for var in model.variables():
                    try:
                        variables[var.name] = var.varValue
                    except Exception:
                        variables[var.name] = None
            except Exception:
                pass

            # --- objective value -----------------------------------------
            objective_value = None
            if is_optimal:
                try:
                    objective_value = pulp.value(model.objective)
                except Exception:
                    pass

            # --- counts --------------------------------------------------
            try:
                num_variables = len(model.variables())
            except Exception:
                num_variables = len(variables)
            try:
                num_constraints = len(model.constraints)
            except Exception:
                num_constraints = 0

            # --- warnings ------------------------------------------------
            warnings_list: list = []
            status_warning = self._STATUS_WARNINGS.get(status_label)
            if status_warning:
                warnings_list.append(status_warning)

            # Add solver routing explanation if there was a fallback
            if (
                hasattr(self, "resolved_explanation")
                and "falling back" in self.resolved_explanation.lower()
            ):
                warnings_list.append(self.resolved_explanation)

            overrides = {}
            if isinstance(self.solver, pulp.HiGHS):
                overrides["solver_name"] = "PuLP HiGHS"
                warnings_list.append(
                    "CBC cannot run on this machine; solved with HiGHS instead."
                )

            result = self._empty_result(
                **overrides,
                status=status_label,
                is_optimal=is_optimal,
                objective_value=objective_value,
                variables=variables,
                num_variables=num_variables,
                num_constraints=num_constraints,
                solve_time=solve_time,
                warnings=warnings_list,
            )
            return self._maybe_cache_and_return(result)

        except Exception as e:
            return self._empty_result(
                solve_time=round(time.time() - start_time, 4),
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    #  CVXPY solver
    # ------------------------------------------------------------------

    _CVXPY_STATUS_MAP = {
        "optimal": "Optimal",
        "infeasible": "Infeasible",
        "unbounded": "Unbounded",
        "infeasible_inaccurate": "Infeasible",
        "unbounded_inaccurate": "Unbounded",
        "optimal_inaccurate": "Optimal (warning: inaccurate)",
        "solver_error": "Error",
    }

    def _solve_with_cvxpy(
        self,
        model,
        max_time: int = 60,
    ) -> Dict[str, Any]:
        """
        Solve using CVXPY.  Accepts a ``pulp.LpProblem`` (auto-converted)
        or a raw dict ``{'objective': cp.Minimize/Maximize, 'constraints': [...],
        'variables': {name: cp.Variable}}``.

        Returns the same result dict shape as the PuLP path.
        """
        try:
            import cvxpy as cp
        except ImportError:
            return self._empty_result(
                error_message="cvxpy is not installed. Run: pip install cvxpy",
            )

        start = time.time()

        try:
            prob = None
            cp_vars: Dict[str, Any] = {}
            cp_cons = []

            # --- unwrap model-generator CVXPY dict ------------------------
            _suggested_solver = None
            if isinstance(model, dict) and model.get("type") == "cvxpy":
                inner = model.get("problem")
                cp_vars = model.get("variables", {})
                _suggested_solver = model.get("solver")
                if isinstance(inner, cp.Problem):
                    prob = inner
                    cp_cons = inner.constraints
                elif isinstance(inner, pulp.LpProblem):
                    model = inner  # fall through to PuLP conversion
                else:
                    return self._empty_result(
                        error_message=(
                            "CVXPY dict has unsupported problem type: "
                            f"{type(inner).__name__}"
                        ),
                    )

            # --- accept a raw CVXPY dict (objective + constraints) --------
            if prob is None and isinstance(model, dict) and "objective" in model:
                cp_obj = model["objective"]
                cp_cons = model.get("constraints", [])
                cp_vars = model.get("variables", {})
                prob = cp.Problem(cp_obj, cp_cons)

            # --- convert PuLP → CVXPY ------------------------------------
            if prob is None and isinstance(model, pulp.LpProblem):
                cp_vars = {}
                bound_cons = []
                for v in model.variables():
                    if v.cat == "Binary":
                        cp_vars[v.name] = cp.Variable(
                            name=v.name,
                            boolean=True,
                        )
                        continue
                    cvar = cp.Variable(
                        name=v.name,
                        integer=(v.cat == "Integer"),
                    )
                    cp_vars[v.name] = cvar
                    if v.lowBound is not None:
                        bound_cons.append(cvar >= float(v.lowBound))
                    if v.upBound is not None:
                        bound_cons.append(cvar <= float(v.upBound))

                # objective (iterate only the terms present, keep the constant)
                obj_expr = float(model.objective.constant or 0) + sum(
                    float(coef) * cp_vars[v.name] for v, coef in model.objective.items()
                )
                if model.sense == pulp.constants.LpMaximize:
                    cp_obj = cp.Maximize(obj_expr)
                else:
                    cp_obj = cp.Minimize(obj_expr)

                # constraints
                cp_cons = []
                for cname, cobj in model.constraints.items():
                    lhs = sum(float(coef) * cp_vars[v.name] for v, coef in cobj.items())
                    rhs = -float(cobj.constant)
                    sense = cobj.sense
                    if sense == pulp.constants.LpConstraintLE:
                        cp_cons.append(lhs <= rhs)
                    elif sense == pulp.constants.LpConstraintGE:
                        cp_cons.append(lhs >= rhs)
                    else:
                        cp_cons.append(lhs == rhs)

                # bound constraints are kept out of cp_cons so the reported
                # constraint count matches the original model
                prob = cp.Problem(cp_obj, cp_cons + bound_cons)

            if prob is None:
                return self._empty_result(
                    error_message=(
                        f"Unsupported model type for CVXPY: {type(model).__name__}"
                    ),
                )

            # --- solve with proper solver availability checking -----------
            solver_used = ""
            warnings_list = []

            # Check if the requested solver is actually available
            _solver_chain = []

            if hasattr(
                self, "resolved_solver_key"
            ) and self.resolved_solver_key.startswith("cvxpy_"):
                requested_key = self.resolved_solver_key

                if requested_key == "cvxpy_scip":
                    # Check if SCIP is available in CVXPY
                    scip_cls = getattr(cp, "SCIP", None)
                    if scip_cls is None or "SCIP" not in cp.installed_solvers():
                        # Also check if pyscipopt is installed
                        try:
                            import pyscipopt  # noqa: F401  (availability check)

                            # Even with pyscipopt, SCIP might not be in CVXPY
                            if "SCIP" in cp.installed_solvers():
                                scip_cls = cp.SCIP
                                _solver_chain.append(scip_cls)
                            else:
                                raise ImportError("SCIP not in CVXPY installed solvers")
                        except ImportError:
                            warnings_list.append(
                                "SCIP not available — falling back. Install with: pip"
                                " install pyscipopt"
                            )
                            # Get fallback solver from router
                            from src.solvers.solver_router import resolve_solver

                            fallback_key, _ = resolve_solver(
                                "cvxpy_glpk", self.problem_data
                            )
                            requested_key = fallback_key
                            # Now add the fallback solver
                            if fallback_key == "cvxpy_glpk" and hasattr(cp, "GLPK_MI"):
                                _solver_chain.append(cp.GLPK_MI)
                            elif fallback_key == "cvxpy_osqp" and cp.OSQP:
                                _solver_chain.append(cp.OSQP)
                    else:
                        _solver_chain.append(scip_cls)

                if requested_key == "cvxpy_osqp":
                    if cp.OSQP:
                        _solver_chain.append(cp.OSQP)
                    else:
                        warnings_list.append("OSQP not available")

                elif requested_key == "cvxpy_glpk":
                    if hasattr(cp, "GLPK_MI") and cp.GLPK_MI:
                        _solver_chain.append(cp.GLPK_MI)
                    else:
                        warnings_list.append("GLPK not available")
                        # Fallback to OSQP
                        if cp.OSQP:
                            _solver_chain.append(cp.OSQP)

            elif _suggested_solver:
                # Legacy path for suggested solver
                _suggested_cls = getattr(cp, _suggested_solver, None)
                if _suggested_cls:
                    _solver_chain.append(_suggested_cls)

            # cp.OSQP / cp.GLPK_MI are plain name strings and always truthy, so
            # drop any solver CVXPY can't actually use.
            _installed = set(cp.installed_solvers())
            for s in _solver_chain:
                if s not in _installed:
                    warnings_list.append(f"{s} not installed — skipped")
            _solver_chain = [s for s in _solver_chain if s in _installed]
            # OSQP is a QP solver; on linear objectives it is the wrong tool and
            # its Linux wheels can segfault (uncatchable) during setup, so let
            # CVXPY pick an LP solver instead.
            if cp.OSQP in _solver_chain and prob.objective.expr.is_affine():
                _solver_chain.remove(cp.OSQP)
                warnings_list.append(
                    "OSQP skipped for a linear objective; used CVXPY's LP solver."
                )
            # Always finish with CVXPY's own choice (None) so a missing or
            # failing solver (e.g. OSQP on an integer model) isn't fatal.
            _solver_chain.append(None)

            # Try each solver in the chain
            for solver_cls in _solver_chain:
                try:
                    if solver_cls is None:
                        prob.solve(verbose=False)
                        solver_used = "CVXPY default"
                    else:
                        # Build solver-specific kwargs for time limits
                        # Each CVXPY solver accepts time limits via different keywords
                        kwargs = {"solver": solver_cls, "verbose": False}

                        if max_time is not None and max_time > 0:
                            solver_name = getattr(
                                solver_cls, "__name__", str(solver_cls)
                            ).upper()

                            if "SCIP" in solver_name:
                                kwargs["scip_params"] = {"limits/time": float(max_time)}
                            elif "GLPK" in solver_name:
                                kwargs["glpk"] = {"tmlim": int(max_time)}
                            elif "OSQP" in solver_name:
                                kwargs["time_limit"] = float(max_time)
                            elif "HIGHS" in solver_name:
                                kwargs["time_limit"] = float(max_time)
                            elif "CBC" in solver_name:
                                kwargs["maximumSeconds"] = int(max_time)
                            elif "SCS" in solver_name:
                                kwargs["time_limit_secs"] = float(max_time)

                        prob.solve(**kwargs)
                        solver_used = getattr(solver_cls, "__name__", str(solver_cls))

                    # Check if solve was successful
                    if prob.status in [
                        "optimal",
                        "optimal_inaccurate",
                        "unbounded_inaccurate",
                    ]:
                        break
                    elif prob.status in ["infeasible", "infeasible_inaccurate"]:
                        # Infeasible is a valid result, not an error
                        break
                    else:
                        # Status like solver_error, try next solver
                        continue

                except Exception:
                    # Actual error, try next solver
                    continue

            # Store which solver actually succeeded
            self.solver_actually_used = solver_used

            # If we used a different solver than requested, note it
            if hasattr(
                self, "resolved_solver_key"
            ) and self.resolved_solver_key.startswith("cvxpy_"):
                expected_name = self.resolved_solver_key.replace("cvxpy_", "").upper()
                if (
                    solver_used
                    and expected_name not in solver_used.upper()
                    and solver_used != "CVXPY default"
                ):
                    warnings_list.append(
                        f"Used {solver_used} solver (requested {expected_name})"
                    )

            solve_time = round(time.time() - start, 4)

            # --- map status -----------------------------------------------
            raw_status = prob.status or "solver_error"
            status_label = self._CVXPY_STATUS_MAP.get(raw_status, raw_status)
            is_optimal = raw_status in ("optimal", "optimal_inaccurate")

            # Check if we hit time limit (indicated by inaccurate status + time close to limit)
            hit_time_limit = False
            if max_time and solve_time >= (max_time * 0.95):  # Within 95% of time limit
                if raw_status in ("optimal_inaccurate", "feasible"):
                    status_label = "Feasible (time limit)"
                    hit_time_limit = True
                    is_optimal = False

            # --- extract variable values ----------------------------------
            variables: Dict[str, Any] = {}
            for vname, cvar in cp_vars.items():
                try:
                    val = cvar.value
                    variables[vname] = float(val) if val is not None else None
                except Exception:
                    variables[vname] = None

            # --- objective value ------------------------------------------
            objective_value = None
            if is_optimal:
                try:
                    objective_value = float(prob.value)
                except Exception:
                    pass

            # --- warnings -------------------------------------------------
            # warnings_list already populated during solver selection
            if hit_time_limit:
                warnings_list.append(
                    f"Solver reached time limit of {max_time} seconds. Solution may not"
                    " be optimal."
                )
            elif raw_status == "optimal_inaccurate" and not hit_time_limit:
                warnings_list.append(
                    "Solution is optimal but may be numerically inaccurate."
                )
            status_warning = self._STATUS_WARNINGS.get(status_label)
            if status_warning:
                warnings_list.append(status_warning)

            # Try to get MIP gap for time limit cases
            mip_gap = None
            if hit_time_limit and hasattr(prob, "solver_stats"):
                # Different solvers report gap differently
                if "mipgap" in prob.solver_stats:
                    mip_gap = prob.solver_stats["mipgap"]
                elif "gap" in prob.solver_stats:
                    mip_gap = prob.solver_stats["gap"]

            result = self._empty_result(
                status=status_label,
                is_optimal=is_optimal,
                objective_value=objective_value,
                variables=variables,
                num_variables=len(cp_vars),
                num_constraints=len(cp_cons),
                solve_time=solve_time,
                solver_name=f"cvxpy ({solver_used})",
                warnings=warnings_list,
            )

            # Add MIP gap if available
            if mip_gap is not None:
                result["mip_gap"] = mip_gap

            return self._maybe_cache_and_return(result)

        except Exception as e:
            return self._empty_result(
                solve_time=round(time.time() - start, 4),
                solver_name="cvxpy",
                error_message=str(e),
            )

    def _solve_with_pyscipopt(
        self,
        mps_path: str,
        max_time: int = 60,
    ) -> Dict[str, Any]:
        """
        Solve an MPS file directly with PySCIPOpt — bypasses PuLP and CVXPY
        entirely, which is dramatically faster for large MIPLIB instances.
        """
        try:
            from pyscipopt import Model as SCIPModel
        except ImportError:
            return None  # caller falls back to normal path

        import gzip
        import os

        start = time.time()
        warnings_list = []

        try:
            # SCIP can't read .gz if the compression plugin is missing;
            # decompress to a temp file alongside the original.
            actual_path = mps_path
            if mps_path.endswith(".gz"):
                plain = mps_path[:-3]  # strip .gz
                if not os.path.exists(plain):
                    with gzip.open(mps_path, "rb") as fin:
                        with open(plain, "wb") as fout:
                            fout.write(fin.read())
                actual_path = plain

            m = SCIPModel()
            m.readProblem(actual_path)

            if max_time and max_time > 0:
                m.setParam("limits/time", float(max_time))

            # Start timing only the optimization
            opt_start = time.time()
            m.optimize()
            solve_time = round(time.time() - opt_start, 4)

            scip_status = m.getStatus()  # 'optimal', 'timelimit', 'infeasible', …
            is_optimal = scip_status == "optimal"

            if scip_status == "timelimit":
                status_label = "Feasible (time limit)"
            elif scip_status == "optimal":
                status_label = "Optimal"
            elif scip_status == "infeasible":
                status_label = "Infeasible"
            elif scip_status == "unbounded":
                status_label = "Unbounded"
            else:
                status_label = scip_status.capitalize()

            obj_val = None
            variables: Dict[str, Any] = {}

            best_sol = m.getBestSol() if m.getNSols() > 0 else None
            if best_sol is not None:
                obj_val = m.getObjVal()
                for v in m.getVars():
                    variables[v.name] = m.getSolVal(best_sol, v)

            mip_gap = None
            if m.getNSols() > 0 and not is_optimal:
                try:
                    mip_gap = m.getGap()
                    warnings_list.append(
                        f"MIP gap: {mip_gap:.2%}. Time limit of {max_time}s reached —"
                        " solution is feasible but may not be optimal."
                    )
                except Exception:
                    pass

            result = self._empty_result(
                status=status_label,
                is_optimal=is_optimal,
                objective_value=obj_val,
                variables=variables,
                num_variables=m.getNVars(),
                num_constraints=m.getNConss(),
                solve_time=solve_time,
                solver_name="PySCIPOpt (SCIP direct)",
                warnings=warnings_list,
            )
            if mip_gap is not None:
                result["mip_gap"] = mip_gap
            return self._maybe_cache_and_return(result)

        except Exception as e:
            return self._empty_result(
                solve_time=round(time.time() - start, 4),
                solver_name="PySCIPOpt",
                error_message=f"PySCIPOpt failed: {e}",
            )

    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about the current solver."""

        return {
            "solver_type": self.solver_type,
            "solver_object": str(self.solver),
            "available": True,  # Could check actual availability
        }

    @staticmethod
    def get_sensitivity(model: pulp.LpProblem) -> Dict[str, Any]:
        """
        Extract sensitivity analysis data from an optimally-solved model.

        Returns a dict with ``shadow_prices``, ``reduced_costs``, and an
        ``available`` flag.  If the solver did not expose dual values
        (``pi`` / ``dj`` are None), returns empty dicts with a ``note``.

        Raises:
            ValueError: If the model has not been solved to optimality.
        """
        if model.status != 1:
            raise ValueError(
                "Sensitivity data requires an optimal solution. "
                f"Current status: {pulp.LpStatus.get(model.status, 'Unknown')} "
                f"(code {model.status})"
            )

        try:
            shadow_prices = {}
            for name in model.constraints:
                pi = model.constraints[name].pi
                if pi is None:
                    raise AttributeError("pi is None")
                shadow_prices[name] = pi

            reduced_costs = {}
            for var in model.variables():
                dj = var.dj
                if dj is None:
                    raise AttributeError("dj is None")
                reduced_costs[var.name] = dj

            return {
                "shadow_prices": shadow_prices,
                "reduced_costs": reduced_costs,
                "available": True,
            }

        except (AttributeError, TypeError):
            return {
                "shadow_prices": {},
                "reduced_costs": {},
                "available": False,
                "note": (
                    "Sensitivity data not available. The solver did not "
                    "return dual values (shadow prices / reduced costs). "
                    "This is common with MIP solvers or when the model "
                    "uses integer/binary variables."
                ),
            }

    @staticmethod
    def list_available_solvers() -> Dict[str, bool]:
        """List all solvers and their availability."""

        solvers = {
            "pulp_cbc": True,  # Always available with PuLP
            "gurobi": False,
            "cplex": False,
            "glpk": False,
            "ortools": False,
            "cvxpy": False,
        }

        for name, factory in (
            ("gurobi", pulp.GUROBI_CMD),
            ("cplex", pulp.CPLEX_CMD),
            ("glpk", pulp.GLPK_CMD),
        ):
            try:
                if factory(msg=0).available():
                    solvers[name] = True
            except Exception:
                pass

        return solvers


class SensitivityAnalyzer:
    """Perform sensitivity analysis on solutions."""

    def __init__(self, solution: Dict[str, Any]):
        """Initialize with a solution."""
        self.solution = solution

    def analyze_shadow_prices(self) -> Dict[str, Any]:
        """Analyze shadow prices (dual values) of constraints."""
        if not self.solution.get("is_optimal"):
            return {"error": "Solution not optimal"}

        shadow_prices = {}
        constraints = self.solution.get("_constraint_detail", {})
        for name, info in constraints.items():
            if info.get("shadow_price") is not None:
                shadow_prices[name] = {
                    "shadow_price": info["shadow_price"],
                    "slack": info.get("slack", 0),
                    "binding": abs(info.get("slack", 1)) < 1e-6,
                }
        return shadow_prices

    def analyze_variable_slack(self) -> Dict[str, Any]:
        """Analyze variable slack (distance from bounds)."""
        if not self.solution.get("is_optimal"):
            return {"error": "Solution not optimal"}

        slack_analysis = {}
        for name, value in self.solution.get("variables", {}).items():
            slack_analysis[name] = {
                "value": value,
                "at_bound": value is not None and abs(value) < 1e-6,
            }
        return slack_analysis


if __name__ == "__main__":
    # Example usage
    print("Available solvers:", SolverInterface.list_available_solvers())

    # Create a simple LP problem
    prob = pulp.LpProblem("Test", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)

    prob += 2 * x + 3 * y, "Cost"
    prob += x + y >= 10, "Constraint1"
    prob += 2 * x + y >= 15, "Constraint2"

    # Solve
    solver = SolverInterface("pulp")
    solution = solver.solve(prob)

    print("\nSolution:")
    print(f"  Status       : {solution['status']}")
    print(f"  Is Optimal   : {solution['is_optimal']}")
    print(f"  Objective    : {solution['objective_value']}")
    print(f"  Solver       : {solution['solver_name']}")
    print(f"  Solve time   : {solution['solve_time']}s")
    print(f"  Variables    : {solution['num_variables']}")
    print(f"  Constraints  : {solution['num_constraints']}")
    print("\nVariable values:")
    for name, value in solution["variables"].items():
        print(f"  {name} = {value}")
