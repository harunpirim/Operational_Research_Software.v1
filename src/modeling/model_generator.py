"""
Model Generator - Creates mathematical models from problem data
Supports both Anthropic Claude and OpenAI models.
"""

import re
from typing import Dict, Any, Optional, List
import pulp
import os
import json
from src.utils.api_client import APIClient


class ModelGenerator:
    """
    Generates mathematical optimization models from structured problem data.
    Can use AI (Anthropic Claude or OpenAI) to help generate model code dynamically.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the model generator.

        Args:
            api_key: API key (optional, reads from environment if not provided)
            provider: 'anthropic' or 'openai' (optional, auto-detects if not provided)
            model: Model name (optional, uses default if not provided)
        """
        try:
            self.api_client = APIClient(provider=provider, api_key=api_key, model=model)
            self.model = self.api_client.get_model_name()
        except ValueError:
            # API not configured - model generation without AI will still work
            self.api_client = None
            self.model = None

    def generate(
        self,
        problem_data: Dict[str, Any],
        solver_preference: str = "pulp",
    ):
        """
        Generate an optimization model from problem data.

        Args:
            problem_data: Structured problem information from classifier.
            solver_preference: ``'pulp'`` (default) returns a
                ``pulp.LpProblem``; ``'cvxpy'`` returns a dict with keys
                ``type``, ``problem``, ``variables`` that the CVXPY solver
                path in ``SolverInterface`` can consume directly.

        Returns:
            A ``pulp.LpProblem`` or a CVXPY model dict.

        Raises:
            ValueError: If problem_type is unrecognised/unknown or generation
                        fails for any reason.
        """
        problem_type = problem_data.get("problem_type", "unknown")
        confidence = problem_data.get("confidence", 0.0)
        warnings = problem_data.get("warnings", [])

        if (
            problem_type in ("unknown", "general", "other", "", None)
            or confidence < 0.1
        ):
            raise ValueError(
                "Cannot build model: problem was not recognized as an "
                "Operations Research problem.\n"
                f"Warnings: {warnings}\n"
                "Please describe your problem with: what you want to optimize "
                "(minimize/maximize), what the decision variables are, and "
                "what constraints apply."
            )

        if solver_preference == "cvxpy":
            return self._generate_cvxpy_model(problem_data)

        try:
            if problem_type == "linear_programming":
                return self._generate_lp_model(problem_data)
            elif problem_type == "integer_programming":
                return self._generate_ip_model(problem_data)
            elif problem_type == "mixed_integer_programming":
                return self._generate_mip_model(problem_data)
            elif problem_type == "transportation":
                return self._generate_transportation_model(problem_data)
            elif problem_type == "assignment":
                return self._generate_assignment_model(problem_data)
            elif problem_type == "knapsack":
                return self._generate_knapsack_model(problem_data)
            else:
                return self._generate_with_ai(problem_data)
        except ValueError:
            raise
        except Exception as e:
            import traceback

            tb_str = traceback.format_exc()
            raise ValueError(
                f"Model generation failed for problem type '{problem_type}': {e}\n"
                f"Full error trace:\n{tb_str}\n"
                "Try rephrasing your problem with explicit numbers, variable "
                "names, and constraints."
            ) from e

    # ------------------------------------------------------------------
    #  LP / IP / MIP builders
    # ------------------------------------------------------------------

    _CAT_MAP = {"continuous": "Continuous", "integer": "Integer", "binary": "Binary"}

    def _generate_lp_model(self, problem_data: Dict[str, Any]) -> pulp.LpProblem:
        """Build a pure-LP model (all variables default to Continuous)."""
        return self._build_structured_model(problem_data, "LP_Problem")

    def _generate_ip_model(self, problem_data: Dict[str, Any]) -> pulp.LpProblem:
        """Build an IP model with explicit binary bounds (lowBound=0, upBound=1)."""
        return self._build_structured_model(
            problem_data,
            "IP_Problem",
            explicit_binary_bounds=True,
        )

    def _generate_mip_model(self, problem_data: Dict[str, Any]) -> pulp.LpProblem:
        """Build a MIP model — same builder as IP, mixed types come from the data."""
        return self._generate_ip_model(problem_data)

    def _build_structured_model(
        self,
        problem_data: Dict[str, Any],
        problem_name: str,
        explicit_binary_bounds: bool = False,
    ) -> pulp.LpProblem:
        """
        Shared builder for LP / IP / MIP models.

        Args:
            problem_data: Classifier output dict.
            problem_name: Name for the PuLP LpProblem.
            explicit_binary_bounds: If True, binary variables are created with
                lowBound=0 and upBound=1 (standard for IP/MIP).  If False,
                binary variables omit explicit bounds (PuLP defaults).
        """
        if os.getenv("OR_DEBUG", "0") == "1":
            print(
                f"\n[DEBUG] _build_structured_model({problem_name}) — problem_data received:"
            )
            print(json.dumps(problem_data, indent=2, default=str))
            print(
                "[DEBUG] parameters keys:",
                list(problem_data.get("parameters", {}).keys()),
            )
            print(
                "[DEBUG] decision_variables:",
                [v.get("name") for v in problem_data.get("decision_variables", [])],
            )

        # 1 — objective direction
        sense = (
            pulp.LpMaximize
            if problem_data.get("objective") == "maximize"
            else pulp.LpMinimize
        )
        prob = pulp.LpProblem(problem_name, sense)

        # 2 — decision variables
        var_defs = problem_data.get("decision_variables", [])
        if not var_defs:
            raise ValueError("No decision_variables in problem_data")

        lp_vars: Dict[str, pulp.LpVariable] = {}
        for v in var_defs:
            name = v["name"]
            cat = self._CAT_MAP.get(v.get("type", "continuous"), "Continuous")
            if cat == "Binary" and explicit_binary_bounds:
                lp_vars[name] = pulp.LpVariable(
                    name,
                    lowBound=0,
                    upBound=1,
                    cat="Binary",
                )
            else:
                low = None if cat == "Binary" else 0
                lp_vars[name] = pulp.LpVariable(name, lowBound=low, cat=cat)

        # 3 — objective coefficients (multi-layer)
        var_list = list(lp_vars.values())
        var_names = list(lp_vars.keys())
        n = len(var_list)

        lp_extracted = self._extract_lp_data(problem_data)
        obj_coeffs = lp_extracted["obj_coeffs"]

        if not obj_coeffs:
            obj_coeffs = self._extract_objective_coefficients(
                problem_data,
                var_names,
            )

        # 4 — constraints (first pass)
        constraints = self._extract_constraints_list(problem_data)

        if (
            not constraints
            and lp_extracted["constraint_coefficients"]
            and lp_extracted["constraint_rhs"]
        ):
            matrix = lp_extracted["constraint_coefficients"]
            rhs_vals = lp_extracted["constraint_rhs"]
            parsed_cons = problem_data.get("constraints", [])
            for idx, row in enumerate(matrix):
                sense = "<="
                if idx < len(parsed_cons) and isinstance(parsed_cons[idx], dict):
                    sense = parsed_cons[idx].get("sense", "<=")
                rhs_val = rhs_vals[idx] if idx < len(rhs_vals) else 0.0
                constraints.append(
                    {
                        "name": f"c_{idx}",
                        "coefficients": row,
                        "sense": sense,
                        "rhs": rhs_val,
                    }
                )

        # 5 — AI fallback if coefficients or constraints are missing
        if not obj_coeffs or not constraints:
            ai_data = self._extract_model_data_with_ai(problem_data)
            if ai_data:
                if not obj_coeffs:
                    ai_coeffs = ai_data.get("objective_coefficients")
                    if isinstance(ai_coeffs, list) and ai_coeffs:
                        obj_coeffs = [float(c) for c in ai_coeffs]
                        print("INFO: Objective coefficients recovered via AI fallback.")
                if not constraints:
                    ai_cons = ai_data.get("constraints")
                    if isinstance(ai_cons, list) and ai_cons:
                        constraints = ai_cons
                        print("INFO: Constraints recovered via AI fallback.")

        if not obj_coeffs:
            raise ValueError(
                "Could not determine objective coefficients from parameters, "
                "decision_variables, or AI extraction."
            )

        if len(obj_coeffs) != n:
            action = "Padded with zeros" if len(obj_coeffs) < n else "Truncated"
            print(
                f"WARNING: Coefficient count mismatch. "
                f"{action} to match {n} variables."
            )
            if len(obj_coeffs) < n:
                obj_coeffs += [0.0] * (n - len(obj_coeffs))
            else:
                obj_coeffs = obj_coeffs[:n]

        # 6 — objective function
        prob += (
            pulp.lpSum(c * v for c, v in zip(obj_coeffs, var_list)),
            "Objective",
        )

        # 7 — add constraints to model
        for i, con in enumerate(constraints):
            self._add_parsed_constraint(prob, con, lp_vars, index=i)

        return prob

    # ------------------------------------------------------------------
    #  Helpers for structured model builders
    # ------------------------------------------------------------------

    _COEFF_KEYS = (
        "objective_coefficients",
        "coefficients",
        "obj_coefficients",
        "costs",
        "profits",
        "objective_values",
        "unit_profits",
        "unit_costs",
        "cost_per_unit",
        "profit_per_unit",
        "values",
        "prices",
    )

    _VAR_COEFF_FIELDS = (
        "coefficient",
        "cost",
        "profit",
        "value",
        "objective_coefficient",
    )

    def _extract_objective_coefficients(
        self,
        problem_data: Dict[str, Any],
        var_names: List[str],
    ) -> Optional[List[float]]:
        """
        Return a list of floats for the objective function.

        Four-tier extraction:
          1. Search ``parameters`` and ``problem_data`` for known key names.
          2. Read per-variable coefficient fields from ``decision_variables``.
          3. Ask the AI to extract them.
          4. Return ``None`` only if all three tiers fail.
        """
        params = problem_data.get("parameters", {})
        search_dicts = [params, problem_data]

        # --- Tier 1: known keys in parameters / problem_data -------------
        for d in search_dicts:
            for key in self._COEFF_KEYS:
                val = d.get(key)
                if (
                    isinstance(val, list)
                    and val
                    and all(isinstance(v, (int, float)) for v in val)
                ):
                    return [float(x) for x in val]

        # --- Tier 2: per-variable fields in decision_variables -----------
        var_defs = problem_data.get("decision_variables", [])
        if var_defs:
            extracted = []
            for v in var_defs:
                coeff = None
                for field in self._VAR_COEFF_FIELDS:
                    if field in v and isinstance(v[field], (int, float)):
                        coeff = float(v[field])
                        break
                if coeff is None:
                    break
                extracted.append(coeff)
            else:
                if extracted:
                    return extracted

        # --- Tier 3: AI extraction ---------------------------------------
        if not self.api_client:
            return None

        problem_json = json.dumps(problem_data, indent=2, default=str)
        prompt = (
            "Extract the objective function coefficients for each decision "
            "variable. Return ONLY a JSON array of numbers in the same order "
            "as the decision_variables array.\n\n"
            f"```json\n{problem_json}\n```"
        )
        try:
            response = self.api_client.create_message(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0,
            )
            raw = response["content"].strip()
            fence = re.search(r"\[[\s\S]*?\]", raw)
            text = fence.group(0) if fence else raw
            coeffs = json.loads(text)
            if isinstance(coeffs, list) and all(
                isinstance(c, (int, float)) for c in coeffs
            ):
                return [float(c) for c in coeffs]
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    #  AI fallback for missing coefficients / constraints
    # ------------------------------------------------------------------

    _FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.DOTALL)

    @classmethod
    def _parse_json(cls, text: str) -> Dict[str, Any]:
        """
        Three-layer JSON extraction (mirrors ProblemClassifier._parse_json):
          1. Direct ``json.loads`` on raw text.
          2. Strip markdown fences then ``json.loads``.
          3. Substring from first ``{`` to last ``}`` then ``json.loads``.
        Raises ``json.JSONDecodeError`` if all three fail.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fence = cls._FENCE_RE.search(text)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(text[first : last + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)

    def _extract_model_data_with_ai(
        self,
        problem_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        AI fallback called when structured extraction yields no coefficients
        or no constraints.

        Sends the full ``problem_data`` JSON to the AI and asks for a
        compact structure containing ``objective_coefficients`` and a
        constraint list with per-constraint coefficient arrays + sense + rhs.

        Returns the parsed dict, or ``None`` if the call or parsing fails.
        """
        if not self.api_client:
            return None

        problem_json = json.dumps(problem_data, indent=2, default=str)
        prompt = (
            "Given this problem data JSON:\n\n"
            f"```json\n{problem_json}\n```\n\n"
            "Extract and return ONLY this JSON structure, no other text:\n"
            "{\n"
            '  "objective_coefficients": [list of numbers, one per '
            "decision variable, in order],\n"
            '  "constraints": [\n'
            "    {\n"
            '      "name": "constraint name",\n'
            '      "coefficients": [list of numbers matching variable order],\n'
            '      "sense": "<= or >= or =",\n'
            '      "rhs": number\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        try:
            response = self.api_client.create_message(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
            )
            parsed = self._parse_json(response["content"])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    #  Constraint extraction & parsing
    # ------------------------------------------------------------------

    _CONSTRAINT_LIST_KEYS = (
        "constraints",
        "constraint_list",
        "conditions",
        "restrictions",
        "limits",
    )
    _EXPR_KEYS = ("expression", "formula", "lhs", "constraint", "equation")
    _RHS_KEYS = ("rhs", "right_hand_side", "value", "bound", "limit")
    _SENSE_KEYS = ("sense", "type", "operator", "direction")

    _SENSE_MAP_LE = {"leq", "le", "<=", "max", "upper"}
    _SENSE_MAP_GE = {"geq", "ge", ">=", "min", "lower"}
    _SENSE_MAP_EQ = {"eq", "=", "=="}

    def _extract_constraints_list(
        self,
        problem_data: Dict[str, Any],
    ) -> List[Any]:
        """
        Find the constraint list from ``problem_data``, trying several keys
        in both ``parameters`` and the top level.
        """
        params = problem_data.get("parameters", {})
        for d in (params, problem_data):
            for key in self._CONSTRAINT_LIST_KEYS:
                val = d.get(key)
                if isinstance(val, list) and val:
                    return val
        return []

    def _resolve_sense(self, raw: str) -> int:
        """Map a raw sense string to a PuLP comparator constant."""
        token = raw.strip().lower()
        if token in self._SENSE_MAP_LE:
            return pulp.constants.LpConstraintLE
        if token in self._SENSE_MAP_GE:
            return pulp.constants.LpConstraintGE
        if token in self._SENSE_MAP_EQ:
            return pulp.constants.LpConstraintEQ
        return pulp.constants.LpConstraintLE

    @staticmethod
    def _get_first(d: Dict[str, Any], keys: tuple, default=None):
        """Return the first truthy value found under *keys* in *d*."""
        for k in keys:
            val = d.get(k)
            if val is not None and val != "":
                return val
        return default

    def _add_parsed_constraint(
        self,
        prob: pulp.LpProblem,
        constraint: Any,
        lp_vars: Dict[str, pulp.LpVariable],
        index: int = 0,
    ) -> None:
        """
        Parse one constraint entry and add it to *prob*.

        The constraint may be:
          - A dict with various field names for expression / sense / rhs.
          - A plain string like ``"3*x + 2*y <= 240"``.

        On any parse failure the constraint is **skipped** with a warning
        rather than crashing the entire model build.
        """
        try:
            self._do_add_constraint(prob, constraint, lp_vars, index)
        except Exception:
            label = (
                constraint
                if isinstance(constraint, str)
                else (
                    self._get_first(constraint, self._EXPR_KEYS, constraint)
                    if isinstance(constraint, dict)
                    else str(constraint)
                )
            )
            print(f"WARNING: Skipped constraint {index}: " f"could not parse '{label}'")

    def _do_add_constraint(
        self,
        prob: pulp.LpProblem,
        constraint: Any,
        lp_vars: Dict[str, pulp.LpVariable],
        index: int,
    ) -> None:
        """Inner implementation of constraint parsing (may raise)."""
        if isinstance(constraint, str):
            constraint = {"expression": constraint}

        if not isinstance(constraint, dict):
            raise ValueError(f"Unexpected constraint type: {type(constraint)}")

        name = constraint.get("name", f"c_{index}")
        expr_str = str(self._get_first(constraint, self._EXPR_KEYS, "") or "")
        explicit_rhs = self._get_first(constraint, self._RHS_KEYS)
        raw_sense = str(self._get_first(constraint, self._SENSE_KEYS, "<=") or "<=")

        if not expr_str:
            raise ValueError("Empty expression")

        # --- Detect sense operator embedded in the expression string ------
        embedded_sense = None
        for op in ("<=", ">=", "==", "="):
            if op in expr_str:
                parts = expr_str.split(op, 1)
                lhs_str = parts[0].strip()
                rhs_str = parts[1].strip()
                embedded_sense = op
                break
        else:
            lhs_str = expr_str.strip()
            rhs_str = None

        # Determine the actual sense (prefer embedded, fall back to field)
        if embedded_sense:
            sense_const = self._resolve_sense(embedded_sense)
        else:
            sense_const = self._resolve_sense(raw_sense)

        # Determine RHS value
        if explicit_rhs is not None:
            rhs = float(explicit_rhs)
        elif rhs_str is not None:
            rhs = float(rhs_str)
        else:
            rhs = 0.0

        if not lp_vars:
            return

        # --- Parse LHS for coefficient * variable terms ------------------
        var_pattern = "|".join(
            re.escape(v) for v in sorted(lp_vars, key=len, reverse=True)
        )
        term_re = re.compile(r"([+-]?\s*\d*\.?\d*)\s*\*?\s*(" + var_pattern + r")")

        lhs_expr = pulp.lpSum(0)
        for m in term_re.finditer(lhs_str):
            coef_str = m.group(1).replace(" ", "")
            var_name = m.group(2)
            if coef_str in ("", "+"):
                coef = 1.0
            elif coef_str == "-":
                coef = -1.0
            else:
                coef = float(coef_str)
            lhs_expr += coef * lp_vars[var_name]

        # --- Add to model ------------------------------------------------
        if sense_const == pulp.constants.LpConstraintLE:
            prob += (lhs_expr <= rhs, name)
        elif sense_const == pulp.constants.LpConstraintGE:
            prob += (lhs_expr >= rhs, name)
        else:
            prob += (lhs_expr == rhs, name)

    # ------------------------------------------------------------------
    #  LP / Knapsack / Assignment extraction helpers
    # ------------------------------------------------------------------

    def _extract_lp_data(
        self,
        problem_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Multi-layer extraction for LP/IP/MIP objective coefficients,
        constraint coefficient matrix, and constraint RHS values.

        Returns a dict with keys ``obj_coeffs``, ``constraint_coefficients``,
        and ``constraint_rhs`` (any may be ``None``).
        """
        params = problem_data.get("parameters", {})
        search_dicts = [params, problem_data]
        var_defs = problem_data.get("decision_variables", [])

        # --- Objective coefficients --------------------------------------
        _OBJ_KEYS = (
            "objective_coefficients",
            "coefficients",
            "obj_coefficients",
            "profits",
            "costs",
            "objective_values",
            "unit_profits",
            "unit_costs",
            "cost_per_unit",
            "profit_per_unit",
            "prices",
        )
        obj_coeffs = self._first_numeric_list(search_dicts, _OBJ_KEYS)

        if not obj_coeffs and var_defs:
            _VAR_OBJ_FIELDS = (
                "coefficient",
                "profit",
                "cost",
                "value",
                "objective_coefficient",
                "unit_profit",
                "unit_cost",
            )
            extracted = []
            for v in var_defs:
                found = None
                for f in _VAR_OBJ_FIELDS:
                    if f in v and isinstance(v[f], (int, float)):
                        found = float(v[f])
                        break
                if found is None:
                    break
                extracted.append(found)
            else:
                if extracted:
                    obj_coeffs = extracted

        # --- Constraint RHS ----------------------------------------------
        _RHS_KEYS = (
            "constraint_rhs",
            "rhs",
            "right_hand_side",
            "bounds",
            "rhs_values",
            "limits",
            "capacities",
        )
        constraint_rhs = self._first_numeric_list(search_dicts, _RHS_KEYS)

        # --- Constraint coefficient matrix --------------------------------
        _MATRIX_KEYS = (
            "constraint_coefficients",
            "constraint_matrix",
            "lhs_coefficients",
            "A_matrix",
        )
        constraint_coefficients = None
        for d in search_dicts:
            for k in _MATRIX_KEYS:
                val = d.get(k)
                if isinstance(val, list) and val and isinstance(val[0], list):
                    constraint_coefficients = val
                    break
            if constraint_coefficients:
                break

        return {
            "obj_coeffs": obj_coeffs,
            "constraint_rhs": constraint_rhs,
            "constraint_coefficients": constraint_coefficients,
        }

    def _extract_knapsack_data(
        self,
        problem_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Multi-layer extraction for knapsack problems.

        Returns ``{'capacity', 'weights', 'values', 'item_names'}``.
        Raises ``ValueError`` if capacity or weights cannot be found.
        """
        params = problem_data.get("parameters", {})
        search_dicts = [params, problem_data]
        var_defs = problem_data.get("decision_variables", [])

        # --- Capacity ----------------------------------------------------
        _CAP_KEYS = (
            "capacity",
            "max_weight",
            "weight_limit",
            "knapsack_capacity",
            "bag_capacity",
            "limit",
            "total_capacity",
            "budget",
            "investment_budget",  # FIX: classifier uses 'budget' for investment problems
        )
        capacity = None
        for d in search_dicts:
            for k in _CAP_KEYS:
                val = d.get(k)
                if isinstance(val, (int, float)):
                    capacity = float(val)
                    break
            if capacity is not None:
                break

        # --- Weights -----------------------------------------------------
        _WEIGHT_KEYS = (
            "weights",
            "weight",
            "item_weights",
            "sizes",
            "volumes",
            "costs",
            "cost",
            "investment_costs",
            "prices",  # FIX: investment problems use 'costs' as weights
        )
        weights = self._first_numeric_list(search_dicts, _WEIGHT_KEYS)

        # --- Values / profits --------------------------------------------
        _VALUE_KEYS = (
            "values",
            "value",
            "profits",
            "item_values",
            "benefits",
            "rewards",
            "returns",
            "return",
            "investment_returns",
            "gains",  # FIX: investment problems use 'returns' as values
        )
        values = self._first_numeric_list(search_dicts, _VALUE_KEYS)

        # --- Item names (optional) ----------------------------------------
        _NAME_KEYS = (
            "item_names",
            "items",
            "names",
            "project_names",
        )  # FIX: classifier uses 'project_names'
        item_names = None
        for d in search_dicts:
            for k in _NAME_KEYS:
                val = d.get(k)
                if isinstance(val, list) and val:
                    item_names = [str(x) for x in val]
                    break
            if item_names:
                break

        # --- Fallback: read from decision_variables ----------------------
        if (not weights or not values) and var_defs:
            _W_FIELDS = ("weight", "size")
            _V_FIELDS = ("value", "profit", "benefit")
            w_list, v_list, n_list = [], [], []
            for v in var_defs:
                w = next(
                    (
                        float(v[f])
                        for f in _W_FIELDS
                        if f in v and isinstance(v[f], (int, float))
                    ),
                    None,
                )
                vv = next(
                    (
                        float(v[f])
                        for f in _V_FIELDS
                        if f in v and isinstance(v[f], (int, float))
                    ),
                    None,
                )
                if w is None or vv is None:
                    break
                w_list.append(w)
                v_list.append(vv)
                n_list.append(v.get("name", f"item_{len(n_list)}"))
            else:
                if w_list:
                    if not weights:
                        weights = w_list
                    if not values:
                        values = v_list
                    if not item_names:
                        item_names = n_list

        # --- Validation --------------------------------------------------
        if capacity is None:
            raise ValueError(
                "Could not find knapsack capacity. "
                f"Keys in parameters: {list(params.keys())}\n"
                "Run debug_classifier.py to inspect the AI output."
            )
        if not weights:
            raise ValueError(
                "Could not find item weights. "
                f"Keys in parameters: {list(params.keys())}\n"
                "Run debug_classifier.py to inspect the AI output."
            )

        n = len(weights)
        if not values:
            values = [1.0] * n
        if not item_names:
            item_names = [f"item_{i}" for i in range(n)]

        return {
            "capacity": capacity,
            "weights": weights,
            "values": values,
            "item_names": item_names,
        }

    def _extract_assignment_data(
        self,
        problem_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Multi-layer extraction for assignment problems.

        Returns ``{'cost_matrix', 'agent_names', 'task_names'}``.
        Raises ``ValueError`` if the cost matrix cannot be found.
        """
        params = problem_data.get("parameters", {})
        search_dicts = [params, problem_data]

        # --- Cost / profit matrix ----------------------------------------
        _MATRIX_KEYS = (
            "cost_matrix",
            "assignment_matrix",
            "costs",
            "profit_matrix",
            "efficiency_matrix",
            "time_matrix",
            "distance_matrix",
        )
        matrix = None
        for d in search_dicts:
            for k in _MATRIX_KEYS:
                val = d.get(k)
                if isinstance(val, list) and val and isinstance(val[0], list):
                    matrix = val
                    break
            if matrix:
                break

        if not matrix:
            raise ValueError(
                "Could not find assignment cost matrix. "
                f"Keys in parameters: {list(params.keys())}\n"
                "Run debug_classifier.py to inspect the AI output."
            )

        # --- Agent names (optional) --------------------------------------
        _AGENT_KEYS = (
            "agent_names",
            "agents",
            "workers",
            "machines",
            "resources",
            "employees",
        )
        agent_names = None
        for d in search_dicts:
            for k in _AGENT_KEYS:
                val = d.get(k)
                if isinstance(val, list) and val:
                    agent_names = [str(x) for x in val]
                    break
            if agent_names:
                break

        # --- Task names (optional) ---------------------------------------
        _TASK_KEYS = (
            "task_names",
            "tasks",
            "jobs",
            "assignments",
            "activities",
            "projects",
        )
        task_names = None
        for d in search_dicts:
            for k in _TASK_KEYS:
                val = d.get(k)
                if isinstance(val, list) and val:
                    task_names = [str(x) for x in val]
                    break
            if task_names:
                break

        n = len(matrix)
        if not agent_names:
            agent_names = [f"agent_{i}" for i in range(n)]
        m = len(matrix[0]) if matrix else n
        if not task_names:
            task_names = [f"task_{j}" for j in range(m)]

        return {
            "cost_matrix": matrix,
            "agent_names": agent_names,
            "task_names": task_names,
        }

    @staticmethod
    def _first_numeric_list(
        sources: List[Dict[str, Any]],
        keys: tuple,
    ) -> Optional[List[float]]:
        """Return the first list of numbers found under any of *keys*."""
        for d in sources:
            for k in keys:
                val = d.get(k)
                if isinstance(val, list) and val:
                    try:
                        return [float(x) for x in val]
                    except (TypeError, ValueError):
                        continue
        return None

    # ------------------------------------------------------------------
    #  Transportation / Assignment / AI generators
    # ------------------------------------------------------------------

    def _extract_transportation_data(
        self,
        problem_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Three-layer extraction for transportation supply / demand / costs.

        Layer 1: Standard keys guaranteed by DataExtractor (``supply``,
                 ``demand``, ``costs``).
        Layer 2: Extended fallback covering every reasonable AI key-name
                 variation.
        Layer 3: Direct CSV structure reading via ``_raw_rows`` /
                 ``_raw_headers`` injected by DataExtractor.

        Returns ``{'supply': [...], 'demand': [...], 'costs': [[...]]}``.
        Raises ``ValueError`` only when all three layers fail.
        """
        params = problem_data.get("parameters", {})
        search_dicts = [params, problem_data]

        supply = None
        demand = None
        raw_costs = None

        # === Layer 1 — standard keys from DataExtractor =================
        _L1_SUPPLY = ("supply",)
        _L1_DEMAND = ("demand",)
        _L1_COSTS = ("costs",)

        supply = self._first_list(search_dicts, _L1_SUPPLY)
        demand = self._first_list(search_dicts, _L1_DEMAND)
        raw_costs = self._first_value(search_dicts, _L1_COSTS)

        # Convert empty lists to None
        if supply == []:
            supply = None
        if demand == []:
            demand = None

        if supply and demand and raw_costs:
            return {"supply": supply, "demand": demand, "costs": raw_costs}

        # === Layer 2 — extended key-name fallback ========================
        _L2_SUPPLY = (
            "supply_capacities",
            "warehouse_supply",
            "capacities",
            "sources",
            "warehouse_capacities",
            "available",
            "supply_values",
            "row_supply",
            "warehouse_capacity",
            "source_supply",
            "supplies",
            "stock",
            "inventory",
            "production_capacity",
        )
        _L2_DEMAND = (
            "store_demands",
            "destination_demand",
            "demands",
            "requirements",
            "needed",
            "stores",
            "demand_values",
            "destination_demands",
            "customer_demand",
            "consumption",
            "orders",
            "requirements_list",
        )
        _L2_COSTS = (
            "cost_matrix",
            "shipping_costs",
            "transportation_costs",
            "cost_data",
            "unit_costs",
            "per_unit_costs",
            "route_costs",
            "distance_matrix",
            "shipping_matrix",
            "cost_table",
            "rates",
            "tariffs",
            "freight_costs",
        )

        if not supply:
            supply = self._first_list(search_dicts, _L2_SUPPLY)
            if supply == []:
                supply = None
        if not demand:
            demand = self._first_list(search_dicts, _L2_DEMAND)
            if demand == []:
                demand = None
        if not raw_costs:
            raw_costs = self._first_value(search_dicts, _L2_COSTS)

        if supply and demand and raw_costs:
            return {"supply": supply, "demand": demand, "costs": raw_costs}

        # === Layer 3 — direct CSV structure reading ======================
        raw_headers = params.get("_raw_headers")
        raw_rows = params.get("_raw_rows")
        if raw_headers and raw_rows:
            try:
                from src.ingestion.data_extractor import DataExtractor

                auto = DataExtractor._auto_extract_transportation(
                    None,
                    raw_headers,
                    raw_rows,
                )
                if not supply and auto.get("supply"):
                    supply = auto["supply"]
                if not demand and auto.get("demand"):
                    demand = auto["demand"]
                if not raw_costs and auto.get("costs"):
                    raw_costs = auto["costs"]
            except Exception:
                pass

        # === Final check =================================================
        missing = []
        if not supply:
            missing.append("supply")
        if not demand:
            missing.append("demand")
        if missing:
            all_keys = list(params.keys())
            raise ValueError(
                f"Could not find {' or '.join(missing)} data after trying "
                f"3 extraction layers.\n"
                f"Keys found in parameters: {all_keys}\n"
                f"Run debug_classifier.py to inspect the AI output."
            )

        # Validate the extracted data
        if supply:
            try:
                supply = [float(x) for x in supply]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid supply values (must be numeric). "
                    f"Got: {supply} (type: {type(supply).__name__})"
                ) from e

        if demand:
            try:
                demand = [float(x) for x in demand]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid demand values (must be numeric). "
                    f"Got: {demand} (type: {type(demand).__name__})"
                ) from e

        if raw_costs and isinstance(raw_costs, list):
            try:
                if isinstance(raw_costs[0], list):
                    raw_costs = [[float(x) for x in row] for row in raw_costs]
                else:
                    raw_costs = [float(x) for x in raw_costs]
            except (TypeError, ValueError, IndexError):
                # Costs might be None or empty, which is OK
                pass

        return {"supply": supply, "demand": demand, "costs": raw_costs}

    def _generate_transportation_model(
        self, problem_data: Dict[str, Any]
    ) -> pulp.LpProblem:
        """
        Generate a transportation problem model.

        Uses ``_extract_transportation_data()`` (3-layer fallback) to
        locate supply, demand, and cost data regardless of key names.
        """
        if os.getenv("OR_DEBUG", "0") == "1":
            print("\n[DEBUG] _generate_transportation_model() — problem_data received:")
            print(json.dumps(problem_data, indent=2, default=str))
            print(
                "[DEBUG] parameters keys:",
                list(problem_data.get("parameters", {}).keys()),
            )
            print(
                "[DEBUG] decision_variables:",
                [v.get("name") for v in problem_data.get("decision_variables", [])],
            )

        extracted = self._extract_transportation_data(problem_data)
        supply = extracted["supply"]
        demand = extracted["demand"]
        raw_costs = extracted["costs"]
        notes: List[str] = []

        # Validate that we have valid data
        if not supply or not isinstance(supply, list):
            raise ValueError(f"Invalid or empty supply data: {supply}")
        if not demand or not isinstance(demand, list):
            raise ValueError(f"Invalid or empty demand data: {demand}")

        num_sources = len(supply)
        num_destinations = len(demand)

        if raw_costs is None:
            costs = [[1.0] * num_destinations for _ in range(num_sources)]
            notes.append(
                "No cost data found — using uniform cost of 1.0 for every route."
            )
        else:
            costs = self._ensure_2d_costs(raw_costs, num_sources, num_destinations)

        # --- Balance check -----------------------------------------------
        total_supply = sum(supply)
        total_demand = sum(demand)

        if total_supply > total_demand:
            slack = total_supply - total_demand
            demand = list(demand) + [slack]
            costs = [row + [0] for row in costs]
            num_destinations += 1
            notes.append(
                f"Supply ({total_supply}) > Demand ({total_demand}): "
                f"added dummy destination (sink) with demand={slack} and zero cost."
            )
        elif total_demand > total_supply:
            slack = total_demand - total_supply
            supply = list(supply) + [slack]
            costs = costs + [[0] * num_destinations]
            num_sources += 1
            notes.append(
                f"Demand ({total_demand}) > Supply ({total_supply}): "
                f"added dummy source with supply={slack} and zero cost."
            )

        # --- Build the PuLP model ----------------------------------------
        prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            "ship",
            ((i, j) for i in range(num_sources) for j in range(num_destinations)),
            lowBound=0,
            cat="Continuous",
        )

        prob += (
            pulp.lpSum(
                costs[i][j] * x[i, j]
                for i in range(num_sources)
                for j in range(num_destinations)
            ),
            "Total_Cost",
        )

        for i in range(num_sources):
            prob += (
                pulp.lpSum(x[i, j] for j in range(num_destinations)) == supply[i],
                f"Supply_{i}",
            )

        for j in range(num_destinations):
            prob += (
                pulp.lpSum(x[i, j] for i in range(num_sources)) == demand[j],
                f"Demand_{j}",
            )

        prob._or_notes = notes
        return prob

    # --- tiny helpers used by the transportation builder -----------------

    @staticmethod
    def _first_list(
        sources: List[Dict[str, Any]],
        keys: tuple,
    ) -> List:
        """
        Search *sources* (a list of dicts) for the first non-empty list
        found under any of *keys*.
        """
        for d in sources:
            for k in keys:
                val = d.get(k)
                if isinstance(val, list) and val:
                    return val
        return []

    @staticmethod
    def _first_value(
        sources: List[Dict[str, Any]],
        keys: tuple,
    ) -> Any:
        """
        Search *sources* (a list of dicts) for the first truthy value
        found under any of *keys*.  Returns ``None`` if nothing is found.
        """
        for d in sources:
            for k in keys:
                val = d.get(k)
                if val:
                    return val
        return None

    @staticmethod
    def _ensure_2d_costs(
        raw: Any,
        num_sources: int,
        num_destinations: int,
    ) -> List[List[float]]:
        """
        Accept a 2-D list or a flat list and return a 2-D cost matrix.

        Raises ValueError when dimensions do not match.
        """
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return raw

        flat = list(raw)
        expected = num_sources * num_destinations
        if len(flat) != expected:
            raise ValueError(
                f"Flat cost list has {len(flat)} elements, but "
                f"{num_sources} sources x {num_destinations} destinations = "
                f"{expected} expected."
            )
        return [
            flat[i * num_destinations : (i + 1) * num_destinations]
            for i in range(num_sources)
        ]

    def _generate_assignment_model(
        self, problem_data: Dict[str, Any]
    ) -> pulp.LpProblem:
        """Generate an assignment problem model."""

        extracted = self._extract_assignment_data(problem_data)
        costs = extracted["cost_matrix"]

        n = len(costs)
        m = len(costs[0]) if costs else n

        prob = pulp.LpProblem("Assignment_Problem", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            "assign",
            ((i, j) for i in range(n) for j in range(m)),
            cat="Binary",
        )

        prob += (
            pulp.lpSum(costs[i][j] * x[i, j] for i in range(n) for j in range(m)),
            "Total_Cost",
        )

        for i in range(n):
            prob += (
                pulp.lpSum(x[i, j] for j in range(m)) == 1,
                f"Agent_{i}",
            )

        for j in range(m):
            prob += (
                pulp.lpSum(x[i, j] for i in range(n)) == 1,
                f"Task_{j}",
            )

        return prob

    def _generate_knapsack_model(self, problem_data: Dict[str, Any]) -> pulp.LpProblem:
        """Generate a 0-1 knapsack problem model."""

        extracted = self._extract_knapsack_data(problem_data)
        capacity = extracted["capacity"]
        weights = extracted["weights"]
        values = extracted["values"]
        item_names = extracted["item_names"]

        n = len(weights)

        sense = (
            pulp.LpMaximize
            if problem_data.get("objective", "maximize") == "maximize"
            else pulp.LpMinimize
        )
        prob = pulp.LpProblem("Knapsack_Problem", sense)

        x = [pulp.LpVariable(item_names[i], cat="Binary") for i in range(n)]

        prob += pulp.lpSum(values[i] * x[i] for i in range(n)), "Total_Value"

        prob += (
            pulp.lpSum(weights[i] * x[i] for i in range(n)) <= capacity,
            "Weight_Limit",
        )

        return prob

    _CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)

    def _generate_with_ai(self, problem_data: Dict[str, Any]) -> pulp.LpProblem:
        """
        Use AI to generate a complete PuLP model from the classified problem data.

        The AI is asked to produce a self-contained script that creates a
        ``prob`` variable of type ``pulp.LpProblem``.  The script is executed
        in a namespace that already contains ``pulp``, and ``prob`` is
        extracted and returned.
        """
        if not self.api_client:
            raise ValueError("AI API not configured for dynamic model generation")

        # 1 — build prompt with the full problem_data JSON
        problem_json = json.dumps(problem_data, indent=2, default=str)
        prompt = (
            "You are an expert in Operations Research and Python PuLP.\n\n"
            "Write COMPLETE, executable Python code that builds a PuLP model "
            "for the problem described in the JSON below.\n\n"
            f"```json\n{problem_json}\n```\n\n"
            "Rules:\n"
            "- `import pulp` at the top.\n"
            "- Create a variable called `prob` (type `pulp.LpProblem`).\n"
            "- Add ALL decision variables, the objective function, and "
            "ALL constraints to `prob`.\n"
            "- End the code with the comment `# END MODEL`.\n"
            "- Do NOT wrap the code in a function — write it at module level.\n"
            "- Do NOT call `prob.solve()`.\n"
            "- Return ONLY the Python code inside a single ```python ``` block."
        )

        response = self.api_client.create_message(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.3,
            effort="medium",
        )
        raw = response["content"]

        # 2 — extract code from ```python ... ``` fences (fall back to raw)
        fence = self._CODE_FENCE_RE.search(raw)
        code = fence.group(1).strip() if fence else raw.strip()

        # 3-6 — execute, verify, return (or raise with diagnostics)
        try:
            namespace: Dict[str, Any] = {"pulp": pulp}
            exec(code, namespace)

            prob = namespace.get("prob")
            # LLMs often wrap the model in a function despite the prompt.
            if prob is None and callable(namespace.get("create_model")):
                prob = namespace["create_model"]()
            if not isinstance(prob, pulp.LpProblem):
                raise TypeError(
                    f"Expected namespace['prob'] to be pulp.LpProblem, "
                    f"got {type(prob).__name__}"
                )
            return prob

        except Exception as exc:
            print("=" * 60)
            print("AI-GENERATED CODE (failed):")
            print("=" * 60)
            print(code)
            print("=" * 60)
            print(f"Exception: {exc}")
            print("=" * 60)
            raise RuntimeError(
                f"AI model generation failed: {exc}\n\n"
                f"--- generated code ---\n{code}"
            ) from exc

    # ------------------------------------------------------------------
    #  CVXPY model builder (quadratic / convex non-linear)
    # ------------------------------------------------------------------

    _RETURN_KEYS = (
        "expected_returns",
        "returns",
        "mean_returns",
        "expected_return",
        "avg_returns",
    )
    _COV_KEYS = ("covariance_matrix", "covariance", "cov_matrix", "cov", "risk_matrix")

    def _generate_cvxpy_model(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a CVXPY model dict for problem types that benefit from
        CVXPY's native quadratic / convex support.

        Handles:
          - ``portfolio_optimization`` (quadratic risk minimisation or
            linear return maximisation with a sum-to-1 constraint).
          - ``resource_allocation`` with a ``quadratic_costs`` key
            (least-squares objective).
          - Everything else: builds a PuLP model first, then wraps it in
            the dict format that ``SolverInterface._solve_with_cvxpy``
            can auto-convert.

        Returns:
            ``{'type': 'cvxpy', 'problem': cp.Problem,
              'variables': {name: cp.Variable}, 'constraints': [...]}``
        """
        try:
            import cvxpy as cp
            import numpy as np
        except ImportError as exc:
            raise ValueError(
                "CVXPY solver requested but cvxpy/numpy is not installed. "
                "Run: pip install cvxpy numpy"
            ) from exc

        ptype = problem_data.get("problem_type", "")
        params = problem_data.get("parameters", {})

        # ------- portfolio_optimization -----------------------------------
        if ptype == "portfolio_optimization":
            returns = None
            for k in self._RETURN_KEYS:
                val = params.get(k)
                if isinstance(val, list) and val:
                    returns = np.array(val, dtype=float)
                    break

            cov = None
            for k in self._COV_KEYS:
                val = params.get(k)
                if isinstance(val, list) and val:
                    cov = np.array(val, dtype=float)
                    break

            n = len(returns) if returns is not None else 0
            if n == 0:
                raise ValueError(
                    "portfolio_optimization requires 'expected_returns' "
                    "in parameters."
                )

            weights = cp.Variable(n, nonneg=True, name="weights")
            cp_vars = {f"w_{i}": weights[i] for i in range(n)}
            cp_vars["weights"] = weights

            constraints = [cp.sum(weights) == 1]

            budget_min = params.get("min_weight")
            budget_max = params.get("max_weight")
            if budget_min is not None:
                constraints.append(weights >= float(budget_min))
            if budget_max is not None:
                constraints.append(weights <= float(budget_max))

            target_return = params.get("target_return")
            objective = problem_data.get("objective", "minimize")

            if objective == "minimize" and cov is not None:
                cp_obj = cp.Minimize(cp.quad_form(weights, cov))
                if target_return is not None:
                    constraints.append(returns @ weights >= float(target_return))
            elif objective == "maximize" and returns is not None:
                cp_obj = cp.Maximize(returns @ weights)
                risk_limit = params.get("risk_limit", params.get("max_risk"))
                if risk_limit is not None and cov is not None:
                    constraints.append(cp.quad_form(weights, cov) <= float(risk_limit))
            elif cov is not None:
                cp_obj = cp.Minimize(cp.quad_form(weights, cov))
            else:
                cp_obj = cp.Maximize(returns @ weights)

            prob = cp.Problem(cp_obj, constraints)
            return {
                "type": "cvxpy",
                "problem": prob,
                "variables": cp_vars,
                "constraints": constraints,
            }

        # ------- resource_allocation with quadratic costs -----------------
        if ptype == "resource_allocation" and "quadratic_costs" in params:
            A = np.array(params["quadratic_costs"], dtype=float)
            b = np.array(params.get("targets", params.get("demand", [])), dtype=float)
            n_vars = A.shape[1] if A.ndim == 2 else len(A)

            x = cp.Variable(n_vars, nonneg=True, name="x")
            cp_vars = {f"x_{i}": x[i] for i in range(n_vars)}
            cp_vars["x"] = x

            constraints = []

            budget = params.get("budget", params.get("capacity"))
            if budget is not None:
                constraints.append(cp.sum(x) <= float(budget))

            lb = params.get("lower_bounds")
            ub = params.get("upper_bounds")
            if lb is not None:
                constraints.append(x >= np.array(lb, dtype=float))
            if ub is not None:
                constraints.append(x <= np.array(ub, dtype=float))

            cp_obj = cp.Minimize(cp.sum_squares(A @ x - b))
            prob = cp.Problem(cp_obj, constraints)
            return {
                "type": "cvxpy",
                "problem": prob,
                "variables": cp_vars,
                "constraints": constraints,
            }

        # ------- fallback: build a PuLP model, wrap for CVXPY solver ------
        pulp_model = self.generate(problem_data, solver_preference="pulp")
        return {
            "type": "cvxpy",
            "problem": pulp_model,
            "variables": {},
            "constraints": [],
        }

    def validate_model(self, model: pulp.LpProblem) -> Dict[str, Any]:
        """
        Validate a model for common issues.

        Returns:
            Dictionary with validation results
        """

        issues = []
        warnings = []

        # Check for variables
        if not model.variables():
            issues.append("No decision variables defined")

        # Check for objective
        if model.objective is None:
            issues.append("No objective function defined")

        # Check for constraints
        if not model.constraints:
            warnings.append("No constraints defined - problem may be unbounded")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "num_variables": len(model.variables()),
            "num_constraints": len(model.constraints),
        }

    # ------------------------------------------------------------------
    #  MPS → Model (CVXPY primary, PuLP fallback — no AI needed)
    # ------------------------------------------------------------------

    def generate_from_mps(
        self,
        parsed_mps: Dict[str, Any],
        solver_preference: str = "auto",
    ):
        """
        Build a model directly from the structured output of
        ``FileParser._parse_mps()``.

        Builds a PuLP model from the parsed dict; ``SolverInterface``
        converts it to CVXPY when a CVXPY solver is selected.

        Args:
            parsed_mps: Dict produced by ``FileParser.parse()`` with
                ``type == 'mps'``.
            solver_preference: kept for API compatibility; the model is
                always built with PuLP.

        Returns:
            ``(model, problem_data)`` — *model* is a ``pulp.LpProblem``;
            *problem_data* is a classifier-
            compatible dict for the rest of the pipeline.
        """
        file_path = parsed_mps.get("file_path")

        # --- Detect problem type from variable bounds -------------------
        var_types = [
            b.get("type", "continuous")
            for b in parsed_mps.get("variable_bounds", {}).values()
        ]
        n_binary = sum(1 for t in var_types if t == "binary")
        n_integer = sum(1 for t in var_types if t == "integer")
        n_continuous = len(var_types) - n_binary - n_integer

        if n_binary:
            detected_type = "mixed_integer_programming"
        elif n_integer:
            detected_type = "integer_programming"
        else:
            detected_type = "linear_programming"

        # CVXPY has no MPS reader; build a PuLP model, which SolverInterface
        # converts to CVXPY when a CVXPY solver is selected.
        model = self._build_mps_with_pulp(parsed_mps)

        # --- Build problem_data compatible with ResultInterpreter -------
        _obj_sense = parsed_mps.get("objective_sense", "minimize")
        problem_data: Dict[str, Any] = {
            "problem_type": detected_type,
            "objective": _obj_sense,
            "objective_description": (
                f"{_obj_sense.capitalize()} {parsed_mps.get('objective_name', '?')} "
                f"(MPS benchmark)"
            ),
            "source": "miplib",
            "instance_name": parsed_mps.get("name", ""),
            "mps_file_path": file_path,
            "confidence": 1.0,
            "assumptions": [],
            "warnings": [
                "Loaded from MPS file.",
                f"Detected type: {detected_type} ({n_binary} binary, "
                f"{n_integer} integer, {n_continuous} continuous variables).",
            ],
            "parameters": {},
            "decision_variables": [
                {
                    "name": v,
                    "type": parsed_mps.get("variable_bounds", {})
                    .get(
                        v,
                        {},
                    )
                    .get("type", "continuous"),
                    "description": "",
                }
                for v in parsed_mps.get("variables", [])[:20]
            ],
            "constraints": [],
            "notes": "",
        }

        return model, problem_data

    @staticmethod
    def _build_mps_with_pulp(parsed_mps: Dict[str, Any]) -> pulp.LpProblem:
        """Build a ``pulp.LpProblem`` from the structured MPS dict."""
        name = parsed_mps.get("name", "mps_problem") or "mps_problem"
        _sense = (
            pulp.LpMaximize
            if parsed_mps.get("objective_sense") == "maximize"
            else pulp.LpMinimize
        )
        prob = pulp.LpProblem(name, _sense)

        var_names = parsed_mps.get("variables", [])
        vbounds = parsed_mps.get("variable_bounds", {})
        obj_coefs = parsed_mps.get("objective_coefficients", {})

        _CAT_MAP = {
            "continuous": pulp.LpContinuous,
            "integer": pulp.LpInteger,
            "binary": pulp.LpBinary,
        }

        lp_vars: Dict[str, pulp.LpVariable] = {}
        for vname in var_names:
            info = vbounds.get(vname, {})
            cat = _CAT_MAP.get(info.get("type", "continuous"), pulp.LpContinuous)
            lb = info.get("lb", 0.0)
            ub = info.get("ub", None)
            if cat == pulp.LpBinary:
                lb, ub = 0, 1
            lp_vars[vname] = pulp.LpVariable(
                vname,
                lowBound=lb,
                upBound=ub,
                cat=cat,
            )

        prob += (
            pulp.lpSum(
                obj_coefs.get(v, 0.0) * lp_vars[v]
                for v in var_names
                if v in lp_vars and obj_coefs.get(v, 0.0) != 0.0
            ),
            "objective",
        )

        _SENSE_MAP = {
            "<=": pulp.LpConstraintLE,
            ">=": pulp.LpConstraintGE,
            "=": pulp.LpConstraintEQ,
        }
        for con in parsed_mps.get("constraints", []):
            cname = con.get("name", "")
            coeffs = con.get("coefficients", {})
            rhs_val = con.get("rhs", 0.0)
            sense_code = _SENSE_MAP.get(
                con.get("type", "<="),
                pulp.LpConstraintLE,
            )
            lhs = pulp.lpSum(
                coef * lp_vars[v] for v, coef in coeffs.items() if v in lp_vars
            )
            prob += pulp.LpConstraint(
                lhs,
                sense=sense_code,
                rhs=rhs_val,
                name=cname,
            )

        return prob


if __name__ == "__main__":
    # Example usage
    generator = ModelGenerator()

    # Test with a simple problem
    test_data = {
        "problem_type": "integer_programming",
        "objective": "maximize",
        "objective_description": "profit from producing widgets",
        "decision_variables": ["number of widgets to produce"],
        "constraints": [
            "Cannot exceed 100 hours of labor",
            "Cannot exceed 120 kg of material",
        ],
        "parameters": {
            "profit_per_widget": 50,
            "labor_hours_per_widget": 2,
            "material_kg_per_widget": 3,
            "available_labor_hours": 100,
            "available_material_kg": 120,
        },
        "confidence": 0.95,
        "notes": "Widget production must be integer values",
    }

    print("Generating model from problem data...")
    model = generator.generate(test_data)

    validation = generator.validate_model(model)
    print("\nModel validation:", validation)

    if validation["valid"]:
        print("\n✓ Model successfully created!")
        print(f"  Variables: {validation['num_variables']}")
        print(f"  Constraints: {validation['num_constraints']}")

        # Try to solve it
        print("\nAttempting to solve...")
        model.solve()
        print(f"Status: {pulp.LpStatus[model.status]}")

        if model.status == 1:  # Optimal
            print(f"Objective value: {pulp.value(model.objective)}")
            print("\nVariable values:")
            for v in model.variables():
                print(f"  {v.name} = {v.varValue}")
