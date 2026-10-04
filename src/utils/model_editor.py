"""
Model Editor — PuLP ↔ LaTeX conversion, sympy parsing, model rebuilding, and diffing.

Provides the Python-side logic for the Model Playground tab:
  - Convert PuLP models to LaTeX-based editor state dicts
  - Parse user-edited LaTeX back into PuLP models
  - Diff two editor states to show what changed
  - Generate blank OR model templates
"""

import re
from typing import Dict, List, Tuple

import pulp
import sympy
from sympy import Symbol, latex as sympy_latex

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _pulp_var_to_sympy_name(pulp_name: str) -> str:
    """Convert a PuLP variable name to a LaTeX-safe sympy symbol name."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", pulp_name)
    if safe and safe[0].isdigit():
        safe = "x_" + safe
    return f"x_{{{safe}}}"


def _pulp_cat_label(cat: str) -> str:
    """Normalise PuLP category string."""
    cat_map = {"Continuous": "Continuous", "Integer": "Integer", "Binary": "Binary"}
    return cat_map.get(cat, "Continuous")


# ---------------------------------------------------------------------------
#  1.1  PuLP model → editor state dict
# ---------------------------------------------------------------------------


def pulp_to_editor_state(model: pulp.LpProblem) -> dict:
    """
    Convert a PuLP model into a JSON-serialisable dict that the Playground
    stores in ``session_state`` and pre-populates the MathLive editors with.
    """
    sense = "maximize" if model.sense == pulp.constants.LpMaximize else "minimize"

    variables = []
    sym_map: Dict[str, Symbol] = {}

    for v in model.variables():
        sname = _pulp_var_to_sympy_name(v.name)
        sym = Symbol(sname)
        sym_map[v.name] = sym
        variables.append(
            {
                "name": v.name,
                "sympy_name": sname,
                "lb": v.lowBound,
                "ub": v.upBound,
                "cat": _pulp_cat_label(v.cat),
            }
        )

    # Objective
    obj_expr = sympy.Integer(0)
    if model.objective is not None:
        for v, coeff in model.objective.items():
            if v.name in sym_map:
                obj_expr += coeff * sym_map[v.name]
    objective_latex = sympy_latex(obj_expr)

    # Constraints
    constraints = []
    for cname, cobj in model.constraints.items():
        lhs_expr = sympy.Integer(0)
        for v, coeff in cobj.items():
            if v.name in sym_map:
                lhs_expr += coeff * sym_map[v.name]

        rhs_val = -cobj.constant
        sense_sym = cobj.sense

        if sense_sym == pulp.constants.LpConstraintLE:
            con_sense = "<="
            latex_op = r"\leq"
        elif sense_sym == pulp.constants.LpConstraintGE:
            con_sense = ">="
            latex_op = r"\geq"
        else:
            con_sense = "="
            latex_op = "="

        con_latex = (
            f"{sympy_latex(lhs_expr)} {latex_op} {sympy_latex(sympy.Float(rhs_val))}"
        )

        constraints.append(
            {
                "name": cname,
                "latex": con_latex,
                "sense": con_sense,
                "rhs": float(rhs_val),
            }
        )

    return {
        "name": model.name,
        "sense": sense,
        "objective_latex": objective_latex,
        "constraints": constraints,
        "variables": variables,
    }


# ---------------------------------------------------------------------------
#  1.2  LaTeX expression → coefficient dict
# ---------------------------------------------------------------------------


def latex_expr_to_coefficients(
    latex_str: str,
    variable_sympy_names: List[str],
) -> Dict[str, float]:
    """
    Parse a LaTeX expression and extract the coefficient for each known variable.

    Falls back to regex-based extraction if ``sympy.parsing.latex`` fails.
    """
    coeffs: Dict[str, float] = {}

    # Build symbol objects
    symbols_by_name = {n: Symbol(n) for n in variable_sympy_names}

    # --- Try sympy LaTeX parser first ---
    # parse_latex mis-interprets multi-char subscripts (e.g. x_{chairs} becomes
    # c*h*a*i*r*s), so we validate that at least one expected symbol was found.
    # It also creates nested Add trees for 3+ terms, so we must flatten first.
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import Add

        expr = parse_latex(latex_str)
        parsed_names = {fs.name for fs in expr.free_symbols}
        expected_names = {sym.name for sym in symbols_by_name.values()}
        if parsed_names & expected_names:
            # Flatten nested Add(a, Add(b, Add(c, ...))) into Add(a, b, c, ...)
            def _flatten_add(e):
                if isinstance(e, Add):
                    parts = []
                    for arg in e.args:
                        parts.extend(_flatten_add(arg))
                    return parts
                return [e]

            flat_expr = Add(*_flatten_add(expr))

            for sname, sym in symbols_by_name.items():
                matched = None
                for fs in flat_expr.free_symbols:
                    if fs.name == sym.name:
                        matched = fs
                        break
                if matched is not None:
                    c = flat_expr.coeff(matched)
                    try:
                        coeffs[sname] = float(c)
                    except (TypeError, ValueError):
                        coeffs[sname] = 0.0
                else:
                    coeffs[sname] = 0.0
            return coeffs
        # No expected symbols found — fall through to regex
    except Exception:
        pass

    # --- Fallback: regex-based coefficient extraction ---
    # Normalise MathLive output the regex can't read: explicit products
    # (10\cdot x, 10\times x, 10*x) and numeric fractions (\frac{1}{2}x).
    latex_str = re.sub(r"\\cdot|\\times|\*", " ", latex_str)
    latex_str = re.sub(
        r"\\[dt]?frac\s*\{\s*(\d+\.?\d*)\s*\}\s*\{\s*(\d+\.?\d*)\s*\}",
        lambda m: repr(float(m.group(1)) / float(m.group(2))),
        latex_str,
    )
    for sname in variable_sympy_names:
        escaped = re.escape(sname)
        # Match patterns like "50 x_{foo}", "50x_{foo}", "-3.5 x_{foo}"
        pattern = rf"([+-]?\s*\d*\.?\d*)\s*{escaped}"
        m = re.search(pattern, latex_str)
        if m:
            raw = m.group(1).replace(" ", "")
            if raw in ("", "+"):
                coeffs[sname] = 1.0
            elif raw == "-":
                coeffs[sname] = -1.0
            else:
                try:
                    coeffs[sname] = float(raw)
                except ValueError:
                    coeffs[sname] = 0.0
        else:
            coeffs[sname] = 0.0

    return coeffs


# ---------------------------------------------------------------------------
#  1.3  Parse constraint LaTeX → (LHS, sense, RHS)
# ---------------------------------------------------------------------------

_SENSE_TOKENS = [
    (r"\leq", "<="),
    (r"\le", "<="),
    (r"\geq", ">="),
    (r"\ge", ">="),
    ("<=", "<="),
    (">=", ">="),
    ("=", "="),
]


def parse_constraint_latex(latex_str: str) -> dict:
    """Split a constraint LaTeX string into LHS, sense, and RHS."""
    for token, sense in _SENSE_TOKENS:
        idx = latex_str.find(token)
        if idx != -1 and token != "=":
            lhs = latex_str[:idx].strip()
            rhs = latex_str[idx + len(token) :].strip()
            return {"lhs_latex": lhs, "sense": sense, "rhs_latex": rhs}

    # Check bare '=' last (avoid matching \leq as '=')
    if "=" in latex_str:
        parts = latex_str.split("=", 1)
        return {
            "lhs_latex": parts[0].strip(),
            "sense": "=",
            "rhs_latex": parts[1].strip(),
        }

    return {"lhs_latex": latex_str.strip(), "sense": "<=", "rhs_latex": "0"}


def _parse_rhs_value(rhs_latex: str) -> float:
    """Extract a numeric value from a RHS LaTeX string."""
    try:
        from sympy.parsing.latex import parse_latex

        expr = parse_latex(rhs_latex)
        return float(expr.evalf())
    except Exception:
        pass
    cleaned = re.sub(r"[^0-9.eE+-]", "", rhs_latex)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
#  1.4  Editor state → PuLP model
# ---------------------------------------------------------------------------


def editor_state_to_pulp(state: dict) -> Tuple[pulp.LpProblem, List[str]]:
    """
    Rebuild a PuLP ``LpProblem`` from an editor state dict.

    Returns ``(model, warnings)`` — warnings list any constraints that failed
    to parse and were skipped.
    """
    warnings: List[str] = []

    sense_pulp = (
        pulp.LpMaximize if state.get("sense") == "maximize" else pulp.LpMinimize
    )
    prob = pulp.LpProblem(state.get("name", "playground_model"), sense_pulp)

    # 1. Create PuLP variables
    pulp_vars: Dict[str, pulp.LpVariable] = {}
    sympy_names: List[str] = []
    sympy_to_pulp: Dict[str, str] = {}

    for vinfo in state.get("variables", []):
        pname = vinfo["name"]
        sname = vinfo.get("sympy_name", _pulp_var_to_sympy_name(pname))
        cat = vinfo.get("cat", "Continuous")
        lb = vinfo.get("lb", 0)
        ub = vinfo.get("ub")

        cat_map = {
            "Continuous": pulp.LpContinuous,
            "Integer": pulp.LpInteger,
            "Binary": pulp.LpBinary,
        }
        pulp_vars[pname] = pulp.LpVariable(
            pname,
            lowBound=lb,
            upBound=ub,
            cat=cat_map.get(cat, pulp.LpContinuous),
        )
        sympy_names.append(sname)
        sympy_to_pulp[sname] = pname

    # 2. Parse objective
    try:
        obj_coeffs = latex_expr_to_coefficients(
            state.get("objective_latex", "0"),
            sympy_names,
        )
        obj_expr = pulp.lpSum(
            obj_coeffs.get(sn, 0) * pulp_vars[sympy_to_pulp[sn]]
            for sn in sympy_names
            if sympy_to_pulp[sn] in pulp_vars
        )
        prob += obj_expr
    except Exception as e:
        warnings.append(f"Objective parse failed: {e}")

    # 3. Parse constraints
    for cidx, con in enumerate(state.get("constraints", [])):
        cname = con.get("name", f"constraint_{cidx}")
        try:
            parsed = parse_constraint_latex(con.get("latex", "0 <= 0"))
            lhs_coeffs = latex_expr_to_coefficients(parsed["lhs_latex"], sympy_names)
            rhs_val = con.get("rhs")
            if rhs_val is None:
                rhs_val = _parse_rhs_value(parsed["rhs_latex"])

            lhs_expr = pulp.lpSum(
                lhs_coeffs.get(sn, 0) * pulp_vars[sympy_to_pulp[sn]]
                for sn in sympy_names
                if sympy_to_pulp[sn] in pulp_vars
            )

            sense = parsed["sense"]
            if sense == "<=":
                prob += (lhs_expr <= rhs_val, cname)
            elif sense == ">=":
                prob += (lhs_expr >= rhs_val, cname)
            else:
                prob += (lhs_expr == rhs_val, cname)
        except Exception as e:
            warnings.append(f"Constraint '{cname}' skipped: {e}")

    return prob, warnings


# ---------------------------------------------------------------------------
#  1.5  Diff two editor states
# ---------------------------------------------------------------------------


def diff_states(original: dict, edited: dict) -> List[str]:
    """Return human-readable list of changes between original and edited state."""
    changes: List[str] = []

    if original.get("sense") != edited.get("sense"):
        changes.append(
            f"Objective direction changed: {original.get('sense')} → {edited.get('sense')}"
        )

    if original.get("objective_latex") != edited.get("objective_latex"):
        changes.append("Objective expression was modified")

    orig_cons = {c["name"]: c for c in original.get("constraints", [])}
    edit_cons = {c["name"]: c for c in edited.get("constraints", [])}

    for name in set(orig_cons) - set(edit_cons):
        changes.append(f"Constraint removed: {name}")
    for name in set(edit_cons) - set(orig_cons):
        changes.append(f"Constraint added: {name}")
    for name in set(orig_cons) & set(edit_cons):
        if orig_cons[name].get("latex") != edit_cons[name].get("latex"):
            changes.append(f"Constraint modified: {name}")
        if orig_cons[name].get("rhs") != edit_cons[name].get("rhs"):
            old_rhs = orig_cons[name].get("rhs", "?")
            new_rhs = edit_cons[name].get("rhs", "?")
            changes.append(f"Constraint '{name}' RHS: {old_rhs} → {new_rhs}")

    orig_vars = {v["name"]: v for v in original.get("variables", [])}
    edit_vars = {v["name"]: v for v in edited.get("variables", [])}

    for name in set(orig_vars) - set(edit_vars):
        changes.append(f"Variable removed: {name}")
    for name in set(edit_vars) - set(orig_vars):
        changes.append(f"Variable added: {name}")
    for name in set(orig_vars) & set(edit_vars):
        if orig_vars[name].get("cat") != edit_vars[name].get("cat"):
            changes.append(
                f"Variable '{name}' type: "
                f"{orig_vars[name].get('cat')} → {edit_vars[name].get('cat')}"
            )
        if orig_vars[name].get("lb") != edit_vars[name].get("lb"):
            changes.append(
                f"Variable '{name}' lower bound: "
                f"{orig_vars[name].get('lb')} → {edit_vars[name].get('lb')}"
            )
        if orig_vars[name].get("ub") != edit_vars[name].get("ub"):
            changes.append(
                f"Variable '{name}' upper bound: "
                f"{orig_vars[name].get('ub')} → {edit_vars[name].get('ub')}"
            )

    if not changes:
        changes.append("No changes detected")

    return changes


# ---------------------------------------------------------------------------
#  1.6  OR template generator
# ---------------------------------------------------------------------------


def get_or_template(problem_type: str, n_vars: int = 3, n_constraints: int = 2) -> dict:
    """Return a starter editor_state dict for a blank model of the given type."""

    if problem_type == "transportation":
        variables = []
        for i in range(1, 3):
            for j in range(1, 4):
                vname = f"x_{i}_{j}"
                sname = f"x_{{{i}{j}}}"
                variables.append(
                    {
                        "name": vname,
                        "sympy_name": sname,
                        "lb": 0,
                        "ub": None,
                        "cat": "Continuous",
                    }
                )
        return {
            "name": "Transportation_Template",
            "sense": "minimize",
            "objective_latex": r"\sum_{i=1}^{2}\sum_{j=1}^{3} c_{ij} x_{ij}",
            "constraints": [
                {
                    "name": "supply_1",
                    "latex": r"x_{11} + x_{12} + x_{13} \leq 100",
                    "sense": "<=",
                    "rhs": 100,
                },
                {
                    "name": "supply_2",
                    "latex": r"x_{21} + x_{22} + x_{23} \leq 150",
                    "sense": "<=",
                    "rhs": 150,
                },
                {
                    "name": "demand_1",
                    "latex": r"x_{11} + x_{21} \geq 80",
                    "sense": ">=",
                    "rhs": 80,
                },
                {
                    "name": "demand_2",
                    "latex": r"x_{12} + x_{22} \geq 60",
                    "sense": ">=",
                    "rhs": 60,
                },
                {
                    "name": "demand_3",
                    "latex": r"x_{13} + x_{23} \geq 70",
                    "sense": ">=",
                    "rhs": 70,
                },
            ],
            "variables": variables,
        }

    if problem_type == "knapsack":
        variables = [
            {
                "name": f"x_{i}",
                "sympy_name": f"x_{{{i}}}",
                "lb": 0,
                "ub": 1,
                "cat": "Binary",
            }
            for i in range(1, n_vars + 1)
        ]
        obj_terms = " + ".join(f"v_{{{i}}} x_{{{i}}}" for i in range(1, n_vars + 1))
        cap_terms = " + ".join(f"w_{{{i}}} x_{{{i}}}" for i in range(1, n_vars + 1))
        return {
            "name": "Knapsack_Template",
            "sense": "maximize",
            "objective_latex": obj_terms,
            "constraints": [
                {
                    "name": "capacity",
                    "latex": cap_terms + r" \leq W",
                    "sense": "<=",
                    "rhs": 100,
                },
            ],
            "variables": variables,
        }

    if problem_type == "assignment":
        variables = []
        for i in range(1, 4):
            for j in range(1, 4):
                vname = f"x_{i}_{j}"
                sname = f"x_{{{i}{j}}}"
                variables.append(
                    {
                        "name": vname,
                        "sympy_name": sname,
                        "lb": 0,
                        "ub": 1,
                        "cat": "Binary",
                    }
                )
        return {
            "name": "Assignment_Template",
            "sense": "minimize",
            "objective_latex": r"\sum_{i=1}^{3}\sum_{j=1}^{3} c_{ij} x_{ij}",
            "constraints": [
                {
                    "name": "agent_1",
                    "latex": r"x_{11} + x_{12} + x_{13} = 1",
                    "sense": "=",
                    "rhs": 1,
                },
                {
                    "name": "agent_2",
                    "latex": r"x_{21} + x_{22} + x_{23} = 1",
                    "sense": "=",
                    "rhs": 1,
                },
                {
                    "name": "agent_3",
                    "latex": r"x_{31} + x_{32} + x_{33} = 1",
                    "sense": "=",
                    "rhs": 1,
                },
                {
                    "name": "task_1",
                    "latex": r"x_{11} + x_{21} + x_{31} = 1",
                    "sense": "=",
                    "rhs": 1,
                },
                {
                    "name": "task_2",
                    "latex": r"x_{12} + x_{22} + x_{32} = 1",
                    "sense": "=",
                    "rhs": 1,
                },
                {
                    "name": "task_3",
                    "latex": r"x_{13} + x_{23} + x_{33} = 1",
                    "sense": "=",
                    "rhs": 1,
                },
            ],
            "variables": variables,
        }

    # Default: LP or IP
    cat = "Integer" if problem_type == "ip" else "Continuous"
    variables = [
        {"name": f"x{i}", "sympy_name": f"x_{{{i}}}", "lb": 0, "ub": None, "cat": cat}
        for i in range(1, n_vars + 1)
    ]
    obj_terms = " + ".join(f"c_{{{i}}} x_{{{i}}}" for i in range(1, n_vars + 1))
    constraints = []
    for k in range(1, n_constraints + 1):
        lhs = " + ".join(f"a_{{{k}{i}}} x_{{{i}}}" for i in range(1, n_vars + 1))
        constraints.append(
            {
                "name": f"constraint_{k}",
                "latex": lhs + r" \leq b_{" + str(k) + "}",
                "sense": "<=",
                "rhs": 100,
            }
        )

    return {
        "name": f'{"IP" if problem_type == "ip" else "LP"}_Template',
        "sense": "maximize",
        "objective_latex": obj_terms,
        "constraints": constraints,
        "variables": variables,
    }
