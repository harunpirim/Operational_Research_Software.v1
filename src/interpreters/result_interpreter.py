"""
Result Interpreter - Uses AI to explain solutions in plain language
Supports both Anthropic Claude and OpenAI models.
"""

import re
from typing import Dict, Any, Optional
import json
from src.utils.api_client import APIClient


class ResultInterpreter:
    """
    Interprets optimization results and generates human-readable explanations
    using AI (Anthropic Claude or OpenAI).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_client = APIClient(provider=provider, api_key=api_key, model=model)
        self.model = self.api_client.get_model_name()

    # ------------------------------------------------------------------
    #  3-layer JSON parser (same logic as ProblemClassifier)
    # ------------------------------------------------------------------

    _FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.DOTALL)

    @classmethod
    def _parse_json(cls, text: str) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def interpret(
        self,
        solution: Dict[str, Any],
        problem_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate an AI-powered, plain-language interpretation of a solution.

        Returns a dict with keys: summary, key_findings, recommendations,
        warnings, business_impact.
        """
        if not solution.get("is_optimal", False):
            return self._fallback_interpret(solution, problem_data)

        prompt = self._build_prompt(solution, problem_data)

        try:
            response = self.api_client.create_message(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.4,
            )
            return self._parse_json(response["content"])

        except Exception:
            return self._fallback_interpret(solution, problem_data)

    # ------------------------------------------------------------------
    #  Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _top_nonzero(variables: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """Return the top *limit* non-zero variable values."""
        nonzero = {
            k: v for k, v in variables.items() if v is not None and abs(v) > 1e-8
        }
        return dict(list(nonzero.items())[:limit])

    def _build_prompt(
        self,
        solution: Dict[str, Any],
        problem_data: Dict[str, Any],
    ) -> str:
        top_vars = self._top_nonzero(solution.get("variables", {}))

        # For knapsack problems, enhance variable names with actual item names
        if problem_data.get("problem_type") == "knapsack":
            params = problem_data.get("parameters", {})
            item_names = params.get("item_names") or params.get("project_names")
            if item_names:
                enhanced_vars = {}
                for var_name, value in top_vars.items():
                    # Extract index from variable name (e.g., 'item_0' -> 0)
                    try:
                        if var_name.startswith("item_"):
                            idx = int(var_name.split("_")[1])
                            if 0 <= idx < len(item_names):
                                enhanced_vars[f"{var_name} ({item_names[idx]})"] = value
                            else:
                                enhanced_vars[var_name] = value
                        else:
                            enhanced_vars[var_name] = value
                    except (ValueError, IndexError):
                        enhanced_vars[var_name] = value
                top_vars = enhanced_vars

        vars_json = json.dumps(top_vars, indent=2, default=str)

        return (
            "You are an Operations Research consultant explaining "
            "optimization results to a business stakeholder.\n\n"
            f"Problem type: {problem_data.get('problem_type', 'unknown')}\n"
            f"Objective: {problem_data.get('objective', 'minimize')} — "
            f"{problem_data.get('objective_description', '')}\n"
            f"Optimal objective value: {solution.get('objective_value')}\n"
            f"Number of variables: {solution.get('num_variables', '?')}\n"
            f"Number of constraints: {solution.get('num_constraints', '?')}\n\n"
            f"Top non-zero decision-variable values:\n{vars_json}\n\n"
            "Explain what these numbers mean in plain business language.\n\n"
            "Return ONLY this JSON (no other text):\n"
            "{\n"
            '  "summary": "2-sentence plain English answer — what is the optimal solution?",\n'
            '  "key_findings": ["Finding 1 with specific number", "Finding 2", "Finding 3"],\n'
            '  "recommendations": ["Actionable step 1", "Actionable step 2"],\n'
            '  "warnings": ["Any tight constraints or risks noted"],\n'
            '  "business_impact": "One paragraph about what this means practically"\n'
            "}"
        )

    # ------------------------------------------------------------------
    #  Deterministic fallback (no AI call)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_interpret(
        solution: Dict[str, Any],
        problem_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Basic interpretation without calling the AI."""
        ptype = problem_data.get("problem_type", "optimization")
        objective = problem_data.get("objective", "optimal")
        obj_val = solution.get("objective_value")

        try:
            obj_str = f"{float(obj_val):.4f}"
        except (TypeError, ValueError):
            obj_str = str(obj_val)

        summary = (
            f"The {ptype} was solved optimally. " f"The {objective} value is {obj_str}."
        )

        variables = solution.get("variables", {})

        # For knapsack problems, enhance variable names with actual item names
        if problem_data.get("problem_type") == "knapsack":
            params = problem_data.get("parameters", {})
            item_names = params.get("item_names") or params.get("project_names")
            if item_names:
                enhanced_vars = {}
                for var_name, value in variables.items():
                    if value is not None and abs(value) > 1e-8:
                        try:
                            if var_name.startswith("item_"):
                                idx = int(var_name.split("_")[1])
                                if 0 <= idx < len(item_names):
                                    enhanced_vars[f"{var_name} ({item_names[idx]})"] = (
                                        value
                                    )
                                else:
                                    enhanced_vars[var_name] = value
                            else:
                                enhanced_vars[var_name] = value
                        except (ValueError, IndexError):
                            enhanced_vars[var_name] = value
                variables = enhanced_vars

        sorted_vars = sorted(
            ((k, v) for k, v in variables.items() if v is not None and abs(v) > 1e-8),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        key_findings = [
            f"Variable {name} = {val} units" for name, val in sorted_vars[:5]
        ]

        num_vars = solution.get("num_variables", "?")
        num_cons = solution.get("num_constraints", "?")
        solve_time = solution.get("solve_time")
        try:
            time_str = f"{float(solve_time):.2f}"
        except (TypeError, ValueError):
            time_str = str(solve_time)

        return {
            "summary": summary,
            "key_findings": key_findings,
            "recommendations": [
                "Review decision variables above zero for resource allocation insights",
            ],
            "warnings": solution.get("warnings", []),
            "business_impact": (
                f"Solved {num_vars} variables with "
                f"{num_cons} constraints in {time_str} seconds."
            ),
        }

    # ------------------------------------------------------------------
    #  Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        solution: Dict[str, Any],
        problem_data: Dict[str, Any],
        fmt: str = "markdown",
    ) -> str:
        interpretation = self.interpret(solution, problem_data)

        if fmt == "markdown":
            return self._markdown_report(interpretation, solution, problem_data)
        return self._text_report(interpretation, solution, problem_data)

    @staticmethod
    def _markdown_report(
        interp: Dict[str, Any],
        solution: Dict[str, Any],
        problem_data: Dict[str, Any],
    ) -> str:
        lines = [
            "# Optimization Results Report\n",
            "## Summary",
            interp.get("summary", ""),
            "",
            "## Key Findings",
        ]
        for f in interp.get("key_findings", []):
            lines.append(f"- {f}")

        lines += ["", "## Recommendations"]
        for r in interp.get("recommendations", []):
            lines.append(f"- {r}")

        if interp.get("warnings"):
            lines += ["", "## Warnings"]
            for w in interp["warnings"]:
                lines.append(f"- ⚠️ {w}")

        lines += ["", "## Business Impact", interp.get("business_impact", "")]

        top = {
            k: v
            for k, v in solution.get("variables", {}).items()
            if v is not None and abs(v) > 1e-8
        }

        # For knapsack problems, enhance variable names with actual item names
        if problem_data.get("problem_type") == "knapsack":
            params = problem_data.get("parameters", {})
            item_names = params.get("item_names") or params.get("project_names")
            if item_names:
                enhanced_top = {}
                for var_name, value in top.items():
                    try:
                        if var_name.startswith("item_"):
                            idx = int(var_name.split("_")[1])
                            if 0 <= idx < len(item_names):
                                enhanced_top[f"{var_name} ({item_names[idx]})"] = value
                            else:
                                enhanced_top[var_name] = value
                        else:
                            enhanced_top[var_name] = value
                    except (ValueError, IndexError):
                        enhanced_top[var_name] = value
                top = enhanced_top

        if top:
            lines += ["", "## Decision Variables (non-zero)"]
            for name, val in list(top.items())[:20]:
                lines.append(f"- **{name}** = {val}")

        return "\n".join(lines)

    @staticmethod
    def _text_report(
        interp: Dict[str, Any],
        solution: Dict[str, Any],
        problem_data: Dict[str, Any],
    ) -> str:
        parts = [
            "=== Optimization Results ===",
            interp.get("summary", ""),
            "",
            "Key Findings:",
        ]
        for f in interp.get("key_findings", []):
            parts.append(f"  - {f}")
        parts += ["", "Business Impact:", interp.get("business_impact", "")]
        return "\n".join(parts)


if __name__ == "__main__":
    interpreter = ResultInterpreter()

    test_solution = {
        "status": "Optimal",
        "is_optimal": True,
        "objective_value": 1250.0,
        "solve_time": 2.3,
        "num_variables": 12,
        "num_constraints": 7,
        "solver_name": "pulp",
        "variables": {"x_0_0": 50, "x_0_1": 50, "x_1_0": 30},
        "warnings": [],
        "error_message": "",
    }

    test_problem = {
        "problem_type": "transportation",
        "objective": "minimize",
        "objective_description": "total transportation cost",
    }

    result = interpreter.interpret(test_solution, test_problem)
    print(json.dumps(result, indent=2))
