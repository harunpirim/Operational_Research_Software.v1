"""
Problem Classifier - Uses AI to identify problem type and extract parameters
Supports both Anthropic Claude and OpenAI models.
"""

import re
from typing import Dict, Any, Optional, List
import json
from src.utils.api_client import APIClient


class ProblemClassifier:
    """
    Classifies optimization problems and extracts relevant parameters
    using AI (Anthropic Claude or OpenAI).
    """

    PROBLEM_TYPES = {
        "linear_programming": (
            "Linear optimization with continuous variables (e.g., maximize profit from"
            " products given resource limits)"
        ),
        "integer_programming": (
            "Optimization where all decision variables must be whole numbers (e.g., how"
            " many trucks to buy)"
        ),
        "mixed_integer_programming": (
            "Mix of continuous and integer variables (e.g., factory on/off decisions"
            " with production quantities)"
        ),
        "transportation": (
            "Minimize cost of shipping goods from supply points to demand points (e.g.,"
            " warehouses to stores)"
        ),
        "assignment": (
            "One-to-one matching of agents to tasks at minimum cost (e.g., assign"
            " workers to jobs)"
        ),
        "scheduling": (
            "Sequence jobs/tasks over time on machines or resources to meet deadlines"
            " (e.g., job-shop scheduling)"
        ),
        "knapsack": (
            "Select items to maximize value without exceeding weight/capacity (e.g.,"
            " loading a cargo container)"
        ),
        "network_flow": (
            "Optimize flow through a directed graph with capacities (e.g., max flow in"
            " a pipeline network)"
        ),
        "routing": (
            "Find optimal vehicle routes to serve customers (e.g., delivery truck route"
            " planning)"
        ),
        "cutting_stock": (
            "Cut raw material into required sizes with minimal waste (e.g., cutting"
            " rolls of paper or steel bars)"
        ),
        "set_covering": (
            "Select the fewest sets that cover all elements (e.g., minimum fire"
            " stations to cover all neighborhoods)"
        ),
        "facility_location": (
            "Decide where to open facilities to minimize cost of serving customers"
            " (e.g., warehouse placement)"
        ),
        "portfolio_optimization": (
            "Allocate investments across assets to maximize return for a given risk"
            " level"
        ),
        "resource_allocation": (
            "Distribute limited resources across activities to optimize outcome (e.g.,"
            " budget allocation across projects)"
        ),
        "production_planning": (
            "Decide how much to produce each period to meet demand while minimizing"
            " cost (e.g., monthly production schedule)"
        ),
        "blending": (
            "Mix raw materials in optimal proportions to meet quality specs at minimum"
            " cost (e.g., animal feed, gasoline blending)"
        ),
        "crew_scheduling": (
            "Assign crew members to shifts or routes satisfying labor rules (e.g.,"
            " airline pilot scheduling)"
        ),
        "bin_packing": (
            "Pack items into the fewest bins/containers possible (e.g., loading boxes"
            " onto pallets)"
        ),
        "traveling_salesman": (
            "Visit all cities exactly once and return home with minimum travel distance"
        ),
        "shortest_path": (
            "Find the least-cost or shortest route between two nodes in a network"
            " (e.g., GPS navigation)"
        ),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the classifier with AI API.

        Args:
            api_key: API key (optional, reads from environment if not provided)
            provider: 'anthropic' or 'openai' (optional, auto-detects if not provided)
            model: Model name (optional, uses default if not provided)
        """
        self.api_client = APIClient(provider=provider, api_key=api_key, model=model)
        self.model = self.api_client.get_model_name()
        self.last_api_metadata = None

    REQUIRED_KEYS = [
        "problem_type",
        "objective",
        "objective_description",
        "decision_variables",
        "constraints",
        "parameters",
        "confidence",
        "assumptions",
        "warnings",
        "notes",
    ]

    _FALLBACK_PROBLEM_TYPES = {"unknown", "general", "other"}

    _NON_PROBLEM_WORDS = {
        "clear",
        "test",
        "hello",
        "hi",
        "ok",
        "yes",
        "no",
        "exit",
        "quit",
        "help",
        "?",
        "...",
        "asd",
        "asdf",
    }

    @classmethod
    def _is_meaningful_input(cls, text: str) -> bool:
        """
        Quick sanity check — returns False for inputs that are clearly
        not an OR problem so we can skip the AI call entirely.
        """
        stripped = text.strip()
        if len(stripped) < 10:
            return False
        if not any(c.isalpha() for c in stripped):
            return False
        if stripped.lower() in cls._NON_PROBLEM_WORDS:
            return False
        return True

    _SAFE_DEFAULT: Dict[str, Any] = {
        "problem_type": "unknown",
        "objective": "minimize",
        "objective_description": (
            "Not identified — input was too short or not a valid problem description."
        ),
        "decision_variables": [],
        "constraints": [],
        "parameters": {},
        "confidence": 0.0,
        "assumptions": [],
        "warnings": ["Input does not appear to be an optimization problem."],
        "notes": (
            "Provide a detailed problem description with objectives, constraints, and"
            " data."
        ),
    }

    def classify(self, problem_description: str) -> Dict[str, Any]:
        """
        Classify a problem description and extract structured parameters.

        Safety layers (in order):
          0. Reject clearly non-problem inputs without calling the AI.
          1. Try json.loads() on the raw response.
          2. Strip markdown code fences and retry json.loads().
          3. Find the first '{' to last '}' substring and retry json.loads().
          4. If all three fail, make ONE retry call asking for clean JSON.
          5. Validate the parsed result with _validate_classification().

        Args:
            problem_description: Natural language description of the problem

        Returns:
            Validated dictionary matching the classification schema
        """
        if not self._is_meaningful_input(problem_description):
            self.last_api_metadata = None
            return dict(self._SAFE_DEFAULT)

        prompt = self._build_classification_prompt(problem_description)
        messages = [{"role": "user", "content": prompt}]

        total_input_tokens = 0
        total_output_tokens = 0

        try:
            response = self.api_client.create_message(
                messages=messages, max_tokens=4096, temperature=0.3
            )
            result_text = response["content"]
            total_input_tokens += response["usage"]["input_tokens"]
            total_output_tokens += response["usage"]["output_tokens"]

            result = self._parse_json(result_text)

        except (json.JSONDecodeError, ValueError):
            try:
                retry_messages = messages + [
                    {"role": "assistant", "content": result_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Return ONLY the JSON object with no other text."
                        ),
                    },
                ]
                retry_response = self.api_client.create_message(
                    messages=retry_messages, max_tokens=4096, temperature=0
                )
                total_input_tokens += retry_response["usage"]["input_tokens"]
                total_output_tokens += retry_response["usage"]["output_tokens"]
                result = self._parse_json(retry_response["content"])
            except Exception as retry_err:
                raise RuntimeError(
                    "Classification failed — could not parse JSON even after retry:"
                    f" {retry_err}\nOriginal response preview: {result_text[:200]}..."
                )
        except Exception as e:
            raise RuntimeError(f"Classification failed: {e}")

        self._validate_classification(result)

        self.last_api_metadata = {
            "provider": self.api_client.get_provider_name(),
            "model": response.get("model", self.model),
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
        }

        return result

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        Three-stage JSON extraction:
          1. Direct json.loads on raw text.
          2. Strip markdown fences (```json … ``` or ``` … ```) then json.loads.
          3. Substring from first '{' to last '}' then json.loads.
        Raises json.JSONDecodeError if all stages fail.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.DOTALL)
        fence_match = fence_pattern.search(text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
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

    _VALID_VAR_TYPES = {"continuous", "integer", "binary"}
    _VALID_SENSES = {"<=", ">=", "="}
    _VALID_IMPACTS = {"high", "medium", "low"}

    def _validate_classification(self, result: Dict[str, Any]) -> None:
        """
        Validate that the classification result contains all required keys
        and that values conform to expected types and sub-schemas.

        Raises ValueError with a clear message listing every problem found.
        """
        errors: List[str] = []

        for key in self.REQUIRED_KEYS:
            if key not in result:
                errors.append(f"missing required key: '{key}'")

        if "problem_type" in result and (
            result["problem_type"] not in self.PROBLEM_TYPES
            and result["problem_type"] not in self._FALLBACK_PROBLEM_TYPES
        ):
            errors.append(
                f"unknown problem_type '{result['problem_type']}' — "
                f"must be one of: {', '.join(self.PROBLEM_TYPES.keys())}"
            )

        if "objective" in result and result["objective"] not in (
            "minimize",
            "maximize",
        ):
            errors.append(
                "objective must be 'minimize' or 'maximize', got"
                f" '{result['objective']}'"
            )

        if "confidence" in result:
            conf = result["confidence"]
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                errors.append(
                    f"confidence must be a float between 0.0 and 1.0, got {conf!r}"
                )

        if "decision_variables" in result:
            if not isinstance(result["decision_variables"], list):
                errors.append("decision_variables must be a list")
            else:
                for i, var in enumerate(result["decision_variables"]):
                    for k in ("name", "description", "type"):
                        if k not in var:
                            errors.append(f"decision_variables[{i}] missing '{k}'")
                    if var.get("type") not in self._VALID_VAR_TYPES:
                        errors.append(
                            f"decision_variables[{i}].type must be one of "
                            f"{self._VALID_VAR_TYPES}, got '{var.get('type')}'"
                        )

        if "constraints" in result:
            if not isinstance(result["constraints"], list):
                errors.append("constraints must be a list")
            else:
                for i, con in enumerate(result["constraints"]):
                    for k in ("name", "expression", "sense"):
                        if k not in con:
                            errors.append(f"constraints[{i}] missing '{k}'")
                    if con.get("sense") not in self._VALID_SENSES:
                        errors.append(
                            f"constraints[{i}].sense must be one of "
                            f"{self._VALID_SENSES}, got '{con.get('sense')}'"
                        )

        if "assumptions" in result:
            if not isinstance(result["assumptions"], list):
                errors.append("assumptions must be a list")
            else:
                for i, a in enumerate(result["assumptions"]):
                    for k in ("assumption", "confidence", "impact", "field"):
                        if k not in a:
                            errors.append(f"assumptions[{i}] missing '{k}'")
                    if "confidence" in a:
                        ac = a["confidence"]
                        if not isinstance(ac, (int, float)) or not (0.0 <= ac <= 1.0):
                            errors.append(
                                f"assumptions[{i}].confidence must be 0.0–1.0, got"
                                f" {ac!r}"
                            )
                    if a.get("impact") not in self._VALID_IMPACTS:
                        errors.append(
                            f"assumptions[{i}].impact must be one of "
                            f"{self._VALID_IMPACTS}, got '{a.get('impact')}'"
                        )

        if "warnings" in result and not isinstance(result["warnings"], list):
            errors.append("warnings must be a list")

        if errors:
            raise ValueError(
                "Classification validation failed:\n  - " + "\n  - ".join(errors)
            )

    def _build_classification_prompt(self, description: str) -> str:
        """Build the prompt for AI to classify the problem."""

        problem_types_list = "\n".join(
            [f"- {key}: {value}" for key, value in self.PROBLEM_TYPES.items()]
        )

        prompt = f"""You are an expert in Operations Research. Analyze the following problem description.

1. Identify the problem type from these categories:
{problem_types_list}

2. Extract ALL relevant parameters: decision variables, objective, constraints, and data.

3. CRITICAL — Assumptions rule:
   For EVERY piece of information you infer that was NOT explicitly stated in the
   problem description, you MUST add an entry to the "assumptions" array. Each entry
   records what you assumed, how confident you are, the impact level, and which JSON
   field the assumption affects. Never silently fill in missing information.

4. Return ONLY a valid JSON object matching this EXACT schema (every field is required):
{{
    "problem_type": "one_of_the_types_above",
    "objective": "minimize" or "maximize",
    "objective_description": "what is being optimized and why",
    "decision_variables": [
        {{
            "name": "ship[i,j]",
            "description": "units shipped from warehouse i to store j",
            "type": "continuous" or "integer" or "binary"
        }}
    ],
    "constraints": [
        {{
            "name": "supply_limit",
            "expression": "sum(ship[i,j] for all j) <= supply[i]",
            "sense": "<=" or ">=" or "="
        }}
    ],
    "parameters": {{
        "supply": [100, 150, 200],
        "demand": [80, 120, 90, 110],
        "costs": [[5, 8, 6, 7], [6, 7, 9, 5], [8, 6, 7, 9]]
    }},
    "confidence": 0.0 to 1.0,
    "assumptions": [
        {{
            "assumption": "demand values are in the same units as supply",
            "confidence": 0.9,
            "impact": "high" or "medium" or "low",
            "field": "parameters.demand"
        }}
    ],
    "warnings": [
        "total supply (450) exceeds total demand (290) — problem may have slack"
    ],
    "notes": "any other clarifications about the formulation"
}}

Rules:
- Every field shown above is REQUIRED in your response.
- "decision_variables": list every variable with name, description, and type
  (type must be "continuous", "integer", or "binary").
- "constraints": list every constraint with name, a mathematical expression, and sense
  (sense must be "<=", ">=", or "=").
- "parameters": all extracted numerical data as proper JSON types (numbers, arrays, matrices).
  For knapsack problems:
  * Use "capacity" or "budget" for the weight/capacity limit
  * Use "weights" or "costs" for item weights/costs
  * Use "values" or "returns" for item values/profits
  * Use "item_names" or "project_names" for item identifiers
  * For investment knapsack: prefer "budget", "costs", "returns", "project_names"
- "assumptions": one entry per inference you made about data NOT explicitly stated.
  Each needs: assumption (string), confidence (0.0–1.0), impact ("high"/"medium"/"low"),
  and field (the JSON key path this assumption affects, e.g. "parameters.demand").
  If the problem is fully specified with zero ambiguity, use an empty array [].
- "warnings": flag potential issues — infeasibility, mismatched totals, unusual values, etc.
  Use an empty array [] if none.
- "confidence": your overall confidence in the classification (0.0–1.0).

Problem Description:
{description}

Return ONLY the JSON object, no additional text or markdown formatting.
"""  # noqa: E501
        return prompt

    def get_problem_template(self, problem_type: str) -> Dict[str, Any]:
        """
        Get a template structure for a specific problem type.

        Args:
            problem_type: Type of problem

        Returns:
            Template dictionary
        """

        templates = {
            "linear_programming": {
                "variables": [],
                "objective_coefficients": [],
                "constraint_coefficients": [],
                "constraint_bounds": [],
                "variable_bounds": [],
            },
            "transportation": {
                "sources": [],
                "destinations": [],
                "supply": [],
                "demand": [],
                "costs": [],
            },
            "assignment": {"agents": [], "tasks": [], "costs": []},
            # Add more templates as needed
        }

        return templates.get(problem_type, {})


if __name__ == "__main__":
    # Example usage
    classifier = ProblemClassifier()

    test_problem = """
    I need to minimize transportation costs between 3 warehouses and 4 stores.
    Warehouse supplies: [100, 150, 200]
    Store demands: [80, 120, 90, 110]
    Cost per unit from each warehouse to each store (3x4 matrix):
    [[5, 8, 6, 7],
     [6, 7, 9, 5],
     [8, 6, 7, 9]]
    """

    result = classifier.classify(test_problem)
    print(json.dumps(result, indent=2))
