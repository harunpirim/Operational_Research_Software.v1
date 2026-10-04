"""
Data Extractor — analyse parsed file content with AI and produce a
structured problem dict compatible with ``ModelGenerator.generate()``.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.utils.api_client import APIClient


class DataExtractor:
    """
    Takes the output of ``FileParser.parse()`` and uses AI to identify the
    optimisation problem hidden in the data.  The returned dict is guaranteed
    to contain every key that ``ProblemClassifier.classify()`` would produce,
    so it can be fed directly into ``ModelGenerator.generate()``.
    """

    _CLASSIFIER_DEFAULTS: Dict[str, Any] = {
        "problem_type": "unknown",
        "objective": None,  # None means AI didn't specify — caller should not assume minimize
        "objective_description": "",
        "decision_variables": [],
        "constraints": [],
        "parameters": {},
        "confidence": 0.0,
        "assumptions": [],
        "warnings": [],
        "notes": "",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_client = APIClient(
            provider=provider,
            api_key=api_key,
            model=model,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        parsed_file_data: Dict[str, Any],
        user_hint: str = "",
        filename: str = "",
    ) -> Dict[str, Any]:
        """
        Analyse parsed file content and return a classifier-compatible dict.

        Args:
            parsed_file_data: Output of ``FileParser.parse()``.
            user_hint: Optional context from the user, e.g.
                ``"This is a transportation problem"``.
            filename: Original filename (stored in the result for traceability).

        Returns:
            A dict with all ``ProblemClassifier`` keys plus ``source``,
            ``filename``, and ``data_summary``.
        """
        if parsed_file_data.get("type") == "error":
            return self._error_result(
                parsed_file_data.get("error", "File parsing failed"),
                filename,
            )

        prompt = self._build_prompt(parsed_file_data, user_hint)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.api_client.create_message(
                messages=messages,
                max_tokens=4096,
                temperature=0.3,
            )
            result = self._parse_json(response["content"])
        except (json.JSONDecodeError, ValueError):
            try:
                retry_messages = messages + [
                    {"role": "assistant", "content": response["content"]},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Return ONLY the JSON object with no other text."
                        ),
                    },
                ]
                retry_response = self.api_client.create_message(
                    messages=retry_messages,
                    max_tokens=4096,
                    temperature=0,
                )
                result = self._parse_json(retry_response["content"])
            except Exception as retry_err:
                return self._error_result(
                    f"AI JSON parsing failed after retry: {retry_err}",
                    filename,
                )
        except Exception as e:
            return self._error_result(f"AI call failed: {e}", filename)

        merged = self._merge_with_defaults(result)
        merged = self._add_raw_data_to_parameters(merged, parsed_file_data)
        merged["source"] = "file_upload"
        merged["filename"] = filename
        return merged

    # ------------------------------------------------------------------
    #  Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        parsed: Dict[str, Any],
        user_hint: str,
    ) -> str:
        sections: List[str] = []

        sections.append(
            "You are an expert in Operations Research. "
            "Analyse the following data extracted from an uploaded file and "
            "identify the optimisation problem it represents."
        )

        raw = parsed.get("raw_text", "")
        if raw:
            truncated = raw[:3000]
            if len(raw) > 3000:
                truncated += "\n... [truncated]"
            sections.append(f"=== RAW TEXT (first 3000 chars) ===\n{truncated}")

        table_preview = self._format_table_preview(parsed)
        if table_preview:
            sections.append(f"=== STRUCTURED DATA PREVIEW ===\n{table_preview}")

        if user_hint.strip():
            sections.append(f"=== USER HINT ===\n{user_hint.strip()}")

        sections.append(self._json_instructions())

        return "\n\n".join(sections)

    @staticmethod
    def _format_table_preview(parsed: Dict[str, Any]) -> str:
        """Format column headers + first 5 rows for each sheet (Excel/CSV)."""
        ftype = parsed.get("type", "")
        if ftype not in ("excel", "csv"):
            return ""

        parts: List[str] = []
        for sheet in parsed.get("sheets", []):
            name = sheet.get("name", "Sheet")
            headers = sheet.get("headers", [])
            rows = sheet.get("rows", [])
            if not headers:
                continue

            parts.append(f"Sheet: {name}")
            parts.append(" | ".join(str(h) for h in headers))
            parts.append("-+-".join("-" * max(len(str(h)), 3) for h in headers))
            for row in rows[:5]:
                parts.append(" | ".join(str(row.get(h, "")) for h in headers))
            remaining = len(rows) - 5
            if remaining > 0:
                parts.append(f"... ({remaining} more rows)")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _json_instructions() -> str:
        return (
            "Analyse this data and return ONLY this JSON structure.\n\n"
            "CRITICAL RULE FOR 'parameters' KEY NAMES:\n"
            "You MUST use these exact key names — no variations allowed:\n"
            "  For transportation problems:\n"
            "    'supply'       -> list of numbers, one per source/warehouse/factory\n"
            "    'demand'       -> list of numbers, one per destination/store/customer\n"
            "    'costs'        -> 2D list [source][destination] of shipping costs\n"
            "    'source_names' -> list of source names (strings)\n"
            "    'dest_names'   -> list of destination names (strings)\n"
            "  For linear/integer programming:\n"
            "    'objective_coefficients' -> list of numbers matching decision_variables order\n"
            "    'constraint_rhs'         -> list of right-hand-side values per constraint\n"
            "    'constraint_coefficients'-> 2D list [constraint][variable] of coefficients\n"
            "  For assignment problems:\n"
            "    'cost_matrix'  -> 2D list [agent][task] of assignment costs\n"
            "    'agent_names'  -> list of agent names\n"
            "    'task_names'   -> list of task names\n"
            "  For knapsack problems:\n"
            "    'capacity'     -> single number, the knapsack capacity\n"
            "    'weights'      -> list of item weights\n"
            "    'values'       -> list of item values/profits\n"
            "    'item_names'   -> list of item names\n"
            "  For scheduling problems:\n"
            "    'processing_times' -> 2D list [job][machine] of processing times\n"
            "    'due_dates'        -> list of due dates per job\n"
            "    'num_jobs'         -> integer\n"
            "    'num_machines'     -> integer\n"
            "  For all other types: use the most descriptive key name possible.\n\n"
            "JSON STRUCTURE (all fields required):\n"
            "{\n"
            '  "problem_type": "<transportation|linear_programming|integer_programming|'
            "mixed_integer_programming|assignment|scheduling|knapsack|network_flow|"
            "routing|cutting_stock|set_covering|facility_location|portfolio_optimization|"
            'resource_allocation|production_planning|blending|unknown>",\n'
            '  "objective": "minimize" or "maximize",\n'
            '  "objective_description": "one sentence describing what is being optimized",\n'
            '  "decision_variables": [\n'
            '    {"name": "x_ij", "description": "units shipped from i to j", '
            '"type": "continuous"}\n'
            "  ],\n"
            '  "constraints": [\n'
            '    {"name": "supply_A", "expression": "x_A1+x_A2+x_A3+x_A4 <= 100", '
            '"sense": "<=", "rhs": 100}\n'
            "  ],\n"
            '  "parameters": {\n'
            '    "supply": [100, 150, 200],\n'
            '    "demand": [80, 120, 90, 110],\n'
            '    "costs": [[5,8,6,7],[6,7,9,5],[8,6,7,9]],\n'
            '    "source_names": ["warehouse_A","warehouse_B","warehouse_C"],\n'
            '    "dest_names": ["store_1","store_2","store_3","store_4"]\n'
            "  },\n"
            '  "confidence": 0.95,\n'
            '  "assumptions": [],\n'
            '  "warnings": [],\n'
            '  "notes": "",\n'
            '  "data_summary": {\n'
            '    "num_rows": 4,\n'
            '    "num_columns": 6,\n'
            '    "detected_structure": "3x4 transportation cost matrix with supply and demand"\n'
            "  }\n"
            "}\n\n"
            "RULES:\n"
            "- Use EXACTLY the key names listed above for the detected problem type.\n"
            "- Extract ALL numerical values from the file into parameters.\n"
            "- If a row is labeled 'demand' or 'requirement', put those values in 'demand'.\n"
            "- If a column is labeled 'supply' or 'capacity', put those values in 'supply'.\n"
            "- The example parameters block above is for transportation — adapt for other types.\n"
            "- Return ONLY the JSON object, no markdown, no extra text."
        )

    def _add_raw_data_to_parameters(
        self,
        merged: Dict[str, Any],
        parsed_file_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        After AI extraction, inject raw structured data from the file directly
        into parameters so the model generator can always fall back to it.
        This is the safety net — if the AI used wrong key names, the model
        generator can read from these guaranteed keys instead.
        """
        params = merged.setdefault("parameters", {})
        file_type = parsed_file_data.get("type", "")

        if file_type in ("csv", "excel"):
            sheets = parsed_file_data.get("sheets", [])
            if sheets:
                sheet = sheets[0]
                rows = sheet.get("rows", [])
                headers = sheet.get("headers", [])
                params["_raw_headers"] = headers
                params["_raw_rows"] = rows  # list of dicts {col: value}

                # For transportation CSVs: try to auto-extract supply/demand/costs
                # from structure if AI didn't produce them with correct keys
                if merged.get("problem_type") == "transportation":
                    if (
                        not params.get("supply")
                        or not params.get("demand")
                        or not params.get("costs")
                    ):
                        auto = self._auto_extract_transportation(headers, rows)
                        for k, v in auto.items():
                            if v and not params.get(k):
                                params[k] = v

        elif file_type == "docx":
            params["_raw_paragraphs"] = parsed_file_data.get("paragraphs", [])
            params["_raw_tables"] = parsed_file_data.get("tables", [])

        return merged

    def _auto_extract_transportation(
        self,
        headers: List[Any],
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Read a transportation CSV directly from its structure without relying
        on the AI. Works for any CSV where:
          - Rows represent sources (warehouses, factories, suppliers)
          - Last column represents supply capacity
          - Final row represents demand (labeled 'demand', 'requirement', etc.)
          - Middle columns are cost values
        """
        supply: List[float] = []
        demand: List[float] = []
        costs: List[List[float]] = []
        source_names: List[str] = []
        dest_names: List[str] = []

        demand_labels = {
            "demand",
            "demands",
            "requirement",
            "requirements",
            "need",
            "needed",
            "consumption",
            "required",
        }

        # Identify which columns are cost columns (not name col, not supply col)
        # Supply column: last column named 'supply', 'capacity', 'available', etc.
        supply_col_names = {
            "supply",
            "capacity",
            "available",
            "stock",
            "production",
            "max",
            "maximum",
        }
        supply_col = None
        for h in reversed(headers):
            if str(h).lower().strip() in supply_col_names:
                supply_col = h
                break
        if supply_col is None and headers:
            supply_col = headers[-1]  # assume last column is supply

        # Name column: first column
        name_col = headers[0] if headers else None

        # Cost columns: everything between name and supply
        cost_cols = [h for h in headers if h != name_col and h != supply_col]
        dest_names = [str(c) for c in cost_cols]

        for row in rows:
            name_val = str(row.get(name_col, "")).lower().strip()

            if name_val in demand_labels:
                # This is the demand row
                demand = []
                for col in cost_cols:
                    v = row.get(col)
                    try:
                        demand.append(float(v))
                    except (TypeError, ValueError):
                        pass
            else:
                # This is a supply row
                source_names.append(str(row.get(name_col, f"source_{len(supply)}")))
                row_costs: List[float] = []
                for col in cost_cols:
                    v = row.get(col)
                    try:
                        row_costs.append(float(v))
                    except (TypeError, ValueError):
                        row_costs.append(0.0)
                if row_costs:
                    costs.append(row_costs)

                sup_val = row.get(supply_col)
                try:
                    supply.append(float(sup_val))
                except (TypeError, ValueError):
                    pass

        return {
            "supply": supply or None,
            "demand": demand or None,
            "costs": costs or None,
            "source_names": source_names or None,
            "dest_names": dest_names or None,
        }

    # ------------------------------------------------------------------
    #  JSON parsing (3-layer, same pattern as ProblemClassifier)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fence = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.DOTALL)
        m = fence.search(text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(text[first : last + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)

    # ------------------------------------------------------------------
    #  Merge / defaults
    # ------------------------------------------------------------------

    def _merge_with_defaults(self, ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in any keys the AI omitted using classifier defaults."""
        merged = dict(self._CLASSIFIER_DEFAULTS)
        for key, value in ai_result.items():
            if value is not None:
                merged[key] = value
        # If objective still None after merge, default to minimize with a warning
        if not merged.get("objective"):
            merged["objective"] = "minimize"
            merged.setdefault("warnings", []).append(
                "Objective sense not detected — defaulted to minimize."
            )
        return merged

    def _error_result(
        self,
        error_msg: str,
        filename: str,
    ) -> Dict[str, Any]:
        result = dict(self._CLASSIFIER_DEFAULTS)
        result["objective"] = "minimize"  # safe default for error paths
        result.update(
            {
                "source": "file_upload",
                "filename": filename,
                "confidence": 0.0,
                "warnings": [error_msg],
                "notes": "Extraction failed — try pasting the problem text directly.",
                "data_summary": {
                    "num_rows": 0,
                    "num_columns": 0,
                    "detected_structure": "error",
                },
            }
        )
        return result
