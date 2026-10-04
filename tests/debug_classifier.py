"""
OR Assistant — Debug Classifier Output
Run from the project root:
    python tests/debug_classifier.py

Paste your problem, press Enter twice.  This script prints the FULL raw
JSON returned by the classifier so you can see exactly which key names
appeared — useful for diagnosing 'Could not find ... data' errors in
the model generator.
"""

import sys
import json

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from src.agents.problem_classifier import ProblemClassifier
except ImportError as e:
    print(f"\n  ERROR: Could not import ProblemClassifier.\n  Details: {e}")
    sys.exit(1)

# --- Known key lists from ModelGenerator (keep in sync) ------------------
SUPPLY_KEYS = (
    "supply",
    "supply_capacities",
    "warehouse_supply",
    "capacities",
    "sources",
    "warehouse_capacities",
    "available",
)
DEMAND_KEYS = (
    "demand",
    "store_demands",
    "destination_demand",
    "demands",
    "requirements",
    "needed",
    "stores",
)
COST_KEYS = (
    "costs",
    "cost_matrix",
    "shipping_costs",
    "transportation_costs",
    "cost_data",
    "unit_costs",
    "per_unit_costs",
    "route_costs",
    "distance_matrix",
)

# -------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  OR ASSISTANT — Debug Classifier Output")
print("=" * 60)
print("\n  Paste your problem in plain English below.")
print("  Press Enter twice when done.\n")

lines = []
while True:
    try:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    except EOFError:
        break

problem = "\n".join(lines).strip()
if not problem:
    print("\n  No input received. Exiting.")
    sys.exit(1)

print("\n" + "-" * 60)
print("  Running classifier ...")
print("-" * 60)

try:
    classifier = ProblemClassifier()
    result = classifier.classify(problem)
except Exception as e:
    print(f"\n  Classifier FAILED: {e}")
    sys.exit(1)

# --- Print full raw JSON -------------------------------------------------
print("\n  FULL CLASSIFIER JSON:")
print("-" * 60)
print(json.dumps(result, indent=2, default=str))
print("-" * 60)

# --- Key-match report ----------------------------------------------------
params = result.get("parameters", {})
search_dicts = [params, result]


def find_key(sources, keys):
    for d in sources:
        for k in keys:
            val = d.get(k)
            if isinstance(val, (list, dict)) and val:
                return k, val
    return None, None


print("\n  KEY-MATCH REPORT (what the model generator will find):")
print()

for label, keys in [
    ("SUPPLY", SUPPLY_KEYS),
    ("DEMAND", DEMAND_KEYS),
    ("COST", COST_KEYS),
]:
    found_key, found_val = find_key(search_dicts, keys)
    if found_key:
        preview = json.dumps(found_val, default=str)
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  {label:8s}  FOUND  key='{found_key}'  value={preview}")
    else:
        all_param_keys = sorted(params.keys())
        print(f"  {label:8s}  ** NOT FOUND **")
        print(f"           Tried: {', '.join(keys)}")
        print(f"           Actual parameter keys: {all_param_keys}")

print()

# --- Show any keys in parameters that are NOT in any known list ----------
all_known = set(SUPPLY_KEYS) | set(DEMAND_KEYS) | set(COST_KEYS)
unknown = [k for k in params if k not in all_known]
if unknown:
    print("  UNKNOWN PARAMETER KEYS (may need adding to generator):")
    for k in unknown:
        val = params[k]
        preview = json.dumps(val, default=str)
        if len(preview) > 60:
            preview = preview[:57] + "..."
        print(f"    '{k}': {preview}")
    print()

print("  Done. Copy the JSON above into your Cursor prompt to fix key mismatches.")
print()
