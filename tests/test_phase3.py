"""
OR Assistant - Phase 3 Interactive Tester
Run from your project root:
    python tests/test_phase3.py

Paste your problem in plain English when prompted.
Shows the mathematical model that gets built.
"""

import sys

sys.path.insert(0, ".")

# Load .env file so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from src.agents.problem_classifier import ProblemClassifier
    from src.modeling.model_generator import ModelGenerator
    import pulp
except ImportError as e:
    print("\n  ERROR: Could not import required modules.")
    print(f"  Details: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  OR ASSISTANT — Phase 3: Model Generator Test")
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
print("  Step 1: Classifying problem...")
print("-" * 60)
try:
    classifier = ProblemClassifier()
    problem_data = classifier.classify(problem)
    print(f"  Type       : {problem_data.get('problem_type', 'N/A')}")
    print(f"  Objective  : {problem_data.get('objective', 'N/A')}")
    print(f"  Confidence : {problem_data.get('confidence', 0):.0%}")
except Exception as e:
    print(f"\n  FAILED at classifier: {e}")
    sys.exit(1)

print("\n" + "-" * 60)
print("  Step 2: Building mathematical model...")
print("-" * 60)
try:
    generator = ModelGenerator()
    model = generator.generate(problem_data)

    if not isinstance(model, pulp.LpProblem):
        print("\n  FAILED: generate() did not return an LpProblem.")
        sys.exit(1)

    sense = "Maximize" if model.sense == pulp.LpMaximize else "Minimize"
    variables = model.variables()
    binary_vars = [v for v in variables if v.cat == "Binary"]
    integer_vars = [v for v in variables if v.cat == "Integer"]
    cont_vars = [v for v in variables if v.cat == "Continuous"]

    print(f"\n  Model Name     : {model.name}")
    print(f"  Direction      : {sense}")
    print(f"  Variables      : {len(variables)} total")
    if cont_vars:
        print(f"    Continuous   : {len(cont_vars)}")
    if integer_vars:
        print(f"    Integer      : {len(integer_vars)}")
    if binary_vars:
        print(f"    Binary (0/1) : {len(binary_vars)}")

    names = [v.name for v in variables[:10]]
    print(f"\n  Variable names : {names}")
    if len(variables) > 10:
        print(f"    ... and {len(variables)-10} more")

    constraints = model.constraints
    print(f"\n  Constraints    : {len(constraints)}")
    for name, c in list(constraints.items())[:5]:
        print(f"    - {name}: {c}")
    if len(constraints) > 5:
        print(f"    ... and {len(constraints)-5} more")

    print("\n" + "-" * 60)
    print("  RESULT: Model built successfully.")
    print(f"  {len(variables)} variables, {len(constraints)} constraints.")
    print("-" * 60 + "\n")

except Exception as e:
    print(f"\n  FAILED at model generator: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
