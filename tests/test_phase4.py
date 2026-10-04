"""
OR Assistant - Phase 4 Interactive Tester
Run from your project root:
    python tests/test_phase4.py

Paste your problem in plain English when prompted.
Runs classify + build + solve and prints the answer.
"""

import sys
import time

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
    from src.solvers.solver_interface import SolverInterface
    import pulp  # noqa: F401  (dependency check)
except ImportError as e:
    print("\n  ERROR: Could not import required modules.")
    print(f"  Details: {e}")
    sys.exit(1)


def _print_guidance(step: int, err_str: str) -> None:
    """Print actionable guidance based on the error content."""
    low = err_str.lower()

    if "not recognized" in low or "unknown" in low or "confidence" in low:
        print("\n  GUIDANCE: Your input was not recognized as an OR problem.")
        print("  Try describing: WHAT to optimize, WHAT the variables are,")
        print("  and WHAT constraints apply. Example:")
        print("  'Maximize profit. I make tables ($50 each) and chairs ($30 each).")
        print("   Tables need 3 labor hours, chairs need 2. I have 240 hours total.'")
    elif "coefficient" in low or "variable count" in low:
        print("\n  GUIDANCE: Run Prompt 3.5 in Cursor to fix coefficient extraction.")
        print("  Then run: python tests/debug_classifier.py")
        print("  to see exactly what data the classifier returned.")
    elif "cost data" in low or "supply" in low or "demand" in low:
        print("\n  GUIDANCE: Run Prompt 3.3 in Cursor to fix key name lookups.")
        print("  Then run: python tests/debug_classifier.py")
    elif "infeasible" in low:
        print("\n  GUIDANCE: Your constraints contradict each other.")
        print("  Check that your >= and <= constraints are compatible.")
    else:
        print("\n  GUIDANCE: Run with debug mode to see raw data:")
        print("  Windows: $env:OR_DEBUG='1'; python tests/test_phase4.py")


print("\n" + "=" * 60)
print("  OR ASSISTANT — Phase 4: Solver Test")
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

if len(problem.strip()) < 15:
    print("\n  INPUT TOO SHORT — not a valid OR problem.")
    print("  Please describe an optimization problem with numbers and constraints.")
    print("  Example: 'Maximize profit making tables and chairs...'")
    sys.exit(0)

start_total = time.time()

# -----------------------------------------------------------------
#  Step 1: Classify
# -----------------------------------------------------------------
print("\n" + "-" * 60)
print("  Step 1: Classifying...")
print("-" * 60)
try:
    classifier = ProblemClassifier()
    problem_data = classifier.classify(problem)
    print(f"  Type       : {problem_data.get('problem_type', 'N/A')}")
    print(f"  Objective  : {problem_data.get('objective', 'N/A')}")
    print(f"  Confidence : {problem_data.get('confidence', 0):.0%}")
except Exception as e:
    err = str(e)
    print(f"\n  FAILED at Step 1: {err}")
    _print_guidance(1, err)
    import traceback

    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------
#  Step 2: Build model
# -----------------------------------------------------------------
print("\n" + "-" * 60)
print("  Step 2: Building model...")
print("-" * 60)
try:
    generator = ModelGenerator()
    model = generator.generate(problem_data)
    print(f"  Variables   : {len(model.variables())}")
    print(f"  Constraints : {len(model.constraints)}")
except Exception as e:
    err = str(e)
    print(f"\n  FAILED at Step 2: {err}")
    _print_guidance(2, err)
    import traceback

    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------
#  Step 3: Solve
# -----------------------------------------------------------------
print("\n" + "-" * 60)
print("  Step 3: Solving...")
print("-" * 60)
try:
    solver = SolverInterface("pulp")
    solution = solver.solve(model)

    status = solution.get("status", "Unknown")
    obj = solution.get("objective_value")
    t = solution.get("solve_time", 0)

    print(f"\n  Status          : {status}")
    if obj is not None:
        print(f"  Objective Value : {obj:.4f}")
    print(f"  Solve Time      : {t:.3f}s")

    # Show warnings from the solver
    for w in solution.get("warnings", []):
        print(f"  WARNING: {w}")

    if solution.get("error_message"):
        print(f"  ERROR: {solution['error_message']}")

    vars_dict = solution.get("variables", {})
    nonzero = {k: v for k, v in vars_dict.items() if v is not None and abs(v) > 0.0001}
    zero = {k: v for k, v in vars_dict.items() if v is not None and abs(v) <= 0.0001}

    print("\n  Active Variables (non-zero):")
    if nonzero:
        for name, val in list(nonzero.items())[:15]:
            print(f"    {name} = {val:.4f}")
        if len(nonzero) > 15:
            print(f"    ... and {len(nonzero)-15} more")
    else:
        print("    None — all variables are zero.")

    if zero:
        zkeys = list(zero.keys())[:5]
        print(f"\n  Zero Variables  : {zkeys}" + (" ..." if len(zero) > 5 else ""))

    total = time.time() - start_total
    print("\n" + "-" * 60)
    if solution.get("is_optimal"):
        print("  RESULT: OPTIMAL SOLUTION FOUND")
        print(f"  Objective Value = {obj:.4f}")
        print(f"  Total time      : {total:.1f}s")
    else:
        print(f"  RESULT: {status} — no optimal solution.")
        for w in solution.get("warnings", []):
            print(f"  {w}")
        if not solution.get("warnings"):
            print("  Check your constraints — they may contradict each other.")
    print("-" * 60 + "\n")

except Exception as e:
    err = str(e)
    print(f"\n  FAILED at Step 3: {err}")
    _print_guidance(3, err)
    import traceback

    traceback.print_exc()
    sys.exit(1)
