"""
OR Assistant - Phase 5 Interactive Tester (Full Pipeline)
Run from your project root:
    python tests/test_phase5.py

Paste your problem in plain English when prompted.
Runs the FULL pipeline: classify → model → solve → AI explanation.
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
    from src.interpreters.result_interpreter import ResultInterpreter
except ImportError as e:
    print("\n  ERROR: Could not import required modules.")
    print(f"  Details: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  OR ASSISTANT — Phase 5: Full Pipeline Test")
print("  Classify → Model → Solve → AI Explanation")
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

start_total = time.time()

print("\n" + "-" * 60)
print("  Step 1: Classifying...")
try:
    classifier = ProblemClassifier()
    problem_data = classifier.classify(problem)
    ptype = problem_data.get("problem_type", "?")
    obj = problem_data.get("objective", "?")
    conf = problem_data.get("confidence", 0)
    print(f"  Type: {ptype}  |  Objective: {obj}  |  Confidence: {conf:.0%}")
    assumptions = problem_data.get("assumptions", [])
    if assumptions:
        print(f"  AI made {len(assumptions)} assumption(s) — visible in the app UI.")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\n  Step 2: Building model...")
try:
    generator = ModelGenerator()
    model = generator.generate(problem_data)
    print(f"  {len(model.variables())} variables, {len(model.constraints)} constraints")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n  Step 3: Solving...")
try:
    solver = SolverInterface("pulp")
    solution = solver.solve(model)
    status = solution.get("status", "Unknown")
    obj_val = solution.get("objective_value")
    t = solution.get("solve_time", 0)
    obj_str = f"{obj_val:.4f}" if obj_val is not None else "N/A"
    print(f"  Status: {status}  |  Objective: {obj_str}  |  Time: {t:.3f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\n  Step 4: Generating AI explanation...")
try:
    interpreter = ResultInterpreter()
    interpretation = interpreter.interpret(solution, problem_data)
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

total = time.time() - start_total

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\n  Objective Value : {solution.get('objective_value', 'N/A')}")
print(f"  Status          : {solution.get('status', 'N/A')}")
print(f"  Total Time      : {total:.1f}s")

vars_dict = solution.get("variables", {})
nonzero = {k: v for k, v in vars_dict.items() if v is not None and abs(v) > 0.0001}
if nonzero:
    print("\n  Active Variables:")
    for name, val in list(nonzero.items())[:10]:
        print(f"    {name} = {val:.4f}")

summary = interpretation.get("summary", "")
if summary:
    print("\n  AI Summary:")
    words = summary.split()
    line_buf, out = [], []
    for w in words:
        line_buf.append(w)
        if len(" ".join(line_buf)) > 55:
            out.append("  " + " ".join(line_buf[:-1]))
            line_buf = [w]
    if line_buf:
        out.append("  " + " ".join(line_buf))
    for ln in out:
        print(ln)

findings = interpretation.get("key_findings", [])
if findings:
    print("\n  Key Findings:")
    for f in findings:
        print(f"    * {f}")

recs = interpretation.get("recommendations", [])
if recs:
    print("\n  Recommendations:")
    for i, r in enumerate(recs, 1):
        print(f"    {i}. {r}")

warnings = interpretation.get("warnings", [])
if warnings:
    print("\n  Warnings:")
    for w in warnings:
        print(f"    ! {w}")

print("\n" + "=" * 60)
if total < 30:
    print(f"  ALL STEPS PASSED — Pipeline done in {total:.1f}s")
    print("  Ready to wire the UI in Phase 6.")
else:
    print(f"  WARNING: Took {total:.1f}s (over 30s target).")
print("=" * 60 + "\n")
