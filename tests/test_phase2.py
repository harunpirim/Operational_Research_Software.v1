"""
OR Assistant - Phase 2 Interactive Tester
Run from your project root:
    python tests/test_phase2.py

Paste your problem in plain English when prompted.
The classifier will show you what it detected.
"""

import sys

sys.path.insert(0, ".")

# Load .env file so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

try:
    from src.agents.problem_classifier import ProblemClassifier
except ImportError as e:
    print("\n  ERROR: Could not import ProblemClassifier.")
    print("  Make sure you are running from the or-assistant folder.")
    print(f"  Details: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  OR ASSISTANT — Phase 2: Problem Classifier Test")
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
print("  Running classifier...")
print("-" * 60)

try:
    classifier = ProblemClassifier()
    result = classifier.classify(problem)

    # --- AI Model & API Info ---
    meta = classifier.last_api_metadata
    if meta:
        print(f"\n  AI Provider    : {meta['provider']}")
        print(f"  Model Used     : {meta['model']}")
        print(f"  Tokens (in)    : {meta['usage']['input_tokens']:,}")
        print(f"  Tokens (out)   : {meta['usage']['output_tokens']:,}")
        print(f"  Tokens (total) : {meta['usage']['total_tokens']:,}")

    # --- Classification Results ---
    print(f"\n  Problem Type   : {result.get('problem_type', 'N/A')}")
    print(f"  Objective      : {result.get('objective', 'N/A')}")

    conf = result.get("confidence", 0)
    print(f"  Confidence     : {conf:.0%}")

    # --- Confidence Explanation ---
    num_assumptions = len(result.get("assumptions", []))
    num_warnings = len(result.get("warnings", []))
    print("\n  Confidence Breakdown:")
    print(f"    The AI model rated its own confidence at {conf:.0%}.")
    print(
        f"    Factors: {num_assumptions} assumption(s) made, {num_warnings} warning(s) flagged."
    )
    if num_assumptions == 0 and num_warnings == 0:
        print("    No assumptions or warnings — the problem was fully specified.")
    else:
        if num_assumptions > 0:
            print(
                "    More assumptions = more ambiguity in the input = lower confidence."
            )
        if num_warnings > 0:
            print(
                "    Warnings indicate potential data issues that may affect accuracy."
            )

    variables = result.get("decision_variables", [])
    print(f"\n  Decision Variables ({len(variables)} found):")
    for v in variables:
        print(
            f"    - {v.get('name', '?')} ({v.get('type', '?')}): {v.get('description', '')}"
        )

    constraints = result.get("constraints", [])
    print(f"\n  Constraints ({len(constraints)} found):")
    for c in constraints:
        print(f"    - {c.get('name', '?')} : {c.get('expression', '')}")

    assumptions = result.get("assumptions", [])
    if assumptions:
        print(f"\n  Assumptions ({len(assumptions)} made by AI):")
        for a in assumptions:
            impact = a.get("impact", "?").upper()
            aconf = a.get("confidence", 0)
            print(f"    [{impact}] {a.get('assumption', '')} (confidence: {aconf:.0%})")

    warnings = result.get("warnings", [])
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    ! {w}")

    print("\n" + "-" * 60)
    if conf >= 0.8:
        print("  RESULT: HIGH CONFIDENCE — classifier is sure about this problem.")
    elif conf >= 0.6:
        print("  RESULT: MEDIUM CONFIDENCE — review assumptions above before solving.")
    else:
        print("  RESULT: LOW CONFIDENCE — problem is vague. Add more details.")
    print("-" * 60 + "\n")

except Exception as e:
    print(f"\n  FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
