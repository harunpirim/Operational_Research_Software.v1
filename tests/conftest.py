# The test_phase*.py files are interactive scripts meant to be run directly
# (python tests/test_phase2.py); they read stdin at import time, so pytest
# must not collect them.
collect_ignore_glob = ["test_phase*.py"]
