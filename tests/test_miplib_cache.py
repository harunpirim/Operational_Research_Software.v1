"""
Tests for the MPS solution cache: entries are keyed by model content, not by name.
"""

import gzip
import sqlite3

import pytest

from src.ingestion.file_parser import FileParser
from src.modeling.model_generator import ModelGenerator
from src.solvers.solver_interface import SolverInterface
from src.storage.miplib_cache import MIPLIBCache

# Same NAME line, different data: optima are 1000 (TABLES=20) and 1500 (TABLES=30).
_MPS_TEMPLATE = """NAME          PROBLEM
OBJSENSE
    MAX
ROWS
 N  PROFIT
 L  LABOR
 L  WOOD
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    CHAIRS    PROFIT         30.0   LABOR           2.0
    CHAIRS    WOOD            1.0
    TABLES    PROFIT         50.0   LABOR           3.0
    TABLES    WOOD            4.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       LABOR          {labor}   WOOD            {wood}
BOUNDS
 UP BND       CHAIRS         25.0
ENDATA
"""


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """MIPLIBCache writes ./data/miplib_cache.db; keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


def _write_mps(path, labor, wood, gz=False):
    text = _MPS_TEMPLATE.format(labor=labor, wood=wood)
    if gz:
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


def _solve_file(path, solver="pulp"):
    parsed = FileParser().parse(str(path), filename=path.name)
    generator = ModelGenerator()
    generator.api_client = None
    model, problem_data = generator.generate_from_mps(parsed)
    solver_interface = SolverInterface(solver, problem_data=problem_data)
    return solver_interface.solve(model), problem_data, solver_interface


def test_models_with_same_name_do_not_share_cached_solution(tmp_path):
    a = _write_mps(tmp_path / "a.mps", labor=60, wood=80)
    b = _write_mps(tmp_path / "b.mps", labor=90, wood=120)

    sol_a, pdata_a, _ = _solve_file(a)
    sol_b, pdata_b, _ = _solve_file(b)

    assert pdata_a["instance_name"] == pdata_b["instance_name"] == "PROBLEM"
    assert pdata_a["cache_key"] != pdata_b["cache_key"]
    assert sol_a["objective_value"] == pytest.approx(1000)
    assert sol_b["objective_value"] == pytest.approx(1500)
    assert "cache_metadata" not in sol_b


def test_same_model_is_served_from_cache_even_when_gzipped(tmp_path):
    plain = _write_mps(tmp_path / "m.mps", labor=60, wood=80)
    gzipped = _write_mps(tmp_path / "copy.mps.gz", labor=60, wood=80, gz=True)

    first, pdata_plain, _ = _solve_file(plain)
    second, pdata_gz, _ = _solve_file(gzipped)

    assert pdata_plain["cache_key"] == pdata_gz["cache_key"]
    assert "cache_metadata" not in first
    assert "cache_metadata" in second
    assert second["status"].endswith("(from cache)")
    assert second["objective_value"] == pytest.approx(1000)


def test_local_mps_file_is_not_labelled_miplib(tmp_path):
    _, problem_data, _ = _solve_file(_write_mps(tmp_path / "m.mps", labor=60, wood=80))
    assert problem_data["source"] == "mps_file"


def test_miplib_download_is_labelled_miplib(tmp_path, monkeypatch):
    from src.ingestion.miplib_loader import MIPLIBLoader

    local = _write_mps(tmp_path / "p0033.mps.gz", labor=60, wood=80, gz=True)
    monkeypatch.setattr(MIPLIBLoader, "download_instance", lambda self, name: local)

    parsed = MIPLIBLoader().load_instance("p0033")
    generator = ModelGenerator()
    generator.api_client = None
    _, problem_data = generator.generate_from_mps(parsed)

    assert problem_data["source"] == "miplib"
    assert problem_data["instance_name"] == "p0033"
    assert problem_data["cache_key"]


def test_will_use_cache_matches_solve_for_other_solver_family(tmp_path):
    path = _write_mps(tmp_path / "m.mps", labor=60, wood=80)
    _solve_file(path)  # caches a PuLP solution

    _, problem_data, _ = _solve_file(path)
    pulp_si = SolverInterface("pulp", problem_data=problem_data)
    cvxpy_si = SolverInterface("cvxpy", problem_data=problem_data)

    assert pulp_si.will_use_cache is True
    assert cvxpy_si.will_use_cache is False


def test_legacy_name_keyed_table_is_left_untouched():
    """Old databases keep their table and rows; the new cache ignores them."""
    import os

    os.makedirs("data", exist_ok=True)
    with sqlite3.connect("data/miplib_cache.db") as conn:
        conn.execute(
            "CREATE TABLE miplib_cache (instance_name TEXT PRIMARY KEY, objective_value REAL,"
            " solution_json TEXT, solver_used TEXT, original_solve_time REAL, timestamp TEXT)"
        )
        conn.execute(
            "INSERT INTO miplib_cache VALUES ('PROBLEM', 1, '{}', 'x', 0.1, '2026-01-01')"
        )

    cache = MIPLIBCache()

    assert cache.list_cached() == []
    assert cache.get("PROBLEM") is None
    with sqlite3.connect("data/miplib_cache.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM miplib_cache").fetchone()[0] == 1
