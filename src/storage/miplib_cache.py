"""
MIPLIB Cache Module

This module handles caching of MIPLIB problem solutions in a separate SQLite database.
It provides fast retrieval of previously solved MIPLIB instances to avoid redundant
computation of benchmark problems.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


class MIPLIBCache:
    """
    Manages cached solutions for MIPLIB problem instances.

    Uses a separate SQLite database to avoid interference with the main history database.
    """

    def __init__(self, db_path: str = "./data/miplib_cache.db"):
        """
        Initialize the MIPLIB cache database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Create table if it doesn't exist
        self._create_table()

    def _create_table(self):
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS miplib_cache (
                    instance_name TEXT PRIMARY KEY,
                    objective_value REAL,
                    solution_json TEXT,
                    solver_used TEXT,
                    original_solve_time REAL,
                    timestamp TEXT
                )
            """)
            conn.commit()

    def get(self, instance_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached solution for a MIPLIB instance.

        Args:
            instance_name: Name of the MIPLIB instance

        Returns:
            Solution dict with added cache_metadata, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM miplib_cache WHERE instance_name = ?", (instance_name,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        # Reconstruct solution dict from JSON
        try:
            solution = json.loads(row["solution_json"])
        except (json.JSONDecodeError, TypeError):
            return None

        # Add cache metadata
        solution["cache_metadata"] = {
            "original_solve_time": row["original_solve_time"],
            "solver_used": row["solver_used"],
            "timestamp": row["timestamp"],
            "instance_name": instance_name,
        }

        # Update status to indicate it came from cache
        solution["status"] = f"{solution.get('status', 'Unknown')} (from cache)"

        return solution

    def set(self, instance_name: str, solution: Dict[str, Any]):
        """
        Save a solution to the cache.

        Uses INSERT OR REPLACE to update existing entries.

        Args:
            instance_name: Name of the MIPLIB instance
            solution: Solution dictionary to cache
        """
        # Extract relevant fields
        objective_value = solution.get("objective_value")
        solver_used = solution.get("solver_name", "Unknown")
        solve_time = solution.get("solve_time", 0.0)
        timestamp = datetime.now().isoformat()

        # Create a copy without cache metadata to avoid recursion
        solution_copy = {k: v for k, v in solution.items() if k != "cache_metadata"}
        solution_json = json.dumps(solution_copy)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO miplib_cache
                (instance_name, objective_value, solution_json, solver_used,
                 original_solve_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    instance_name,
                    objective_value,
                    solution_json,
                    solver_used,
                    solve_time,
                    timestamp,
                ),
            )
            conn.commit()

    def clear(self):
        """Delete all cached entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM miplib_cache")
            conn.commit()

    def list_cached(self) -> List[Dict[str, Any]]:
        """
        List all cached instances with summary information.

        Returns:
            List of dicts with instance details
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT instance_name, objective_value, solver_used,
                       original_solve_time, timestamp
                FROM miplib_cache
                ORDER BY timestamp DESC
            """)
            rows = cursor.fetchall()

        return [
            {
                "instance_name": row["instance_name"],
                "objective_value": row["objective_value"],
                "solver_used": row["solver_used"],
                "solve_time": row["original_solve_time"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_cache_size(self) -> Dict[str, Any]:
        """
        Get information about the cache size and statistics.

        Returns:
            Dict with cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Count entries
            count = conn.execute("SELECT COUNT(*) FROM miplib_cache").fetchone()[0]

            # Get database file size
            try:
                file_size = os.path.getsize(self.db_path)
                size_mb = file_size / (1024 * 1024)
            except Exception:
                size_mb = 0

            # Get date range
            cursor = conn.execute("""
                SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest
                FROM miplib_cache
            """)
            row = cursor.fetchone()
            oldest = row[0] if row and row[0] else None
            newest = row[1] if row and row[1] else None

        return {
            "num_entries": count,
            "size_mb": round(size_mb, 2),
            "oldest_entry": oldest,
            "newest_entry": newest,
        }
