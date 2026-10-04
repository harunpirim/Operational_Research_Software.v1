"""
MIPLIB Loader — download and parse real-world MIP benchmark instances
from https://miplib.zib.de without manual setup.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.file_parser import FileParser

_BASE_URL = "https://miplib.zib.de"
_EASY_LIST_URL = f"{_BASE_URL}/downloads/easy-v18.test"
_INSTANCE_DL_URL = f"{_BASE_URL}/WebData/instances"
_INSTANCE_INFO_URL = f"{_BASE_URL}/instance_details_"

_DATA_DIR = Path("data")
_MIPLIB_DIR = _DATA_DIR / "miplib"
_EASY_CACHE = _DATA_DIR / "miplib_easy.txt"
_CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days in seconds


RECOMMENDED_INSTANCES: List[Dict[str, str]] = [
    {"name": "eil33-2", "desc": "Vehicle routing, 4516 vars"},
    {"name": "air04", "desc": "Airline crew scheduling, IP"},
    {"name": "cap6000", "desc": "Facility location, 6000 vars"},
    {"name": "neos-1067731", "desc": "Production planning"},
    {"name": "pb-grow22", "desc": "Blending problem"},
    {"name": "gen-ip054", "desc": "General integer programming"},
    {"name": "misc03", "desc": "Miscellaneous MIP, 160 vars"},
    {"name": "p0033", "desc": "Set packing, 33 vars"},
    {"name": "pk1", "desc": "Packing problem, 86 vars"},
    {"name": "sentoy", "desc": "Toy scheduling, 60 vars"},
]


class MIPLIBLoader:
    """
    Download, cache, and parse MIPLIB benchmark instances.

    All files are stored under ``./data/miplib/``.  The easy-instance
    list is cached in ``./data/miplib_easy.txt`` with a 7-day expiry.
    """

    def __init__(self):
        self._parser = FileParser()
        _MIPLIB_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def list_easy_instances(self) -> List[str]:
        """
        Return the list of "easy" MIPLIB instance names.

        Fetches from the MIPLIB website on first call and caches the
        result locally for 7 days.
        """
        if self._cache_is_fresh():
            return self._read_cache()

        import requests

        resp = requests.get(_EASY_LIST_URL, timeout=30)
        resp.raise_for_status()

        names = [
            line.strip()
            for line in resp.text.splitlines()
            if line.strip() and not line.startswith("#")
        ]

        _EASY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _EASY_CACHE.write_text("\n".join(names), encoding="utf-8")
        return names

    def download_instance(self, instance_name: str) -> Path:
        """
        Download ``{instance_name}.mps.gz`` from MIPLIB.

        Skips the download if the file already exists locally.
        Returns the local file path.
        """
        fname = f"{instance_name}.mps.gz"
        local_path = _MIPLIB_DIR / fname

        if local_path.exists():
            return local_path

        import requests

        url = f"{_INSTANCE_DL_URL}/{fname}"
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)

        return local_path

    def load_instance(self, instance_name: str) -> Dict[str, Any]:
        """
        Download (if needed) and parse a MIPLIB instance.

        Returns the same dict that ``FileParser.parse()`` produces for
        MPS files (type ``'mps'``), with ``file_path`` pointing to the
        local ``.mps.gz`` file.
        """
        local_path = self.download_instance(instance_name)
        result = self._parser.parse(
            str(local_path),
            filename=f"{instance_name}.mps.gz",
        )
        result["file_path"] = str(local_path)
        return result

    def get_instance_info(self, instance_name: str) -> Dict[str, Any]:
        """
        Scrape key statistics for an instance from the MIPLIB website.

        Returns a dict with keys: ``name``, ``variables``, ``constraints``,
        ``binary_variables``, ``best_known_objective``, ``status``,
        and ``url``.  Values are ``None`` when the page cannot be parsed.
        """
        import requests

        url = f"{_INSTANCE_INFO_URL}{instance_name}.html"
        info: Dict[str, Any] = {
            "name": instance_name,
            "variables": None,
            "constraints": None,
            "binary_variables": None,
            "best_known_objective": None,
            "status": None,
            "url": url,
        }

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            return info

        info["variables"] = self._extract_int(
            html, r"Variables\s*</td>\s*<td[^>]*>\s*([\d,]+)"
        )
        info["constraints"] = self._extract_int(
            html, r"Constraints\s*</td>\s*<td[^>]*>\s*([\d,]+)"
        )
        info["binary_variables"] = self._extract_int(
            html, r"Binaries\s*</td>\s*<td[^>]*>\s*([\d,]+)"
        )
        info["best_known_objective"] = self._extract_float(
            html,
            r"Objective Value\s*</td>\s*<td[^>]*>\s*([^\s<]+)",
        )
        status_match = re.search(r"Status\s*</td>\s*<td[^>]*>\s*(\w+)", html)
        if status_match:
            info["status"] = status_match.group(1)

        return info

    @staticmethod
    def get_recommended() -> List[Dict[str, str]]:
        """Return the curated list of small, solvable instances."""
        return list(RECOMMENDED_INSTANCES)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_is_fresh() -> bool:
        if not _EASY_CACHE.exists():
            return False
        age = time.time() - _EASY_CACHE.stat().st_mtime
        return age < _CACHE_MAX_AGE

    @staticmethod
    def _read_cache() -> List[str]:
        text = _EASY_CACHE.read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def _extract_int(html: str, pattern: str) -> Optional[int]:
        m = re.search(pattern, html)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                pass
        return None

    @staticmethod
    def _extract_float(html: str, pattern: str) -> Optional[float]:
        m = re.search(pattern, html)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        return None
