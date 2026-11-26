# -*- coding: utf-8 -*-
"""Checksums for the released parameter tables."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_optimal_kb_checksum():
    path = ROOT / "parameters" / "optimal_kb_M4.csv"
    assert _sha(path).startswith("D4778452F54582EE")
    assert len(path.read_text(encoding="utf-8-sig").splitlines()) == 121  # header + 120


def test_released_gen_correction_checksum():
    path = ROOT / "parameters" / "gen_correction_M4prod_final.csv"
    # Post-Feb-fix released table (NOT the older pre-Feb 3128FFB1 file).
    assert _sha(path).startswith("8CBDD8B3111EDFE0")


def test_pre2012_table_is_utf8_and_complete():
    path = ROOT / "parameters" / "pre2012_official.csv"
    text = path.read_text(encoding="utf-8-sig")
    assert "全国" in text
    for year in range(2005, 2013):
        assert str(year) in text
