# -*- coding: utf-8 -*-
"""Checksums for the released parameter tables.

The checksums are computed on LF-normalized bytes so they are identical
on Windows (CRLF working tree) and on Linux/GitHub (LF).
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest().upper()


def test_optimal_kb_checksum():
    path = ROOT / "parameters" / "optimal_kb_M4.csv"
    assert _sha_lf(path).startswith("5ECE24C6B98680DB")
    assert len(path.read_text(encoding="utf-8-sig").splitlines()) == 121  # header + 120


def test_released_gen_correction_checksum():
    path = ROOT / "parameters" / "gen_correction_M4prod_final.csv"
    # Post-Feb-fix released table (NOT the older pre-Feb 3128FFB1 file).
    assert _sha_lf(path).startswith("38988788DB5CB660")


def test_pre2012_table_is_utf8_and_complete():
    path = ROOT / "parameters" / "pre2012_official.csv"
    text = path.read_text(encoding="utf-8-sig")
    assert "全国" in text
    for year in range(2005, 2013):
        assert str(year) in text
