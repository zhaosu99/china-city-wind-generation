# -*- coding: utf-8 -*-
"""Central path configuration for the paper-figure plotting project.

This module is the **single** place where absolute paths live.  Every figure
script imports from here instead of hard-coding ``D:\\HKU PhD\\wind data
paper\\...``.

Environment overrides
---------------------
===========================  ==================================================
``WINDPAPER_PLOT_DATA``      Pipeline results root.
                             Default: ``D:\\HKU PhD\\wind data paper\\_hpc_download``
``WINDPAPER_PAPER_ROOT``     Paper working directory (inventory CSV, capacity CSV).
                             Default: ``D:\\HKU PhD\\wind data paper``
``WINDPAPER_ADMIN_DATA``     Administrative boundary data (geojson / 九段线 shp).
                             Default: ``D:\\Data\\admin``  *(see ADMIN note below)*
``WINDPAPER_FONT_DIR``       Font directory. Default: ``C:\\Windows\\Fonts``
``WINDPAPER_TURBINE_CSV``    ``D:\\Data\\turbines\\wind_turbines.csv``
``WINDPAPER_PLOT_OUT``       Default output root (defaults to *this* directory).
``WINDPAPER_KB_ADJUST_ROOT`` The ``kb_adjust`` generation cube used by Fig.7.
                             Default: ``<data root>\\calibrated_gen\\gen\\city``
                             *(see KB_ADJUST note below)*
===========================  ==================================================

KB_ADJUST note
--------------
The original Fig.7 script compared three generation sources:
``no_kb``, ``kb_adjust`` and ``final``.  The historical ``kb_adjust`` input was
the cluster-only intermediate ``<paper root>\\farm_aggr_with_kb\\city``, which is
**not** part of the downloadable pipeline products and does not exist locally.
Its semantically equivalent product in the current pipeline is the
KB-adjusted generation cube ``<data root>\\calibrated_gen\\gen\\city`` (the same
cube Fig.11 uses as the "after calibration" state), so ``KB_ADJUST_GEN_DIR``
resolves there.  ``WINDPAPER_KB_ADJUST_ROOT`` pins a different directory (for
example the original ``farm_aggr_with_kb\\city`` when it is available);
alternatively point ``WINDPAPER_PLOT_DATA`` at a root that contains it.

Discovery / resolution
----------------------
Some inputs are looked up through ``resolve_first_existing`` with a small list
of candidate locations, so the project works both on this workstation and on a
re-cloned machine.  A missing input raises ``FileNotFoundError`` with the full
candidate list; ``run_all.py --dry-run`` reports the same list without
crashing.

ADMIN note
----------
The original scripts pointed at ``D:\\Data\\admin\\``.  That directory does not
exist on this machine; the same files live under
``D:\\网络数据爬取\\_数据\\_外挂参考\\admin\\``.  ``ADMIN_DIR`` therefore defaults
to ``D:\\Data\\admin`` (so the project stays faithful to the original layout)
and the resolvers fall back to the real locations.  Set
``WINDPAPER_ADMIN_DATA`` to pin a specific directory.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PLOT_ROOT",
    "PAPER_ROOT",
    "ADMIN_DIR",
    "FONT_DIR",
    "DATA_ROOT",
    "DISPLAY_DATA_ROOT",
    "FINAL_GEN_DIR",
    "CALIBRATED_GEN_DIR",
    "NO_KB_GEN_DIR",
    "KB_ADJUST_GEN_DIR",
    "KB_ADJUST_GEN_DIR_CANDIDATES",
    "FARM_AGGR_WITH_KB_GEN_DIR",
    "OPTIMAL_KB_DIR",
    "OPTIMAL_KB_CSV",
    "OPTIMAL_KB_PLOTS_DIR",
    "KB_VALIDATION_DIR",
    "KB_VALIDATION_DETAIL_CSV",
    "KB_VALIDATION_SUMMARY_CSV",
    "FARM_POWER_CURVES_DIR",
    "PROV_GEN_FULL_SMAPE_DIR",
    "PROV_W_SMAPE_ENUMS_DIR",
    "PROV_GEN_ENUMS_DIR",
    "PROV_W_COMPARE_DIR",
    "ADMIN_GEOJSON_CANDIDATES",
    "NINE_DOTTED_LINE_CANDIDATES",
    "TURBINE_CSV_CANDIDATES",
    "INVENTORY_CSV_CANDIDATES",
    "CITY_MONTHLY_CAPACITY_CANDIDATES",
    "FONT_CANDIDATES",
    "WINDPAPER_PLOT_DATA",
    "WINDPAPER_PAPER_ROOT",
    "WINDPAPER_ADMIN_DATA",
    "WINDPAPER_FONT_DIR",
    "WINDPAPER_TURBINE_CSV",
    "WINDPAPER_PLOT_OUT",
    "WINDPAPER_KB_ADJUST_ROOT",
    "env_path",
    "first_existing",
    "select_existing_roots",
    "resolve_first_existing",
    "resolve_admin_geojson",
    "resolve_nine_dotted_line",
    "describe_missing",
]


# --------------------------------------------------------------------------- #
# 1. roots
# --------------------------------------------------------------------------- #
PLOT_ROOT = Path(__file__).resolve().parent

_env_data = os.environ.get("WINDPAPER_PLOT_DATA")
if _env_data:
    DATA_ROOT = Path(_env_data).expanduser()
else:
    DATA_ROOT = PLOT_ROOT.parents[1] / "results"

_env_paper = os.environ.get("WINDPAPER_PAPER_ROOT")
if _env_paper:
    PAPER_ROOT = Path(_env_paper).expanduser()
else:
    PAPER_ROOT = PLOT_ROOT.parents[1]

_env_admin = os.environ.get("WINDPAPER_ADMIN_DATA")
if _env_admin:
    ADMIN_DIR = Path(_env_admin).expanduser()
else:
    ADMIN_DIR = PLOT_ROOT.parents[1] / "data" / "admin"

_env_font = os.environ.get("WINDPAPER_FONT_DIR")
if _env_font:
    FONT_DIR = Path(_env_font).expanduser()
else:
    FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

# Root used when a script references the *outer* download folder that contains
# ``final_gen/`` as a subdirectory (stage-1 layout) rather than the results
# root that contains ``city/`` (stage-2 layout, used by the figure scripts).
DISPLAY_DATA_ROOT = DATA_ROOT.parent


# --------------------------------------------------------------------------- #
# 2. pipeline product directories
# --------------------------------------------------------------------------- #
FINAL_GEN_DIR = DATA_ROOT / "final_gen" / "city"
CALIBRATED_GEN_DIR = DATA_ROOT / "calibrated_gen" / "gen" / "city"
NO_KB_GEN_DIR = DATA_ROOT / "no_kb_gen" / "city"

# Fig.7's ``kb_adjust`` source.  The historical cluster-only intermediate
# ``farm_aggr_with_kb/city`` does not exist in the local download; its
# semantically equivalent product is the KB-adjusted generation cube
# ``calibrated_gen/gen/city``.  Keeping the mapping here (instead of in the
# figure script) makes it visible and overridable: set
# ``WINDPAPER_KB_ADJUST_ROOT`` to pin any other directory.
KB_ADJUST_ENV_VAR = "WINDPAPER_KB_ADJUST_ROOT"
FARM_AGGR_WITH_KB_GEN_DIR = DATA_ROOT / "farm_aggr_with_kb" / "city"
KB_ADJUST_GEN_DIR_CANDIDATES = (
    CALIBRATED_GEN_DIR,          # current pipeline equivalent (preferred)
    FARM_AGGR_WITH_KB_GEN_DIR,   # legacy cluster-only intermediate, if present
)
_env_kb_adjust = os.environ.get(KB_ADJUST_ENV_VAR)
if _env_kb_adjust:
    KB_ADJUST_GEN_DIR = Path(_env_kb_adjust).expanduser()
else:
    # ``first_existing`` is defined in section 5; resolve with an inline scan so
    # this constant is final by the time any figure script imports it.
    KB_ADJUST_GEN_DIR = next(
        (p for p in KB_ADJUST_GEN_DIR_CANDIDATES if p.exists()),
        KB_ADJUST_GEN_DIR_CANDIDATES[0],
    )

OPTIMAL_KB_DIR = DATA_ROOT / "optimal_kb"
OPTIMAL_KB_CSV = OPTIMAL_KB_DIR / "optimal_kb.csv"
OPTIMAL_KB_PLOTS_DIR = OPTIMAL_KB_DIR / "plots"

KB_VALIDATION_DIR = DATA_ROOT / "kb_validation"
KB_VALIDATION_DETAIL_CSV = KB_VALIDATION_DIR / "kb_validation_detail.csv"
KB_VALIDATION_SUMMARY_CSV = KB_VALIDATION_DIR / "kb_validation_summary.csv"

FARM_POWER_CURVES_DIR = DATA_ROOT / "farm_power_curves"
PROV_GEN_FULL_SMAPE_DIR = DATA_ROOT / "prov_gen_full_smape"
PROV_W_SMAPE_ENUMS_DIR = DATA_ROOT / "prov_w_smape_enums"
PROV_GEN_ENUMS_DIR = DATA_ROOT / "prov_gen_enums"
PROV_W_COMPARE_DIR = DATA_ROOT / "prov_w_compare"


# --------------------------------------------------------------------------- #
# 3. candidate lists (checked in order)
# --------------------------------------------------------------------------- #
_ADMIN_ALT = PAPER_ROOT / "data" / "admin"

ADMIN_GEOJSON_CANDIDATES = {
    "中国_省.geojson": (
        ADMIN_DIR / "中国_省.geojson",
        _ADMIN_ALT / "中国_省.geojson",
        Path(r"D:\HongyuEnergy\EnerMap\res\中国_省.geojson"),
        Path(r"D:\HongyuEnergy\OmniNet\res\common\中国_省.geojson"),
        PLOT_ROOT / "_mapdata" / "中国_省.geojson",
    ),
    "中国_市.geojson": (
        ADMIN_DIR / "中国_市.geojson",
        _ADMIN_ALT / "中国_市.geojson",
        Path(r"D:\HongyuEnergy\OmniNet\res\common\中国_市.geojson"),
        PLOT_ROOT / "_mapdata" / "中国_市.geojson",
    ),
    "中国_市2.geojson": (
        ADMIN_DIR / "中国_市2.geojson",
        _ADMIN_ALT / "中国_市2.geojson",
        PLOT_ROOT / "_mapdata" / "中国_市2.geojson",
    ),
}

NINE_DOTTED_LINE_CANDIDATES = (
    ADMIN_DIR / "九段线" / "china_nine_dotted_line.shp",
    _ADMIN_ALT / "九段线" / "china_nine_dotted_line.shp",
    _ADMIN_ALT / "九段线" / "九段线" / "china_nine_dotted_line.shp",
    Path(r"D:\HongyuEnergy\EnerMap\res\china_nine_dotted_line.geojson"),
    PLOT_ROOT / "_mapdata" / "九段线" / "china_nine_dotted_line.shp",
)

_env_turbine = os.environ.get("WINDPAPER_TURBINE_CSV")
TURBINE_CSV_CANDIDATES = tuple(
    p
    for p in (
        Path(_env_turbine).expanduser() if _env_turbine else None,
        PAPER_ROOT / "data" / "turbines" / "wind_turbines.csv",
        PAPER_ROOT / "DATA_INPUT" / "wind_turbines.csv",
    )
    if p is not None
)

INVENTORY_CSV_CANDIDATES = (
    PAPER_ROOT / "wind_farms_inventory_till_2023.csv",
    PAPER_ROOT / "DATA_INPUT" / "wind_farms_inventory_till_2023.csv",
)

CITY_MONTHLY_CAPACITY_CANDIDATES = (
    PAPER_ROOT / "city_monthly_wind_capacity.csv",
    PAPER_ROOT / "DATA_INPUT" / "city_monthly_wind_capacity.csv",
)

FONT_CANDIDATES = {
    "msyh": (FONT_DIR / "msyh.ttc", Path(r"C:\Windows\Fonts\msyh.ttc")),
    "msyhbd": (FONT_DIR / "msyhbd.ttc", Path(r"C:\Windows\Fonts\msyhbd.ttc")),
    "simhei": (FONT_DIR / "simhei.ttf", Path(r"C:\Windows\Fonts\simhei.ttf")),
    "simsun": (FONT_DIR / "simsun.ttc", Path(r"C:\Windows\Fonts\simsun.ttc")),
    "arial": (FONT_DIR / "arial.ttf", Path(r"C:\Windows\Fonts\arial.ttf")),
    "arialbd": (FONT_DIR / "arialbd.ttf", Path(r"C:\Windows\Fonts\arialbd.ttf")),
}


# --------------------------------------------------------------------------- #
# 4. environment variables, re-exported for convenience in README/runner
# --------------------------------------------------------------------------- #
WINDPAPER_PLOT_DATA = "WINDPAPER_PLOT_DATA"
WINDPAPER_PAPER_ROOT = "WINDPAPER_PAPER_ROOT"
WINDPAPER_ADMIN_DATA = "WINDPAPER_ADMIN_DATA"
WINDPAPER_FONT_DIR = "WINDPAPER_FONT_DIR"
WINDPAPER_TURBINE_CSV = "WINDPAPER_TURBINE_CSV"
WINDPAPER_PLOT_OUT = "WINDPAPER_PLOT_OUT"
WINDPAPER_KB_ADJUST_ROOT = KB_ADJUST_ENV_VAR


# --------------------------------------------------------------------------- #
# 5. helpers
# --------------------------------------------------------------------------- #
def env_path(name: str, default: Path | str | None = None) -> Path | None:
    """Return an environment variable as a :class:`Path` (or ``default``)."""
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser()
    if default is None:
        return None
    return Path(default)


def first_existing(candidates, *, kind: str = "path") -> Path | None:
    """Return the first candidate that exists on disk, else ``None``."""
    if isinstance(candidates, (str, Path)):
        candidates = (candidates,)
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def select_existing_roots(roots: dict) -> tuple[dict, dict]:
    """Split ``{source_key: path}`` into ``(available, missing)`` dictionaries.

    Used by figure scripts whose sources are *optional*: a source whose directory
    is absent must be dropped (with a printed warning) instead of aborting the
    figure.  Existence — not content — is what is tested here.
    """
    available: dict = {}
    missing: dict = {}
    for key, value in roots.items():
        path = Path(value)
        (available if path.exists() else missing)[key] = path
    return available, missing


def resolve_first_existing(candidates, *, kind: str = "path", hint: str = "") -> Path:
    """Like :func:`first_existing` but raises a helpful error when nothing hits.

    The error message lists every candidate that was tried so that
    ``run_all.py --dry-run`` output is actionable.
    """
    found = first_existing(candidates, kind=kind)
    if found is not None:
        return found
    tried = "\n".join(f"    - {Path(c)}" for c in candidates if c is not None)
    extra = f"\n  hint: {hint}" if hint else ""
    raise FileNotFoundError(
        f"No existing {kind} found.  Tried:\n{tried}{extra}\n"
        f"  (override with the {WINDPAPER_ADMIN_DATA!r} / {WINDPAPER_PLOT_DATA!r} "
        f"environment variables)"
    )


def resolve_admin_geojson(name: str) -> Path:
    """Resolve one of the administrative geojson layers by file name."""
    if name not in ADMIN_GEOJSON_CANDIDATES:
        raise KeyError(
            f"Unknown admin geojson {name!r}; known: "
            f"{sorted(ADMIN_GEOJSON_CANDIDATES)}"
        )
    return resolve_first_existing(
        ADMIN_GEOJSON_CANDIDATES[name],
        kind=f"admin geojson {name}",
        hint=f"set {WINDPAPER_ADMIN_DATA} to the directory holding {name}",
    )


def resolve_nine_dotted_line() -> Path:
    """Resolve the nine-dotted-line shapefile (南沙群岛 inset)."""
    return resolve_first_existing(
        NINE_DOTTED_LINE_CANDIDATES,
        kind="九段线 shapefile",
        hint=f"set {WINDPAPER_ADMIN_DATA} to the parent of 九段线/",
    )


def describe_missing(candidates) -> str:
    """Human-readable list of the candidate paths that were tried."""
    if isinstance(candidates, (str, Path)):
        candidates = (candidates,)
    return "; ".join(str(Path(c)) for c in candidates if c is not None)
