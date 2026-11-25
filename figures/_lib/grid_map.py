# -*- coding: utf-8 -*-
"""Canonical province -> regional-grid mapping and Inner Mongolia east/west split.

WHY THIS MODULE EXISTS
----------------------
Before 2026-10 the same mapping was hand-copied into 47 places across the paper
pipeline.  36 of those copies carried a superseded 蒙东 definition that included
锡林郭勒盟 (Xilingol), which moves ~33 TWh/yr of generation from 华北电网 to
东北电网 and flipped the 华北 share for 2023 from 29.3% to 25.2% -- contradicting
the paper's own Methods section and its Fig.9 numbers.  The audit
(``D:\\HongyuEnergy Copilot\\_run\\grid_mapping_audit\\REPORT.md``, recommendation #7)
therefore asked for one machine-checkable source of truth.

This module is that source of truth.  Import it; do not re-declare the maps.

RATIFIED DECISIONS (user, 2026-10)
----------------------------------
1. 蒙东电网 = 赤峰 / 通辽 / 呼伦贝尔 / 兴安盟  -- ONLY these four.
   锡林郭勒盟 belongs to 蒙西 -> 华北电网.
2. 山东 belongs to 华北电网.
3. 香港 is excluded from the study entirely: it is counted in no grid, appears in
   no legend, panel, ranking or share denominator.

INNER MONGOLIA COMPLETENESS
---------------------------
``verify_inner_mongolia_partition()`` asserts that the 12 盟市 of 内蒙古 are each
assigned to exactly one of 蒙东 / 蒙西 -- no overlap, no gap -- and raises
:class:`GridMapError` otherwise.  It is executed at import time (cheap, pure
Python) and can be called again from tests.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# 研究范围排除地区 (whole-region exclusion, applied at load time)
# --------------------------------------------------------------------------- #
EXCLUDED_REGIONS: frozenset[str] = frozenset({"香港"})
#: data-directory province keys (``<province_key>__<city_key>``) of the same regions
EXCLUDED_REGION_KEYS: frozenset[str] = frozenset({"xianggang"})

#: Regions that appear in the national administrative-unit tables (the 375 市 of
#: ``中国_市2.csv``) but are out of scope: no wind farms, no grid assignment.
#: They stay as rows in such tables, but must never be silently defaulted into a
#: grid -- that silent default is how 香港 ended up inside 南方电网 in
#: fig09/fig10/plot-sunburst before 2026-10-14.  (The sunbursts simply never plot
#: them: fig09/fig10 render 305/300 cities from the generation cube.)
OUT_OF_STUDY_REGIONS: frozenset[str] = frozenset({"香港", "澳门", "台湾"})


class GridMapError(AssertionError):
	"""Raised when the region mapping is not a valid partition."""


# --------------------------------------------------------------------------- #
# grid order / English names
# --------------------------------------------------------------------------- #
GRID_ORDER: tuple[str, ...] = ("华北", "东北", "华东", "华中", "西北", "西南", "南方")

GRID_NAME_EN: dict[str, str] = {
	"华北": "North China Grid",
	"东北": "Northeast China Grid",
	"华东": "East China Grid",
	"华中": "Central China Grid",
	"西北": "Northwest China Grid",
	"西南": "Southwest China Grid",
	"南方": "China Southern Power Grid",
}

# --------------------------------------------------------------------------- #
# province -> grid  (蒙东 / 蒙西 are the two pseudo-provinces of 内蒙古)
# --------------------------------------------------------------------------- #
PROVINCE_TO_GRID: dict[str, str] = {
	"北京": "华北",
	"天津": "华北",
	"河北": "华北",
	"山西": "华北",
	"山东": "华北",
	"辽宁": "东北",
	"吉林": "东北",
	"黑龙江": "东北",
	"蒙东": "东北",
	"上海": "华东",
	"江苏": "华东",
	"浙江": "华东",
	"安徽": "华东",
	"福建": "华东",
	"河南": "华中",
	"湖北": "华中",
	"湖南": "华中",
	"江西": "华中",
	"四川": "西南",
	"重庆": "西南",
	"西藏": "西南",
	"陕西": "西北",
	"甘肃": "西北",
	"青海": "西北",
	"宁夏": "西北",
	"新疆": "西北",
	"广东": "南方",
	"广西": "南方",
	"云南": "南方",
	"贵州": "南方",
	"海南": "南方",
	"蒙西": "华北",
	# "香港": excluded study-wide via EXCLUDED_REGIONS -- see the module docstring.
}

# --------------------------------------------------------------------------- #
# Inner Mongolia split
# --------------------------------------------------------------------------- #
#: canonical (bare, suffix-stripped) 盟市 names of 蒙东电网 -- the ratified four
INNER_MONGOLIA_EAST_CITIES: frozenset[str] = frozenset({"呼伦贝尔", "兴安", "通辽", "赤峰"})

#: every variant spelling seen in the inputs/pipelines for the four 蒙东 盟市
INNER_MONGOLIA_EAST_ALIASES: frozenset[str] = frozenset(
	{"呼伦贝尔", "呼伦贝尔市", "兴安", "兴安盟", "通辽", "通辽市", "赤峰", "赤峰市"}
)

#: canonical names of 蒙西 (everything else in 内蒙古)
INNER_MONGOLIA_WEST_CITIES: frozenset[str] = frozenset(
	{"呼和浩特", "包头", "乌海", "鄂尔多斯", "巴彦淖尔", "乌兰察布", "锡林郭勒", "阿拉善"}
)

#: every variant spelling seen for the 蒙西 盟市
INNER_MONGOLIA_WEST_ALIASES: frozenset[str] = frozenset(
	{"呼和浩特", "呼和浩特市", "包头", "包头市", "乌海", "乌海市", "鄂尔多斯", "鄂尔多斯市",
	 "巴彦淖尔", "巴彦淖尔市", "乌兰察布", "乌兰察布市", "锡林郭勒", "锡林郭勒盟",
	 "阿拉善", "阿拉善盟"}
)

#: the 12 盟市 of 内蒙古, canonical bare names (GB/T 2260 level-2 units)
INNER_MONGOLIA_CITIES_12: tuple[str, ...] = (
	"呼和浩特", "包头", "乌海", "赤峰", "通辽", "鄂尔多斯",
	"呼伦贝尔", "巴彦淖尔", "乌兰察布", "兴安", "锡林郭勒", "阿拉善",
)

#: fig1 / plot_nature use GB codes instead of names; 15E = 蒙东, everything else 15W
INNER_MONGOLIA_EAST_GB_CODES: frozenset[str] = frozenset(
	{"156150400", "156150500", "156150700", "156152200"}
)  # 150400 赤峰 / 150500 通辽 / 150700 呼伦贝尔 / 152200 兴安盟
INNER_MONGOLIA_WEST_GB_CODES: frozenset[str] = frozenset(
	{"156150100", "156150200", "156150300", "156150600", "156150800",
	 "156150900", "156152500", "156152900"}
)  # 150100 呼和浩特 / 150200 包头 / 150300 乌海 / 150600 鄂尔多斯 /
   # 150800 巴彦淖尔 / 150900 乌兰察布 / 152500 锡林郭勒 / 152900 阿拉善
   # (152500 锡林郭勒 is 蒙西 -- it must never appear in the EAST set)

#: data-directory city keys (``<province_key>__<city_key>``) of the four 蒙东 盟市.
#: This is what the generation pipelines match on -- they key off the directory
#: name, not the (Chinese) ``city_short_name`` attribute.
INNER_MONGOLIA_EAST_CITY_KEYS: frozenset[str] = frozenset(
	{"chifeng", "hulunbeier", "tongliao", "xingan"}
)
#: the complementary key set, kept for completeness / assertions
INNER_MONGOLIA_WEST_CITY_KEYS: frozenset[str] = frozenset(
	{"huhehaote", "baotou", "wuhai", "eerduosi", "bayannaoer",
	 "wulanchabu", "xilinguolei", "alashan"}
)

CITY_SUFFIXES: tuple[str, ...] = (
	"特别行政区", "维吾尔自治区", "壮族自治区", "回族自治区", "自治区",
	"自治州", "地区", "盟", "市", "省",
)


def normalize_city_label(value: object) -> str:
	"""Strip one administrative suffix (盟/市/地区/...) so lookups are consistent.

	``normalize_city_label("兴安盟") == "兴安"`` -- the mismatch between a set
	holding ``"兴安盟"`` and a lookup on the stripped ``"兴安"`` is exactly the bug
	that sent 兴安盟 into 华北电网 in ``table\\reshape_city_subgrid_table.py``.
	"""
	text = str(value).strip()
	for suffix in CITY_SUFFIXES:
		if text.endswith(suffix) and len(text) > len(suffix):
			return text[: -len(suffix)].strip()
	return text


def is_excluded_region(value: object) -> bool:
	"""True for a province label or province key that is out of study scope (香港)."""
	raw = str(value).strip()
	if raw in EXCLUDED_REGIONS or raw in EXCLUDED_REGION_KEYS:
		return True
	return normalize_city_label(raw) in EXCLUDED_REGIONS


def split_inner_mongolia_city(city: object) -> str:
	"""Return ``"蒙东"`` or ``"蒙西"`` for a 内蒙古 盟市 name (any suffix variant).

	Raises :class:`GridMapError` for a name that is not one of the 12 盟市, so a
	new/renamed unit can never be silently defaulted into the wrong grid.
	"""
	base = normalize_city_label(city)
	if base in INNER_MONGOLIA_EAST_CITIES or str(city).strip() in INNER_MONGOLIA_EAST_ALIASES:
		return "蒙东"
	if base in INNER_MONGOLIA_WEST_CITIES or str(city).strip() in INNER_MONGOLIA_WEST_ALIASES:
		return "蒙西"
	raise GridMapError(
		f"内蒙古 city {city!r} (normalised {base!r}) is not one of the 12 盟市; "
		f"it cannot be silently defaulted into 蒙东/蒙西"
	)


def province_group_to_grid(province_label: object) -> str:
	"""Map a province (or 蒙东/蒙西) label to its grid; raise if unmapped."""
	label = normalize_city_label(province_label)
	grid = PROVINCE_TO_GRID.get(label)
	if grid is None:
		if is_excluded_region(label):
			raise GridMapError(f"{province_label!r} is excluded from the study (EXCLUDED_REGIONS)")
		raise GridMapError(f"province {province_label!r} (normalised {label!r}) has no grid mapping")
	return grid


def grid_to_provinces() -> dict[str, tuple[str, ...]]:
	"""Reverse view, ordered by :data:`GRID_ORDER` (fig09/fig10 sunburst input)."""
	reverse: dict[str, list[str]] = {grid: [] for grid in GRID_ORDER}
	for province, grid in PROVINCE_TO_GRID.items():
		reverse[grid].append(province)
	return {grid: tuple(reverse[grid]) for grid in GRID_ORDER}


def verify_inner_mongolia_partition() -> None:
	"""Assert 内蒙古's 12 盟市 are each in exactly one of 蒙东 / 蒙西.

	Raises :class:`GridMapError` on any overlap, gap, or unknown member.
	Called at import time.
	"""
	east, west = INNER_MONGOLIA_EAST_CITIES, INNER_MONGOLIA_WEST_CITIES
	overlap = east & west
	if overlap:
		raise GridMapError(f"内蒙古 overlap: {sorted(overlap)} appear in BOTH 蒙东 and 蒙西")
	covered = east | west
	missing = set(INNER_MONGOLIA_CITIES_12) - covered
	extra = covered - set(INNER_MONGOLIA_CITIES_12)
	if missing:
		raise GridMapError(f"内蒙古 gap: {sorted(missing)} are in neither 蒙东 nor 蒙西")
	if extra:
		raise GridMapError(f"内蒙古 unknown member(s): {sorted(extra)} are not among the 12 盟市")
	if len(INNER_MONGOLIA_CITIES_12) != 12:
		raise GridMapError("INNER_MONGOLIA_CITIES_12 must list exactly 12 盟市")
	for city in INNER_MONGOLIA_CITIES_12:
		split_inner_mongolia_city(city)  # must not raise
	# every alias must land in the same bucket as its canonical name
	for alias in INNER_MONGOLIA_EAST_ALIASES:
		if split_inner_mongolia_city(alias) != "蒙东":
			raise GridMapError(f"alias {alias!r} does not resolve to 蒙东")
	for alias in INNER_MONGOLIA_WEST_ALIASES:
		if split_inner_mongolia_city(alias) != "蒙西":
			raise GridMapError(f"alias {alias!r} does not resolve to 蒙西")
	# 锡林郭勒 must never be east again (the 2026-10 regression this module guards)
	if split_inner_mongolia_city("锡林郭勒盟") != "蒙西":
		raise GridMapError("锡林郭勒盟 must be 蒙西 (-> 华北电网)")


def verify_city_key_partition() -> None:
	"""The 蒙东/蒙西 *data-directory keys* must also partition the 12 盟市 exactly."""
	east, west = INNER_MONGOLIA_EAST_CITY_KEYS, INNER_MONGOLIA_WEST_CITY_KEYS
	overlap = east & west
	if overlap:
		raise GridMapError(f"city-key overlap: {sorted(overlap)}")
	if len(east) + len(west) != len(INNER_MONGOLIA_CITIES_12):
		raise GridMapError(
			f"city keys cover {len(east) + len(west)} of {len(INNER_MONGOLIA_CITIES_12)} 盟市"
		)
	if "xilinguolei" in east:
		raise GridMapError("xilinguolei (锡林郭勒) must NOT be in INNER_MONGOLIA_EAST_CITY_KEYS")


def verify_grid_coverage() -> None:
	"""Every Chinese province mainland + 蒙东/蒙西 must appear exactly once."""
	expected = {
		"北京", "天津", "河北", "山西", "山东", "辽宁", "吉林", "黑龙江",
		"上海", "江苏", "浙江", "安徽", "福建", "河南", "湖北", "湖南", "江西",
		"四川", "重庆", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
		"广东", "广西", "云南", "贵州", "海南", "蒙东", "蒙西",
	}
	present = set(PROVINCE_TO_GRID)
	if present != expected:
		raise GridMapError(
			f"province set mismatch; missing={sorted(expected - present)} "
			f"extra={sorted(present - expected)}"
		)
	unknown = set(PROVINCE_TO_GRID.values()) - set(GRID_ORDER)
	if unknown:
		raise GridMapError(f"grid names not in GRID_ORDER: {sorted(unknown)}")
	counts = {grid: 0 for grid in GRID_ORDER}
	for grid in PROVINCE_TO_GRID.values():
		counts[grid] += 1
	if sum(counts.values()) != len(PROVINCE_TO_GRID):
		raise GridMapError("a province is mapped to more than one grid")


verify_inner_mongolia_partition()
verify_city_key_partition()
verify_grid_coverage()

if __name__ == "__main__":
	print("grid_map OK")
	print(f"  grids            : {len(GRID_ORDER)}")
	print(f"  provinces/伪省   : {len(PROVINCE_TO_GRID)}")
	print(f"  蒙东 ({len(INNER_MONGOLIA_EAST_CITIES)}): {sorted(INNER_MONGOLIA_EAST_CITIES)}")
	print(f"  蒙西 ({len(INNER_MONGOLIA_WEST_CITIES)}): {sorted(INNER_MONGOLIA_WEST_CITIES)}")
	print(f"  excluded         : {sorted(EXCLUDED_REGIONS)}")
	for grid, provinces in grid_to_provinces().items():
		print(f"  {grid}: {provinces}")
