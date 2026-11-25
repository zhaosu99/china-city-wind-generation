from __future__ import annotations

import argparse
import ast
from typing import Any, Optional, Union


def parse_list(val: Any, typ=str):
    if val is None:
        return None
    if isinstance(val, list):
        return val
    return [typ(x.strip()) for x in str(val).split(",") if x.strip()]


def parse_tf(val: Any) -> bool:
    """将常见字符串布尔值转为 bool。"""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        text = val.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    raise argparse.ArgumentTypeError(f"布尔参数只能为 True/False，收到: {val}")


def parse_none(val: Any) -> Any:
    """将字符串 'None'（不区分大小写）转为 Python None。"""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() == "none":
        return None
    return val


def parse_csv_list(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        return None
    return items


def parse_select_period(value: Optional[str]) -> Optional[Union[str, int, list[Union[str, int]]]]:
    value = parse_none(value)
    if value is None:
        return None

    text = str(value).strip()
    lowered = text.lower()
    if lowered == "all":
        return "all"

    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) == 0:
        raise argparse.ArgumentTypeError("select_period 不能为空")

    def _cast_one(token: str) -> Union[str, int]:
        t = token.lower()
        if t in {"spring", "summer", "autumn", "winter"}:
            return t
        try:
            return int(t)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"非法 select_period 元素: {token}，应为季节名或月份整数"
            ) from exc

    if len(parts) == 1:
        return _cast_one(parts[0])
    return [_cast_one(item) for item in parts]


def parse_kb_info(value: Optional[str]) -> Optional[Union[str, tuple[float, float]]]:
    value = parse_none(value)
    if value is None:
        return None

    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return text

    if isinstance(parsed, tuple) and len(parsed) == 2:
        try:
            return (float(parsed[0]), float(parsed[1]))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError("kb_info tuple 必须是可转换为 float 的 (k, b)") from exc

    return text


def parse_gen_adjust_info(value: Optional[str]) -> Optional[dict]:
    value = parse_none(value)
    if value is None:
        return None

    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except Exception as exc:
        raise argparse.ArgumentTypeError("gen_adjust_info 必须是可 literal_eval 的 dict 字符串") from exc

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("gen_adjust_info 解析后必须为 dict")
    return parsed
