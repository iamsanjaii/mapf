"""Kinds of removable obstacle: glyph, colour and the weight of one push step (cost = kappa * weight).

A kind is data, not code: adding one is one more row. Incident kinds (family D) are included so every obstacle
that can appear on a map has a weight; `needs_human` incidents are never pushed whatever their kind.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Kind:
    name: str
    glyph: str
    weight: float
    color: Tuple[float, float, float]


KINDS: Dict[str, Kind] = {k.name: k for k in (
    Kind("pallet", "L", 1.0, (0.85, 0.52, 0.18)),
    Kind("crate", "C", 0.5, (0.55, 0.38, 0.20)),
    Kind("shelf_unit", "S", 2.0, (0.45, 0.45, 0.62)),
    Kind("spill", "W", 0.5, (0.30, 0.62, 0.80)),
    Kind("debris", "R", 0.5, (0.62, 0.62, 0.55)),
    Kind("rack_damage", "K", 2.0, (0.70, 0.30, 0.55)),
)}
NAMES = tuple(KINDS)
GLYPHS: Dict[str, str] = {k.glyph: k.name for k in KINDS.values()}
DEFAULT_KIND = "pallet"


def weight(kind: Optional[str]) -> float:
    """Push-step weight of a kind; unknown or missing kinds count as a pallet."""
    return KINDS[kind].weight if kind in KINDS else KINDS[DEFAULT_KIND].weight


def code(kind: Optional[str]) -> int:
    """Small integer for a kind (0 = unknown), used in the belief's kind register."""
    return NAMES.index(kind) + 1 if kind in KINDS else 0


def name_of(c: int) -> Optional[str]:
    return NAMES[c - 1] if 1 <= c <= len(NAMES) else None
