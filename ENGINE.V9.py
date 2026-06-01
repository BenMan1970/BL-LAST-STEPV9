"""
BLUESTAR ENGINE v10 — Hybrid Absolute/Cross-Sectional (V4 architecture)
========================================================================
Single-file monolithic engine. Source of truth = merged JSON.
Reuses the robust v9 backbone (Pydantic validation, graceful degraded mode,
tiered calendar, synthetic-ATR cap, preflight, audit trail) and REPLACES the
additive 13-component scoring with the V4 factor engine:

  - 7 orthogonal bounded factors (F1 HWA, F2 RMG, F3 EXT, F4 TRG, F5 XCTX,
    F6 THEME, F7 MACRO), each scored in [0,1] with intrinsic meaning.
  - Equal-weight absolute mean -> CONVICTION (absolute thresholds AAA..B).
  - Cross-sectional quantile used ONLY for tie-breaking and diversification,
    NEVER for conviction (the hybrid invariant).
  - Contradictions C1..C5 (minor/major) feeding the grading grid.
  - Hard vetos -> caps (synthetic ATR -> BBB, macro risk -> AA) -> grid.
  - Diversification by risk cluster BEFORE top-N (corrects v9 greedy tail).

Pipeline is linear and pure; every section is testable in isolation.

Usage:
  python v10.py --merged merge.json --calendar-json calendar.json -o report.html
  from v10 import run_pipeline
  html = run_pipeline(merged_path="merge.json", calendar_json_path="calendar.json")
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional

import jinja2
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger("bluestar.v10")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — OPTIONAL upstream import (graceful fallback, never blocking)
# ════════════════════════════════════════════════════════════════════════════
try:  # pragma: no cover
    from merge_appbackup import Direction as _UpstreamDirection  # noqa: F401
    _HAS_UPSTREAM = True
except Exception:
    _HAS_UPSTREAM = False


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS  (ported from v9; Conviction extended with BB/B)
# ════════════════════════════════════════════════════════════════════════════
class Direction(str, Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class ImpactLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EventTier(str, Enum):
    S = "S"      # NFP, FOMC, CPI, Rate Decision
    A = "A"      # GDP, PMI, ADP, PCE
    B = "B"      # Speeches / press conf
    NONE = "NONE"


class GateCode(str, Enum):
    PASS = "PASS"
    G0_SCHEMA_ASSET_ERROR = "SCHEMA_ASSET_ERROR"
    G1_CAL_BLACKOUT = "CAL_BLACKOUT"
    G2_LOW_QUALITY = "LOW_QUALITY"
    G3_NO_DIRECTION = "NO_DIRECTION"
    G4_LOW_CONSENSUS = "LOW_CONSENSUS"
    G5_NO_ATR = "NO_ATR"


class Conviction(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"


class CalStatus(str, Enum):
    OK = "OK"
    BLACKOUT = "BLACKOUT"
    PROXIMITY = "PROXIMITY"
    WATCH = "WATCH"


# Ordinal rank for diversification preference (higher = stronger conviction).
_CONVICTION_ORDINAL: Mapping[str, int] = MappingProxyType({
    "AAA": 6, "AA": 5, "A": 4, "BBB": 3, "BB": 2, "B": 1,
})


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPERS  (ported verbatim from v9)
# ════════════════════════════════════════════════════════════════════════════
def _dir_eq(a: Any, b: Any) -> bool:
    av = a.value if hasattr(a, "value") else str(a)
    bv = b.value if hasattr(b, "value") else str(b)
    return av.lower() == bv.lower()


def _norm_dir(v: Any) -> Direction:
    if isinstance(v, Direction):
        return v
    s = str(v).lower()
    if "bull" in s:
        return Direction.BULLISH
    if "bear" in s:
        return Direction.BEARISH
    return Direction.NEUTRAL


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def _opposite_dir(d: Direction) -> Direction:
    if d is Direction.BULLISH:
        return Direction.BEARISH
    if d is Direction.BEARISH:
        return Direction.BULLISH
    return Direction.NEUTRAL


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CALENDAR MODELS  (ported verbatim from v9 — do not break)
# ════════════════════════════════════════════════════════════════════════════
_TIER_S = ("non-farm", "nonfarm", "nfp", "fomc", "cpi", "cash rate",
           "bank rate", "rate statement", "interest rate", "monetary policy")
_TIER_A = ("gdp", "pmi", "adp", "pce", "employment change", "unemployment",
           "average hourly", "retail sales", "ppi")
_TIER_B = ("speaks", "speech", "press conference", "testifies", "testimony")


def classify_tier(name: str) -> EventTier:
    n = (name or "").lower()
    if any(k in n for k in _TIER_S):
        return EventTier.S
    if any(k in n for k in _TIER_A):
        return EventTier.A
    if any(k in n for k in _TIER_B):
        return EventTier.B
    return EventTier.NONE


def classify_impact(name: str) -> ImpactLevel:
    return ImpactLevel.HIGH if classify_tier(name) != EventTier.NONE else ImpactLevel.MEDIUM


# (hours_before, hours_after) blackout windows by tier
TIER_WINDOWS: Mapping[EventTier, tuple[float, float]] = MappingProxyType({
    EventTier.S: (4.0, 48.0),
    EventTier.A: (2.0, 24.0),
    EventTier.B: (1.0, 6.0),
})
PROXIMITY_MAX_H = 48.0
WATCH_MAX_H = 168.0


class CalendarEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    currency: str = Field(..., min_length=3, max_length=3)
    event_name: str = Field(..., max_length=256)
    datetime_utc: datetime
    impact: Optional[ImpactLevel] = None
    tier: EventTier = EventTier.NONE

    @field_validator("currency")
    @classmethod
    def _up(cls, v: str) -> str:
        return v.upper()

    @field_validator("datetime_utc")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _derive(self) -> "CalendarEvent":
        if self.tier is EventTier.NONE:
            self.tier = classify_tier(self.event_name)
        if self.impact is None:
            self.impact = classify_impact(self.event_name)
        return self


class CalendarSets(BaseModel):
    model_config = ConfigDict(extra="ignore")
    blackout: list[CalendarEvent] = Field(default_factory=list)
    proximity: list[CalendarEvent] = Field(default_factory=list)
    watch: list[CalendarEvent] = Field(default_factory=list)
    suspended_ccy: set[str] = Field(default_factory=set)
    proximity_ccy: set[str] = Field(default_factory=set)
    watch_ccy: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def _sets(self) -> "CalendarSets":
        self.suspended_ccy = {e.currency for e in self.blackout}
        self.proximity_ccy = {e.currency for e in self.proximity}
        self.watch_ccy = {e.currency for e in self.watch}
        return self


class CalendarData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    events: list[CalendarEvent] = Field(default_factory=list)
    timezone_source: str = "UTC"
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_html_hash: str = ""

    def bucket(self, now: datetime) -> CalendarSets:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        blackout, proximity, watch = [], [], []
        for ev in self.events:
            if ev.impact != ImpactLevel.HIGH:
                continue
            before, after = TIER_WINDOWS.get(ev.tier, (2.0, 24.0))
            delta = (ev.datetime_utc - now).total_seconds() / 3600.0
            if -after <= delta <= before:
                blackout.append(ev)
            elif before < delta <= PROXIMITY_MAX_H:
                proximity.append(ev)
            elif PROXIMITY_MAX_H < delta <= WATCH_MAX_H:
                watch.append(ev)
        return CalendarSets(blackout=blackout, proximity=proximity, watch=watch)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CANONICAL ASSET VIEW  (ported from v9; +conviction_cap)
# ════════════════════════════════════════════════════════════════════════════
class MTFView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pct: int = 0
    direction: Direction = Direction.NEUTRAL
    quality: str = ""
    nc: int = 0
    age_d1: int = 0
    atr_h1: Optional[float] = None
    atr_h4: Optional[float] = None
    atr_daily: Optional[float] = None
    biases: dict[str, str] = Field(default_factory=dict)

    @field_validator("direction", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Direction:
        return _norm_dir(v)


class StructureEventView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    signal_id: str = ""
    kind: str = ""
    direction: Direction = Direction.NEUTRAL
    timeframe: str = ""
    level: Optional[float] = None
    confluence_score: float = 0.0
    status: str = ""
    distance_pct: Optional[float] = None
    distance_atr_multiple: Optional[float] = None
    volatility: str = ""
    force: str = ""
    bb_regime: str = "Normal"
    session: str = ""
    candles_elapsed: int = 999

    @field_validator("direction", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Direction:
        return _norm_dir(v)


class ZoneView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    level: float
    side: str = ""
    score: float = 0.0
    distance_pct: float = 999.0


class CanonicalAsset(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    base: str = ""
    quote: Optional[str] = None
    asset_class: str = "forex"
    current_price: Optional[float] = None
    rsi_by_tf: dict[str, dict] = Field(default_factory=dict)
    rsi_h4_status: Optional[str] = None
    mtf: Optional[MTFView] = None
    zones: list[ZoneView] = Field(default_factory=list)
    structure_events: list[StructureEventView] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    atr_effective: Optional[float] = None
    atr_source: Optional[str] = None
    nearest_aligned_zone: Optional[ZoneView] = None
    hot_zone_primary: Optional[ZoneView] = None
    conviction_cap: Optional[str] = None  # V4: mapped from JSON (e.g. "BBB" for synthetic ATR)


class MergeMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = ""
    assets_count: int = 0
    signals_count: int = 0

    @field_validator("generated_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


class Clock(BaseModel):
    now_utc: datetime
    now_local: datetime
    date_hdr: str

    @classmethod
    def from_meta(cls, generated_at: datetime) -> "Clock":
        now_utc = generated_at if generated_at.tzinfo else generated_at.replace(tzinfo=timezone.utc)
        now_local = now_utc.astimezone(timezone(timedelta(hours=1)))
        return cls(now_utc=now_utc, now_local=now_local,
                   date_hdr=now_local.strftime("%Y-%m-%d %H:%M GMT+1"))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — V4 MODELS & DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
_FACTOR_NAMES = ("f1_hwa", "f2_rmg", "f3_ext", "f4_trg", "f5_xctx", "f6_theme", "f7_macro")


@dataclass(frozen=True)
class ScoredFactor:
    name: str
    raw: Optional[float]
    score: float            # bounded [0,1]
    is_missing: bool
    detail: str


@dataclass
class FactorVector:
    symbol: str
    factors: dict[str, ScoredFactor]

    @property
    def present(self) -> list[str]:
        return [n for n, f in self.factors.items() if not f.is_missing]

    @property
    def missing(self) -> list[str]:
        return [n for n, f in self.factors.items() if f.is_missing]

    @property
    def absolute_mean(self) -> float:
        present = [f.score for f in self.factors.values() if not f.is_missing]
        if not present:
            return 0.0
        return sum(present) / len(present)

    def get(self, name: str) -> float:
        f = self.factors.get(name)
        return f.score if f else 0.0


@dataclass(frozen=True)
class Flag:
    code: str               # "C1".."C5"
    severity: str           # "minor" | "major"
    detail: str


@dataclass
class RiskCluster:
    key: str
    members: list[str] = field(default_factory=list)


class FactorScores(BaseModel):
    model_config = ConfigDict(extra="ignore")
    f1_hwa: float = 0.0
    f2_rmg: float = 0.0
    f3_ext: float = 0.0
    f4_trg: float = 0.0
    f5_xctx: float = 0.0
    f6_theme: float = 0.0
    f7_macro: float = 0.0
    absolute_mean: float = 0.0
    quantile: float = 0.0
    missing: list[str] = Field(default_factory=list)
    details: dict[str, str] = Field(default_factory=dict)


class FlagModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    code: str
    severity: str
    detail: str


class SetupV4(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # — ported from v9 —
    symbol: str
    direction: Direction
    scenario_hint: str = ""
    entry: float = 0.0
    entry_type: str = "Market"
    sl: float = 0.0
    sl_atr_multiple: float = 0.0
    tp1: float = 0.0
    tp1_atr_multiple: Optional[float] = None
    tp2: Optional[float] = None
    tp2_atr_multiple: Optional[float] = None
    rr: float = 0.0
    rr_synthetic: bool = False
    atr_effective: float = 0.0
    atr_source: str = "unknown"
    distance_atr: float = 0.0
    choch_score: Optional[float] = None
    gps_quality: Optional[str] = None
    mtf_pct: int = 0
    rsi_h4: Optional[float] = None
    rsi_h4_status: Optional[str] = None
    age_d1: int = 0
    cal_status: CalStatus = CalStatus.OK
    cal_note: str = ""
    htf_aligned: bool = False
    sl_detail: str = ""
    rr_detail: str = ""
    rationale: str = ""
    # — V4 —
    conviction: Conviction = Conviction.BBB
    factor_scores: FactorScores = Field(default_factory=FactorScores)
    flags: list[FlagModel] = Field(default_factory=list)
    cluster: str = ""
    capped_reason: Optional[str] = None
    reject_code: Optional[str] = None
    reject_detail: Optional[str] = None


class Universe(BaseModel):
    model_config = ConfigDict(extra="ignore")
    passed: list[CanonicalAsset] = Field(default_factory=list)
    rejected: list[tuple[CanonicalAsset, GateCode, str]] = Field(default_factory=list)


class Eliminated(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    direction: Direction = Direction.NEUTRAL
    scenario: Optional[str] = None
    reject_code: str
    reject_detail: str
    rsi_h4: Optional[float] = None
    age_d1: int = 0
    cal_status: CalStatus = CalStatus.OK
    rr: Optional[float] = None


@dataclass
class MarketThemes:
    strong: dict[str, str] = field(default_factory=dict)        # ccy -> "Bullish"/"Bearish"
    cohesion: dict[str, float] = field(default_factory=dict)    # ccy -> [0,1] consensus strength

    def bonus_for(self, base: str, quote: Optional[str], direction: Direction) -> float:
        """F6 score in [0,1]: how well the trade rides dominant currency themes."""
        d = direction.value
        inv = "Bearish" if d == "Bullish" else "Bullish"
        contributions: list[float] = []
        # base leg
        if base in self.strong:
            coh = self.cohesion.get(base, 0.0)
            contributions.append(coh if self.strong[base] == d else -coh)
        # quote leg (inverse)
        if quote and quote in self.strong:
            coh = self.cohesion.get(quote, 0.0)
            contributions.append(coh if self.strong[quote] == inv else -coh)
        if not contributions:
            return 0.5  # neutral when no theme touches the pair
        signed = sum(contributions) / len(contributions)  # [-1,1]
        return _clamp01((signed + 1.0) / 2.0)

    def is_counter_theme(self, base: str, quote: Optional[str], direction: Direction) -> tuple[bool, float]:
        """True + cohesion if the trade fights a high-cohesion dominant theme."""
        d = direction.value
        inv = "Bearish" if d == "Bullish" else "Bullish"
        worst = 0.0
        counter = False
        if base in self.strong and self.strong[base] != d:
            counter = True
            worst = max(worst, self.cohesion.get(base, 0.0))
        if quote and quote in self.strong and self.strong[quote] != inv:
            counter = True
            worst = max(worst, self.cohesion.get(quote, 0.0))
        return counter, worst


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CONFIG  (V4 thresholds; structural common-sense, not calibrated)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class V4Config:
    # universe (ported from v9)
    MIN_QUALITY: frozenset = frozenset({"A+", "A"})
    MIN_CONSENSUS_PCT: int = 50
    # F1 HWA — ordinal seniority weights
    HWA_WEIGHTS: Mapping[str, int] = field(default_factory=lambda: MappingProxyType(
        {"MN": 6, "W1": 5, "D1": 4, "H4": 3, "H1": 2, "M15": 1}))
    # F2 RMG
    RMG_FAST: tuple = ("M15", "H1")
    RMG_SLOW: tuple = ("D1", "W1")
    RMG_SCALE: float = 15.0
    RMG_MIN_TF: int = 3
    # F3 EXT
    EXT_TF_COUNT: int = 5
    # F4 TRG
    TRG_SCORE_CAP: float = 85.0
    TRG_FRESH_MAX: int = 6
    TRG_DIST_ATR_MAX: float = 1.0
    # F6 THEME
    THEME_MIN_VOTES: int = 3
    THEME_BULL_HI: float = 0.8
    THEME_BULL_LO: float = 0.2
    THEME_COHESION_C5: float = 0.8
    # F7 MACRO
    MACRO_TAU_HOURS: float = 48.0
    # conviction (ABSOLUTE thresholds)
    AAA_MIN: float = 0.80
    AA_MIN: float = 0.68
    A_MIN: float = 0.55
    BBB_MIN: float = 0.42
    BB_MIN: float = 0.30
    MACRO_CAP_RISK_THRESHOLD: float = 0.50   # macro RISK >= 0.5 -> cap AA
    # contradictions
    C1_TRG_MIN: float = 0.5
    C1_RMG_MAX: float = 0.35
    C2_EXT_MAX: float = 0.3
    C2_HWA_MAX: float = 0.5
    C3_MACRO_MAX: float = 0.5
    C4_DIST_ATR: float = 1.0
    # preflight (ported from v9)
    RR_MIN: float = 1.5
    RR_MAX: float = 20.0
    SL_FLOOR_MULT: float = 0.8
    DEFAULT_BB_MULT: float = 1.5
    BB_REGIME_MULT: Mapping[str, float] = field(default_factory=lambda: MappingProxyType(
        {"Squeeze": 1.0, "Normal": 1.5, "Expansion": 2.0}))
    FRESH_ATR_MAX: float = 0.3
    LIMIT_ZONE_MAX_DIST: float = 2.0
    TP1_ATR_MULT: float = 2.0
    TP2_ATR_MULT: float = 1.0
    # selection
    MAX_SETUPS: int = 5
    MAX_EXPOSURE_PER_CCY: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "V4Config":
        base = cls()
        kw = {}
        for k, v in (d or {}).items():
            if hasattr(base, k):
                kw[k] = v
        return cls(**kw)


CONFIG = V4Config()

# Execution-context point maps for F5 XCTX (bounded, categorical)
_XCTX_FORCE = MappingProxyType({"fort": 1.0, "strong": 1.0, "": 0.5, "faible": 0.0, "weak": 0.0})
_XCTX_VOL = MappingProxyType({"haute": 1.0, "high": 1.0, "": 0.5, "faible": 0.3, "basse": 0.3, "low": 0.3})
_XCTX_SESSION = MappingProxyType({"london": 1.0, "newyork": 1.0, "ny": 1.0, "us": 1.0,
                                  "asian": 0.5, "tokyo": 0.5, "sydney": 0.5, "off": 0.0, "": 0.3})
_XCTX_BB = MappingProxyType({"squeeze": 1.0, "normal": 0.6, "expansion": 0.3, "": 0.6})

_EXT_STATUSES = ("extreme_overbought", "extreme_oversold", "overbought", "oversold")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THEME DETECTION  (extended from v9 with cohesion)
# ════════════════════════════════════════════════════════════════════════════
def detect_currency_themes(assets: Mapping[str, CanonicalAsset], cfg: V4Config = CONFIG) -> MarketThemes:
    votes: dict[str, list[str]] = defaultdict(list)
    for a in assets.values():
        if not a.mtf or a.mtf.direction is Direction.NEUTRAL:
            continue
        d = a.mtf.direction.value
        inv = "Bearish" if d == "Bullish" else "Bullish"
        votes[a.base].append(d)
        if a.quote:
            votes[a.quote].append(inv)
    strong: dict[str, str] = {}
    cohesion: dict[str, float] = {}
    for ccy, vs in votes.items():
        if len(vs) < cfg.THEME_MIN_VOTES:
            continue
        bull = vs.count("Bullish") / len(vs)
        if bull >= cfg.THEME_BULL_HI:
            strong[ccy] = "Bullish"
            cohesion[ccy] = bull
        elif bull <= cfg.THEME_BULL_LO:
            strong[ccy] = "Bearish"
            cohesion[ccy] = 1.0 - bull
    return MarketThemes(strong=strong, cohesion=cohesion)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FACTORS F1..F7  (pure functions, all bounded [0,1])
# ════════════════════════════════════════════════════════════════════════════
def _rsi_value(a: CanonicalAsset, tf: str) -> Optional[float]:
    d = a.rsi_by_tf.get(tf) or a.rsi_by_tf.get(tf.upper()) or a.rsi_by_tf.get(tf.lower())
    if isinstance(d, dict):
        return _safe_float(d.get("value"))
    return _safe_float(d)


def _rsi_status(a: CanonicalAsset, tf: str) -> str:
    d = a.rsi_by_tf.get(tf) or a.rsi_by_tf.get(tf.upper()) or a.rsi_by_tf.get(tf.lower())
    if isinstance(d, dict):
        return str(d.get("status") or "").lower()
    return ""


def _aligned_trigger(a: CanonicalAsset) -> Optional[StructureEventView]:
    """Most recent Fresh CHoCH aligned with MTF direction (lowest candles_elapsed)."""
    if a.mtf is None:
        return None
    want = a.mtf.direction
    cands = [ev for ev in a.structure_events
             if ev.status.lower() == "fresh" and _dir_eq(ev.direction, want)]
    if not cands:
        return None
    return min(cands, key=lambda e: e.candles_elapsed)


def f1_hwa(a: CanonicalAsset, cfg: V4Config = CONFIG) -> ScoredFactor:
    if a.mtf is None:
        return ScoredFactor("f1_hwa", None, 0.5, True, "MTF absent")
    biases = a.mtf.biases or {}
    direction = a.mtf.direction
    num = 0
    den = 0
    conflicts: list[str] = []
    for tf, w in cfg.HWA_WEIGHTS.items():
        b = (biases.get(tf) or "Range")
        if _dir_eq(_norm_dir(b), direction) and _norm_dir(b) is not Direction.NEUTRAL:
            s = 1
        elif _norm_dir(b) is _opposite_dir(direction) and _opposite_dir(direction) is not Direction.NEUTRAL:
            s = -1
            conflicts.append(tf)
        else:
            s = 0
        num += w * s
        den += w
    raw = (num / den) if den else 0.0
    score = _clamp01((raw + 1.0) / 2.0)
    detail = f"HWA num={num}/den={den} raw={raw:.2f} conflits={conflicts or '∅'}"
    return ScoredFactor("f1_hwa", raw, score, a.mtf is None, detail)


def f2_rmg(a: CanonicalAsset, cfg: V4Config = CONFIG) -> ScoredFactor:
    if a.mtf is None:
        return ScoredFactor("f2_rmg", None, 0.5, True, "MTF absent")
    fast_vals = [v for v in (_rsi_value(a, tf) for tf in cfg.RMG_FAST) if v is not None]
    slow_vals = [v for v in (_rsi_value(a, tf) for tf in cfg.RMG_SLOW) if v is not None]
    n_available = len(fast_vals) + len(slow_vals)
    if not fast_vals or not slow_vals or n_available < cfg.RMG_MIN_TF:
        return ScoredFactor("f2_rmg", None, 0.5, True,
                            f"RSI insuffisant ({n_available} TF)")
    fast = sum(fast_vals) / len(fast_vals)
    slow = sum(slow_vals) / len(slow_vals)
    grad = fast - slow
    signed = grad if a.mtf.direction is Direction.BULLISH else -grad
    score = _clamp01(0.5 + 0.5 * math.tanh(signed / cfg.RMG_SCALE))
    detail = f"RMG fast={fast:.1f} slow={slow:.1f} grad={grad:.1f} signed={signed:.1f}"
    return ScoredFactor("f2_rmg", grad, score, False, detail)


def f3_ext(a: CanonicalAsset, cfg: V4Config = CONFIG) -> ScoredFactor:
    if a.mtf is None:
        return ScoredFactor("f3_ext", None, 0.5, True, "MTF absent")
    direction = a.mtf.direction
    ext_in_dir = 0
    checked = 0
    for tf in cfg.HWA_WEIGHTS.keys():
        st = _rsi_status(a, tf)
        if not st:
            continue
        checked += 1
        is_ext = any(k in st for k in _EXT_STATUSES)
        if not is_ext:
            continue
        # overheated in the direction of the trade = bad
        if direction is Direction.BULLISH and ("overbought" in st):
            ext_in_dir += 1
        elif direction is Direction.BEARISH and ("oversold" in st):
            ext_in_dir += 1
    score = _clamp01(1.0 - ext_in_dir / cfg.EXT_TF_COUNT)
    detail = f"{ext_in_dir}/{cfg.EXT_TF_COUNT} TF surchauffés dans le sens (checked={checked})"
    return ScoredFactor("f3_ext", float(ext_in_dir), score, False, detail)


def f4_trg(a: CanonicalAsset, cfg: V4Config = CONFIG) -> ScoredFactor:
    ev = _aligned_trigger(a)
    if ev is None:
        return ScoredFactor("f4_trg", None, 0.0, True, "pas de trigger aligné")
    score_n = min(ev.confluence_score, cfg.TRG_SCORE_CAP) / cfg.TRG_SCORE_CAP
    fresh = 1.0 - min(ev.candles_elapsed, cfg.TRG_FRESH_MAX) / cfg.TRG_FRESH_MAX
    dist = ev.distance_atr_multiple if ev.distance_atr_multiple is not None else cfg.TRG_DIST_ATR_MAX
    proximity = 1.0 - min(dist, cfg.TRG_DIST_ATR_MAX) / cfg.TRG_DIST_ATR_MAX
    score = _clamp01(0.4 * score_n + 0.3 * fresh + 0.3 * proximity)
    detail = (f"TRG score_n={score_n:.2f} fresh={fresh:.2f} prox={proximity:.2f} "
              f"(conf={ev.confluence_score:.0f}, {ev.candles_elapsed}c, {dist:.2f}ATR)")
    return ScoredFactor("f4_trg", ev.confluence_score, score, False, detail)


def f5_xctx(a: CanonicalAsset, cfg: V4Config = CONFIG) -> ScoredFactor:
    ev = _aligned_trigger(a)
    if ev is None:
        return ScoredFactor("f5_xctx", None, 0.5, True, "trigger absent (contexte neutre)")
    force = _XCTX_FORCE.get((ev.force or "").lower(), 0.5)
    vol = _XCTX_VOL.get((ev.volatility or "").lower(), 0.5)
    session = _XCTX_SESSION.get((ev.session or "").lower(), 0.3)
    bb = _XCTX_BB.get((ev.bb_regime or "").lower(), 0.6)
    score = _clamp01((force + vol + session + bb) / 4.0)
    detail = (f"XCTX force={force:.1f} vol={vol:.1f} sess={session:.1f} bb={bb:.1f} "
              f"({ev.force}/{ev.volatility}/{ev.session}/{ev.bb_regime})")
    return ScoredFactor("f5_xctx", None, score, False, detail)


def f6_theme(a: CanonicalAsset, themes: MarketThemes, cfg: V4Config = CONFIG) -> ScoredFactor:
    if a.mtf is None:
        return ScoredFactor("f6_theme", None, 0.5, True, "MTF absent")
    score = themes.bonus_for(a.base, a.quote, a.mtf.direction)
    detail = (f"THEME {a.base}/{a.quote or '—'} dir={a.mtf.direction.value} "
              f"strong={themes.strong} -> {score:.2f}")
    return ScoredFactor("f6_theme", None, score, False, detail)


def f7_macro(a: CanonicalAsset, cal: Optional[CalendarSets], clock: Clock,
             cfg: V4Config = CONFIG) -> ScoredFactor:
    if cal is None:
        return ScoredFactor("f7_macro", None, 1.0, False, "calendrier absent (risque nul)")
    sides = {a.base, (a.quote or "")}
    # Blackout active -> score 0 (hard veto handled in preflight)
    if sides & cal.suspended_ccy:
        return ScoredFactor("f7_macro", 1.0, 0.0, False, "BLACKOUT actif")
    now = clock.now_utc
    horizon: list[CalendarEvent] = list(cal.blackout) + list(cal.proximity) + list(cal.watch)
    relevant_h: list[float] = []
    for ev in horizon:
        if ev.tier not in (EventTier.S, EventTier.A):
            continue
        if ev.currency not in sides:
            continue
        delta = (ev.datetime_utc - now).total_seconds() / 3600.0
        if delta >= 0:
            relevant_h.append(delta)
    if not relevant_h:
        return ScoredFactor("f7_macro", 0.0, 1.0, False, "aucun event S/A futur")
    hours = min(relevant_h)
    risk = math.exp(-hours / cfg.MACRO_TAU_HOURS)
    score = _clamp01(1.0 - risk)
    detail = f"MACRO event S/A dans {hours:.1f}h risk={risk:.2f} -> {score:.2f}"
    return ScoredFactor("f7_macro", risk, score, False, detail)


def build_factor_vector(a: CanonicalAsset, themes: MarketThemes,
                        cal: Optional[CalendarSets], clock: Clock,
                        cfg: V4Config = CONFIG) -> FactorVector:
    factors = {
        "f1_hwa": f1_hwa(a, cfg),
        "f2_rmg": f2_rmg(a, cfg),
        "f3_ext": f3_ext(a, cfg),
        "f4_trg": f4_trg(a, cfg),
        "f5_xctx": f5_xctx(a, cfg),
        "f6_theme": f6_theme(a, themes, cfg),
        "f7_macro": f7_macro(a, cal, clock, cfg),
    }
    return FactorVector(symbol=a.symbol, factors=factors)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SCORING (absolute) + CROSS-SECTION (tie-break/diversif ONLY)
# ════════════════════════════════════════════════════════════════════════════
def score_absolute(fv: FactorVector) -> float:
    return fv.absolute_mean


def compute_quantiles(vectors: list[FactorVector]) -> dict[str, float]:
    """Pure-python percentile rank of absolute_mean within the universe."""
    means = [(v.symbol, v.absolute_mean) for v in vectors]
    if not means:
        return {}
    values = sorted(m for _, m in means)
    n = len(values)
    out: dict[str, float] = {}
    for sym, m in means:
        below = sum(1 for x in values if x < m)
        equal = sum(1 for x in values if x == m)
        # mid-rank percentile, deterministic
        out[sym] = (below + 0.5 * equal) / n if n else 0.0
    return out


def rank_setups(setups: list[SetupV4], cfg: V4Config = CONFIG) -> list[SetupV4]:
    """Sort DESC by absolute_mean; tie-break f4 -> f1 -> low-macro-risk -> quantile.
    Quantile NEVER influences conviction; only ordering here. Symbol is the
    final stable secondary key for bit-for-bit reproducibility."""
    def key(s: SetupV4):
        fs = s.factor_scores
        return (
            -fs.absolute_mean,
            -fs.f4_trg,
            -fs.f1_hwa,
            -fs.f7_macro,      # higher f7 = lower macro risk preferred
            -fs.quantile,
            s.symbol,
        )
    return sorted(setups, key=key)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CONTRADICTIONS C1..C5
# ════════════════════════════════════════════════════════════════════════════
def _c1_struct_vs_momentum(fv: FactorVector, cfg: V4Config) -> Optional[Flag]:
    if fv.get("f4_trg") > cfg.C1_TRG_MIN and fv.get("f2_rmg") < cfg.C1_RMG_MAX:
        return Flag("C1", "minor",
                    f"Structure forte (TRG={fv.get('f4_trg'):.2f}) mais momentum faible "
                    f"(RMG={fv.get('f2_rmg'):.2f})")
    return None


def _c2_momentum_vs_trend(fv: FactorVector, cfg: V4Config) -> Optional[Flag]:
    if fv.get("f3_ext") < cfg.C2_EXT_MAX and fv.get("f1_hwa") < cfg.C2_HWA_MAX:
        return Flag("C2", "major",
                    f"Parabolique : surchauffe (EXT={fv.get('f3_ext'):.2f}) + alignement "
                    f"faible (HWA={fv.get('f1_hwa'):.2f})")
    return None


def _c3_trend_vs_calendar(a: CanonicalAsset, fv: FactorVector,
                          cal: Optional[CalendarSets], cfg: V4Config) -> Optional[Flag]:
    if fv.get("f7_macro") >= cfg.C3_MACRO_MAX:
        return None
    if cal is None:
        return None
    sides = {a.base, (a.quote or "")}
    tier_s = [e for e in (list(cal.blackout) + list(cal.proximity))
              if e.tier is EventTier.S and e.currency in sides]
    if tier_s:
        names = ", ".join(f"{e.currency} {e.event_name}" for e in tier_s[:2])
        return Flag("C3", "major",
                    f"Risque calendaire élevé (MACRO={fv.get('f7_macro'):.2f}) : {names}")
    return None


def _c4_quality_vs_potential(a: CanonicalAsset, cfg: V4Config) -> Optional[Flag]:
    ev = _aligned_trigger(a)
    if ev is None or ev.distance_atr_multiple is None:
        return None
    if ev.distance_atr_multiple > cfg.C4_DIST_ATR:
        return Flag("C4", "minor",
                    f"Chasing : prix à {ev.distance_atr_multiple:.2f}×ATR du trigger "
                    f"(> {cfg.C4_DIST_ATR})")
    return None


def _c5_trade_vs_theme(a: CanonicalAsset, themes: MarketThemes, cfg: V4Config) -> Optional[Flag]:
    if a.mtf is None:
        return None
    counter, coh = themes.is_counter_theme(a.base, a.quote, a.mtf.direction)
    if counter and coh >= cfg.THEME_COHESION_C5:
        return Flag("C5", "major",
                    f"Trade contre thème devise dominant (cohésion={coh:.2f})")
    return None


def detect_contradictions(a: CanonicalAsset, fv: FactorVector, themes: MarketThemes,
                          cal: Optional[CalendarSets], cfg: V4Config = CONFIG) -> list[Flag]:
    flags: list[Flag] = []
    for f in (
        _c1_struct_vs_momentum(fv, cfg),
        _c2_momentum_vs_trend(fv, cfg),
        _c3_trend_vs_calendar(a, fv, cal, cfg),
        _c4_quality_vs_potential(a, cfg),
        _c5_trade_vs_theme(a, themes, cfg),
    ):
        if f is not None:
            flags.append(f)
    return flags


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — GRADE  (vetos -> caps -> grid). Quantile NEVER enters here.
# ════════════════════════════════════════════════════════════════════════════
def apply_caps(a: CanonicalAsset, fv: FactorVector, cfg: V4Config = CONFIG
               ) -> tuple[Optional[Conviction], Optional[str]]:
    """Returns the most restrictive cap and a human reason, or (None, None)."""
    caps: list[tuple[Conviction, str]] = []
    # Synthetic ATR -> BBB (from JSON conviction_cap or atr_source)
    if (a.conviction_cap or "").upper() == "BBB" or (a.atr_source or "").lower() == "synthetic":
        caps.append((Conviction.BBB, "ATR synthétique"))
    # High macro risk -> AA  (risk = 1 - f7_score)
    macro_risk = 1.0 - fv.get("f7_macro")
    if macro_risk >= cfg.MACRO_CAP_RISK_THRESHOLD:
        caps.append((Conviction.AA, f"risque macro élevé ({macro_risk:.2f})"))
    if not caps:
        return None, None
    cap = min(caps, key=lambda c: _CONVICTION_ORDINAL[c[0].value])
    return cap[0], cap[1]


def grade(absolute_mean: float, flags: list[Flag], cap: Optional[Conviction],
          cfg: V4Config = CONFIG) -> Conviction:
    """Map absolute score x contradictions to AAA..B. Caps applied last."""
    minors = sum(1 for f in flags if f.severity == "minor")
    majors = sum(1 for f in flags if f.severity == "major")
    k = minors + 2 * majors
    m = absolute_mean
    if m >= cfg.AAA_MIN and k == 0:
        base = Conviction.AAA
    elif m >= cfg.AA_MIN and k <= 1:
        base = Conviction.AA
    elif m >= cfg.A_MIN and k <= 1:
        base = Conviction.A
    elif (m >= cfg.BBB_MIN or cap is not None) and k <= 2:
        base = Conviction.BBB
    elif m >= cfg.BB_MIN:
        base = Conviction.BB
    else:
        base = Conviction.B
    if cap is not None:
        if _CONVICTION_ORDINAL[cap.value] < _CONVICTION_ORDINAL[base.value]:
            base = cap
    return base


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — LEVELS  (ported from v9 §8, logic unchanged) + preflight
# ════════════════════════════════════════════════════════════════════════════
def _is_opposite(zone: ZoneView, direction: Direction) -> bool:
    side = (zone.side or "").upper()
    if direction is Direction.BULLISH:
        return side in ("SELL", "RESISTANCE", "SUPPLY")
    if direction is Direction.BEARISH:
        return side in ("BUY", "SUPPORT", "DEMAND")
    return False


def _get_opposite_zone(a: CanonicalAsset, direction: Direction) -> Optional[ZoneView]:
    zs = [z for z in a.zones if _is_opposite(z, direction)]
    return min(zs, key=lambda z: z.distance_pct) if zs else None


def atr_for_signal(a: CanonicalAsset, ev: Optional[StructureEventView]) -> tuple[float, str]:
    if ev is not None and a.mtf:
        tf = (ev.timeframe or "").upper()
        m = {"H1": a.mtf.atr_h1, "H4": a.mtf.atr_h4, "D1": a.mtf.atr_daily}
        v = m.get(tf)
        if v and v > 0:
            return float(v), f"atr_{tf.lower()}"
    return (a.atr_effective or 0.0), (a.atr_source or "h4")


def compute_entry(a: CanonicalAsset, ev: Optional[StructureEventView], atr: float,
                  cfg: V4Config) -> tuple[float, str]:
    price = a.current_price or 0.0
    if ev and ev.candles_elapsed <= 1 and (ev.distance_atr_multiple or 999) <= cfg.FRESH_ATR_MAX:
        return price, "Market"
    z = a.nearest_aligned_zone
    if z and z.distance_pct <= cfg.LIMIT_ZONE_MAX_DIST:
        return z.level, "Limit"
    if a.hot_zone_primary:
        return a.hot_zone_primary.level, "Limit"
    return price, "Market"


def compute_sl(a: CanonicalAsset, entry: float, atr: float,
               ev: Optional[StructureEventView], cfg: V4Config) -> tuple[float, float, str]:
    direction = a.mtf.direction if a.mtf else Direction.NEUTRAL
    bb_regime = ev.bb_regime if ev else "Normal"
    bb_mult = cfg.BB_REGIME_MULT.get(bb_regime, cfg.DEFAULT_BB_MULT)
    if direction is Direction.BULLISH:
        sl_raw = entry - atr * bb_mult
    elif direction is Direction.BEARISH:
        sl_raw = entry + atr * bb_mult
    else:
        sl_raw = entry
    sl = sl_raw
    detail = f"Raw SL={sl_raw:.5f} ({bb_regime} ×{bb_mult})"
    z = a.nearest_aligned_zone
    if z and z.distance_pct <= cfg.LIMIT_ZONE_MAX_DIST:
        if direction is Direction.BULLISH:
            sl = min(sl_raw, z.level - 0.3 * atr)
        elif direction is Direction.BEARISH:
            sl = max(sl_raw, z.level + 0.3 * atr)
        detail += f" zone-adj→{sl:.5f}"
    min_dist = atr * cfg.SL_FLOOR_MULT
    if abs(entry - sl) < min_dist:
        sl = entry - min_dist if direction is Direction.BULLISH else entry + min_dist
        detail += f" [floored {cfg.SL_FLOOR_MULT}×ATR]"
    return sl, bb_mult, detail


def compute_tp1(a: CanonicalAsset, entry: float, atr: float,
                cfg: V4Config) -> tuple[float, Optional[float], bool]:
    direction = a.mtf.direction if a.mtf else Direction.NEUTRAL
    opp = _get_opposite_zone(a, direction)
    if opp:
        return opp.level, (round(abs(opp.level - entry) / atr, 2) if atr > 0 else None), False
    tp1 = entry + cfg.TP1_ATR_MULT * atr if direction is Direction.BULLISH else entry - cfg.TP1_ATR_MULT * atr
    return tp1, cfg.TP1_ATR_MULT, True


def compute_tp2(a: CanonicalAsset, entry: float, tp1: float, atr: float,
                cfg: V4Config) -> tuple[Optional[float], Optional[float], bool]:
    direction = a.mtf.direction if a.mtf else Direction.NEUTRAL
    opp = [z for z in sorted(a.zones, key=lambda z: z.distance_pct)
           if _is_opposite(z, direction)]
    if len(opp) >= 2:
        lvl = opp[1].level
        return lvl, (round(abs(lvl - entry) / atr, 2) if atr > 0 else None), False
    tp2 = tp1 + cfg.TP2_ATR_MULT * atr if direction is Direction.BULLISH else tp1 - cfg.TP2_ATR_MULT * atr
    return tp2, (round(abs(tp2 - entry) / atr, 2) if atr > 0 else None), True


def compute_rr(entry: float, sl: float, tp1: float, tp2: Optional[float],
               tp1_syn: bool, tp2_syn: bool) -> tuple[float, str]:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0, "Risk=0, invalid"
    r1 = abs(tp1 - entry)
    if tp2 is None:
        rr = r1 / risk
        detail = f"RR(TP1 only)={rr:.2f}"
    else:
        r2 = abs(tp2 - entry)
        rr = (0.6 * r1 + 0.4 * r2) / risk
        detail = f"RR=(0.6×{r1:.5f}+0.4×{r2:.5f})/{risk:.5f}={rr:.2f}"
    flags = []
    if tp1_syn:
        flags.append("TP1 synth 2×ATR")
    if tp2_syn:
        flags.append("TP2 synth")
    if flags:
        detail += " [" + ", ".join(flags) + "]"
    return round(rr, 2), detail


@dataclass
class LevelBundle:
    entry: float
    entry_type: str
    sl: float
    sl_atr_multiple: float
    sl_detail: str
    tp1: float
    tp1_atr_multiple: Optional[float]
    tp1_synthetic: bool
    tp2: Optional[float]
    tp2_atr_multiple: Optional[float]
    tp2_synthetic: bool
    rr: float
    rr_detail: str
    atr_effective: float
    atr_source: str
    trigger: Optional[StructureEventView]


def build_levels(a: CanonicalAsset, cfg: V4Config = CONFIG) -> LevelBundle:
    ev = _aligned_trigger(a)
    atr, atr_src = atr_for_signal(a, ev)
    entry, entry_type = compute_entry(a, ev, atr, cfg)
    sl, sl_mult, sl_detail = compute_sl(a, entry, atr, ev, cfg)
    tp1, tp1_mult, tp1_syn = compute_tp1(a, entry, atr, cfg)
    tp2, tp2_mult, tp2_syn = compute_tp2(a, entry, tp1, atr, cfg)
    rr, rr_detail = compute_rr(entry, sl, tp1, tp2, tp1_syn, tp2_syn)
    return LevelBundle(
        entry=round(entry, 5), entry_type=entry_type,
        sl=round(sl, 5), sl_atr_multiple=sl_mult, sl_detail=sl_detail,
        tp1=round(tp1, 5), tp1_atr_multiple=tp1_mult, tp1_synthetic=tp1_syn,
        tp2=(round(tp2, 5) if tp2 is not None else None),
        tp2_atr_multiple=tp2_mult, tp2_synthetic=tp2_syn,
        rr=rr, rr_detail=rr_detail,
        atr_effective=atr, atr_source=atr_src, trigger=ev,
    )


def preflight(setup: SetupV4, cfg: V4Config = CONFIG) -> SetupV4:
    if setup.cal_status is CalStatus.BLACKOUT:
        setup.reject_code = "CAL_BLACKOUT"
        setup.reject_detail = setup.cal_note
        return setup
    if setup.atr_effective <= 0:
        setup.reject_code = "NO_ATR"
        setup.reject_detail = "ATR ≤ 0"
        return setup
    if not (cfg.RR_MIN <= setup.rr <= cfg.RR_MAX):
        setup.reject_code = "RR_OUT_OF_RANGE"
        setup.reject_detail = f"RR {setup.rr} ∉ [{cfg.RR_MIN},{cfg.RR_MAX}]"
        return setup
    if setup.direction is Direction.BULLISH and setup.sl >= setup.entry:
        setup.reject_code = "SL_SIGN"
        setup.reject_detail = "SL ≥ entry (bullish)"
        return setup
    if setup.direction is Direction.BEARISH and setup.sl <= setup.entry:
        setup.reject_code = "SL_SIGN"
        setup.reject_detail = "SL ≤ entry (bearish)"
        return setup
    return setup


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — DIVERSIFY  (cluster by risk -> representative -> cap -> top N)
# ════════════════════════════════════════════════════════════════════════════
def _split_symbol(symbol: str) -> tuple[str, str]:
    if "/" in symbol:
        b, q = symbol.split("/", 1)
        return b, q
    return symbol, ""


def assign_clusters(setups: list[SetupV4], themes: MarketThemes) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in setups:
        base, quote = _split_symbol(s.symbol)
        d = s.direction.value
        inv = "Bearish" if d == "Bullish" else "Bullish"
        key = None
        # dominant currency theme drives the cluster
        if base in themes.strong and themes.strong[base] == d:
            key = f"{base}_{'strong' if d == 'Bullish' else 'weak'}"
        elif quote and quote in themes.strong and themes.strong[quote] == inv:
            key = f"{quote}_{'weak' if d == 'Bullish' else 'strong'}"
        if key is None:
            key = f"isolated:{s.symbol}"
        out[s.symbol] = key
        s.cluster = key
    return out


def diversify(setups: list[SetupV4], themes: MarketThemes,
              cfg: V4Config = CONFIG) -> list[SetupV4]:
    if not setups:
        return []
    assign_clusters(setups, themes)
    # group by cluster, keep best absolute_mean as representative
    groups: dict[str, list[SetupV4]] = defaultdict(list)
    for s in setups:
        groups[s.cluster].append(s)
    representatives: list[SetupV4] = []
    for key, members in groups.items():
        members_sorted = sorted(
            members,
            key=lambda x: (-x.factor_scores.absolute_mean,
                           -_CONVICTION_ORDINAL[x.conviction.value],
                           x.symbol))
        rep = members_sorted[0]
        representatives.append(rep)
        for loser in members_sorted[1:]:
            loser.reject_code = "CLUSTER_DUP"
            loser.reject_detail = f"Représentant cluster {key} = {rep.symbol}"
    # rank representatives, then apply per-currency exposure cap, then top-N
    ranked = sorted(
        representatives,
        key=lambda x: (-_CONVICTION_ORDINAL[x.conviction.value],
                       -x.factor_scores.absolute_mean,
                       x.symbol))
    net: Counter = Counter()
    kept: list[SetupV4] = []
    for s in ranked:
        base, quote = _split_symbol(s.symbol)
        sign = 1 if s.direction is Direction.BULLISH else -1
        over_base = abs(net[base] + sign) > cfg.MAX_EXPOSURE_PER_CCY
        over_quote = bool(quote) and abs(net[quote] - sign) > cfg.MAX_EXPOSURE_PER_CCY
        if over_base or over_quote:
            s.capped_reason = "exposition devise"
            s.cal_note = (s.cal_note + " [capped: exposition devise]").strip()
            continue
        net[base] += sign
        if quote:
            net[quote] -= sign
        kept.append(s)
        if len(kept) >= cfg.MAX_SETUPS:
            break
    return kept


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — PIPELINE  (linear orchestration, V4)
# ════════════════════════════════════════════════════════════════════════════
def _build_universe(assets: Mapping[str, CanonicalAsset], cal: CalendarSets,
                    cfg: V4Config) -> Universe:
    passed: list[CanonicalAsset] = []
    rejected: list[tuple[CanonicalAsset, GateCode, str]] = []
    for asset in assets.values():
        if asset.mtf is None:
            rejected.append((asset, GateCode.G0_SCHEMA_ASSET_ERROR, "MTF manquant"))
            continue
        base, quote = asset.base, (asset.quote or "")
        if base in cal.suspended_ccy or quote in cal.suspended_ccy:
            hit = ({base, quote} & cal.suspended_ccy)
            rejected.append((asset, GateCode.G1_CAL_BLACKOUT, f"Blackout: {sorted(hit)}"))
            continue
        quality = asset.mtf.quality or ""
        if quality not in cfg.MIN_QUALITY:
            rejected.append((asset, GateCode.G2_LOW_QUALITY, f"Quality {quality}"))
            continue
        if asset.mtf.direction is Direction.NEUTRAL:
            rejected.append((asset, GateCode.G3_NO_DIRECTION, "Direction Neutral"))
            continue
        if asset.mtf.pct < cfg.MIN_CONSENSUS_PCT:
            rejected.append((asset, GateCode.G4_LOW_CONSENSUS, f"MTF {asset.mtf.pct}%"))
            continue
        if asset.atr_effective is None or asset.atr_effective <= 0:
            rejected.append((asset, GateCode.G5_NO_ATR, f"ATR {asset.atr_source}"))
            continue
        passed.append(asset)
    return Universe(passed=passed, rejected=rejected)


def _compute_cal_status(a: CanonicalAsset, cal: Optional[CalendarSets]) -> tuple[CalStatus, str]:
    if cal is None:
        return CalStatus.OK, ""
    sides = {a.base, (a.quote or "")}
    hit_black = sides & cal.suspended_ccy
    if hit_black:
        names = [f"{e.currency} {e.event_name}" for e in cal.blackout if e.currency in hit_black]
        return CalStatus.BLACKOUT, "; ".join(names[:3])
    hit_prox = sides & cal.proximity_ccy
    if hit_prox:
        return CalStatus.PROXIMITY, ", ".join(sorted(hit_prox))
    hit_watch = sides & cal.watch_ccy
    if hit_watch:
        return CalStatus.WATCH, ", ".join(sorted(hit_watch))
    return CalStatus.OK, ""


def _scenario_hint(a: CanonicalAsset, lv: LevelBundle) -> str:
    """Descriptive label only — NOT a scoring pivot in V4."""
    parts = []
    age = a.mtf.age_d1 if a.mtf else 0
    if lv.trigger is not None:
        ev = lv.trigger
        parts.append(f"CHoCH {ev.timeframe} {ev.candles_elapsed}c score={ev.confluence_score:.0f}")
    elif a.hot_zone_primary:
        parts.append("Hot Zone")
    if age <= 15:
        parts.append("trend frais")
    elif age <= 30:
        parts.append("trend mûr")
    else:
        parts.append(f"trend âgé {age}j")
    parts.append(lv.entry_type)
    return " · ".join(parts)


def _htf_aligned(a: CanonicalAsset) -> bool:
    if a.mtf is None:
        return False
    d1 = a.mtf.biases.get("D1", "")
    h4 = a.mtf.biases.get("H4", "")
    dt = a.mtf.direction.value.lower()
    return dt in d1.lower() and dt in h4.lower()


def _rationale(a: CanonicalAsset, fv: FactorVector, themes: MarketThemes,
               flags: list[Flag], lv: LevelBundle) -> str:
    parts = [f"Score absolu {fv.absolute_mean:.2f}"]
    top = sorted(fv.present, key=lambda n: -fv.get(n))[:3]
    parts.append("forts: " + ", ".join(f"{n.split('_')[0].upper()}={fv.get(n):.2f}" for n in top))
    if lv.trigger:
        ev = lv.trigger
        parts.append(f"trigger {ev.direction.value} {ev.timeframe} ({ev.session}, {ev.bb_regime})")
    if a.mtf:
        tb = themes.bonus_for(a.base, a.quote, a.mtf.direction)
        if tb > 0.6:
            parts.append(f"thème favorable ({tb:.2f})")
    if flags:
        parts.append("flags: " + ", ".join(f.code for f in flags))
    return " · ".join(parts)


def _make_draft(a: CanonicalAsset, fv: FactorVector, themes: MarketThemes,
                cal: Optional[CalendarSets], cfg: V4Config) -> SetupV4:
    lv = build_levels(a, cfg)
    cal_status, cal_note = _compute_cal_status(a, cal)
    fs = FactorScores(
        f1_hwa=round(fv.get("f1_hwa"), 4),
        f2_rmg=round(fv.get("f2_rmg"), 4),
        f3_ext=round(fv.get("f3_ext"), 4),
        f4_trg=round(fv.get("f4_trg"), 4),
        f5_xctx=round(fv.get("f5_xctx"), 4),
        f6_theme=round(fv.get("f6_theme"), 4),
        f7_macro=round(fv.get("f7_macro"), 4),
        absolute_mean=round(fv.absolute_mean, 4),
        quantile=0.0,
        missing=list(fv.missing),
        details={n: f.detail for n, f in fv.factors.items()},
    )
    return SetupV4(
        symbol=a.symbol,
        direction=(a.mtf.direction if a.mtf else Direction.NEUTRAL),
        scenario_hint=_scenario_hint(a, lv),
        entry=lv.entry, entry_type=lv.entry_type,
        sl=lv.sl, sl_atr_multiple=lv.sl_atr_multiple,
        tp1=lv.tp1, tp1_atr_multiple=lv.tp1_atr_multiple,
        tp2=lv.tp2, tp2_atr_multiple=lv.tp2_atr_multiple,
        rr=lv.rr, rr_synthetic=(lv.tp1_synthetic or lv.tp2_synthetic),
        atr_effective=lv.atr_effective, atr_source=lv.atr_source,
        distance_atr=(lv.trigger.distance_atr_multiple or 0.0) if lv.trigger else 0.0,
        choch_score=(lv.trigger.confluence_score if lv.trigger else None),
        gps_quality=(a.mtf.quality if a.mtf else None),
        mtf_pct=(a.mtf.pct if a.mtf else 0),
        rsi_h4=_rsi_value(a, "H4"), rsi_h4_status=a.rsi_h4_status,
        age_d1=(a.mtf.age_d1 if a.mtf else 0),
        cal_status=cal_status, cal_note=cal_note,
        htf_aligned=_htf_aligned(a),
        sl_detail=lv.sl_detail, rr_detail=lv.rr_detail,
        factor_scores=fs,
    )


def run_pipeline(
    merged_path: str,
    calendar_path: Optional[str] = None,
    calendar_json_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config: V4Config = CONFIG,
) -> str:
    # 1 — ingestion
    meta, assets = load_merged(merged_path)
    if calendar_path and not calendar_json_path:
        raise NotImplementedError(
            "HTML calendar parsing is delegated upstream; pass --calendar-json.")
    calendar_data = load_calendar(calendar_json_path)
    clock = Clock.from_meta(meta.generated_at)

    # 2 — calendar buckets
    cal_sets = calendar_data.bucket(clock.now_utc)

    # 3 — universe gates
    universe = _build_universe(assets, cal_sets, config)

    # 4 — themes
    themes = detect_currency_themes(assets, config)

    # 5 — factor vectors + drafts (levels)
    vectors: list[FactorVector] = []
    drafts: list[SetupV4] = []
    for a in universe.passed:
        fv = build_factor_vector(a, themes, cal_sets, clock, config)
        vectors.append(fv)
        drafts.append(_make_draft(a, fv, themes, cal_sets, config))

    # 6 — cross-section quantiles (tie-break/diversif only)
    quantiles = compute_quantiles(vectors)
    for s in drafts:
        s.factor_scores.quantile = round(quantiles.get(s.symbol, 0.0), 4)

    # 7 — contradictions + grade (vetos -> caps -> grid)
    asset_by_sym = {a.symbol: a for a in universe.passed}
    fv_by_sym = {v.symbol: v for v in vectors}
    for s in drafts:
        a = asset_by_sym[s.symbol]
        fv = fv_by_sym[s.symbol]
        flags = detect_contradictions(a, fv, themes, cal_sets, config)
        s.flags = [FlagModel(code=f.code, severity=f.severity, detail=f.detail) for f in flags]
        cap, cap_reason = apply_caps(a, fv, config)
        if cap_reason:
            s.capped_reason = cap_reason
        s.conviction = grade(fv.absolute_mean, flags, cap, config)
        s.rationale = _rationale(a, fv, themes, flags, build_levels(a, config))

    # 8 — preflight (before rank)
    for s in drafts:
        preflight(s, config)
    valid = [s for s in drafts if s.reject_code is None]
    preflight_rejects = [s for s in drafts if s.reject_code is not None]

    # 9 — rank (absolute_mean DESC + tie-break)
    ranked = rank_setups(valid, config)

    # 10 — diversify (cluster -> representative -> cap -> top N)
    final = diversify(ranked, themes, config)

    # 11 — collect eliminated
    eliminated = _collect_eliminated(universe)
    eliminated.extend(_eliminated_from_setups(preflight_rejects))
    final_syms = {s.symbol for s in final}
    non_reps = [s for s in ranked if s.symbol not in final_syms]
    eliminated.extend(_eliminated_from_setups(non_reps))

    # 12 — render
    html = render_report(final, eliminated, meta, clock, cal_sets, themes,
                         n_passed=len(universe.passed), cfg=config)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


def _collect_eliminated(universe: Universe) -> list[Eliminated]:
    out: list[Eliminated] = []
    for asset, code, detail in universe.rejected:
        m = asset.mtf
        h4 = asset.rsi_by_tf.get("H4") if asset.rsi_by_tf else None
        out.append(Eliminated(
            symbol=asset.symbol,
            direction=(m.direction if m else Direction.NEUTRAL),
            reject_code=code.value, reject_detail=detail,
            rsi_h4=(_safe_float(h4.get("value")) if isinstance(h4, dict) else None),
            age_d1=(m.age_d1 if m else 0),
        ))
    return out


def _eliminated_from_setups(setups: list[SetupV4]) -> list[Eliminated]:
    return [Eliminated(
        symbol=s.symbol, direction=s.direction, scenario=s.scenario_hint,
        reject_code=(s.reject_code or "CLUSTER_DUP"),
        reject_detail=(s.reject_detail or s.capped_reason or "non-représentant cluster"),
        rsi_h4=s.rsi_h4, age_d1=s.age_d1, cal_status=s.cal_status, rr=s.rr,
    ) for s in setups]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — INGESTION  (ported from v9)
# ════════════════════════════════════════════════════════════════════════════
def load_merged(merged_path: str) -> tuple[MergeMeta, dict[str, CanonicalAsset]]:
    with open(merged_path, encoding="utf-8") as f:
        raw = json.load(f)
    meta = MergeMeta.model_validate(raw.get("meta", {}))
    assets: dict[str, CanonicalAsset] = {}
    for sym, a in (raw.get("assets") or {}).items():
        try:
            assets[sym] = CanonicalAsset.model_validate(a)
        except Exception as exc:  # never blocking — degraded mode
            logger.warning("asset %s skipped: %s", sym, exc)
    return meta, assets


def load_calendar(calendar_json_path: Optional[str]) -> CalendarData:
    if not calendar_json_path:
        return CalendarData()
    with open(calendar_json_path, encoding="utf-8") as f:
        raw = f.read()
    data = CalendarData.model_validate_json(raw)
    data.raw_html_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return data


# ════════════════════════════════════════════════════════════════════════════
# SECTION 16 — RENDER  (v9 template extended with V4 factor/flag sections)
# ════════════════════════════════════════════════════════════════════════════
_INLINE_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BLUESTAR FX CASCADE – {{date_hdr}}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root{
  --royal:#1B45B4;--royal-mid:#2355C3;--royal-light:#E8EEFF;--royal-dim:#6B89D8;
  --bg:#f5f7fc;--white:#fff;--card:#f0f3fa;--dark:#0d1f4e;--body:#1a1a2e;--sec:#3a4a7a;--muted:#6B89D8;--th:#E8EEFF;
  --green:#1a7a4a;--grn-bg:#e8f5ee;--grn-bd:#6EE7B7;--grn-tx:#065F46;
  --red:#c0292a;--red-bg:#fdecea;--red-bd:#FCA5A5;--red-tx:#7F1D1D;
  --blue:#2355C3;--purple:#1B45B4;
  --border:#dde3f5;--border2:#bbc6e8;--r:5px;--rl:7px;--gap:12px;
  --sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono','SF Mono','Courier New',monospace
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--body);font-family:var(--sans);font-size:12px;line-height:1.45;-webkit-font-smoothing:antialiased}
#page{max-width:1180px;margin:0 auto;background:var(--bg)}
.wrap{padding:14px 20px}
.section{background:var(--white);border:1px solid var(--border);border-radius:var(--rl);margin-bottom:var(--gap);overflow:hidden;box-shadow:0 1px 3px rgba(13,31,78,.03)}
.sec-hdr{display:flex;align-items:center;gap:10px;padding:9px 16px;border-bottom:1px solid var(--border);background:var(--white)}
.sec-num{width:22px;height:22px;border-radius:50%;background:var(--royal);color:#fff;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-family:var(--mono)}
.sec-ttl{font-size:12px;font-weight:700;color:var(--dark);text-transform:uppercase;letter-spacing:.5px;font-family:var(--mono)}
.sec-sub{margin-left:auto;font-size:9.5px;color:var(--muted);font-style:italic}
.sec-body{padding:12px 16px}
.banner{background:var(--red-bg);border:1px solid var(--red-bd);color:var(--red-tx);border-radius:var(--r);padding:9px 14px;margin-bottom:12px;font-family:var(--mono);font-size:10.5px;font-weight:600}
.setup{border:1px solid var(--border);border-radius:var(--rl);overflow:hidden;margin-bottom:11px;box-shadow:0 1px 2px rgba(13,31,78,.03)}
.setup:last-child{margin-bottom:0}
.setup.aaa{border-left:3px solid var(--royal)}.setup.aa{border-left:3px solid var(--royal-mid)}.setup.a{border-left:3px solid var(--green)}.setup.bbb{border-left:3px solid var(--muted)}.setup.bb{border-left:3px solid var(--border2)}.setup.b{border-left:3px solid var(--border2)}
.setup-hdr{display:flex;align-items:center;gap:10px;padding:9px 16px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.setup-hdr.long{background:var(--grn-bg)}.setup-hdr.short{background:var(--red-bg)}
.pair{font-size:16px;font-weight:700;font-family:var(--mono);color:var(--dark)}
.dir{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:4px;font-size:10.5px;font-weight:700;font-family:var(--mono)}
.dir.long{background:var(--grn-bg);border:1px solid var(--grn-bd);color:var(--grn-tx)}
.dir.short{background:var(--red-bg);border:1px solid var(--red-bd);color:var(--red-tx)}
.conv{display:inline-flex;padding:2px 9px;border-radius:4px;font-size:10.5px;font-weight:700;font-family:var(--mono)}
.conv.aaa{background:var(--royal-light);border:1px solid var(--royal-dim);color:var(--royal)}
.conv.aa{background:var(--royal-light);border:1px solid var(--royal-dim);color:var(--royal-mid)}
.conv.a{background:var(--grn-bg);border:1px solid var(--grn-bd);color:var(--green)}
.conv.bbb,.conv.bb,.conv.b{background:var(--card);border:1px solid var(--border2);color:var(--sec)}
.scen-lbl{margin-left:auto;font-size:9.5px;color:var(--muted);font-family:var(--mono)}
.setup-body{padding:12px 16px;background:var(--white)}
.metrics-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin-bottom:11px;padding:9px;background:var(--card);border:1px solid var(--border);border-radius:var(--r)}
.metric{text-align:center;padding:3px 0}
.metric-lbl{font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;font-family:var(--mono);margin-bottom:2px}
.metric-val{font-size:12px;font-weight:700;font-family:var(--mono)}
.metric-val.ok{color:var(--green)}.metric-val.warn{color:var(--royal)}.metric-val.danger{color:var(--red)}
.factor-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:5px;margin-bottom:11px;padding:9px;background:var(--royal-light);border:1px solid var(--royal-dim);border-radius:var(--r)}
.factor{text-align:center}
.factor-lbl{font-size:7.5px;color:var(--royal);text-transform:uppercase;letter-spacing:.5px;font-family:var(--mono);margin-bottom:2px;font-weight:700}
.factor-val{font-size:12px;font-weight:700;font-family:var(--mono);color:var(--dark)}
.factor-val.miss{color:var(--muted);font-style:italic}
.factor.mean .factor-val{color:var(--royal)}
.px-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;margin-bottom:11px}
.px-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:8px 10px;text-align:center}
.px-card.entry{border-top:2px solid var(--royal)}.px-card.sl{border-top:2px solid var(--red)}.px-card.tp1{border-top:2px solid var(--green)}.px-card.tp2{border-top:2px solid var(--royal-mid)}.px-card.rr{border-top:2px solid var(--royal-dim)}
.px-lbl{font-size:7.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin-bottom:3px;font-family:var(--mono)}
.px-val{font-size:14px;font-weight:700;font-family:var(--mono)}
.px-sub{font-size:8.5px;color:var(--muted);margin-top:2px}
.rationale{background:var(--royal-light);border-left:3px solid var(--royal);padding:9px 12px;font-size:11px;color:var(--dark);margin-bottom:10px;line-height:1.55;border-radius:var(--r)}
.rationale strong{display:block;font-size:8.5px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;color:var(--royal);font-family:var(--mono)}
.flags-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.flag{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;font-family:var(--mono)}
.flag.minor{background:var(--royal-light);border:1px solid var(--royal-dim);color:var(--royal-mid)}
.flag.major{background:var(--red-bg);border:1px solid var(--red-bd);color:var(--red-tx)}
.cap-note{font-size:9.5px;color:var(--red);font-family:var(--mono);font-weight:600;margin-bottom:8px}
.cluster-tag{font-size:9px;font-family:var(--mono);color:var(--sec);background:var(--card);border:1px solid var(--border2);padding:1px 7px;border-radius:4px}
.cal-row{display:flex;align-items:center;gap:8px;font-size:10.5px;color:var(--sec);margin-bottom:10px}
.cal-ok,.cal-prox,.cal-blackout,.cal-watch{padding:2px 8px;border-radius:4px;font-size:9.5px;font-weight:700;font-family:var(--mono)}
.cal-ok{background:var(--grn-bg);border:1px solid var(--grn-bd);color:var(--grn-tx)}
.cal-watch,.cal-prox{background:var(--royal-light);border:1px solid var(--royal-dim);color:var(--royal-mid)}
.cal-blackout{background:var(--red-bg);border:1px solid var(--red-bd);color:var(--red-tx)}
.sub-lbl{font-size:8.5px;font-weight:700;color:var(--royal);text-transform:uppercase;letter-spacing:1px;margin:11px 0 7px;font-family:var(--mono)}
.sub-lbl:first-child{margin-top:0}
.elim{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--border2);border-radius:var(--r);padding:8px 12px;margin-bottom:6px;display:flex;align-items:flex-start;gap:10px}
.elim.sus{border-left-color:var(--red);background:var(--red-bg)}
.elim-pair{font-size:12px;font-weight:700;font-family:var(--mono);color:var(--sec);min-width:84px;flex-shrink:0}
.elim-txt{font-size:10px;color:var(--muted)}
hr.div{border:none;border-top:1px solid var(--border);margin:9px 0}
table{width:100%;border-collapse:collapse;font-size:11px}
thead tr{background:var(--royal)!important}
thead th{padding:7px 10px;text-align:left;font-size:8.5px;font-weight:700;color:#fff;letter-spacing:.8px;text-transform:uppercase;white-space:nowrap;font-family:var(--mono)}
tbody tr{border-bottom:1px solid var(--border)}
tbody tr:nth-child(even){background:var(--card)}
tbody td{padding:5px 10px;vertical-align:middle}
.no-setup{background:var(--card);border:2px dashed var(--border2);border-radius:var(--rl);padding:36px 20px;text-align:center}
.no-setup-icon{font-size:32px;margin-bottom:10px}.no-setup-title{font-size:15px;font-weight:700;color:var(--dark);margin-bottom:6px}.no-setup-sub{font-size:11px;color:var(--muted);font-family:var(--mono)}
.reject-code{font-family:var(--mono);font-size:9.5px;font-weight:700;color:var(--red)}
.audit-block{background:#0d1f4e;color:#E8EEFF;border-radius:var(--r);padding:9px 12px;margin-top:10px;font-family:var(--mono);font-size:9px;line-height:1.55;word-break:break-word}
.audit-block strong{color:#6EE7B7;font-size:8.5px;text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:4px}
.footer{text-align:center;font-family:var(--mono);font-size:7.5px;color:var(--muted);border-top:1px solid var(--border);padding:9px 20px;margin-top:4px;letter-spacing:1.2px}
.page-header{background:linear-gradient(135deg,#F8FAFF 0%,#F0F4FE 100%);border:1px solid var(--border);border-radius:var(--rl) var(--rl) 0 0;display:flex;align-items:center;justify-content:space-between;padding:13px 24px;box-shadow:0 1px 4px rgba(13,31,78,.04),inset 0 1px 0 rgba(255,255,255,.8);position:relative}
.page-header::after{content:'';position:absolute;bottom:0;left:24px;right:24px;height:2px;background:linear-gradient(90deg,var(--royal),var(--royal-dim),transparent);border-radius:2px}
.header-left{display:flex;align-items:center;gap:14px}
.logo-marker{width:34px;height:34px;display:flex;align-items:center;justify-content:center;flex-shrink:0;background:var(--white);border:1px solid var(--border);border-radius:var(--r)}
.sys-label{font-size:8.5px;letter-spacing:.3em;color:var(--royal-dim);font-family:var(--mono);font-weight:600;text-transform:uppercase}
.sys-name{font-size:21px;font-weight:700;color:var(--dark);letter-spacing:-.02em;line-height:1.1;font-family:var(--mono)}
.sys-desc{font-size:8.5px;color:var(--muted);font-family:var(--mono);margin-top:2px;letter-spacing:.02em}
.header-right{text-align:right;border-left:1px solid var(--border2);padding-left:18px}
.briefing-label{font-size:10.5px;color:var(--royal);font-family:var(--mono);letter-spacing:.08em;font-weight:600;text-transform:uppercase}
.briefing-sub{font-size:8.5px;color:var(--sec);font-family:var(--mono);margin-top:4px;letter-spacing:.02em}
.page-subbar{background:rgba(27,69,180,.04);border-left:1px solid var(--border);border-right:1px solid var(--border);border-bottom:1px solid var(--border);padding:7px 24px;display:flex;align-items:center;gap:22px;flex-wrap:wrap;font-size:9.5px;font-family:var(--mono);color:var(--sec)}
.confidential{margin-left:auto;color:var(--royal);font-weight:600;background:rgba(27,69,180,.08);padding:2px 10px;border-radius:20px;font-size:8.5px}
@media print{@page{margin:8mm 7mm;size:A4 portrait}*{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}body{background:#fff!important;font-size:9.5px!important}.section{break-inside:avoid!important}.setup{break-inside:avoid!important}#pdf-fab{display:none!important}}
#pdf-fab{position:fixed;bottom:28px;right:28px;z-index:9999}
#pdf-fab button{background:#1B45B4;color:#fff;border:none;padding:11px 20px;border-radius:8px;font-family:var(--mono);font-size:12px;font-weight:700;cursor:pointer;box-shadow:0 4px 16px rgba(27,69,180,.45)}
</style>
</head>
<body>
<div id="pdf-fab"><button onclick="window.print()">Télécharger PDF</button></div>
<div id="page">
<div class="page-header">
  <div class="header-left">
    <div class="logo-marker"><svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M12 17.27L18.18 21L16.54 13.97L22 9.24L14.81 8.63L12 2L9.19 8.63L2 9.24L7.46 13.97L5.82 21L12 17.27Z" fill="#1B45B4"/></svg></div>
    <div><div class="sys-label">BLUESTAR SYSTEM</div><div class="sys-name">BLUESTAR</div><div class="sys-desc">FX INSTITUTIONAL DESK · v10 HYBRID V4</div></div>
  </div>
  <div class="header-right"><div class="briefing-label">FX CASCADE · TRADER</div><div class="briefing-sub">{{date_hdr}}</div></div>
</div>
<div class="page-subbar">
  <span>{{date_hdr}}</span><span>GMT+1</span>
  <span style="background:rgba(27,69,180,.12);color:var(--royal);padding:2px 10px;border-radius:20px;font-weight:700;border:1px solid var(--royal-dim)">{{n_setups}} setup(s)</span>
  <span>Universe <strong>{{n_passed}}/{{n_total}}</strong></span>
  <span>Event Risk : <strong style="color:{% if event_risk == 'High' %}var(--red){% elif event_risk == 'Medium' %}#EA580C{% else %}var(--green){% endif %}">{{event_risk}}</strong></span>
  {% if themes %}<span>Thèmes : {{themes}}</span>{% endif %}
  <span class="confidential">CONFIDENTIEL</span>
</div>
<div class="wrap">

<div class="section">
  <div class="sec-hdr"><div class="sec-num">1</div><div class="sec-ttl">Setups Valides</div><div class="sec-sub">{{n_setups}} validé(s) · Universe {{n_passed}}/{{n_total}}</div></div>
  <div class="sec-body">
  {% if sr_degraded %}<div class="banner">⚠ SR indisponible — niveaux en mode ATR synthétique (entrées Market, TP 2×ATR)</div>{% endif %}
  {% if setups %}
  {% for s in setups %}
  {% set dc = 'long' if s.direction.value == 'Bullish' else 'short' %}
  {% set arrow = '▲' if s.direction.value == 'Bullish' else '▼' %}
  {% set cv = s.conviction.value|lower %}
  {% set fs = s.factor_scores %}
  <div class="setup {{cv}}">
    <div class="setup-hdr {{dc}}">
      <span class="pair">{{s.symbol}}</span>
      <span class="dir {{dc}}">{{arrow}} {{s.direction.value}}</span>
      <span class="conv {{cv}}">{{s.conviction.value}} ({{ '%.2f'|format(fs.absolute_mean) }})</span>
      <span class="cluster-tag">{{s.cluster}}</span>
      <span class="scen-lbl">{{s.scenario_hint}}{% if s.cal_status.value != 'OK' %} · {{s.cal_status.value}}{% endif %}</span>
    </div>
    <div class="setup-body">
      <div class="factor-grid">
        <div class="factor"><div class="factor-lbl">F1 HWA</div><div class="factor-val {% if 'f1_hwa' in fs.missing %}miss{% endif %}">{{ '%.2f'|format(fs.f1_hwa) }}</div></div>
        <div class="factor"><div class="factor-lbl">F2 RMG</div><div class="factor-val {% if 'f2_rmg' in fs.missing %}miss{% endif %}">{{ '%.2f'|format(fs.f2_rmg) }}</div></div>
        <div class="factor"><div class="factor-lbl">F3 EXT</div><div class="factor-val {% if 'f3_ext' in fs.missing %}miss{% endif %}">{{ '%.2f'|format(fs.f3_ext) }}</div></div>
        <div class="factor"><div class="factor-lbl">F4 TRG</div><div class="factor-val {% if 'f4_trg' in fs.missing %}miss{% endif %}">{{ '%.2f'|format(fs.f4_trg) }}</div></div>
        <div class="factor"><div class="factor-lbl">F5 XCTX</div><div class="factor-val {% if 'f5_xctx' in fs.missing %}miss{% endif %}">{{ '%.2f'|format(fs.f5_xctx) }}</div></div>
        <div class="factor"><div class="factor-lbl">F6 THM</div><div class="factor-val {% if 'f6_theme' in fs.missing %}miss{% endif %}">{{ '%.2f'|format(fs.f6_theme) }}</div></div>
        <div class="factor"><div class="factor-lbl">F7 MAC</div><div class="factor-val {% if 'f7_macro' in fs.missing %}miss{% endif %}">{{ '%.2f'|format(fs.f7_macro) }}</div></div>
        <div class="factor mean"><div class="factor-lbl">Q-rang</div><div class="factor-val">{{ '%.2f'|format(fs.quantile) }}</div></div>
      </div>
      <div class="metrics-grid">
        <div class="metric"><div class="metric-lbl">Distance ATR</div><div class="metric-val {% if (s.distance_atr or 0) <= 0.3 %}ok{% elif (s.distance_atr or 0) <= 1.0 %}warn{% else %}danger{% endif %}">{{s.distance_atr|round(2)}}×</div></div>
        <div class="metric"><div class="metric-lbl">Score CHoCH</div><div class="metric-val {% if (s.choch_score or 0) >= 70 %}ok{% elif (s.choch_score or 0) >= 50 %}warn{% else %}danger{% endif %}">{{s.choch_score|round(0)|int if s.choch_score else '—'}}</div></div>
        <div class="metric"><div class="metric-lbl">Quality</div><div class="metric-val {% if s.gps_quality in ['A+','A'] %}ok{% else %}warn{% endif %}">{{s.gps_quality or '—'}}</div></div>
        <div class="metric"><div class="metric-lbl">MTF %</div><div class="metric-val {% if s.mtf_pct >= 85 %}ok{% elif s.mtf_pct >= 60 %}warn{% else %}danger{% endif %}">{{s.mtf_pct}}%</div></div>
        <div class="metric"><div class="metric-lbl">RSI H4</div><div class="metric-val {% if s.rsi_h4_status == 'favorable' %}ok{% elif 'extreme' in (s.rsi_h4_status or '') %}danger{% else %}warn{% endif %}">{{s.rsi_h4|round(1) if s.rsi_h4 else '—'}}</div></div>
        <div class="metric"><div class="metric-lbl">Age</div><div class="metric-val {% if s.age_d1 <= 15 %}ok{% elif s.age_d1 <= 30 %}warn{% else %}danger{% endif %}">{{s.age_d1}}j</div></div>
      </div>
      <div class="px-grid">
        <div class="px-card entry"><div class="px-lbl">Entry</div><div class="px-val" style="color:var(--royal)">{{s.entry}}</div><div class="px-sub">{{s.entry_type}}</div></div>
        <div class="px-card sl"><div class="px-lbl">Stop Loss</div><div class="px-val" style="color:var(--red)">{{s.sl}}</div><div class="px-sub">{{s.sl_atr_multiple|round(1)}}×ATR</div></div>
        <div class="px-card tp1"><div class="px-lbl">TP1 (60%)</div><div class="px-val" style="color:var(--green)">{{s.tp1}}</div><div class="px-sub">{% if s.tp1_atr_multiple %}{{s.tp1_atr_multiple}}×ATR{% else %}synth{% endif %}</div></div>
        <div class="px-card tp2"><div class="px-lbl">TP2 (40%)</div><div class="px-val" style="color:var(--blue)">{{s.tp2 if s.tp2 else '—'}}</div><div class="px-sub">{% if s.tp2_atr_multiple %}{{s.tp2_atr_multiple}}×ATR{% else %}synth{% endif %}</div></div>
        <div class="px-card rr"><div class="px-lbl">R : R</div><div class="px-val" style="color:var(--purple)">{{s.rr|round(2)}}</div><div class="px-sub">pondéré 60/40</div></div>
      </div>
      {% if s.flags %}<div class="flags-row">{% for f in s.flags %}<span class="flag {{f.severity}}">{{f.code}} · {{f.detail}}</span>{% endfor %}</div>{% endif %}
      {% if s.capped_reason %}<div class="cap-note">Plafond conviction appliqué : {{s.capped_reason}}</div>{% endif %}
      <div class="rationale"><strong>Rationale</strong>{{s.rationale}}{% if s.cal_note %} · <em>{{s.cal_note}}</em>{% endif %}</div>
      <div class="cal-row"><span class="cal-{{s.cal_status.value|lower}}">{{s.cal_status.value}}</span>{% if s.cal_note %}<span>{{s.cal_note}}</span>{% endif %}</div>
      <div class="audit-block"><strong>Audit Trail</strong>{{s.sl_detail}}<br>{{s.rr_detail}}<br>absolute_mean={{ '%.4f'|format(fs.absolute_mean) }} · quantile={{ '%.4f'|format(fs.quantile) }} · missing={{fs.missing}}<br>{% for k,v in fs.details.items() %}{{v}}<br>{% endfor %}ATR={{s.atr_source}} · cluster={{s.cluster}} · htf={{s.htf_aligned}}</div>
    </div>
  </div>
  {% endfor %}
  {% else %}
  <div class="no-setup"><div class="no-setup-icon">∅</div><div class="no-setup-title">Aucun setup conforme aujourd'hui</div><div class="no-setup-sub">Event Risk : {{event_risk}} · Universe {{n_passed}}/{{n_total}}</div></div>
  {% endif %}
  </div>
</div>

<div class="section">
  <div class="sec-hdr"><div class="sec-num">2</div><div class="sec-ttl">Éliminés &amp; Surveillance</div><div class="sec-sub">{{elimines|length}} actif(s) filtré(s)</div></div>
  <div class="sec-body">
  {% set suspendus = elimines | selectattr('reject_code', 'equalto', 'CAL_BLACKOUT') | list %}
  {% set rejets = elimines | rejectattr('reject_code', 'equalto', 'CAL_BLACKOUT') | list %}
  {% if suspendus %}
  <div class="sub-lbl">SUSPENDUS — Calendrier ({{suspendus|length}})</div>
  {% for e in suspendus %}
  <div class="elim sus"><span class="elim-pair">{{e.symbol}}</span><div><span class="cal-blackout" style="display:inline-flex;margin-bottom:4px">BLACKOUT</span><div class="elim-txt">{{e.reject_detail}} · RSI H4 : {{e.rsi_h4|round(2) if e.rsi_h4 else '—'}} · Age : {{e.age_d1}}j</div></div></div>
  {% endfor %}
  <hr class="div">
  {% endif %}
  {% if rejets %}
  <div class="sub-lbl">REJETS — Filtre / Preflight / Cluster ({{rejets|length}})</div>
  <table>
    <thead><tr><th>Paire</th><th>Dir.</th><th>Code</th><th>Détail</th><th>RSI H4</th><th>Age</th><th>Cal.</th></tr></thead>
    <tbody>
    {% for e in rejets %}
    {% set dc = 'long' if e.direction.value == 'Bullish' else 'short' %}
    <tr><td style="font-family:var(--mono);font-weight:700">{{e.symbol}}</td><td><span class="dir {{dc}}" style="font-size:9.5px;padding:1px 6px">{{e.direction.value}}</span></td><td class="reject-code">{{e.reject_code}}</td><td style="font-size:10px">{{e.reject_detail}}</td><td style="font-family:var(--mono);font-size:10px">{{e.rsi_h4|round(2) if e.rsi_h4 else '—'}}</td><td style="font-family:var(--mono);font-size:10px">{{e.age_d1}}j</td><td><span class="cal-{{e.cal_status.value|lower}}" style="font-size:9.5px;padding:1px 6px">{{e.cal_status.value}}</span></td></tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
  {% if not elimines %}<div style="padding:14px;color:var(--muted);font-style:italic;font-size:11px">Aucun actif éliminé ce cycle.</div>{% endif %}
  </div>
</div>

</div>
<div class="footer">CONFIDENTIEL · BLUESTAR SYSTEM v10 HYBRID V4 · {{date_hdr}} · MAX {{max_setups}} SETUPS · RR ∈ [{{rr_min}}, {{rr_max}}] · Score absolu note, quantile départage</div>
</div>
</body></html>"""


def _get_template() -> jinja2.Template:
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    tfile = os.path.join(tdir, "scaffold.html.j2")
    if os.path.isfile(tfile):
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(tdir),
                                 autoescape=jinja2.select_autoescape(["html", "j2"]))
        return env.get_template("scaffold.html.j2")
    return jinja2.Environment(autoescape=jinja2.select_autoescape(["html"])).from_string(_INLINE_TEMPLATE)


def render_report(setups: list[SetupV4], eliminated: list[Eliminated], meta: MergeMeta,
                  clock: Clock, calendar: Optional[CalendarSets], themes: Optional[MarketThemes],
                  n_passed: int, cfg: V4Config = CONFIG) -> str:
    risk = "Low"
    if calendar:
        if calendar.blackout:
            risk = "High"
        elif calendar.proximity:
            risk = "Medium"
    theme_str = ", ".join(f"{k} {v}" for k, v in (themes.strong.items() if themes else []))
    sr_degraded = all(s.rr_synthetic for s in setups) if setups else False
    return _get_template().render(
        date_hdr=clock.date_hdr,
        n_setups=len(setups),
        n_passed=n_passed,
        n_total=meta.assets_count or (n_passed + len(eliminated)),
        event_risk=risk, themes=theme_str, sr_degraded=sr_degraded,
        setups=setups, elimines=eliminated,
        max_setups=cfg.MAX_SETUPS, rr_min=cfg.RR_MIN, rr_max=cfg.RR_MAX,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 17 — CLI
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="BLUESTAR ENGINE v10 (Hybrid V4)")
    p.add_argument("--merged", required=True, help="Path to merge.json")
    p.add_argument("--calendar", help="Path to Forex Factory HTML (requires upstream LLM parse)")
    p.add_argument("--calendar-json", help="Path to pre-parsed calendar.json")
    p.add_argument("--config", help="Optional JSON config overrides")
    p.add_argument("--output", "-o", help="Output HTML path")
    args = p.parse_args()

    cfg = CONFIG
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = V4Config.from_dict(json.load(f))

    html = run_pipeline(
        merged_path=args.merged,
        calendar_path=args.calendar,
        calendar_json_path=args.calendar_json,
        output_path=args.output,
        config=cfg,
    )
    logger.info("Report generated: %d bytes%s", len(html),
                f" → {args.output}" if args.output else "")
    if not args.output:
        print(html)


if __name__ == "__main__":
    main()
