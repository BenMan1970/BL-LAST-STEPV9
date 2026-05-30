"""
BLUESTAR ENGINE v9.1 — Deterministic Execution Engine (consolidated patch)
==========================================================================
Single-file, autonomous. Source of truth = merged JSON. No hardcoded gating,
no blocking stubs. Optional integration with merge_appbackup (graceful fallback).

Phases:
  1. Ingestion   : merged JSON -> CanonicalAsset views (local validators)
  2. Calendar    : calendar JSON -> tiered CalendarSets
  3. DAG         : clock -> universe -> scenario -> conviction -> preflight -> rank
  4. Render      : Jinja2 (filesystem template if present, else inline)

Usage:
  python bluestar_engine_v9.py --merged merged.json --calendar-json calendar.json -o report.html
  from bluestar_engine_v9 import run_pipeline
  html = run_pipeline(merged_path="merged.json", calendar_json_path="calendar.json")
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional

import jinja2
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger("bluestar.v9")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — OPTIONAL upstream import (graceful fallback, never blocking)
# ════════════════════════════════════════════════════════════════════════════
try:  # pragma: no cover
    from merge_appbackup import Direction as _UpstreamDirection  # noqa: F401
    _HAS_UPSTREAM = True
except Exception:
    _HAS_UPSTREAM = False


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
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


class ScenarioCode(str, Enum):
    T1_A_PURE = "A_PURE"
    T2_A_AGED = "A_AGED"
    T3_B_TRANSITION = "B_TRANSITION"
    T4_B_MANUAL_ZONE = "B_MANUAL_ZONE"
    T5_AGE_EXCESS = "AGE_EXCESS"
    T6_STALE_TREND = "STALE_TREND"
    T7_SCENARIO_FAIL = "SCENARIO_FAIL"


class Conviction(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"


class CalStatus(str, Enum):
    OK = "OK"
    BLACKOUT = "BLACKOUT"
    PROXIMITY = "PROXIMITY"
    WATCH = "WATCH"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPERS (direction equality across str/enum)
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


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CALENDAR MODELS (tiered classification, Bug #3 / Axe 3)
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
# SECTION 4 — CANONICAL ASSET VIEW (autonomous validator layer, Axe 5b)
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
    side: str = ""          # "BUY" / "SELL" or tag
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


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DAG DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════
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


@dataclass
class MarketThemes:
    strong: dict[str, str] = field(default_factory=dict)  # ccy -> "Bullish"/"Bearish"

    def bonus_for(self, base: str, quote: Optional[str], direction: Direction) -> int:
        d = direction.value
        inv = "Bearish" if d == "Bullish" else "Bullish"
        pts = 0
        if self.strong.get(base) == d:
            pts += 5
        elif self.strong.get(base):
            pts -= 5
        if quote:
            if self.strong.get(quote) == inv:
                pts += 5
            elif self.strong.get(quote):
                pts -= 5
        return max(-10, min(10, pts))


@dataclass
class ConvictionScore:
    base_scenario: int = 0
    htf_alignment: int = 0
    choch_freshness: int = 0
    choch_score_pts: int = 0
    rsi_quality: int = 0
    bb_regime_pts: int = 0
    session_pts: int = 0
    force_vol_pts: int = 0
    cal_risk: int = 0
    nc_pts: int = 0
    scanner_coverage: int = 0
    dir_consistency: int = 0
    theme_pts: int = 0

    @property
    def total(self) -> int:
        return sum((self.base_scenario, self.htf_alignment, self.choch_freshness,
                    self.choch_score_pts, self.rsi_quality, self.bb_regime_pts,
                    self.session_pts, self.force_vol_pts, self.cal_risk, self.nc_pts,
                    self.scanner_coverage, self.dir_consistency, self.theme_pts))

    @property
    def grade(self) -> Conviction:
        t = self.total
        if self.cal_risk <= -900 or t <= -100:
            return Conviction.BBB
        if t >= 85:
            return Conviction.AAA
        if t >= 70:
            return Conviction.AA
        if t >= 55:
            return Conviction.A
        return Conviction.BBB

    def breakdown(self) -> str:
        return (f"base={self.base_scenario} htf={self.htf_alignment} fresh={self.choch_freshness} "
                f"choch={self.choch_score_pts} rsi={self.rsi_quality} bb={self.bb_regime_pts} "
                f"sess={self.session_pts} fv={self.force_vol_pts} cal={self.cal_risk} "
                f"nc={self.nc_pts} scan={self.scanner_coverage} dir={self.dir_consistency} "
                f"theme={self.theme_pts} → TOTAL={self.total}")


class Setup(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    direction: Direction
    scenario: ScenarioCode
    conviction: Conviction = Conviction.BBB
    conviction_total: int = 0
    conviction_breakdown: str = ""
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
    gate_code: GateCode = GateCode.PASS
    sl_detail: str = ""
    rr_detail: str = ""
    rationale: str = ""
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


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CONFIG (immutable, externalizable; Axe 5d)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class V9Config:
    MIN_QUALITY: frozenset = frozenset({"A+", "A"})
    MIN_CONSENSUS_PCT: int = 50
    A_PURE_MIN_PCT: int = 85
    A_AGED_MIN_AGE: int = 30
    A_AGED_MAX_AGE: int = 60
    B_MANUAL_MIN_PCT: int = 90
    B_MANUAL_MIN_SCORE: float = 75.0
    SL_FLOOR_MULT: float = 0.8
    DEFAULT_BB_MULT: float = 1.5
    BB_REGIME_MULT: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({"Squeeze": 1.0, "Normal": 1.5, "Expansion": 2.0}))
    FRESH_CANDLES_MAX: int = 2
    FRESH_CANDLES_TOLERANT: int = 6
    FRESH_ATR_MAX: float = 0.3
    LIMIT_ZONE_MAX_DIST: float = 2.0
    RR_MIN: float = 1.5
    RR_MAX: float = 20.0
    TP1_ATR_MULT: float = 2.0
    TP2_ATR_MULT: float = 1.0
    CONVICTION_RANK: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({"AAA": 4, "AA": 3, "A": 2, "BBB": 1}))
    MAX_SETUPS: int = 3
    MAX_EXPOSURE_PER_CCY: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "V9Config":
        base = cls()
        kw = {}
        for k, v in (d or {}).items():
            if hasattr(base, k):
                kw[k] = v
        return cls(**kw)


CONFIG = V9Config()

_SESSION_PTS: Mapping[str, int] = MappingProxyType({
    "london": 8, "newyork": 8, "ny": 8, "us": 8,
    "asian": 3, "tokyo": 3, "sydney": 3, "off": -10, "": 0,
})
_RSI_PTS: Mapping[str, int] = MappingProxyType({
    "favorable": 10, "grey_high": 0, "grey_low": 0,
    "overbought": -5, "oversold": -5,
    "extreme_overbought": -15, "extreme_oversold": -15,
})
_CAL_PTS: Mapping[str, int] = MappingProxyType({
    "OK": 0, "WATCH": -5, "PROXIMITY": -15, "BLACKOUT": -999,
})


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THEME DETECTION (Axe 4c)
# ════════════════════════════════════════════════════════════════════════════
def detect_currency_themes(assets: dict[str, CanonicalAsset]) -> MarketThemes:
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
    for ccy, vs in votes.items():
        if len(vs) < 3:
            continue
        bull = vs.count("Bullish") / len(vs)
        if bull >= 0.8:
            strong[ccy] = "Bullish"
        elif bull <= 0.2:
            strong[ccy] = "Bearish"
    return MarketThemes(strong)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DAG ENGINE
# ════════════════════════════════════════════════════════════════════════════
class DAGEngine:
    def __init__(self, config: V9Config = CONFIG) -> None:
        self.cfg = config
        self._cal: Optional[CalendarSets] = None
        self._theme: Optional[MarketThemes] = None

    # ── §2 clock ──
    def node_clock(self, meta: MergeMeta) -> Clock:
        return Clock.from_meta(meta.generated_at)

    # ── §4 calendar ──
    def node_cal_parse(self, calendar_data: CalendarData, clock: Clock) -> CalendarSets:
        self._cal = calendar_data.bucket(clock.now_utc)
        return self._cal

    # ── §5 universe ──
    def node_universe(self, assets: dict[str, CanonicalAsset], calendar: CalendarSets) -> Universe:
        passed: list[CanonicalAsset] = []
        rejected: list[tuple[CanonicalAsset, GateCode, str]] = []
        for asset in assets.values():
            if asset.mtf is None:
                rejected.append((asset, GateCode.G0_SCHEMA_ASSET_ERROR, "MTF manquant"))
                continue
            base, quote = asset.base, (asset.quote or "")
            if base in calendar.suspended_ccy or quote in calendar.suspended_ccy:
                hit = ({base, quote} & calendar.suspended_ccy)
                rejected.append((asset, GateCode.G1_CAL_BLACKOUT, f"Blackout: {sorted(hit)}"))
                continue
            quality = asset.mtf.quality or ""
            if quality not in self.cfg.MIN_QUALITY:
                rejected.append((asset, GateCode.G2_LOW_QUALITY, f"Quality {quality}"))
                continue
            if asset.mtf.direction is Direction.NEUTRAL:
                rejected.append((asset, GateCode.G3_NO_DIRECTION, "Direction Neutral"))
                continue
            if asset.mtf.pct < self.cfg.MIN_CONSENSUS_PCT:
                rejected.append((asset, GateCode.G4_LOW_CONSENSUS, f"MTF {asset.mtf.pct}%"))
                continue
            if asset.atr_effective is None or asset.atr_effective <= 0:
                rejected.append((asset, GateCode.G5_NO_ATR, f"ATR {asset.atr_source}"))
                continue
            passed.append(asset)
        return Universe(passed=passed, rejected=rejected)

    # ── §6 scenario ──
    def node_scenario(self, universe: Universe) -> list[Setup]:
        return [self._classify_scenario(a) for a in universe.passed]

    def _classify_scenario(self, asset: CanonicalAsset) -> Setup:
        mtf = asset.mtf
        assert mtf is not None
        direction = mtf.direction
        quality = mtf.quality or ""
        pct = mtf.pct
        age = mtf.age_d1

        # Counter-trend guard (Bug #4): a fresh CHoCH against MTF with no aligned one -> T7
        counter = any(
            ev.status.lower() == "fresh"
            and ev.candles_elapsed <= self.cfg.FRESH_CANDLES_MAX
            and not _dir_eq(ev.direction, direction)
            for ev in asset.structure_events
        )
        if counter and self._aligned_fresh_choch(asset) is None:
            return self._build_reject(asset, ScenarioCode.T7_SCENARIO_FAIL,
                                      "CHoCH counter-trend vs MTF (reversal non géré)")

        tolerant = (quality == "A+" and pct >= self.cfg.A_PURE_MIN_PCT)

        # T1 A_PURE (tolerant freshness for A+/100%, Bug #5)
        if (quality in self.cfg.MIN_QUALITY and pct >= self.cfg.A_PURE_MIN_PCT
                and self._has_htf_alignment(asset)
                and (self._has_fresh_choch(asset, tolerant=tolerant) or self._has_hot_zone(asset))):
            return self._build_setup(asset, ScenarioCode.T1_A_PURE)

        # T2 A_AGED
        if (quality in self.cfg.MIN_QUALITY and pct >= self.cfg.A_PURE_MIN_PCT
                and self.cfg.A_AGED_MIN_AGE <= age <= self.cfg.A_AGED_MAX_AGE
                and self._has_fresh_choch(asset, tolerant=tolerant)):
            return self._build_setup(asset, ScenarioCode.T2_A_AGED)

        # T3 B_TRANSITION (zone optional if a fresh aligned CHoCH exists)
        if (self._aligned_fresh_choch(asset) is not None
                and not self._is_rsi_extreme(asset)
                and (self._has_near_zone(asset) or True)):
            return self._build_setup(asset, ScenarioCode.T3_B_TRANSITION)

        # T4 B_MANUAL_ZONE
        if (quality == "A+" and pct >= self.cfg.B_MANUAL_MIN_PCT and self._has_high_score_zone(asset)):
            return self._build_setup(asset, ScenarioCode.T4_B_MANUAL_ZONE)

        # T5-T7 rejects
        if age > self.cfg.A_AGED_MAX_AGE:
            return self._build_reject(asset, ScenarioCode.T5_AGE_EXCESS, f"Age {age}j")
        if self._aligned_fresh_choch(asset) is None and not self._has_hot_zone(asset):
            return self._build_reject(asset, ScenarioCode.T6_STALE_TREND, "Ni CHoCH Fresh aligné ni Hot Zone")
        return self._build_reject(asset, ScenarioCode.T7_SCENARIO_FAIL, "Aucun scénario")

    # ── scenario helpers ──
    def _aligned_fresh_choch_cap(self, asset: CanonicalAsset, cap: int) -> Optional[StructureEventView]:
        if asset.mtf is None:
            return None
        want = asset.mtf.direction
        for ev in asset.structure_events:
            if (ev.status.lower() == "fresh" and ev.candles_elapsed <= cap
                    and _dir_eq(ev.direction, want)):
                return ev
        return None

    def _aligned_fresh_choch(self, asset: CanonicalAsset) -> Optional[StructureEventView]:
        return self._aligned_fresh_choch_cap(asset, self.cfg.FRESH_CANDLES_MAX)

    def _has_fresh_choch(self, asset: CanonicalAsset, tolerant: bool = False) -> bool:
        cap = self.cfg.FRESH_CANDLES_TOLERANT if tolerant else self.cfg.FRESH_CANDLES_MAX
        return self._aligned_fresh_choch_cap(asset, cap) is not None

    def _get_aligned_choch(self, asset: CanonicalAsset, tolerant: bool = False) -> Optional[StructureEventView]:
        cap = self.cfg.FRESH_CANDLES_TOLERANT if tolerant else self.cfg.FRESH_CANDLES_MAX
        return self._aligned_fresh_choch_cap(asset, cap)

    def _has_htf_alignment(self, asset: CanonicalAsset) -> bool:
        if asset.mtf is None:
            return False
        d1 = asset.mtf.biases.get("D1", "")
        h4 = asset.mtf.biases.get("H4", "")
        dt = asset.mtf.direction.value.lower()
        return dt in d1.lower() and dt in h4.lower()

    def _has_hot_zone(self, asset: CanonicalAsset) -> bool:
        return asset.hot_zone_primary is not None

    def _is_rsi_extreme(self, asset: CanonicalAsset) -> bool:
        return asset.rsi_h4_status in ("extreme_overbought", "extreme_oversold")

    def _has_near_zone(self, asset: CanonicalAsset) -> bool:
        z = asset.nearest_aligned_zone
        return z is not None and z.distance_pct <= self.cfg.LIMIT_ZONE_MAX_DIST

    def _has_high_score_zone(self, asset: CanonicalAsset) -> bool:
        return any(z.score >= self.cfg.B_MANUAL_MIN_SCORE for z in asset.zones)

    # ── level computation (§8) ──
    def _build_setup(self, asset: CanonicalAsset, scenario: ScenarioCode) -> Setup:
        mtf = asset.mtf
        assert mtf is not None
        tolerant = (mtf.quality == "A+" and mtf.pct >= self.cfg.A_PURE_MIN_PCT)
        ev = self._get_aligned_choch(asset, tolerant=tolerant)
        atr, atr_src = self._atr_for_signal(asset, ev)

        entry, entry_type = self._compute_entry(asset, scenario, ev, atr)
        sl, sl_mult, sl_detail = self._compute_sl(asset, entry, atr, ev)
        tp1, tp1_mult, tp1_syn = self._compute_tp1(asset, entry, atr)
        tp2, tp2_mult, tp2_syn = self._compute_tp2(asset, entry, tp1, atr)
        rr, rr_detail = self._compute_rr(entry, sl, tp1, tp2, tp1_syn, tp2_syn)
        cal_status, cal_note = self._compute_cal_status(asset)

        return Setup(
            symbol=asset.symbol, direction=mtf.direction, scenario=scenario,
            entry=round(entry, 5), entry_type=entry_type,
            sl=round(sl, 5), sl_atr_multiple=sl_mult,
            tp1=round(tp1, 5), tp1_atr_multiple=tp1_mult,
            tp2=(round(tp2, 5) if tp2 is not None else None), tp2_atr_multiple=tp2_mult,
            rr=rr, rr_synthetic=(tp1_syn or tp2_syn),
            atr_effective=atr, atr_source=atr_src,
            distance_atr=(ev.distance_atr_multiple or 0.0) if ev else 0.0,
            choch_score=(ev.confluence_score if ev else None),
            gps_quality=mtf.quality, mtf_pct=mtf.pct,
            rsi_h4=self._get_rsi_h4(asset), rsi_h4_status=asset.rsi_h4_status,
            age_d1=mtf.age_d1, cal_status=cal_status, cal_note=cal_note,
            htf_aligned=self._has_htf_alignment(asset), gate_code=GateCode.PASS,
            sl_detail=sl_detail, rr_detail=rr_detail,
            rationale=self._build_rationale(asset, scenario, entry_type, ev),
        )

    def _build_reject(self, asset: CanonicalAsset, scenario: ScenarioCode, detail: str) -> Setup:
        mtf = asset.mtf
        return Setup(
            symbol=asset.symbol,
            direction=(mtf.direction if mtf else Direction.NEUTRAL),
            scenario=scenario, gps_quality=(mtf.quality if mtf else None),
            mtf_pct=(mtf.pct if mtf else 0), rsi_h4=self._get_rsi_h4(asset),
            rsi_h4_status=asset.rsi_h4_status, age_d1=(mtf.age_d1 if mtf else 0),
            atr_effective=(asset.atr_effective or 0.0), atr_source=(asset.atr_source or "unknown"),
            reject_code=scenario.value, reject_detail=detail,
        )

    def _atr_for_signal(self, asset: CanonicalAsset, ev: Optional[StructureEventView]) -> tuple[float, str]:
        if ev is not None and asset.mtf:
            tf = (ev.timeframe or "").upper()
            m = {"H1": asset.mtf.atr_h1, "H4": asset.mtf.atr_h4, "D1": asset.mtf.atr_daily}
            v = m.get(tf)
            if v and v > 0:
                return float(v), f"atr_{tf.lower()}"
        return (asset.atr_effective or 0.0), (asset.atr_source or "h4")

    def _compute_entry(self, asset, scenario, ev, atr) -> tuple[float, str]:
        price = asset.current_price or 0.0
        if ev and ev.candles_elapsed <= 1 and (ev.distance_atr_multiple or 999) <= self.cfg.FRESH_ATR_MAX:
            return price, "Market"
        z = asset.nearest_aligned_zone
        if z and z.distance_pct <= self.cfg.LIMIT_ZONE_MAX_DIST:
            return z.level, "Limit"
        if scenario == ScenarioCode.T4_B_MANUAL_ZONE:
            return price, "Market"
        if asset.hot_zone_primary:
            return asset.hot_zone_primary.level, "Limit"
        return price, "Market"

    def _compute_sl(self, asset, entry, atr, ev) -> tuple[float, float, str]:
        direction = asset.mtf.direction if asset.mtf else Direction.NEUTRAL
        bb_regime = ev.bb_regime if ev else "Normal"
        bb_mult = self.cfg.BB_REGIME_MULT.get(bb_regime, self.cfg.DEFAULT_BB_MULT)
        if direction is Direction.BULLISH:
            sl_raw = entry - atr * bb_mult
        elif direction is Direction.BEARISH:
            sl_raw = entry + atr * bb_mult
        else:
            sl_raw = entry
        sl = sl_raw
        detail = f"Raw SL={sl_raw:.5f} ({bb_regime} ×{bb_mult})"
        z = asset.nearest_aligned_zone
        if z and z.distance_pct <= self.cfg.LIMIT_ZONE_MAX_DIST:
            if direction is Direction.BULLISH:
                sl = min(sl_raw, z.level - 0.3 * atr)
            elif direction is Direction.BEARISH:
                sl = max(sl_raw, z.level + 0.3 * atr)
            detail += f" zone-adj→{sl:.5f}"
        min_dist = atr * self.cfg.SL_FLOOR_MULT
        if abs(entry - sl) < min_dist:
            sl = entry - min_dist if direction is Direction.BULLISH else entry + min_dist
            detail += f" [floored {self.cfg.SL_FLOOR_MULT}×ATR]"
        return sl, bb_mult, detail

    def _compute_tp1(self, asset, entry, atr) -> tuple[float, Optional[float], bool]:
        direction = asset.mtf.direction if asset.mtf else Direction.NEUTRAL
        opp = self._get_opposite_zone(asset)
        if opp:
            return opp.level, (round(abs(opp.level - entry) / atr, 2) if atr > 0 else None), False
        tp1 = entry + self.cfg.TP1_ATR_MULT * atr if direction is Direction.BULLISH else entry - self.cfg.TP1_ATR_MULT * atr
        return tp1, self.cfg.TP1_ATR_MULT, True

    def _compute_tp2(self, asset, entry, tp1, atr) -> tuple[Optional[float], Optional[float], bool]:
        direction = asset.mtf.direction if asset.mtf else Direction.NEUTRAL
        opp = [z for z in sorted(asset.zones, key=lambda z: z.distance_pct)
               if self._is_opposite(z, direction)]
        if len(opp) >= 2:
            lvl = opp[1].level
            return lvl, (round(abs(lvl - entry) / atr, 2) if atr > 0 else None), False
        tp2 = tp1 + self.cfg.TP2_ATR_MULT * atr if direction is Direction.BULLISH else tp1 - self.cfg.TP2_ATR_MULT * atr
        return tp2, (round(abs(tp2 - entry) / atr, 2) if atr > 0 else None), True

    def _compute_rr(self, entry, sl, tp1, tp2, tp1_syn, tp2_syn) -> tuple[float, str]:
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

    def _get_opposite_zone(self, asset) -> Optional[ZoneView]:
        direction = asset.mtf.direction if asset.mtf else Direction.NEUTRAL
        zs = [z for z in asset.zones if self._is_opposite(z, direction)]
        return min(zs, key=lambda z: z.distance_pct) if zs else None

    @staticmethod
    def _is_opposite(zone: ZoneView, direction: Direction) -> bool:
        side = (zone.side or "").upper()
        if direction is Direction.BULLISH:
            return side in ("SELL", "RESISTANCE", "SUPPLY")
        if direction is Direction.BEARISH:
            return side in ("BUY", "SUPPORT", "DEMAND")
        return False

    def _get_rsi_h4(self, asset) -> Optional[float]:
        h4 = asset.rsi_by_tf.get("H4")
        return _safe_float(h4.get("value")) if isinstance(h4, dict) else None

    # ── §4 cal status (Bug #1, both sides — Axe 3a) ──
    def _compute_cal_status(self, asset: CanonicalAsset) -> tuple[CalStatus, str]:
        if self._cal is None:
            return CalStatus.OK, ""
        sides = {asset.base, (asset.quote or "")}
        hit_black = sides & self._cal.suspended_ccy
        if hit_black:
            names = [f"{e.currency} {e.event_name}" for e in self._cal.blackout if e.currency in hit_black]
            return CalStatus.BLACKOUT, "; ".join(names[:3])
        hit_prox = sides & self._cal.proximity_ccy
        if hit_prox:
            return CalStatus.PROXIMITY, ", ".join(sorted(hit_prox))
        hit_watch = sides & self._cal.watch_ccy
        if hit_watch:
            return CalStatus.WATCH, ", ".join(sorted(hit_watch))
        return CalStatus.OK, ""

    def _build_rationale(self, asset, scenario, entry_type, ev) -> str:
        parts = [f"Scénario {scenario.value}", f"Entry {entry_type}"]
        if ev:
            parts.append(f"CHoCH {ev.direction.value} {ev.timeframe} score={ev.confluence_score:.0f} "
                         f"({ev.candles_elapsed}c, {ev.bb_regime}, {ev.session})")
        if asset.hot_zone_primary:
            parts.append("Hot Zone")
        if self._theme and asset.mtf:
            tb = self._theme.bonus_for(asset.base, asset.quote, asset.mtf.direction)
            if tb:
                parts.append(f"Theme {'+' if tb > 0 else ''}{tb}")
        return " · ".join(parts)

    # ── §9 conviction (composite, Axe 4) ──
    def node_conviction(self, setups: list[Setup], assets: dict[str, CanonicalAsset]) -> list[Setup]:
        for s in setups:
            asset = assets.get(s.symbol)
            if asset is None:
                continue
            tolerant = (asset.mtf and asset.mtf.quality == "A+" and asset.mtf.pct >= self.cfg.A_PURE_MIN_PCT)
            ev = self._get_aligned_choch(asset, tolerant=bool(tolerant))
            score = self._score_conviction(asset, s, ev)
            s.conviction = score.grade
            s.conviction_total = score.total
            s.conviction_breakdown = score.breakdown()
        return setups

    def _score_conviction(self, asset, setup: Setup, ev) -> ConvictionScore:
        cs = ConvictionScore()
        cs.base_scenario = {
            ScenarioCode.T1_A_PURE: 40, ScenarioCode.T2_A_AGED: 30,
            ScenarioCode.T3_B_TRANSITION: 20, ScenarioCode.T4_B_MANUAL_ZONE: 20,
        }.get(setup.scenario, 0)
        cs.htf_alignment = 20 if setup.htf_aligned else -20
        if ev:
            c = ev.candles_elapsed
            cs.choch_freshness = 15 if c <= 1 else (10 if c <= 2 else 0)
            cs.choch_score_pts = round((min(ev.confluence_score, 85.0) / 85.0) * 15)
            cs.bb_regime_pts = {"Squeeze": 10, "Normal": 0, "Expansion": -5}.get(ev.bb_regime, 0)
            cs.session_pts = _SESSION_PTS.get((ev.session or "").lower(), 0)
            fv = 0
            if (ev.force or "").lower() in ("fort", "strong"):
                fv += 5
            elif (ev.force or "").lower() in ("faible", "weak"):
                fv -= 5
            if (ev.volatility or "").lower() in ("haute", "high"):
                fv += 2
            cs.force_vol_pts = fv
            if asset.mtf:
                cs.dir_consistency = 0 if _dir_eq(ev.direction, asset.mtf.direction) else -30
        cs.rsi_quality = _RSI_PTS.get(setup.rsi_h4_status or "", 0)
        nc = asset.mtf.nc if asset.mtf else 0
        cs.nc_pts = 5 if nc >= 5 else (2 if nc >= 3 else 0)
        n_scan = len([k for k, v in (asset.provenance or {}).items() if v])
        cs.scanner_coverage = 10 if n_scan >= 4 else (5 if n_scan == 3 else -10)
        cs.cal_risk = _CAL_PTS.get(setup.cal_status.value, 0)
        if self._theme and asset.mtf:
            cs.theme_pts = self._theme.bonus_for(asset.base, asset.quote, asset.mtf.direction)
        return cs

    # ── §11 preflight (BEFORE rank — Bug #6) ──
    def node_preflight(self, setups: list[Setup]) -> list[Setup]:
        valid: list[Setup] = []
        for s in setups:
            if s.cal_status is CalStatus.BLACKOUT:
                s.reject_code = "CAL_BLACKOUT"
                s.reject_detail = s.cal_note
                continue
            if not (self.cfg.RR_MIN <= s.rr <= self.cfg.RR_MAX):
                s.reject_code = "RR_OUT_OF_RANGE"
                s.reject_detail = f"RR {s.rr} ∉ [{self.cfg.RR_MIN},{self.cfg.RR_MAX}]"
                continue
            if s.direction is Direction.BULLISH and s.sl >= s.entry:
                s.reject_code = "SL_SIGN"
                s.reject_detail = "SL ≥ entry (bullish)"
                continue
            if s.direction is Direction.BEARISH and s.sl <= s.entry:
                s.reject_code = "SL_SIGN"
                s.reject_detail = "SL ≤ entry (bearish)"
                continue
            valid.append(s)
        return valid

    # ── §10 rank (after preflight) ──
    def node_rank(self, setups: list[Setup]) -> list[Setup]:
        cal_order = {CalStatus.OK: 0, CalStatus.WATCH: 1, CalStatus.PROXIMITY: 2, CalStatus.BLACKOUT: 3}

        def key(s: Setup):
            return (
                -self.cfg.CONVICTION_RANK.get(s.conviction.value, 0),
                -s.conviction_total,
                cal_order.get(s.cal_status, 9),
                -s.rr,
                -(s.choch_score or 0),
                -s.mtf_pct,
            )

        setups.sort(key=key)
        return setups[: self.cfg.MAX_SETUPS]

    # ── portfolio exposure cap (Axe 6d) ──
    def node_portfolio(self, setups: list[Setup]) -> list[Setup]:
        net: Counter = Counter()
        kept: list[Setup] = []
        for s in sorted(setups, key=lambda x: -self.cfg.CONVICTION_RANK.get(x.conviction.value, 0)):
            if "/" in s.symbol:
                b, q = s.symbol.split("/", 1)
            else:
                b, q = s.symbol, ""
            sign = 1 if s.direction is Direction.BULLISH else -1
            if abs(net[b] + sign) > self.cfg.MAX_EXPOSURE_PER_CCY or \
               (q and abs(net[q] - sign) > self.cfg.MAX_EXPOSURE_PER_CCY):
                s.cal_note = (s.cal_note + " [capped: exposition devise]").strip()
                continue
            net[b] += sign
            if q:
                net[q] -= sign
            kept.append(s)
        return kept


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — RENDER (filesystem template if present, else inline)
# ════════════════════════════════════════════════════════════════════════════
_INLINE_TEMPLATE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BLUESTAR FX CASCADE – {{date_hdr}}</title>
<style>
:root{--royal:#1B45B4;--green:#1a7a4a;--red:#c0292a;--bg:#f5f7fc}
*{box-sizing:border-box}
body{background:var(--bg);font-family:system-ui,-apple-system,sans-serif;font-size:13px;margin:0;padding:16px;color:#1a2233}
.page-header{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:3px solid var(--royal);padding-bottom:10px;margin-bottom:6px}
.sys-name{font-size:26px;font-weight:800;color:var(--royal);letter-spacing:1px}
.sys-desc{font-size:11px;color:#667}
.briefing-label{font-weight:700;color:var(--royal)}
.briefing-sub{font-size:11px;color:#667;text-align:right}
.page-subbar{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:#556;padding:8px 0 16px}
.page-subbar span{background:#fff;border:1px solid #d7deee;border-radius:4px;padding:3px 8px}
.section{margin-bottom:22px}
.sec-hdr{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.sec-num{background:var(--royal);color:#fff;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px}
.sec-ttl{font-size:16px;font-weight:700}
.setup{border:1px solid #d7deee;border-radius:8px;margin:10px 0;overflow:hidden;background:#fff}
.setup.aaa{border-left:4px solid var(--royal)}
.setup.aa{border-left:4px solid #3a6bd6}
.setup.a{border-left:4px solid #6f96e8}
.setup.bbb{border-left:4px solid #aab6d0}
.setup-hdr{display:flex;gap:12px;align-items:center;padding:9px 13px;background:#eef2fb;font-weight:600}
.pair{font-size:15px;font-weight:800}
.dir.bullish{color:var(--green)}.dir.bearish{color:var(--red)}
.conv{margin-left:auto;background:var(--royal);color:#fff;padding:2px 8px;border-radius:4px;font-size:11px}
.scen-lbl{font-size:11px;color:#667;background:#fff;border:1px solid #d7deee;padding:2px 6px;border-radius:4px}
.cal-tag{font-size:10px;padding:2px 6px;border-radius:4px}
.cal-tag.watch{background:#fff4d6;color:#8a6d00}
.cal-tag.proximity{background:#ffe0d6;color:#9a3b00}
.px-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:12px}
.px-card{background:#f8faff;border:1px solid #e3e9f5;border-radius:6px;padding:8px;text-align:center}
.px-lbl{font-size:10px;color:#778;text-transform:uppercase;letter-spacing:.5px}
.px-val{font-size:16px;font-weight:700}
.px-sub{font-size:10px;color:#889}
.rationale,.audit-block{padding:8px 13px;font-size:12px;border-top:1px solid #eef}
.audit-block{font-family:ui-monospace,monospace;font-size:11px;color:#556;background:#fafbfe}
.elim-table{width:100%;border-collapse:collapse;font-size:11px}
.elim-table th,.elim-table td{border:1px solid #e3e9f5;padding:5px 7px;text-align:left}
.elim-table th{background:#eef2fb}
.footer{margin-top:24px;font-size:11px;color:#889;text-align:center;border-top:1px solid #d7deee;padding-top:10px}
.empty{padding:14px;color:#889;font-style:italic}
</style></head><body>
<div class="page-header">
  <div><div class="sys-name">BLUESTAR</div><div class="sys-desc">FX INSTITUTIONAL DESK · v9.1 DETERMINISTIC</div></div>
  <div><div class="briefing-label">FX CASCADE · TRADER</div><div class="briefing-sub">{{date_hdr}}</div></div>
</div>
<div class="page-subbar">
  <span>{{date_hdr}}</span><span>GMT+1</span>
  <span>{{n_setups}} setup(s)</span>
  <span>Universe {{n_passed}}/{{n_total}}</span>
  <span>Event Risk: {{event_risk}}</span>
  {% if themes %}<span>Thèmes: {{themes}}</span>{% endif %}
</div>
<div class="section">
  <div class="sec-hdr"><div class="sec-num">1</div><div class="sec-ttl">Setups Valides</div></div>
  {% if setups %}{% for s in setups %}
  <div class="setup {{s.conviction.value|lower}}">
    <div class="setup-hdr">
      <span class="pair">{{s.symbol}}</span>
      <span class="dir {{s.direction.value|lower}}">{{s.direction.value}}</span>
      <span class="scen-lbl">{{s.scenario.value}}</span>
      {% if s.cal_status.value != 'OK' %}<span class="cal-tag {{s.cal_status.value|lower}}">{{s.cal_status.value}}</span>{% endif %}
      <span class="conv">{{s.conviction.value}} ({{s.conviction_total}})</span>
    </div>
    <div class="px-grid">
      <div class="px-card"><div class="px-lbl">Entry</div><div class="px-val">{{"%.5f"|format(s.entry)}}</div><div class="px-sub">{{s.entry_type}}</div></div>
      <div class="px-card"><div class="px-lbl">Stop Loss</div><div class="px-val">{{"%.5f"|format(s.sl)}}</div><div class="px-sub">{{"%.1f"|format(s.sl_atr_multiple)}}×ATR</div></div>
      <div class="px-card"><div class="px-lbl">TP1</div><div class="px-val">{{"%.5f"|format(s.tp1)}}</div><div class="px-sub">{% if s.tp1_atr_multiple %}{{s.tp1_atr_multiple}}×ATR{% endif %}</div></div>
      <div class="px-card"><div class="px-lbl">R:R</div><div class="px-val">{{"%.2f"|format(s.rr)}}</div><div class="px-sub">{% if s.rr_synthetic %}synth{% endif %}</div></div>
    </div>
    <div class="rationale"><strong>Rationale · </strong>{{s.rationale}}{% if s.cal_note %} · <em>{{s.cal_note}}</em>{% endif %}</div>
    <div class="audit-block">{{s.sl_detail}}<br>{{s.rr_detail}}<br>{{s.conviction_breakdown}}</div>
  </div>
  {% endfor %}{% else %}<div class="empty">Aucun setup valide ce cycle.</div>{% endif %}
</div>
<div class="section">
  <div class="sec-hdr"><div class="sec-num">2</div><div class="sec-ttl">Éliminés ({{elimines|length}})</div></div>
  {% if elimines %}
  <table class="elim-table"><thead><tr><th>Symbol</th><th>Dir</th><th>Code</th><th>Détail</th><th>RSI H4</th><th>Age</th><th>Cal</th></tr></thead><tbody>
  {% for e in elimines %}<tr><td>{{e.symbol}}</td><td>{{e.direction.value}}</td><td>{{e.reject_code}}</td><td>{{e.reject_detail}}</td><td>{{e.rsi_h4 or '—'}}</td><td>{{e.age_d1}}</td><td>{{e.cal_status.value}}</td></tr>{% endfor %}
  </tbody></table>
  {% else %}<div class="empty">—</div>{% endif %}
</div>
<div class="footer">BLUESTAR v9.1 · {{date_hdr}} · MAX {{max_setups}} SETUPS · RR ∈ [{{rr_min}}, {{rr_max}}]</div>
</body></html>"""


def _get_template() -> jinja2.Template:
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    tfile = os.path.join(tdir, "scaffold.html.j2")
    if os.path.isfile(tfile):
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(tdir),
                                 autoescape=jinja2.select_autoescape(["html", "j2"]))
        return env.get_template("scaffold.html.j2")
    return jinja2.Environment(autoescape=jinja2.select_autoescape(["html"])).from_string(_INLINE_TEMPLATE)


def render_scaffold(setups: list[Setup], elimines: list[Eliminated], meta: MergeMeta,
                    clock: Clock, calendar: Optional[CalendarSets], themes: Optional[MarketThemes],
                    n_passed: int, cfg: V9Config = CONFIG) -> str:
    risk = "Low"
    if calendar:
        if calendar.blackout:
            risk = "High"
        elif calendar.proximity:
            risk = "Medium"
    theme_str = ", ".join(f"{k} {v}" for k, v in (themes.strong.items() if themes else []))
    return _get_template().render(
        date_hdr=clock.date_hdr,
        n_setups=len(setups),
        n_passed=n_passed,
        n_total=meta.assets_count or (n_passed + len(elimines)),
        event_risk=risk, themes=theme_str,
        setups=setups, elimines=elimines,
        max_setups=cfg.MAX_SETUPS, rr_min=cfg.RR_MIN, rr_max=cfg.RR_MAX,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — INGESTION (autonomous JSON parsing)
# ════════════════════════════════════════════════════════════════════════════
def load_merged(merged_path: str) -> tuple[MergeMeta, dict[str, CanonicalAsset]]:
    with open(merged_path, encoding="utf-8") as f:
        raw = json.load(f)
    meta = MergeMeta.model_validate(raw.get("meta", {}))
    assets: dict[str, CanonicalAsset] = {}
    for sym, a in (raw.get("assets") or {}).items():
        try:
            assets[sym] = CanonicalAsset.model_validate(a)
        except Exception as exc:  # never blocking — skip malformed asset
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
# SECTION 11 — ORCHESTRATION (Bug #2 fully implemented)
# ════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    merged_path: str,
    calendar_path: Optional[str] = None,
    calendar_json_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config: V9Config = CONFIG,
) -> str:
    # Phase 1 — ingestion
    meta, assets = load_merged(merged_path)

    # Phase 2 — calendar
    if calendar_path and not calendar_json_path:
        raise NotImplementedError(
            "HTML calendar parsing is delegated to an LLM upstream; pass --calendar-json.")
    calendar_data = load_calendar(calendar_json_path)

    # Phase 3 — DAG
    dag = DAGEngine(config)
    clock = dag.node_clock(meta)
    calendar_sets = dag.node_cal_parse(calendar_data, clock)
    dag._theme = detect_currency_themes(assets)

    universe = dag.node_universe(assets, calendar_sets)
    setups_all = dag.node_scenario(universe)

    playable = [s for s in setups_all if s.reject_code is None]
    eliminated = _collect_eliminated(universe, setups_all)

    playable = dag.node_conviction(playable, assets)
    playable = dag.node_preflight(playable)        # preflight BEFORE rank
    # capture preflight rejects into eliminated
    eliminated.extend(_eliminated_from_setups([s for s in playable if s.reject_code]))
    playable = [s for s in playable if s.reject_code is None]
    playable = dag.node_rank(playable)
    playable = dag.node_portfolio(playable)

    # Phase 4 — render
    html = render_scaffold(playable, eliminated, meta, clock, calendar_sets,
                           dag._theme, n_passed=len(universe.passed), cfg=config)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


def _collect_eliminated(universe: Universe, setups_all: list[Setup]) -> list[Eliminated]:
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
    out.extend(_eliminated_from_setups([s for s in setups_all if s.reject_code is not None]))
    return out


def _eliminated_from_setups(setups: list[Setup]) -> list[Eliminated]:
    return [Eliminated(
        symbol=s.symbol, direction=s.direction, scenario=s.scenario.value,
        reject_code=s.reject_code or "UNKNOWN", reject_detail=s.reject_detail or "",
        rsi_h4=s.rsi_h4, age_d1=s.age_d1, cal_status=s.cal_status, rr=s.rr,
    ) for s in setups]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — CLI
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="BLUESTAR ENGINE v9.1")
    p.add_argument("--merged", required=True, help="Path to merged_pipeline_*.json")
    p.add_argument("--calendar", help="Path to Forex Factory HTML (requires upstream LLM parse)")
    p.add_argument("--calendar-json", help="Path to pre-parsed calendar.json")
    p.add_argument("--config", help="Optional JSON config overrides")
    p.add_argument("--output", "-o", help="Output HTML path")
    args = p.parse_args()

    cfg = CONFIG
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = V9Config.from_dict(json.load(f))

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
