# pylint: disable=invalid-name
# (module file is named ENGINE.V9.py for historical/import reasons — C0103)
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
  python engine.py --merged merged.json --calendar-json calendar.json -o report.html
  from engine import run_pipeline
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
    from merge_appbackup import Direction as _UpstreamDirection  # noqa: F401  # pylint: disable=import-error
    _HAS_UPSTREAM = True
except Exception:  # pylint: disable=broad-exception-caught
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
    PASS = "PASS"  # nosec B105 — gate code label, not a password
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
        # f == f is an intentional NaN check (NaN != NaN)
        return f if f == f and f not in (float("inf"), float("-inf")) else None  # pylint: disable=comparison-with-itself
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
class ConvictionScore:  # pylint: disable=too-many-instance-attributes
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
class V9Config:  # pylint: disable=too-many-instance-attributes,invalid-name
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

    def _classify_scenario(self, asset: CanonicalAsset) -> Setup:  # pylint: disable=too-many-return-statements
        mtf = asset.mtf
        assert mtf is not None  # nosec B101 — type guard, MTF presence verified in node_universe
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
        if (self._aligned_fresh_choch(asset) is not None  # pylint: disable=simplifiable-condition
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
    def _build_setup(self, asset: CanonicalAsset, scenario: ScenarioCode) -> Setup:  # pylint: disable=too-many-locals
        mtf = asset.mtf
        assert mtf is not None  # nosec B101 — type guard, callers pass universe-passed assets
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
            m = {"H1": asset.mtf
