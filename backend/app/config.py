"""Loads config/collections.yaml once and exposes it as a typed-ish object.

Keeping this in one place means a new SPHEREx data release, a changed
SIMBAD mirror, or a tuned concurrency cap is a one-file edit — nothing else
in the codebase should hard-code any of these values (see PLAN.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Optional
import yaml

# repo_root/backend/app/config.py -> repo_root/config/collections.yaml
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "collections.yaml"


@dataclass(frozen=True)
class FetchConfig:
    max_workers: int
    global_concurrency_cap: int
    connect_timeout_s: float
    read_timeout_s: float
    max_retries: int
    backoff_base_s: float
    retry_status_codes: tuple[int, ...]


@dataclass(frozen=True)
class FovConfig:
    default_arcmin: float
    min_arcmin: float
    max_arcmin: float
    warn_above_arcmin: float


@dataclass(frozen=True)
class IrsaConfig:
    base_url: str
    tap_endpoint: str
    sia_collection_prefix: str
    fallback_collections: tuple[str, ...]
    exclude_suffixes: tuple[str, ...]


@dataclass(frozen=True)
class SimbadConfig:
    tap_mirrors: tuple[str, ...]


@dataclass(frozen=True)
class QualityConfig:
    """Max exposures fetched per group, by preset name. 0/None = no cap."""

    presets: dict  # {"low": 5, "mid": 15, "high": 0}
    default: str

    def max_exposures(self, name: Optional[str]) -> Optional[int]:
        value = self.presets.get(name or self.default, self.presets[self.default])
        return value if value else None


@dataclass(frozen=True)
class Config:
    irsa: IrsaConfig
    simbad: SimbadConfig
    detectors_all: tuple[str, ...]
    detectors_default: tuple[str, ...]
    wavelength_um: dict = field(default_factory=dict)
    fov: FovConfig = None
    fetch: FetchConfig = None
    quality: QualityConfig = None
    pixel_scale_arcsec: float = 6.15          # common display/analysis grid
    native_pixel_scale_arcsec: float = 6.15   # SPHEREx detector pixel (SIA s_pixel_scale)
    spectrum_max_exposures: int = 60
    resample_kernel: str = "lanczos3"
    # (true variance of an r=9" aperture sum) / (sum of per-pixel variances),
    # measured per kernel by scripts/benchmark_resampling.py.
    aperture_variance_inflation: float = 5.0


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open(_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    irsa = IrsaConfig(
        base_url=raw["irsa"]["base_url"],
        tap_endpoint=raw["irsa"]["tap_endpoint"],
        sia_collection_prefix=raw["irsa"]["sia_collection_prefix"],
        fallback_collections=tuple(raw["irsa"]["fallback_collections"]),
        exclude_suffixes=tuple(raw["irsa"]["exclude_suffixes"]),
    )
    simbad = SimbadConfig(tap_mirrors=tuple(raw["simbad"]["tap_mirrors"]))
    fov = FovConfig(**raw["fov"])
    fetch = FetchConfig(
        max_workers=raw["fetch"]["max_workers"],
        global_concurrency_cap=raw["fetch"]["global_concurrency_cap"],
        connect_timeout_s=raw["fetch"]["connect_timeout_s"],
        read_timeout_s=raw["fetch"]["read_timeout_s"],
        max_retries=raw["fetch"]["max_retries"],
        backoff_base_s=raw["fetch"]["backoff_base_s"],
        retry_status_codes=tuple(raw["fetch"]["retry_status_codes"]),
    )
    quality_raw = raw.get("quality", {})
    quality = QualityConfig(
        presets={k: v for k, v in quality_raw.items() if k != "default"},
        default=quality_raw.get("default", "high"),
    )
    return Config(
        irsa=irsa,
        simbad=simbad,
        detectors_all=tuple(raw["detectors"]["all"]),
        detectors_default=tuple(raw["detectors"]["default"]),
        wavelength_um=raw["detectors"]["wavelength_um"],
        fov=fov,
        fetch=fetch,
        quality=quality,
        pixel_scale_arcsec=raw["pixel_scale_arcsec"],
        native_pixel_scale_arcsec=raw.get("native_pixel_scale_arcsec", 6.15),
        spectrum_max_exposures=raw.get("spectrum", {}).get("max_exposures", 60),
        resample_kernel=raw.get("resampling", {}).get("kernel", "lanczos3"),
        aperture_variance_inflation=float(
            raw.get("resampling", {}).get("aperture_variance_inflation", {}).get(
                raw.get("resampling", {}).get("kernel", "lanczos3"), 5.0
            )
        ),
    )
