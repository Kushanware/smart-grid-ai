
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PreprocessConfig:
	interval_minutes: int = 15
	rolling_window: int = 4  
	outlier_sigma: float = 3.0


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
	"""Forward/backfill missing kWh per meter and drop any rows still incomplete."""
	df = df.copy()
	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df = df.sort_values(["meter_id", "timestamp"])
	df["kwh"] = (
		df.groupby("meter_id")["kwh"]
		.apply(lambda s: s.ffill().bfill())
		.reset_index(level=0, drop=True)
	)
	return df.dropna(subset=["timestamp", "meter_id", "kwh"])


def remove_noise(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
	"""Apply rolling median smoothing and clip extreme outliers per meter."""
	df = df.copy()
	grouped = df.groupby("meter_id")

	median_smoothed = grouped["kwh"].transform(
		lambda s: s.rolling(cfg.rolling_window, center=True, min_periods=1).median()
	)
	df["kwh_denoised"] = median_smoothed

	# Sigma clip relative to per-meter mean/std to limit spikes
	mean = grouped["kwh_denoised"].transform("mean")
	std = grouped["kwh_denoised"].transform("std").fillna(0)
	upper = mean + cfg.outlier_sigma * std
	lower = mean - cfg.outlier_sigma * std
	df["kwh_denoised"] = df["kwh_denoised"].clip(lower=lower, upper=upper)
	return df


def compute_features(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
	"""Compute engineered features for modeling/monitoring."""
	df = df.copy()
	grouped = df.groupby("meter_id")

	# Rolling average load (same units as kWh)
	df["rolling_avg_kwh"] = grouped["kwh_denoised"].transform(
		lambda s: s.rolling(cfg.rolling_window, min_periods=1).mean()
	)

	# Load difference from meter normal
	meter_mean = grouped["kwh_denoised"].transform("mean")
	df["load_diff_from_normal"] = df["kwh_denoised"] - meter_mean

	# Optional power feature if voltage and current are present
	if {"voltage", "current"}.issubset(df.columns):
		df["power_kw"] = (df["voltage"] * df["current"]) / 1000.0
	else:
		df["power_kw"] = pd.NA
	return df


def preprocess(df: pd.DataFrame, cfg: Optional[PreprocessConfig] = None) -> pd.DataFrame:
	"""Full preprocessing pipeline for clean model-ready data."""
	cfg = cfg or PreprocessConfig()
	clean = handle_missing(df)
	denoised = remove_noise(clean, cfg)
	featured = compute_features(denoised, cfg)
	return featured.reset_index(drop=True)


__all__ = ["PreprocessConfig", "preprocess", "handle_missing", "remove_noise", "compute_features"]

