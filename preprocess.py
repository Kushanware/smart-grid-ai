
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PreprocessConfig:
	interval_minutes: int = 15
	rolling_window: int = 4  
	outlier_sigma: float = 3.0


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
	"""Forward/backfill missing values per meter and ensure required columns."""
	df = df.copy()
	df["timestamp"] = pd.to_datetime(df.get("timestamp"))

	# Drop transformer aggregate rows from feature-building; keep meter-level only
	if "transformer_id" in df.columns:
		df = df[df["meter_id"] != df["transformer_id"]]

	df = df.sort_values(["meter_id", "timestamp"])

	for col in ["voltage", "current", "power", "energy_kwh"]:
		if col not in df.columns:
			df[col] = pd.NA

	for col in ["kwh", "power", "voltage", "current", "energy_kwh"]:
		if col in df.columns:
			df[col] = (
				df.groupby("meter_id")[col]
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

	# Recompute power if possible
	if {"voltage", "current"}.issubset(df.columns):
		df["power"] = (df["voltage"] * df["current"]) / 1000.0

	# Rolling average power
	df["rolling_avg_power"] = grouped["power"].transform(
		lambda s: s.rolling(cfg.rolling_window, min_periods=1).mean()
	)

	# Deviation from normal power
	meter_mean = grouped["power"].transform("mean")
	df["deviation_from_normal"] = df["power"] - meter_mean

	# Cumulative energy per meter (kWh total over time)
	if "energy_kwh" not in df.columns:
		df["energy_kwh"] = grouped["kwh_denoised"].cumsum()

	# Placeholder loss (needs transformer aggregation) and anomaly/risk columns
	df["loss"] = pd.NA
	df["anomaly_score"] = pd.NA
	df["risk_level"] = pd.NA

	return df


def preprocess(df: pd.DataFrame, cfg: Optional[PreprocessConfig] = None) -> pd.DataFrame:
	"""Full preprocessing pipeline for clean model-ready data."""
	cfg = cfg or PreprocessConfig()
	clean = handle_missing(df)
	denoised = remove_noise(clean, cfg)
	featured = compute_features(denoised, cfg)
	return featured.reset_index(drop=True)


__all__ = ["PreprocessConfig", "preprocess", "handle_missing", "remove_noise", "compute_features"]

