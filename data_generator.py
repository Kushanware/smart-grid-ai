
"""Utility to simulate live smart-meter readings and append to CSV."""

import argparse
import random
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd


def round_down_to_interval(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
	minute = (ts.minute // minutes) * minutes
	return ts.replace(minute=minute, second=0, microsecond=0)


def load_existing(path: Path) -> pd.DataFrame:
	if not path.exists() or path.stat().st_size == 0:
		return pd.DataFrame(columns=["timestamp", "meter_id", "kwh"])
	return pd.read_csv(path, parse_dates=["timestamp"])


def latest_values(df: pd.DataFrame, meters: list[str], base: float) -> dict[str, float]:
	values: dict[str, float] = {}
	for meter in meters:
		meter_df = df[df["meter_id"] == meter]
		if meter_df.empty:
			values[meter] = random.uniform(base * 0.8, base * 1.1)
		else:
			values[meter] = float(meter_df.sort_values("timestamp").iloc[-1]["kwh"])
	return values


def prepare_start_timestamp(df: pd.DataFrame, interval_minutes: int) -> pd.Timestamp:
	if df.empty:
		return round_down_to_interval(pd.Timestamp.utcnow(), interval_minutes)
	return df["timestamp"].max() + timedelta(minutes=interval_minutes)


def generate_batch(
	ts: pd.Timestamp,
	values: dict[str, float],
	meters: list[str],
	drift: float,
	noise: float,
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for meter in meters:
		jitter = random.gauss(0, noise)
		next_val = max(0.1, values[meter] + drift + jitter)
		values[meter] = next_val
		rows.append({"timestamp": ts, "meter_id": meter, "kwh": round(next_val, 2)})
	return rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simulate live smart-meter readings.")
	parser.add_argument("--meters", type=str, default="MTR-001,MTR-002,MTR-003", help="Comma-separated meter IDs")
	parser.add_argument("--interval", type=int, default=15, help="Interval in minutes between readings")
	parser.add_argument("--steps", type=int, default=96, help="Number of intervals to emit; ignored when --continuous is set")
	parser.add_argument("--output", type=str, default="data/live_data.csv", help="Output CSV path")
	parser.add_argument("--base", type=float, default=10.0, help="Starting kWh baseline when no history exists")
	parser.add_argument("--drift", type=float, default=0.05, help="Deterministic upward drift per interval (kWh)")
	parser.add_argument("--noise", type=float, default=0.25, help="Gaussian noise standard deviation per interval (kWh)")
	parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between intervals (set to 0 for fast backfill; set to interval*60 for wall-clock pacing)")
	parser.add_argument("--continuous", action="store_true", help="Run indefinitely until interrupted")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.seed is not None:
		random.seed(args.seed)

	meters = [m.strip() for m in args.meters.split(",") if m.strip()]
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	existing = load_existing(output_path)
	current_ts = prepare_start_timestamp(existing, args.interval)
	values = latest_values(existing, meters, args.base)

	write_header = existing.empty
	interval_delta = timedelta(minutes=args.interval)
	total_steps = float("inf") if args.continuous else max(0, args.steps)

	step_index = 0
	while step_index < total_steps:
		batch = generate_batch(current_ts, values, meters, drift=args.drift, noise=args.noise)
		df = pd.DataFrame(batch)
		df.to_csv(output_path, mode="a", index=False, header=write_header)
		write_header = False

		step_index += 1
		current_ts += interval_delta

		print(f"[{step_index}] wrote {len(batch)} rows at {batch[0]['timestamp']:%Y-%m-%d %H:%M}")

		if args.sleep > 0 and step_index < total_steps:
			time.sleep(args.sleep)


if __name__ == "__main__":
	main()

