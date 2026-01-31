
import argparse
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd


@dataclass
class PatternState:
	name: str
	remaining: int


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


def choose_pattern(normal_p: float, theft_p: float, fault_p: float) -> str:
	r = random.random()
	if r < theft_p:
		return "theft"
	if r < theft_p + fault_p:
		return "fault"
	return "normal"


def generate_batch(
	ts: pd.Timestamp,
	values: dict[str, float],
	meters: list[str],
	pattern_state: dict[str, PatternState],
	*,
	drift: float,
	noise: float,
	theft_min_factor: float,
	theft_max_factor: float,
	fault_min_multiplier: float,
	fault_max_multiplier: float,
	min_pattern_steps: int,
	max_pattern_steps: int,
	normal_p: float,
	theft_p: float,
	fault_p: float,
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for meter in meters:
		state = pattern_state[meter]
		if state.remaining <= 0:
			new_pattern = choose_pattern(normal_p, theft_p, fault_p)
			state = PatternState(new_pattern, random.randint(min_pattern_steps, max_pattern_steps))
			pattern_state[meter] = state

		jitter = random.gauss(0, noise)
		base_val = max(0.05, values[meter] + drift + jitter)

		if state.name == "theft":
			factor = random.uniform(theft_min_factor, theft_max_factor)
			next_val = max(0.05, base_val * factor)
		elif state.name == "fault":
			multiplier = random.uniform(fault_min_multiplier, fault_max_multiplier)
			next_val = max(0.05, base_val * multiplier)
		else:
			next_val = base_val

		values[meter] = next_val
		pattern_state[meter] = PatternState(state.name, state.remaining - 1)
		rows.append(
			{
				"timestamp": ts,
				"meter_id": meter,
				"kwh": round(next_val, 2),
				"pattern": state.name,
			}
		)
	return rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simulate live smart-meter readings.")
	parser.add_argument("--meters", type=str, default="MTR-001,MTR-002,MTR-003", help="Comma-separated meter IDs")
	parser.add_argument("--output", type=str, default="data/live_data.csv", help="Output CSV path")
	parser.add_argument("--base", type=float, default=10.0, help="Starting kWh baseline when no history exists")
	parser.add_argument("--drift", type=float, default=0.05, help="Deterministic upward drift per step (kWh)")
	parser.add_argument("--noise", type=float, default=0.25, help="Gaussian noise standard deviation per step (kWh)")
	parser.add_argument("--interval-seconds", type=float, default=2.0, help="Virtual time step between readings (timestamp spacing)")
	parser.add_argument("--sleep", type=float, default=2.0, help="Wall-clock seconds to wait between writes (set 0 for fast simulation)")
	parser.add_argument("--steps", type=int, default=0, help="Number of steps to emit; 0 means run indefinitely")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
	parser.add_argument("--theft-min-factor", type=float, default=0.25, help="Lower bound multiplier applied during theft (0-1, reduces load)")
	parser.add_argument("--theft-max-factor", type=float, default=0.6, help="Upper bound multiplier applied during theft (0-1, reduces load)")
	parser.add_argument("--fault-min-multiplier", type=float, default=1.8, help="Lower bound multiplier applied during fault spikes")
	parser.add_argument("--fault-max-multiplier", type=float, default=3.2, help="Upper bound multiplier applied during fault spikes")
	parser.add_argument("--min-pattern-steps", type=int, default=3, help="Minimum steps a pattern persists")
	parser.add_argument("--max-pattern-steps", type=int, default=8, help="Maximum steps a pattern persists")
	parser.add_argument("--theft-prob", type=float, default=0.18, help="Probability to start a theft pattern when switching")
	parser.add_argument("--fault-prob", type=float, default=0.1, help="Probability to start a fault pattern when switching")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.seed is not None:
		random.seed(args.seed)

	meters = [m.strip() for m in args.meters.split(",") if m.strip()]
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	existing = load_existing(output_path)
	values = latest_values(existing, meters, args.base)

	start_ts = pd.Timestamp.utcnow().floor("S")
	if not existing.empty:
		start_ts = max(start_ts, existing["timestamp"].max() + timedelta(seconds=args.interval_seconds))

	pattern_state = {meter: PatternState("normal", 0) for meter in meters}
	current_ts = start_ts
	write_header = existing.empty
	step_index = 0
	max_steps = float("inf") if args.steps == 0 else max(0, args.steps)

	while step_index < max_steps:
		batch = generate_batch(
			ts=current_ts,
			values=values,
			meters=meters,
			pattern_state=pattern_state,
			drift=args.drift,
			noise=args.noise,
			theft_min_factor=args.theft_min_factor,
			theft_max_factor=args.theft_max_factor,
			fault_min_multiplier=args.fault_min_multiplier,
			fault_max_multiplier=args.fault_max_multiplier,
			min_pattern_steps=args.min_pattern_steps,
			max_pattern_steps=args.max_pattern_steps,
			normal_p=max(0.0, 1.0 - args.theft_prob - args.fault_prob),
			theft_p=args.theft_prob,
			fault_p=args.fault_prob,
		)

		df = pd.DataFrame(batch)
		df.to_csv(output_path, mode="a", index=False, header=write_header)
		write_header = False

		step_index += 1
		current_ts += timedelta(seconds=args.interval_seconds)

		print(f"[{step_index}] wrote {len(batch)} rows at {batch[0]['timestamp']:%Y-%m-%d %H:%M:%S}")

		if args.sleep > 0 and step_index < max_steps:
			time.sleep(args.sleep)


if __name__ == "__main__":
	main()

