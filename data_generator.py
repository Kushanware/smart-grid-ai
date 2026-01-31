"""Live-ish smart-meter simulator emitting voltage/current/power with patterns."""

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
	df = pd.read_csv(path, parse_dates=["timestamp"])
	if "timestamp" in df.columns:
		df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
	return df


def init_state(
	df: pd.DataFrame,
	meters: list[str],
	base_kw: float,
	interval_seconds: float,
) -> dict[str, dict[str, float]]:
	state: dict[str, dict[str, float]] = {}
	for meter in meters:
		meter_df = df[df["meter_id"] == meter].sort_values("timestamp")
		if meter_df.empty:
			power_kw = random.uniform(base_kw * 0.8, base_kw * 1.1)
			energy_cum = 0.0
		else:
			last = meter_df.iloc[-1]
			if "power_kw" in last and not pd.isna(last["power_kw"]):
				power_kw = float(last["power_kw"])
			elif "kwh" in last and not pd.isna(last["kwh"]):
				power_kw = float(last["kwh"]) * 3600.0 / max(interval_seconds, 1.0)
			else:
				power_kw = base_kw
			energy_cum = float(last.get("energy_kwh_cum", meter_df["kwh"].sum()))

		state[meter] = {"power_kw": power_kw, "energy_kwh_cum": energy_cum}
	return state


def choose_pattern(normal_p: float, theft_p: float, fault_p: float) -> str:
	r = random.random()
	if r < theft_p:
		return "theft"
	if r < theft_p + fault_p:
		return "fault"
	return "normal"


def generate_batch(
	ts: pd.Timestamp,
	state: dict[str, dict[str, float]],
	meters: list[str],
	pattern_state: dict[str, PatternState],
	*,
	interval_seconds: float,
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
	voltage_nominal: float,
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []
	for meter in meters:
		pat = pattern_state[meter]
		if pat.remaining <= 0:
			new_pattern = choose_pattern(normal_p, theft_p, fault_p)
			pat = PatternState(new_pattern, random.randint(min_pattern_steps, max_pattern_steps))
			pattern_state[meter] = pat

		jitter = random.gauss(0, noise)
		base_power = max(0.05, state[meter]["power_kw"] + drift + jitter)

		if pat.name == "theft":
			factor = random.uniform(theft_min_factor, theft_max_factor)
			power_kw = max(0.05, base_power * factor)
		elif pat.name == "fault":
			multiplier = random.uniform(fault_min_multiplier, fault_max_multiplier)
			power_kw = max(0.05, base_power * multiplier)
		else:
			power_kw = base_power

		voltage = random.gauss(voltage_nominal, voltage_nominal * 0.015)
		power_factor = min(1.0, max(0.5, random.gauss(0.96, 0.02)))
		current = power_kw * 1000.0 / max(voltage * power_factor, 1.0)

		kwh_interval = power_kw * interval_seconds / 3600.0
		state[meter]["power_kw"] = power_kw
		state[meter]["energy_kwh_cum"] += kwh_interval
		pattern_state[meter] = PatternState(pat.name, pat.remaining - 1)

		rows.append(
			{
				"timestamp": ts,
				"meter_id": meter,
				"voltage": round(voltage, 2),
				"current": round(current, 3),
				"power_kw": round(power_kw, 3),
				"kwh": round(kwh_interval, 4),
				"energy_kwh_cum": round(state[meter]["energy_kwh_cum"], 4),
				"power_factor": round(power_factor, 3),
				"pattern": pat.name,
			}
		)
	return rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simulate live smart-meter readings with electrical fields.")
	parser.add_argument("--meters", type=str, default="MTR-001,MTR-002,MTR-003", help="Comma-separated meter IDs")
	parser.add_argument("--output", type=str, default="data/live_data.csv", help="Output CSV path")
	parser.add_argument("--base-kw", type=float, default=3.2, help="Starting power baseline (kW) when no history exists")
	parser.add_argument("--drift", type=float, default=0.02, help="Deterministic upward drift per step (kW)")
	parser.add_argument("--noise", type=float, default=0.15, help="Gaussian noise standard deviation per step (kW)")
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
	parser.add_argument("--voltage-nominal", type=float, default=230.0, help="Nominal service voltage (V)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.seed is not None:
		random.seed(args.seed)

	meters = [m.strip() for m in args.meters.split(",") if m.strip()]
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	existing = load_existing(output_path)
	state = init_state(existing, meters, args.base_kw, args.interval_seconds)

	start_ts = pd.Timestamp.now(tz=None).floor("s")
	if not existing.empty:
		last_ts = pd.to_datetime(existing["timestamp"]).dt.tz_localize(None).max()
		start_ts = max(start_ts, last_ts + timedelta(seconds=args.interval_seconds))

	pattern_state = {meter: PatternState("normal", 0) for meter in meters}
	current_ts = start_ts
	write_header = existing.empty
	step_index = 0
	max_steps = float("inf") if args.steps == 0 else max(0, args.steps)

	while step_index < max_steps:
		batch = generate_batch(
			ts=current_ts,
			state=state,
			meters=meters,
			pattern_state=pattern_state,
			interval_seconds=args.interval_seconds,
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
			voltage_nominal=args.voltage_nominal,
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

