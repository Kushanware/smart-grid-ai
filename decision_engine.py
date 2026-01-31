

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from model import CAT_COLS, FEATURE_COLS, load_model
from preprocess import preprocess


def load_data(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path, parse_dates=["timestamp"])
	return df.sort_values("timestamp")


def rule_based_labels(processed: pd.DataFrame, *, theft_ratio: float, fault_z: float) -> pd.Series:
	"""Simple heuristic: theft when far below rolling average; fault when z-score high."""
	df = processed.copy()
	grouped = df.groupby("meter_id")
	mean = grouped["kwh_denoised"].transform("mean")
	std = grouped["kwh_denoised"].transform("std").fillna(0.001)
	z = (df["kwh_denoised"] - mean) / std

	theft_mask = df["kwh_denoised"] < theft_ratio * df["rolling_avg_kwh"].clip(lower=0.001)
	fault_mask = z > fault_z

	labels = pd.Series("normal", index=df.index)
	labels[theft_mask] = "theft"
	labels[fault_mask] = "fault"
	return labels


def predict_patterns(processed: pd.DataFrame, model=None, *, theft_ratio: float = 0.55, fault_z: float = 3.0) -> pd.Series:
	labels = rule_based_labels(processed, theft_ratio=theft_ratio, fault_z=fault_z)

	if model is None:
		return labels

	X = processed[FEATURE_COLS + CAT_COLS]
	model_preds = pd.Series(model.predict(X), index=processed.index)

	# Merge: elevate to fault if rule says so (catch spikes); otherwise trust model
	combined = model_preds.copy()
	combined[labels == "fault"] = "fault"
	combined[labels == "theft"] = combined[labels == "theft"].where(combined != "normal", "theft")
	return combined


def run_engine(data_path: Path, model_path: Optional[Path], output_path: Optional[Path]) -> pd.DataFrame:
	raw = load_data(data_path)
	processed = preprocess(raw)

	model = None
	if model_path is not None and model_path.exists():
		model = load_model(model_path)

	preds = predict_patterns(processed, model=model)
	decisions = processed.copy()
	decisions["decision"] = preds
	decisions["alert"] = decisions["decision"].isin(["theft", "fault"])

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		decisions.to_csv(output_path, index=False)

	return decisions


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run decision engine on smart-meter CSV")
	parser.add_argument("--data", type=str, default="data/live_data.csv", help="Input CSV of readings")
	parser.add_argument("--model", type=str, default="artifacts/pattern_model.joblib", help="Trained model path (optional)")
	parser.add_argument("--output", type=str, default="artifacts/decisions.csv", help="Where to write decisions")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	data_path = Path(args.data)
	model_path = Path(args.model) if args.model else None
	output_path = Path(args.output) if args.output else None

	decisions = run_engine(data_path, model_path, output_path)
	alerts = decisions[decisions["alert"]]
	print(f"Processed {len(decisions)} rows; alerts: {len(alerts)}")


if __name__ == "__main__":
	main()

