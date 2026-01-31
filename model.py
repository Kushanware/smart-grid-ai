
"""Training and inference helpers for smart-meter pattern detection.

Trains a simple classifier to label readings as normal/theft/fault using
features produced by preprocess.preprocess.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import preprocess


FEATURE_COLS = [
	"kwh_denoised",
	"rolling_avg_kwh",
	"load_diff_from_normal",
	"power_kw",
	"voltage",
	"current",
	"power_factor",
	"energy_kwh_cum",
]
CAT_COLS = ["meter_id"]


def load_data(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path, parse_dates=["timestamp"])
	return df


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
	processed = preprocess(df)
	if "pattern" not in processed.columns:
		raise ValueError("pattern column required to train classifier")
	labeled = processed.dropna(subset=["pattern"])
	X = labeled[FEATURE_COLS + CAT_COLS]
	y = labeled["pattern"].astype("category")
	return X, y


def make_pipeline() -> Pipeline:
	numeric_features = FEATURE_COLS
	categorical_features = CAT_COLS

	numeric_transformer = Pipeline(
		steps=[
			("scaler", StandardScaler()),
		]
	)

	categorical_transformer = Pipeline(
		steps=[
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		]
	)

	clf = RandomForestClassifier(
		n_estimators=150,
		max_depth=None,
		min_samples_leaf=2,
		n_jobs=-1,
		class_weight="balanced",
		random_state=42,
	)

	return Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("model", clf),
		]
	)


def train_model(df: pd.DataFrame):
	X, y = build_dataset(df)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, stratify=y, random_state=42
	)

	pipe = make_pipeline()
	pipe.fit(X_train, y_train)

	y_pred = pipe.predict(X_test)
	report = classification_report(y_test, y_pred, digits=3)
	return pipe, report


def save_model(model: Pipeline, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
	return joblib.load(path)


def predict(model: Pipeline, rows: Iterable[dict]) -> pd.DataFrame:
	df = pd.DataFrame(rows)
	processed = preprocess(df)
	X = processed[FEATURE_COLS + CAT_COLS]
	preds = model.predict(X)
	processed["predicted_pattern"] = preds
	return processed


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train smart-meter pattern classifier")
	parser.add_argument("--data", type=str, default="data/live_data.csv", help="CSV with pattern labels")
	parser.add_argument("--model-out", type=str, default="artifacts/pattern_model.joblib", help="Where to store trained model")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	data_path = Path(args.data)
	model_out = Path(args.model_out)

	df = load_data(data_path)
	model, report = train_model(df)
	save_model(model, model_out)
	print("Model saved to", model_out)
	print("\nEvaluation:\n")
	print(report)


if __name__ == "__main__":
	main()

