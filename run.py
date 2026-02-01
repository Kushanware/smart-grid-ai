"""Run all Smart Grid components in parallel.

Starts:
- data_generator.py (simulator)
- decision_engine.py (periodic loop)
- app.py via Streamlit
- model.py (optional, or if model missing)
"""

from __future__ import annotations

import argparse
import importlib.util
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable


def module_available(name: str) -> bool:
	spec = importlib.util.find_spec(name)
	return spec is not None


def resolve_path(base_dir: Path, user_path: str) -> Path:
	path = Path(user_path)
	return path if path.is_absolute() else base_dir / path


def build_simulator_command(args: argparse.Namespace, base_dir: Path) -> list[str]:
	script = base_dir / "data_generator.py"
	return [
		sys.executable,
		"-u",
		str(script),
		"--sleep",
		str(args.sim_sleep),
		"--interval-seconds",
		str(args.sim_interval),
		"--steps",
		str(args.sim_steps),
	]


def build_trainer_command(args: argparse.Namespace, base_dir: Path) -> list[str]:
	script = base_dir / "model.py"
	return [
		sys.executable,
		"-u",
		str(script),
		"--data",
		str(args.data_path),
		"--model-out",
		str(args.model_path),
	]


def build_engine_command(args: argparse.Namespace, base_dir: Path) -> list[str]:
	script = base_dir / "decision_engine.py"
	return [
		sys.executable,
		"-u",
		str(script),
		"--data",
		str(args.data_path),
		"--model",
		str(args.model_path),
		"--output",
		str(args.decisions_path),
	]


def build_dashboard_command(base_dir: Path) -> list[str]:
	script = base_dir / "app.py"
	return [sys.executable, "-m", "streamlit", "run", str(script)]


def start_process(command: list[str], name: str) -> subprocess.Popen:
	print(f"Starting {name}...")
	return subprocess.Popen(command)


def run_engine_loop(command: list[str], interval_seconds: float, stop_event: threading.Event) -> None:
	while not stop_event.is_set():
		subprocess.run(command, check=False)
		stop_event.wait(interval_seconds)


def wait_for_processes(processes: Iterable[subprocess.Popen], stop_event: threading.Event) -> None:
	try:
		for proc in processes:
			proc.wait()
	except KeyboardInterrupt:
		stop_event.set()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Smart Grid system in parallel.")

	parser.add_argument("--no-simulator", action="store_true", help="Do not start data generator")
	parser.add_argument("--no-engine", action="store_true", help="Do not start decision engine loop")
	parser.add_argument("--no-dashboard", action="store_true", help="Do not start Streamlit dashboard")

	parser.add_argument("--train-model", action="store_true", help="Train model on startup")
	parser.add_argument("--engine-interval", type=float, default=5.0, help="Seconds between decision engine runs")

	parser.add_argument("--sim-sleep", type=float, default=2.0, help="Simulator wall-clock sleep seconds")
	parser.add_argument("--sim-interval", type=float, default=2.0, help="Simulator virtual interval seconds")
	parser.add_argument("--sim-steps", type=int, default=0, help="Simulator steps (0 = infinite)")

	parser.add_argument("--data-path", type=str, default="data/live_data.csv", help="Live data CSV")
	parser.add_argument("--model-path", type=str, default="artifacts/anomaly_model.joblib", help="Model path")
	parser.add_argument("--decisions-path", type=str, default="artifacts/decisions.csv", help="Decisions output path")

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	base_dir = Path(__file__).parent

	args.data_path = resolve_path(base_dir, args.data_path)
	args.model_path = resolve_path(base_dir, args.model_path)
	args.decisions_path = resolve_path(base_dir, args.decisions_path)

	stop_event = threading.Event()
	processes: list[subprocess.Popen] = []

	def handle_signal(_signum, _frame):
		stop_event.set()
		for proc in processes:
			proc.terminate()

	signal.signal(signal.SIGINT, handle_signal)
	signal.signal(signal.SIGTERM, handle_signal)

	if args.train_model or not args.model_path.exists():
		print("Training model...")
		train_cmd = build_trainer_command(args, base_dir)
		subprocess.run(train_cmd, check=False)

	engine_thread = None
	if not args.no_engine:
		engine_cmd = build_engine_command(args, base_dir)
		engine_thread = threading.Thread(
			target=run_engine_loop,
			args=(engine_cmd, args.engine_interval, stop_event),
			daemon=True,
		)
		engine_thread.start()

	if not args.no_simulator:
		sim_cmd = build_simulator_command(args, base_dir)
		processes.append(start_process(sim_cmd, "simulator"))

	if not args.no_dashboard:
		if not module_available("streamlit"):
			print("Streamlit is not installed; dashboard will not start.")
		else:
			dash_cmd = build_dashboard_command(base_dir)
			processes.append(start_process(dash_cmd, "dashboard"))

	if processes:
		wait_for_processes(processes, stop_event)
	else:
		try:
			stop_event.wait()
		except KeyboardInterrupt:
			stop_event.set()

	for proc in processes:
		if proc.poll() is None:
			proc.terminate()

	if engine_thread is not None:
		engine_thread.join(timeout=2)


if __name__ == "__main__":
	main()
