"""Streamlit Dashboard - Control Room Screen 👀

Displays:
- Power usage graphs with anomalies highlighted in red
- Risk scores and explanations
- Meter table with alerts
- Auto-refresh for live monitoring
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from decision_engine import run_engine
from report_generator import generate_pdf_report


@st.cache_data(show_spinner=False, ttl=60)
def load_and_process_data(data_path: str, model_path: str = None) -> pd.DataFrame:
	"""Load, preprocess, and run decision engine on data."""
	model_path_obj = Path(model_path) if model_path and Path(model_path).exists() else None
	df = run_engine(data_path=Path(data_path), model_path=model_path_obj, output_path=None)
	return df


def load_and_process_data_uncached(data_path: str, model_path: str = None) -> pd.DataFrame:
	"""Uncached load for live updates."""
	model_path_obj = Path(model_path) if model_path and Path(model_path).exists() else None
	df = run_engine(data_path=Path(data_path), model_path=model_path_obj, output_path=None)
	return df


def filter_data(df: pd.DataFrame, meters: list[str], date_range: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
	start, end = date_range
	mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
	if meters:
		mask &= df["meter_id"].isin(meters)
	return df.loc[mask]


def kpi_columns(filtered: pd.DataFrame) -> None:
	total_kwh = filtered["kwh"].sum()
	avg_kw = filtered.get("power", pd.Series(dtype=float)).mean() if not filtered.empty else 0
	n_alerts = filtered["alert"].sum() if "alert" in filtered.columns else 0
	alert_pct = (n_alerts / len(filtered) * 100) if len(filtered) > 0 else 0
	avg_risk = filtered[filtered["alert"]]["risk_score"].mean() if n_alerts > 0 else 0

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Total kWh", f"{total_kwh:,.1f}")
	col2.metric("Avg Power", f"{avg_kw:.2f} kW")
	col3.metric("Alerts", f"{n_alerts}", delta=f"{alert_pct:.1f}%", delta_color="inverse")
	col4.metric("Avg Risk", f"{avg_risk:.2f}", help="Average risk score for alerts (0-1 scale)")


def render_charts(filtered: pd.DataFrame) -> None:
	if filtered.empty:
		st.info("No data for the selected filters.")
		return

	fig = go.Figure()
	for meter_id in filtered["meter_id"].unique():
		meter_data = filtered[filtered["meter_id"] == meter_id]
		normal_data = meter_data[~meter_data["alert"]]
		if not normal_data.empty:
			fig.add_trace(go.Scatter(x=normal_data["timestamp"], y=normal_data["power"], mode='lines+markers', name=f"{meter_id} (normal)", line=dict(width=2), marker=dict(size=4)))
		anomaly_data = meter_data[meter_data["alert"]]
		if not anomaly_data.empty:
			fig.add_trace(go.Scatter(x=anomaly_data["timestamp"], y=anomaly_data["power"], mode='markers', name=f"{meter_id} (alert)", marker=dict(size=12, color='red', symbol='x', line=dict(width=2))))
	
	fig.update_layout(title="Power Usage Over Time (Anomalies in RED)", xaxis_title="Time", yaxis_title="Power (kW)", hovermode='x unified', margin=dict(l=10, r=10, t=40, b=10))
	st.plotly_chart(fig, use_container_width=True, theme="streamlit")
	
	if "pattern" in filtered.columns:
		pattern_counts = filtered["pattern"].value_counts().reset_index()
		pattern_counts.columns = ["Pattern", "Count"]
		pie_fig = px.pie(pattern_counts, values="Count", names="Pattern", title="Pattern Distribution", color="Pattern", color_discrete_map={"normal": "lightgreen", "theft": "red", "fault": "orange", "anomaly": "yellow", "transformer_overload": "darkred"})
		pie_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
		col1, col2 = st.columns(2)
		with col1:
			st.plotly_chart(pie_fig, use_container_width=True, theme="streamlit")
		with col2:
			risk_data = filtered[filtered["risk_score"] > 0]
			if not risk_data.empty:
				hist_fig = px.histogram(risk_data, x="risk_score", nbins=20, title="Risk Score Distribution", labels={"risk_score": "Risk Score"}, color_discrete_sequence=["coral"])
				hist_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
				st.plotly_chart(hist_fig, use_container_width=True, theme="streamlit")


def show_alerts_table(filtered: pd.DataFrame) -> None:
	alerts = filtered[filtered["alert"]].copy()
	if alerts.empty:
		st.success("No alerts in the selected time range!")
		return
	st.warning(f"{len(alerts)} alerts detected")
	display_cols = ["timestamp", "meter_id", "pattern", "risk_score", "power", "voltage", "current", "explanation"]
	available_cols = [col for col in display_cols if col in alerts.columns]
	alerts_display = alerts[available_cols].copy()
	alerts_display["timestamp"] = alerts_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
	def highlight_pattern(row):
		if row["pattern"] == "theft":
			return ["background-color: #ff6b6b; color: #1a1a1a; font-weight: 500"] * len(row)
		elif row["pattern"] == "fault":
			return ["background-color: #ffd93d; color: #1a1a1a; font-weight: 500"] * len(row)
		elif row["pattern"] == "transformer_overload":
			return ["background-color: #ff4757; color: white; font-weight: 600"] * len(row)
		else:
			return ["background-color: #fff5ba; color: #1a1a1a"] * len(row)
	st.dataframe(alerts_display.style.apply(highlight_pattern, axis=1), use_container_width=True, height=400)


def main() -> None:
	st.set_page_config(page_title="Smart Grid Dashboard", layout="wide", page_icon="⚡")
	col1, col2 = st.columns([4, 1])
	with col1:
		st.title("⚡ Smart Grid Dashboard")
		st.caption("Real-time monitoring of energy theft & anomaly detection")
	with col2:
		live_mode = st.checkbox("Live mode", value=False, help="Show new rows as they arrive")
		refresh_seconds = st.number_input("Refresh (sec)", min_value=2, max_value=60, value=2, step=1, disabled=not live_mode)
		if live_mode:
			st.markdown("🟢 **LIVE**")
			try:
				st.autorefresh(interval=int(refresh_seconds * 1000), key="live_refresh")
			except Exception:
				pass

	base_dir = Path(__file__).parent
	data_path = str(base_dir / "data" / "live_data.csv")
	model_path = str(base_dir / "artifacts" / "anomaly_model.joblib")

	with st.spinner("Loading and analyzing data..."):
		if live_mode:
			data = load_and_process_data_uncached(data_path=data_path, model_path=model_path)
		else:
			data = load_and_process_data(data_path=data_path, model_path=model_path)

	st.sidebar.header("Filters")
	meters = sorted(data["meter_id"].unique())
	selected_meters = st.sidebar.multiselect("Meters", options=meters, default=meters[:5] if len(meters) > 5 else meters)
	min_date, max_date = data["timestamp"].min(), data["timestamp"].max()
	start_date, end_date = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
	if "pattern" in data.columns:
		patterns = ["All"] + sorted(data["pattern"].unique().tolist())
		selected_pattern = st.sidebar.selectbox("Pattern", options=patterns, index=0)
	else:
		selected_pattern = "All"

	filtered = filter_data(data, meters=selected_meters, date_range=(pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.Timedelta(days=1)))
	if selected_pattern != "All":
		filtered = filtered[filtered["pattern"] == selected_pattern]

	kpi_columns(filtered)
	st.divider()
	
	# PDF Report Generation
	st.subheader("📄 Generate Report")
	col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 4])
	with col_btn1:
		if st.button("📥 Download PDF Report", type="primary", use_container_width=True):
			with st.spinner("Generating PDF report..."):
				try:
					pdf_buffer = generate_pdf_report(filtered)
					st.download_button(
						label="💾 Download Report",
						data=pdf_buffer,
						file_name=f"smart_grid_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
						mime="application/pdf",
						use_container_width=True
					)
					st.success("✓ Report generated successfully!")
				except Exception as e:
					st.error(f"Error generating report: {str(e)}")
	with col_btn2:
		st.caption(f"Report includes {len(filtered)} records")
	
	st.divider()
	render_charts(filtered)
	st.divider()
	st.subheader("Alerts & Explanations")
	show_alerts_table(filtered)
	st.divider()
	st.subheader("Live Tail (Latest Rows)")
	if not filtered.empty:
		tail_rows = filtered.sort_values("timestamp").tail(50)
		last_seen = st.session_state.get("last_seen_ts")
		max_ts = tail_rows["timestamp"].max()
		def highlight_new(row):
			if last_seen is not None and row["timestamp"] > last_seen:
				return ["background-color: #e6ffed"] * len(row)
			return [""] * len(row)
		st.dataframe(tail_rows.style.apply(highlight_new, axis=1), use_container_width=True, height=300)
		st.caption(f"Latest timestamp: {max_ts:%Y-%m-%d %H:%M:%S}")
		st.session_state["last_seen_ts"] = max_ts
	else:
		st.info("No rows available yet.")
	st.divider()
	with st.expander("View All Data", expanded=False):
		display_cols = ["timestamp", "meter_id", "power", "voltage", "current", "pattern", "risk_score", "alert"]
		available_cols = [col for col in display_cols if col in filtered.columns]
		st.dataframe(filtered[available_cols], use_container_width=True, height=300)
	st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Total records: {len(filtered)}")


if __name__ == "__main__":
	main()

