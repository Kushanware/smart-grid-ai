"""PDF Report Generator for Smart Grid Dashboard

Generates professional PDF reports with:
- Executive summary with key metrics
- Visual charts and graphs
- Detailed alert tables
- Recommendations for action
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def create_summary_section(filtered_df: pd.DataFrame) -> list:
    """Create executive summary section."""
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("⚡ Smart Grid Monitoring Report", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
    
    if not filtered_df.empty:
        date_range = f"{filtered_df['timestamp'].min().strftime('%Y-%m-%d')} to {filtered_df['timestamp'].max().strftime('%Y-%m-%d')}"
        elements.append(Paragraph(f"<b>Period:</b> {date_range}", meta_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Key metrics
    total_kwh = filtered_df["kwh"].sum()
    n_alerts = filtered_df["alert"].sum() if "alert" in filtered_df.columns else 0
    alert_pct = (n_alerts / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    avg_risk = filtered_df[filtered_df["alert"]]["risk_score"].mean() if n_alerts > 0 else 0
    n_meters = filtered_df["meter_id"].nunique()
    
    # Create metrics table
    metrics_data = [
        ["Metric", "Value"],
        ["Total Energy Consumption", f"{total_kwh:,.1f} kWh"],
        ["Total Alerts Detected", f"{int(n_alerts)}"],
        ["Alert Rate", f"{alert_pct:.1f}%"],
        ["Average Risk Score", f"{avg_risk:.2f}"],
        ["Meters Monitored", f"{n_meters}"],
        ["Total Records", f"{len(filtered_df):,}"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.5*inch))
    
    return elements


def create_chart_image(filtered_df: pd.DataFrame, chart_type: str) -> BytesIO:
    """Create chart images for the report."""
    plt.figure(figsize=(8, 4))
    
    if chart_type == "pattern_distribution" and "pattern" in filtered_df.columns:
        pattern_counts = filtered_df["pattern"].value_counts()
        colors_map = {'normal': 'lightgreen', 'theft': 'red', 'fault': 'orange', 
                      'anomaly': 'yellow', 'transformer_overload': 'darkred'}
        chart_colors = [colors_map.get(p, 'grey') for p in pattern_counts.index]
        
        plt.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%', 
                colors=chart_colors, startangle=90)
        plt.title('Pattern Distribution', fontsize=14, fontweight='bold')
        
    elif chart_type == "risk_distribution":
        risk_data = filtered_df[filtered_df["risk_score"] > 0]["risk_score"]
        if not risk_data.empty:
            plt.hist(risk_data, bins=20, color='coral', edgecolor='black', alpha=0.7)
            plt.xlabel('Risk Score', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('Risk Score Distribution', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
    
    elif chart_type == "alerts_timeline":
        alerts = filtered_df[filtered_df["alert"]].copy()
        if not alerts.empty:
            alerts['date'] = alerts['timestamp'].dt.date
            daily_alerts = alerts.groupby('date').size()
            plt.plot(daily_alerts.index, daily_alerts.values, marker='o', color='red', linewidth=2)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('Number of Alerts', fontsize=10)
            plt.title('Daily Alert Trend', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
    
    # Save to BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer


def create_alerts_table(filtered_df: pd.DataFrame) -> list:
    """Create detailed alerts table."""
    elements = []
    styles = getSampleStyleSheet()
    
    # Section header
    header_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], 
                                  fontSize=16, textColor=colors.HexColor('#1f77b4'), 
                                  spaceAfter=12)
    elements.append(Paragraph("Detailed Alerts", header_style))
    
    alerts = filtered_df[filtered_df["alert"]].copy()
    
    if alerts.empty:
        elements.append(Paragraph("✓ No alerts detected in this period.", styles['Normal']))
        return elements
    
    # Top 20 alerts
    alerts = alerts.head(20)
    
    # Prepare table data
    table_data = [["Time", "Meter", "Pattern", "Risk", "Power (kW)"]]
    
    for _, row in alerts.iterrows():
        table_data.append([
            row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            str(row["meter_id"]),
            row.get("pattern", "N/A"),
            f"{row.get('risk_score', 0):.2f}",
            f"{row.get('power', 0):.2f}"
        ])
    
    # Create table
    alerts_table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1.3*inch, 0.8*inch, 1*inch])
    alerts_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d62728')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    elements.append(alerts_table)
    elements.append(Spacer(1, 0.3*inch))
    
    if len(filtered_df[filtered_df["alert"]]) > 20:
        note_style = ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, 
                                    textColor=colors.grey, fontStyle='italic')
        elements.append(Paragraph(f"Note: Showing top 20 alerts out of {len(filtered_df[filtered_df['alert']])} total alerts.", note_style))
    
    return elements


def create_recommendations(filtered_df: pd.DataFrame) -> list:
    """Create recommendations section."""
    elements = []
    styles = getSampleStyleSheet()
    
    header_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], 
                                  fontSize=16, textColor=colors.HexColor('#1f77b4'), 
                                  spaceAfter=12)
    elements.append(Paragraph("Recommendations", header_style))
    
    alerts = filtered_df[filtered_df["alert"]]
    
    if alerts.empty:
        elements.append(Paragraph("✓ System operating normally. Continue routine monitoring.", styles['Normal']))
        return elements
    
    recommendations = []
    
    # High-risk meters
    high_risk = alerts[alerts["risk_score"] > 0.7]
    if not high_risk.empty:
        high_risk_meters = high_risk["meter_id"].unique()[:5]
        recommendations.append(f"<b>URGENT:</b> Inspect high-risk meters immediately: {', '.join(map(str, high_risk_meters))}")
    
    # Pattern-specific recommendations
    if "pattern" in alerts.columns:
        if (alerts["pattern"] == "theft").any():
            recommendations.append("<b>Energy Theft Detected:</b> Coordinate with field team for physical inspection and possible legal action.")
        
        if (alerts["pattern"] == "transformer_overload").any():
            recommendations.append("<b>Transformer Overload:</b> Consider load balancing or capacity upgrade to prevent equipment failure.")
        
        if (alerts["pattern"] == "fault").any():
            recommendations.append("<b>Faults Detected:</b> Schedule maintenance to prevent service disruption.")
    
    # General recommendations
    alert_rate = (len(alerts) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    if alert_rate > 10:
        recommendations.append(f"<b>High Alert Rate ({alert_rate:.1f}%):</b> Consider system-wide audit and predictive maintenance.")
    
    if not recommendations:
        recommendations.append("Continue regular monitoring and periodic audits.")
    
    for i, rec in enumerate(recommendations, 1):
        elements.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        elements.append(Spacer(1, 0.15*inch))
    
    return elements


def generate_pdf_report(filtered_df: pd.DataFrame, output_path: str = None) -> BytesIO:
    """Generate complete PDF report."""
    buffer = BytesIO()
    
    # Create PDF
    doc = SimpleDocTemplate(
        buffer if output_path is None else output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    elements = []
    
    # Add summary section
    elements.extend(create_summary_section(filtered_df))
    
    # Add charts
    if not filtered_df.empty:
        styles = getSampleStyleSheet()
        header_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], 
                                      fontSize=16, textColor=colors.HexColor('#1f77b4'), 
                                      spaceAfter=12)
        
        elements.append(Paragraph("Visual Analytics", header_style))
        
        # Pattern distribution chart
        if "pattern" in filtered_df.columns:
            try:
                img_buffer = create_chart_image(filtered_df, "pattern_distribution")
                img = Image(img_buffer, width=5*inch, height=2.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
            except Exception:
                pass
        
        # Risk distribution chart
        if filtered_df["risk_score"].sum() > 0:
            try:
                img_buffer = create_chart_image(filtered_df, "risk_distribution")
                img = Image(img_buffer, width=5*inch, height=2.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
            except Exception:
                pass
        
        # Alert timeline
        if filtered_df["alert"].sum() > 0:
            try:
                img_buffer = create_chart_image(filtered_df, "alerts_timeline")
                img = Image(img_buffer, width=5*inch, height=2.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.5*inch))
            except Exception:
                pass
    
    # Add page break before alerts table
    elements.append(PageBreak())
    
    # Add alerts table
    elements.extend(create_alerts_table(filtered_df))
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Add recommendations
    elements.extend(create_recommendations(filtered_df))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    styles = getSampleStyleSheet()
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                                  textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph("Smart Grid Monitoring System | Energy Theft & Anomaly Detection", footer_style))
    
    # Build PDF
    doc.build(elements)
    
    if output_path is None:
        buffer.seek(0)
        return buffer
    
    return None
