import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import settings
from app.schemas import QueryResponse


class ReportExporter:
    def export_geojson(self, session_id: str, query_response: QueryResponse) -> Path:
        export_dir = settings.EXPORTS_DIR / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        out_path = export_dir / f"satquery_audit_{session_id[:8]}.geojson"

        payload = {
            "type": "FeatureCollection",
            "satquery_audit": {
                "session_id": session_id,
                "query": query_response.query,
                "final_answer": query_response.final_answer,
                "confidence": query_response.confidence,
                "sensor_calibration_badge": query_response.sensor_calibration_badge,
                "run_signature_hash": query_response.run_signature_hash,
                "report_tamper_token": query_response.report_tamper_token,
                "generated_at": query_response.generated_at
            },
            "features": query_response.composite_overlays.get("features", []),
            "bboxes": query_response.composite_overlays.get("bboxes", [])
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return out_path

    def export_pdf(self, session_id: str, query_response: QueryResponse) -> Path:
        export_dir = settings.EXPORTS_DIR / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = export_dir / f"satquery_intelligence_report_{session_id[:8]}.pdf"

        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f172a")
        )
        subtitle_style = ParagraphStyle(
            'ReportSubTitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor("#475569")
        )
        heading2_style = ParagraphStyle(
            'H2',
            parent=styles['Heading2'],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155")
        )
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#0369a1")
        )

        elements = []

        # Header
        elements.append(Paragraph("SatQuery AI — Intelligence & Audit Report", title_style))
        elements.append(Paragraph(f"ISRO Problem Statement SIH26167 | Generated: {query_response.generated_at}", subtitle_style))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=15))

        # Metadata Table
        meta_data = [
            [Paragraph("<b>Session ID:</b>", body_style), Paragraph(session_id, body_style)],
            [Paragraph("<b>Calibration Badge:</b>", body_style), Paragraph(query_response.sensor_calibration_badge, body_style)],
            [Paragraph("<b>Execution Mode:</b>", body_style), Paragraph(query_response.execution_mode.upper(), body_style)],
            [Paragraph("<b>Fused Confidence:</b>", body_style), Paragraph(f"{query_response.confidence * 100:.1f}%", body_style)],
            [Paragraph("<b>Query:</b>", body_style), Paragraph(f"<i>'{query_response.query}'</i>", body_style)],
        ]
        meta_table = Table(meta_data, colWidths=[130, 410])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 15))

        # Key Findings
        elements.append(Paragraph("Key Findings & Grounded Synthesis", heading2_style))
        elements.append(Paragraph(query_response.final_answer, body_style))
        elements.append(Spacer(1, 15))

        # ReAct Trace Table
        elements.append(Paragraph("Auditable ReAct Execution Trace", heading2_style))
        trace_data = [
            [Paragraph("<b>Step</b>", body_style), Paragraph("<b>Specialist Tool</b>", body_style), Paragraph("<b>Reason / Why This Tool</b>", body_style), Paragraph("<b>Confidence</b>", body_style)]
        ]
        for step in query_response.trace:
            trace_data.append([
                Paragraph(str(step.step_number), body_style),
                Paragraph(step.tool_called, body_style),
                Paragraph(step.why_this_tool, body_style),
                Paragraph(f"{step.step_confidence * 100:.1f}%", body_style)
            ])
        trace_table = Table(trace_data, colWidths=[40, 110, 330, 60])
        trace_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(trace_table)
        elements.append(Spacer(1, 15))

        # Cryptographic Audit Verification
        elements.append(Paragraph("Cryptographic Tamper-Evidence & Verification", heading2_style))
        audit_data = [
            [Paragraph("<b>Deterministic Run Signature (SHA-256):</b>", body_style)],
            [Paragraph(query_response.run_signature_hash, code_style)],
            [Paragraph("<b>Report Tamper-Proof Token (HMAC-SHA256):</b>", body_style)],
            [Paragraph(query_response.report_tamper_token, code_style)],
        ]
        audit_table = Table(audit_data, colWidths=[540])
        audit_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86efac")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(audit_table)

        doc.build(elements)
        return pdf_path

report_exporter = ReportExporter()
