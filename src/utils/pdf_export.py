import logging
import os
from datetime import datetime

from fpdf import FPDF

from src.semantics.schemas import FullAnalysisResponse

logger = logging.getLogger("PDFExporter")

class MeetingPdfExporter:
    """
    Independent utility class to generate and save meeting minutes as PDF logs.
    """
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = os.path.join(os.getcwd(), export_dir)
        os.makedirs(self.export_dir, exist_ok=True)
        
    def _clean_text(self, text: str) -> str:
        """
        Sanitizes LLM output by replacing unsupported Unicode typography 
        (smart quotes, em-dashes, etc.) with standard ASCII equivalents.
        """
        if not text: 
            return ""
            
        replacements = {
            '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-', # Dashes and hyphens
            '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'", # Single quotes
            '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"', # Double quotes
            '\u2026': '...', # Ellipsis
            '\u00A0': ' ',   # Non-breaking space
        }
        for unicode_char, ascii_replacement in replacements.items():
            text = text.replace(unicode_char, ascii_replacement)
            
        return text.encode('latin-1', errors='replace').decode('latin-1')

    def generate_and_save(self, data: FullAnalysisResponse) -> str:
        """
        Parses the structured Pydantic response and compiles it into a formatted PDF.
        Saves the file directly to the local disk as an audit log.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_log_{timestamp}.pdf"
        filepath = os.path.join(self.export_dir, filename)

        try:
            pdf = FPDF()
            pdf.add_page()
            
            # --- HEADER ---
            pdf.set_font("Helvetica", style="B", size=16)
            pdf.cell(0, 10, "SoundPulse - Meeting Minutes", ln=True, align="C")
            pdf.set_font("Helvetica", size=10)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(5)

            # --- EXECUTIVE SUMMARY ---
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(0, 8, "Executive Summary", ln=True)
            pdf.set_font("Helvetica", size=10)

            safe_summary = self._clean_text(data.analysis.executive_summary)
            pdf.multi_cell(0, 6, safe_summary)
            pdf.ln(5)

            # --- SENTIMENT ---
            pdf.set_font("Helvetica", style="B", size=12)
            safe_sentiment = self._clean_text(data.analysis.sentiment)
            pdf.cell(0, 8, f"Overall Sentiment: {safe_sentiment}", ln=True)
            pdf.ln(5)

            # --- ACTION ITEMS ---
            if data.analysis.action_items:
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(0, 8, "Action Items", ln=True)
                pdf.set_font("Helvetica", size=10)
                for item in data.analysis.action_items:
                    safe_desc = self._clean_text(item.task_description)
                    safe_assignee = self._clean_text(item.assignee)
                    task = f"- [{safe_assignee}] {safe_desc} (Ref: {item.start_time}s)"
                    pdf.multi_cell(0, 6, task)
                pdf.ln(5)

            # --- DIVIDER ---
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            # --- TRANSCRIPT ---
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(0, 8, "Full Transcript", ln=True)
            pdf.set_font("Helvetica", size=10)
            
            for seg in data.transcript:
                safe_speaker = self._clean_text(seg.speaker)
                safe_text = self._clean_text(seg.text)
                time_str = f"[{seg.start:05.1f} - {seg.end:05.1f}]"
                line = f"{time_str} {safe_speaker}: {safe_text}"
                pdf.multi_cell(0, 6, line)
                pdf.ln(1) 

            # Output to disk
            pdf.output(filepath)
            
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to generate PDF log: {str(e)}")
            return None