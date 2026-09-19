import asyncio
import json
import os
import time

from celery.utils.log import get_task_logger

from src.audio.pipeline import AudioProcessingPipeline
from src.semantics.analyzer import SemanticAnalyzer
from src.semantics.providers import GroqProvider
from src.semantics.schemas import FullAnalysisResponse
from src.utils.pdf_export import MeetingPdfExporter
from src.worker.worker import app

logger = get_task_logger(__name__)

# =====================================================================
# GPU Worker
# =====================================================================

@app.task(bind=True)
def transcribe_audio_task(self, file_path: str, num_speakers:int, original_filename:str):
    """
    Uses GPU to transcribe and diarize audio. 
    Passes the result to the next task in the chain.
    """
    job_id = self.request.id

    self.update_state(
        state="PROCESSING",
        meta={'step': 'Running Audio Pipeline (Transcription & Diarization)...', 'progress': 25}
    )

    logger.info(f"[{job_id}] Initiating Audio Pipeline for {file_path}")

    pipeline = AudioProcessingPipeline()
    segments = pipeline.process_audio(file_path, num_speakers=num_speakers)

    if not segments:
        raise ValueError("Audio provessing failed or returned empty segments.")

    return {
        "segments":segments,
        "file_path":file_path,
        "original_filename":original_filename
    }

# =====================================================================
# CPU Worker
# =====================================================================
async def _async_analyze_and_report(task_instance, parent_data: dict):
    """
    Core async logic for semantic analysis and PDF generation
    """
    job_id = task_instance.request.id

    segments = parent_data.get("segments")
    file_path = parent_data.get("file_path")
    original_filename = parent_data.get("original_filename")

    base_name = os.path.splitext(original_filename)[0].replace(" ", "_")
    meeting_id = f"{base_name}_{int(time.time())}"

    BASE_DIR = os.path.abspath(os.getcwd())
    output_dir = os.path.join(BASE_DIR, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        task_instance.update_state(
            state="PROCESSEING",
            meta={'step': 'Running Semantic Analysis via LLM...', 'progress': 50}
        )

        logger.info(f"[{job_id}] Sending segments to Semantic Analyzer...")

        provider= GroqProvider(model_name=os.getenv("GROQ_MODEL"))
        analyzer = SemanticAnalyzer(provider=provider)
        analysis_result = await analyzer.run_analysis(segments)

        task_instance.update_state(
            state="PROCESSING",
            meta={'step': 'Saving analysis data...', 'progress': 75}
        )

        transcript_path = os.path.join(output_dir, f"{meeting_id}_segments.json")
        transcript_tmp_path = f"{transcript_path}.tmp"
        with open(transcript_tmp_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        os.rename(transcript_tmp_path, transcript_path)

        analysis_path = os.path.join(output_dir, f"{meeting_id}_analysis.json")
        analysis_tmp_path = f"{analysis_path}.tmp"
        with open(analysis_tmp_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result.model_dump(), f, ensure_ascii=False, indent=2)
        os.rename(analysis_tmp_path, analysis_path)

        response_data = FullAnalysisResponse(
            meeting_id=meeting_id,
            transcript=segments,
            analysis=analysis_result
        )

        task_instance.update_state(
            state="PROCESSING",
            meta={'step': 'Generating PDF audit log...', 'progress': 90}
        )

        logger.info(f"[{job_id}] Generating PDF audit log...")

        try:
            pdf_exporter = MeetingPdfExporter()
            pdf_filepath = pdf_exporter.generate_and_save(response_data)

            if pdf_filepath:
                response_data.pdf_filepath = pdf_filepath

        except Exception as e:
            logger.warning(f"[{job_id}] PDF generation failed: {e}")

        return{
            "status":"success",
            "meeting_id":meeting_id,
            "transcript_path":str(transcript_path),
            "analysis_path":str(analysis_path),
            "pdf_filepath":str(response_data.pdf_filepath) if getattr(response_data, 'pdf_filepath', None) else None
        }

    except Exception as e:
        logger.error(f"[{job_id}] CPU pipeline failed: {e}", exc_info=True)
        raise e

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[{job_id}] Cleaned up shared temp file: {file_path}")

@app.task(bind=True)
def analyze_and_report_task(self, parent_data: dict):
    """
    Synchronous Celery task wrapper that executes the async CPU logic.
    Celery automatically injects 'parent_data' from the return value of Task 1.
    """
    return asyncio.run(_async_analyze_and_report(self, parent_data))