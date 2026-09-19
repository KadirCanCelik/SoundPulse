import json
import logging
import os
import shutil
from typing import Dict, List, Optional

from celery import Celery, chain
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.semantics.providers import BaseLLMProvider, GroqProvider
from src.semantics.qa_agent import InteractiveQAAgent
from src.semantics.schemas import MeetingAnalysis

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_client = Celery(
    "api_client",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)
celery_client.conf.task_routes = {
    'src.worker.tasks.transcribe_audio_task': {'queue': 'gpu_queue'},
    'src.worker.tasks.analyze_and_report_task': {'queue': 'cpu_queue'}
}

celery_client.conf.update(result_extended=True)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["SoundPulse"])

BASE_DIR = os.path.abspath(os.getcwd())
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_llm_provider() -> BaseLLMProvider:
    model = os.getenv("GROQ_MODEL")
    return GroqProvider(model_name=model)

def get_qa_agent(provider: BaseLLMProvider = Depends(get_llm_provider)) -> InteractiveQAAgent:
    try:
        return InteractiveQAAgent(provider=provider)
    except Exception as e:
        logger.error(f"Failed to initialize Interactive QA Agent: {e}")
        raise

class QARequest(BaseModel):
    meeting_id: str
    question: str
    chat_history: Optional[List[Dict[str,str]]] = []


@router.post("/analyze")
async def anaylze_audio(
    file: UploadFile = File(...),
    num_speakers: int = Form(default=2, description="Expected number of speakers")
):
    valid_extensions = (".wav", ".mp3", ".m4a", ".flac")
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    try:
        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        task_chain = chain(
            celery_client.signature(
                'src.worker.tasks.transcribe_audio_task', 
                args=[file_path, num_speakers, file.filename]
            ).set(queue='gpu_queue'), 
            
            celery_client.signature(
                'src.worker.tasks.analyze_and_report_task'
            ).set(queue='cpu_queue') 
        )
        task_result = task_chain.apply_async()

        task_a_id = task_result.parent.id if task_result.parent else "none"
        task_b_id = task_result.id
        composite_job_id = f"{task_a_id}::{task_b_id}"

        return {
            "message": "Analysis pipeline chained and queued successfully.",
            "job_id": composite_job_id, 
            "status": "PENDING"
        }

    except Exception as e:
        logger.error(f"Analysis routing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Endpoint for the UI to poll the real-time processing status.
    Uses Composite ID to track both GPU and CPU tasks flawlessly.
    """
    if "::" in job_id:
        task_a_id, task_b_id = job_id.split("::")
    else:
        task_a_id = task_b_id = job_id

    node_b = celery_client.AsyncResult(task_b_id)
    
    active_node = node_b
    if node_b.state == 'PENDING' and task_a_id != "none":

        node_a = celery_client.AsyncResult(task_a_id)
        if node_a.state in ['PENDING', 'STARTED', 'PROCESSING', 'FAILED']:
            active_node = node_a
        elif node_a.state == 'SUCCESS':

            return {
                "job_id": job_id,
                "status": "PROCESSING",
                "result": None,
                "meta": {"step": "GPU processing completed. Initializing Semantic Analysis...", "progress": 45}
            }

    response = {
        "job_id": job_id,
        "status": active_node.state,
        "result": None,
        "meta": None
    }

    if active_node.state == 'SUCCESS':
        response["result"] = active_node.result
    elif active_node.state == 'PROCESSING':
        response["meta"] = active_node.info 
    elif active_node.state == 'FAILED':
        response["result"] = str(active_node.info)
        
    return response

@router.post("/qa")
async def ask_question(request: QARequest, agent: InteractiveQAAgent = Depends(get_qa_agent)):

    try:
        transcript_path = os.path.join("data", "output", f"{request.meeting_id}_segments.json")
        analysis_path = os.path.join("data", "output", f"{request.meeting_id}_analysis.json")

        if not os.path.exists(transcript_path) or not os.path.exists(analysis_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Meeting data not found for ID: {request.meeting_id}"
            )

        with open(transcript_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
            analysis = MeetingAnalysis(**analysis_data)

        if not segments:
            return {"answer": "I cannot find the answer in the meeting records because the transcript is empty or could not be loaded."}

        answer = await agent.ask(
            segments=segments,
            analysis=analysis,
            question=request.question,
            chat_history=request.chat_history
        )

        if isinstance(answer, dict):
            final_answer = answer.get("answer") or answer.get("Answer") or str(answer)
        else:
            final_answer = str(answer).strip()

        return {"answer": final_answer}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QA Agent failed unexpectedly: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your question.")