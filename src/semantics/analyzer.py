import logging
from typing import Any, Dict, List

from pydantic import ValidationError

from .providers import BaseLLMProvider
from .schemas import MeetingAnalysis

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """
    Orchestrates the semantic analysis process. 
    Acts as a bridge between the raw ASR/Diarization output and the LLM Providers
    """

    def __init__(self, provider: BaseLLMProvider):
        """
        Dependency Injection in action. 
        The analyzer doesn't know if it's using Groq or Ollama.
        It only knows that 'provider' adheres to the BaseLLMProvider contract.
        """
        self.provider = provider

    def _format_transcript(self, segments: List[Dict[str, Any]]) ->str:
        """
        Converts the structured ASR/Diarization JSON segments into a highly readble text format tailored for LLM comprehension

        Expected segment format:
        {
            "speaker": "SPEAKER_00",
            "start": 12.5,
            "end": 15.2,
            "text": "We need to finalize the database architecture."
        }
        """
        formatted_lines = []

        for seg in segments:
            speaker = seg.get("speaker","Unknown")
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = seg.get("text", "").strip()

            line = f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}"
            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    async def run_analysis(self, segments: List[Dict[str, Any]]) -> MeetingAnalysis:
        """
        The main entry point for the service.
        1. Formats the raw segments.
        2. Injects the formatted text into the configured LLM provider.
        3. Returns the strictly validated Pydantic MeetingAnalysis object.
        """
        logger.info("SemanticAnalyzer processing started.")

        if not segments:
            logger.warning("Empty segments provided to SemanticAnalyzer.")
            # Return an empty but valid schema if there is no input.

            return MeetingAnalysis(
                executive_summary = "No meeting data provided.",
                sentiment="Neutral",
                action_items=[],
                key_decisions=[]
            )

        transcript_context = self._format_transcript(segments)
        logger.debug(f"Formatted transcript lenght: {len(transcript_context)} characters.")

        try:
            # Delegate the actual AI inference to the injected provider

            analysis_result = await self.provider.analyze_meeting(transcript_context)
            logger.info("Semantic analysis completed successfully")

            return analysis_result

        except ValidationError as ve:
            logger.error("LLm output did not match the Pydantic schema")
            raise ve
        except Exception as e:
            logger.error(f"An unexpected error occured during analysis: {e}")
            raise e
        