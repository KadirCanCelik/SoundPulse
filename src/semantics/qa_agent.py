import logging
from typing import Any, Dict, List, Optional

from .providers import BaseLLMProvider
from .schemas import MeetingAnalysis

logger = logging.getLogger(__name__)

class InteractiveQAAgent:
    """
    Handles user queries about the meeting.
    Combines the raw transcript with the semantic analysis to provide heavily grounded answers.
    """
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def _format_transcript_safe(self, segments: List[Dict[str, Any]]) -> str:

        lines = []

        for seg in segments:
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg.get("text", "")

                lines.append(f"[{start:05.1f} - {end:05.1f}] {speaker}: {text}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping corrupted segment during formatting: {seg} | Error: {e}")
                continue

        return "\n".join(lines)

    async def ask(self, segments: List[Dict[str, Any]], analysis: MeetingAnalysis, question: str, chat_history: Optional[List[Dict[str, str]]] = None) ->str:
        """
        Constructs the context and asks the LLM the question.
        """
        logger.info(f"QAAgent processing question: '{question}'")

        if chat_history is None:
            chat_history = []

        # Format the raw JSON segments into a readable chronological transcript
        formatted_transcript = self._format_transcript_safe(segments)

        # Serialize the strictly typed Pydantic model into a JSON string
        analysis_json = analysis.model_dump_json(indent=2)

        system_context = (
            "You are an analytical AI assistant. You answer questions based ONLY on the provided meeting context.\n"
            "The transcript has imperfect speaker tags (e.g., 'speaker_0' might represent multiple people due to overlapping audio). "
            "You must logically deduce who is speaking based on conversational context, names, and pronouns.\n\n"
            
            "<instructions>\n"
            "You MUST use a step-by-step thinking process before providing the final answer. Format your response EXACTLY as follows:\n\n"
            "<thinking>\n"
            "0. ADVERSARIAL CHECK: Does the user's question contain false assumptions or contradict the transcript? If yes, explicitly correct it.\n"
            "1. Analyze the dialogue flow: Who is asking the question? Who is answering?\n"
            "2. Locate ALL sentences and segments that contribute to the answer.\n"
            "3. Extract the EXACT timestamp brackets (e.g., [010.0 - 032.5]) for EVERY piece of evidence you use.\n"
            "</thinking>\n\n"
            "<answer>\n"
            "Write your final, concise answer here. You MUST append ALL relevant timestamps at the end of your answer to ground your claims.\n"
            "If the text does not contain the information, explicitly say 'I cannot find this information in the transcript.' Do not invent facts.\n"
            "</answer>\n"
            "</instructions>\n\n"
            
            "<meeting_analysis>\n"
            f"{analysis_json}\n"
            "</meeting_analysis>\n\n"
            
            "<transcript>\n"
            f"{formatted_transcript}\n"
            "</transcript>\n\n"
        )

        if chat_history:
            system_context += "<conversation_history>\n"
            for msg in chat_history:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                system_context += f"{role}: {content}\n"
            system_context += "</conversation_history>\n\n"

        answer = await self.provider.answer_question(
            context_data=system_context,
            question=question
        )

        return answer