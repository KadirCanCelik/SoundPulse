import json
import logging
import os
import re
from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from pydantic import ValidationError

from .schemas import MeetingAnalysis

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Ensures all providers implement a unified interface for meeting analysis.
    """

    @abstractmethod
    async def analyze_meeting(self, transcript_context: str) -> MeetingAnalysis:
        """
        Takes timestamped transcript text and returns a validated MeetingAnalysis object.
        """
        pass

    @abstractmethod
    async def answer_question(self, context_data: str, question: str) -> str:
        """
        Takes the meeting context and a user question, returning a grounded answer.
        """
        pass

class GroqProvider(BaseLLMProvider):

    def __init__(self, model_name:str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model_name = model_name

    async def analyze_meeting(self, transcript_context) -> MeetingAnalysis:
        logger.info(f"Initiating analysis via Groq using model: {self.model_name}")

        system_prompt = (
            "You are an elite corporate AI assistant analyzing a timestamped meeting transcript.\n"
            "Your task is to extract an executive summary, sentiment, key decisions, and action items.\n"
            "CRITICAL RULES:\n"
            "1. You MUST respond ONLY with a valid JSON object matching the requested schema.\n"
            "2. Extract action items and decisions with EXACT start_time and end_time (in seconds).\n"
            "3. If no action items or decisions exist, return empty arrays [].\n"
            "4. Do NOT include markdown code blocks (like ```json), explanations, or greetings.\n"
            f"Required JSON Schema:\n{json.dumps(MeetingAnalysis.model_json_schema(), indent=2)}"
        )


        response = await self.client.chat.completions.create(
            model = self.model_name,
            response_format={"type":"json_object"},
            temperature=0.1,
            max_tokens=4096,
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content":f"Meeting Transcript:\n\n{transcript_context}"}
            ]
        )

        raw_json_str = response.choices[0].message.content

        try:

            return MeetingAnalysis.model_validate_json(raw_json_str)

        except ValidationError as e:
            logger.error(f"Pydantic Validation faied: {e}")
            logger.error(f"Raw output from LLM: { raw_json_str}")
            raise

    async def answer_question(self, context_data, question) -> str:

        logger.info(f"Q-A with Groq is starting... Question: {question}")

        system_prompt = (
            "You are a strict, forensic AI assistant analyzing a meeting.\n"
            "You have been provided with the full meeting transcript and its structured summary.\n"
            "CRITICAL RULES:\n"
            "1. You MUST answer the user's question based ONLY on the provided context.\n"
            "2. If the answer is not in the context, say 'I cannot find the answer in the meeting records.' Do NOT invent facts.\n"
            "3. Whenever you refer to an event, task, or dialogue, you MUST cite the exact timestamp from the transcript in this format: [start_time - end_time]s.\n"
            "4. Be concise, professional, and precise."
        )

        response = await self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {"role":"system", "content": system_prompt},
                {"role": "user", "content":f"Context Data:\n{context_data}\n\nUser Question:\n{question}"}
            ]
        )

        raw_answer = response.choices[0].message.content

        clean_answer = re.sub(r'<thinking>.*?</thinking>', '', raw_answer, flags=re.DOTALL)

        clean_answer = re.sub(r'</?answer>', '', clean_answer, flags=re.IGNORECASE)

        return clean_answer.strip()