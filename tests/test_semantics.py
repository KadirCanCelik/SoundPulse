import pytest
from unittest.mock import AsyncMock, MagicMock

from src.semantics.qa_agent import InteractiveQAAgent
from src.semantics.schemas import MeetingAnalysis

@pytest.fixture
def dummy_analysis():
    """
    Returns a minimal mock of the MeetingAnalysis schema.
    """
    return MeetingAnalysis(
        executive_summary="This is a test summary about AI deployment.",
        sentiment="Neutral",
        key_decisions=[],
        action_items=[],
    )

@pytest.fixture
def dummy_transcript():
    """
    Returns a minimal mock of the whisper transcript out.
    """

    return [
        {"speaker": "SPEAKER_00", "text": "Are we ready for production?", "start": 0.0, "end": 2.5},
        {"speaker": "SPEAKER_01", "text": "Yes, the tests are passing.", "start": 2.6, "end": 5.0}
    ]

pytestmark = pytest.mark.asyncio

async def test_qa_agent_rag_response(dummy_analysis, dummy_transcript):
    """
    Ensures the InteractiveQAAgent correctly parses the inputs and 
    returns the LLM's response without actually hitting the external provider.
    """

    mock_provider = MagicMock()

    expected_mock_answer = "Yes, according to SPEAKER_01, the tests are passing [02.6 - 05.0s]."
    mock_provider.answer_question = AsyncMock(return_value=expected_mock_answer)

    agent = InteractiveQAAgent(provider=mock_provider)

    test_question = "Is the system ready for production?"

    response = await agent.ask(
        segments=dummy_transcript,
        analysis=dummy_analysis,
        question=test_question
    )

    assert response == expected_mock_answer

    mock_provider.answer_question.assert_called_once()