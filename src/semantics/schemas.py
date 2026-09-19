from typing import List, Optional

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """
    Represents a single task, duty, or action item assigned during the meeting.
    """

    assignee: str = Field(
        description= "The name or speaker ID (e.g., 'speaker_1') of the person assigned to the task. Use 'Unassigned' if unclear.")

    task_description:str = Field(
        description= "A clear, concise explanation of the task or responsibility.")

    start_time: float = Field(
        description= "The exact start time (in seconds) of the dialogue where this task was mentioned.")

    end_time: float = Field(
        description= "The exact end time (in seconds) of the dialogue where this task was mentioned.")


class KeyDecision(BaseModel):
    """
    Represents an important decision made or agreed upon during the meeting
    """
    topic: str = Field(
        description="The main topic of the decision")

    decision_summary:str = Field(
        description="What was ultimately decided regarding this topic.")

    start_time:float = Field(
        description="The exact start time (in seconds) of the dialogue whe this decision was finalized."
    )

class MeetingAnalysis(BaseModel):
    """
    The ultimate root schema. The LLM must return its entire analysis structured exactly like this
    """ 
    executive_summary: str = Field(
        description="A high-level, professional executive summary of the entire meeting 3-4 sentences."
    )

    sentiment: str = Field(
        description="The overall emotional tone of the meeting (e.g., 'Positive', 'Tense', 'Collaborative', 'Neutral').")

    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="A list of all tasks assigned during the meeting. Empty list if none.")

    key_decisions: List[KeyDecision] = Field(
        default_factory=list,
        description="A list of all major decisions made. Empty list if none."
    )

class TranscriptSegment(BaseModel):
    speaker: str = Field(description="Identifier for the speaker (e.g., SPEAKER_00)") 
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="the transcribed text for this segment")

class FullAnalysisResponse(BaseModel):
    meeting_id: str
    transcript: List[TranscriptSegment] = Field(description="Full timestamped transcript of the audio")
    analysis: MeetingAnalysis = Field(description="LLM generated structural analysis and action items")
    pdf_filepath: Optional[str] = None