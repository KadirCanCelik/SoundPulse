import gc
import logging
import os
from typing import Any, Dict, List

import torch

from src.audio.speaker_diarizer import SpeakerDiarizer
from src.audio.transcriber import AudioTranscriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")

logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    An orchestrator that runs audio transcription and speaker diarization sequentially.
    """
    def __init__(self):

        logger.info("Initializing AudioProcessingPipeline")

        try:
            self.transcriber = AudioTranscriber(model_size="medium")
            self.diarizer = SpeakerDiarizer()
            logger.info("Pipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise

    def _clear_memory(self) -> None:
        """
        Forces garbage collection and clears CUDA cache to manage VRAM limits.
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA memory cache cleared.")

    def process_audio(self, audio_path: str, num_speakers: int = 2) -> List[Dict[str, Any]]:
        """
        Executes the end-to-end audio processing workflow.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting pipeline for {audio_path} with {num_speakers} expected speakers.")

        try:
            # 1/2 Transcription and Word Alignment
            logger.info("Running transcription...")
            words_data = self.transcriber.process(audio_path)

            self._clear_memory()

            # 2/2 Speaker Diarization
            logger.info("Running speaker diarization...")
            final_segments = self.diarizer.process(audio_path, words_data, num_speakers)

            self._clear_memory()

            logger.info(f"Pipeline completed. Generated {len(final_segments)} final segments.")

            return final_segments

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

if __name__ == "__main__":
    test_audio = "audio_samples/test_audio.wav"

    if os.path.exists(test_audio):
        pipeline = AudioProcessingPipeline()
        segments = pipeline.process_audio(test_audio, num_speakers=2)

        for seg in segments[:5]:
            print(f"[{seg['start']:05.2f} - {seg['end']:05.2f}] {seg['speaker']}: {seg['text']}")

    else:
        print(f"Test file not found: {test_audio}. Please place a valid audio file in the directory.")
 