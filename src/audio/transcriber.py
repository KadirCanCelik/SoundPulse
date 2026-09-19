import logging
import os
from typing import Any, Dict, List

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_size: str = "medium", device: str = "cuda"):
        """
        Initializes the WhisperModel
        """
        self.device = device

        self.compute_type = "int8_float16" if device == "cuda" else "int8"

        logger.info(f"Initializing WhisperModel '{model_size}' on {self.device} (Compute: {self.compute_type})...")

        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
            logger.info("WhisperModel loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load WhisperModel: {str(e)}")
            raise RuntimeError(f"ASR Model initialization failed: {str(e)}")

    def process(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes the audio file and extracts word-level timestamps.
        This output serves as the foundational data for the subsequent speaker diarization module.
        """

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        logger.info(f"Starting transcription and word aligment for: {audio_path}")

        try:
            # Enabling word_timestamps=True is critical for micro-chunking diarization.
            # vad_filter=True ignores silent segments, reducing GPU workload.
            segments, info = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            logger.info(f"Language detected: {info.language} (Probability: {info.language_probability:.2f})")

            aligned_words = []

            for segment in segments:
                # Ensure word_timestamps were generated successfully
                if segment.words is None:
                    continue
                
                for word in segment.words:
                    aligned_words.append({
                        "word": word.word.strip(),
                        "start":word.start,
                        "end":word.end,
                        "probability":word.probability
                    })

            logger.info(f"Transcription completed. Extracted {len(aligned_words)} words with precise timestamps.")
            return aligned_words

        except Exception as e:
            logger.error(f"An error occured during transcription: {str(e)}")
            raise

if __name__ == "__main__":

    test_audio = "audio_samples/test_audio.wav"

    if os.path.exists(test_audio):
        transcriber = AudioTranscriber(model_size="medium")
        words = transcriber.process(test_audio)
        
        for w in words[:10]:
            print(f"[{w['start']:05.2f} - {w['end']:05.2f}] : '{w['word']}' (Confidence: {w['probability']:.2f})")