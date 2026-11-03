"""Brain2Speech-Net: fast and intelligible brain-to-speech synthesis without text decoding."""
from .aligner import DeepHMMAligner
from .content_encoder import BrainContentEncoder
from .model import Brain2SpeechNet

__version__ = "2.0.0"
__all__ = ["Brain2SpeechNet", "BrainContentEncoder", "DeepHMMAligner"]
