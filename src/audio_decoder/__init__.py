"""WAV LSB audio decoder and verification for the INF2005 steganography workflow."""

from .decoder import (
    AudioDecodeResult,
    decode_audio_file,
)

__all__ = [
    "AudioDecodeResult",
    "decode_audio_file",
]
