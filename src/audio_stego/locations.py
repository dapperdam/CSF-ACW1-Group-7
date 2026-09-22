"""Shared start-location derivation for audio LSB embedding.

Audio has the same chicken-and-egg problem images do: the payload's start
sample depends on the framed payload's length (in carrier samples), but a
decoder can't know that length before it has extracted something. The fix
mirrors image_encoder's ``resolve_header_start_channel`` /
``resolve_payload_start_channel`` two-step locator, translated from RGB
channels to PCM samples:

1. A small, FIXED-length locator header (magic + 4-byte big-endian length)
   is embedded at a start sample derived from ``f"{media_id}:header"`` — a
   context that needs no knowledge of the main payload's length, so both
   sides can find it independently.
2. The header's declared length is then used to derive the main payload's
   start sample (context = plain ``media_id``), nudged out of the header's
   own region if the two would otherwise overlap.

``audio_stego.encoder`` (embedding) and ``audio_decoder`` (extraction) both
import this module so the two sides can never drift apart.
"""

from __future__ import annotations

from crypto_payload import derive_start_location

from .common import carrier_count_for_bytes

HEADER_MAGIC = b"CSFAHD1"
_LENGTH_BYTES = 4


def header_length_bytes() -> int:
    """Fixed byte length of the locator header (magic + declared length)."""
    return len(HEADER_MAGIC) + _LENGTH_BYTES


def resolve_header_start_sample(
    capacity: int, lsb_depth: int, start_secret: bytes, media_id: str
) -> int:
    """Return the secret-derived sample where the locator header starts."""
    header_carriers = carrier_count_for_bytes(header_length_bytes(), lsb_depth)
    return derive_start_location(
        start_secret, f"{media_id}:header", "audio", capacity, header_carriers
    )


def resolve_payload_start_sample(
    capacity: int,
    lsb_depth: int,
    start_secret: bytes,
    media_id: str,
    framed_length: int,
) -> int:
    """Return the secret-derived sample where the framed payload starts.

    Never overlaps the locator header's own region.
    """
    if framed_length <= 0:
        raise ValueError("framed_length must be positive")

    payload_carriers = carrier_count_for_bytes(framed_length, lsb_depth)
    header_start = resolve_header_start_sample(capacity, lsb_depth, start_secret, media_id)
    header_carriers = carrier_count_for_bytes(header_length_bytes(), lsb_depth)

    start = derive_start_location(start_secret, media_id, "audio", capacity, payload_carriers)
    return _avoid_range_overlap(start, payload_carriers, header_start, header_carriers, capacity)


def _avoid_range_overlap(
    start: int,
    length: int,
    blocked_start: int,
    blocked_length: int,
    capacity: int,
) -> int:
    """Push ``start`` past ``[blocked_start, blocked_start + blocked_length)``
    if the two ranges would overlap. Mirrors image_encoder's private helper
    of the same name exactly, one carrier unit translated to samples."""
    if start + length <= blocked_start or blocked_start + blocked_length <= start:
        return start
    after = blocked_start + blocked_length
    if after + length <= capacity:
        return after
    before = blocked_start - length
    if before >= 0:
        return before
    raise ValueError("audio file is too small for separated locator and payload regions")
