"""WAV LSB extraction and verification, matching audio_stego/encoder.py's
wire format exactly (Person 4's deliverable: FR8, FR9, FR10 for audio).

TEAM CONVENTION for media_hash: reproducible from the STEGO file alone,
since per the spec's demo scenario the verifier only ever receives the
stego WAV (never the original cover). ``audio_stego.common.stable_audio_bytes``
already implements this — it masks out the low ``lsb_depth`` bits of every
PCM sample before hashing, which are exactly the bits LSB embedding is
allowed to touch. ``audio_stego.encoder.create_signed_audio_payload`` hashes
the cover the same way, so encode and decode agree.

Start-location convention: mirrors image_decoder's two-step locator exactly
(see ``audio_stego.locations`` for why a single derivation isn't possible —
the payload's start depends on its own length). A fixed-length locator
header is read first to learn the payload's declared length, then the real
payload location is derived from that length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crypto_payload import (
    CryptoPayloadError,
    SerializationFormatError,
    Verdict,
    VerificationPayload,
    media_hash_matches,
    parse_and_verify,
    verdict_for_error,
)
from audio_stego.common import (
    AudioStegoError,
    LENGTH_BYTES as _FRAME_LENGTH_BYTES,
    MAGIC as FRAME_MAGIC,
    load_wav_pcm,
    stable_audio_bytes,
)
from audio_stego.locations import (
    HEADER_MAGIC,
    header_length_bytes,
    resolve_header_start_sample,
    resolve_payload_start_sample,
)


@dataclass(frozen=True, slots=True)
class AudioDecodeResult:
    """What the GUI needs to display after audio decoding (FR10)."""

    verdict: Verdict
    payload: VerificationPayload | None
    error: Exception | None


def decode_audio_file(
    stego_path: str | Path,
    *,
    lsb_depth: int,
    start_secret: bytes,
    media_id: str,
    public_key_pem: bytes,
) -> AudioDecodeResult:
    """Extract, verify, and produce a verdict for a stego WAV.

    ``lsb_depth``, ``start_secret``, and ``media_id`` must be the same
    values used at encode time — these are agreed out-of-band (e.g. known
    to both parties, not derived from the file itself), exactly like
    Person 3's ``encode_audio_file`` requires them as inputs.
    """
    error: Exception | None = None
    hash_matches: bool | None = None
    payload: VerificationPayload | None = None

    try:
        payload, hash_matches = _decode_and_verify(
            stego_path, lsb_depth=lsb_depth, start_secret=start_secret,
            media_id=media_id, public_key_pem=public_key_pem,
        )
    except CryptoPayloadError as exc:
        error = exc
    except AudioStegoError as exc:
        # Not a valid/uncompressed PCM WAV, wrong sample width, etc. — treat
        # as "we could not make sense of this as a signed stego WAV".
        error = SerializationFormatError(str(exc))
    except (OSError, ValueError) as exc:
        error = SerializationFormatError(str(exc))

    verdict = verdict_for_error(error, hash_matches)
    return AudioDecodeResult(verdict=verdict, payload=payload, error=error)


def _decode_and_verify(
    stego_path: str | Path,
    *,
    lsb_depth: int,
    start_secret: bytes,
    media_id: str,
    public_key_pem: bytes,
) -> tuple[VerificationPayload, bool]:
    if not isinstance(lsb_depth, int) or not 1 <= lsb_depth <= 8:
        raise ValueError("lsb_depth must be an integer from 1 to 8")

    wav = load_wav_pcm(stego_path)
    capacity = len(wav.samples)

    # --- 1. locate + read the fixed-size locator header ---
    header_start = resolve_header_start_sample(capacity, lsb_depth, start_secret, media_id)
    header_bit_len = header_length_bytes() * 8
    header_bytes = _extract_bytes(wav.samples, header_start, header_bit_len, lsb_depth)

    if header_bytes[: len(HEADER_MAGIC)] != HEADER_MAGIC:
        raise SerializationFormatError("locator header magic mismatch — no payload found")
    framed_length = int.from_bytes(header_bytes[len(HEADER_MAGIC):], "big")
    if framed_length <= 0:
        raise SerializationFormatError("declared framed payload length is not positive")

    # --- 2. locate + read the framed, signed payload ---
    start_location = resolve_payload_start_sample(
        capacity, lsb_depth, start_secret, media_id, framed_length
    )
    framed_bytes = _extract_bytes(wav.samples, start_location, framed_length * 8, lsb_depth)

    if framed_bytes[: len(FRAME_MAGIC)] != FRAME_MAGIC:
        raise SerializationFormatError("frame magic mismatch — payload is corrupted or absent")
    declared_len = int.from_bytes(
        framed_bytes[len(FRAME_MAGIC): len(FRAME_MAGIC) + _FRAME_LENGTH_BYTES], "big"
    )
    envelope_bytes = framed_bytes[len(FRAME_MAGIC) + _FRAME_LENGTH_BYTES:]
    if len(envelope_bytes) != declared_len:
        raise SerializationFormatError("truncated or malformed payload envelope")

    # --- 3. verify signature and parse payload (Person 5) ---
    payload = parse_and_verify(envelope_bytes, public_key_pem)

    # --- 4. FR9: recompute stable audio bytes and compare (shared
    #     convention with audio_stego.encoder.create_signed_audio_payload —
    #     see this module's docstring) ---
    media_bytes = stable_audio_bytes(wav, lsb_depth)
    hash_matches = media_hash_matches(media_bytes, payload.media_hash)

    return payload, hash_matches


def _extract_bytes(
    samples: tuple[int, ...], start_location: int, num_bits: int, lsb_depth: int
) -> bytes:
    """Extract ``num_bits`` starting at ``start_location``, undoing
    encoder.py's ``embed_lsb`` MSB-first packing and last-sample zero
    padding exactly."""
    bits: list[int] = []
    bit_index = 0
    sample_index = start_location
    mask = (1 << lsb_depth) - 1

    while bit_index < num_bits:
        remaining = num_bits - bit_index
        take = min(lsb_depth, remaining)
        if sample_index >= len(samples):
            raise SerializationFormatError("payload runs past the end of the audio file")
        value = samples[sample_index] & mask
        if take < lsb_depth:
            value >>= (lsb_depth - take)  # undo the zero-padding on a partial final chunk
        for i in range(take - 1, -1, -1):
            bits.append((value >> i) & 1)
        bit_index += take
        sample_index += 1

    return _bits_to_bytes(bits)


def _bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)
