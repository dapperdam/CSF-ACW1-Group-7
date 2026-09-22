"""
Person 4 — Audio Decoder & Verification.

Builds the required positive/negative audio cases (AUD-P1, AUD-N1, AUD-N2,
plus a payload-missing case) and confirms audio_decoder.decode_audio_file
produces the correct verdict for each. Mirrors tests/test_image_decoder.py.
"""
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crypto_payload import Verdict, generate_ed25519_keypair
from audio_stego import encode_audio_file
from audio_stego.common import load_wav_pcm
from audio_decoder import decode_audio_file

LSB_DEPTH = 2
MEDIA_ID = "AUDIO001"
SECRET = b"team-shared-secret-2026"
WRONG_SECRET = b"an-incorrect-key"

GENERATED_FILES = ["cover.wav", "stego_short.wav", "stego_large.wav", "stego_tampered.wav"]

SHORT_MSG = "Explain how steganography can be used to embed hidden verification data."
LARGE_MSG = ("This undergraduate project requires student teams to design, implement and "
             "demonstrate a GUI-based LSB Replacement steganography program that protects "
             "and verifies both image and audio cover objects using steganography, hashing "
             "and digital signatures.")


def make_cover(path, num_samples=20000, frame_rate=44100):
    """16-bit mono PCM WAV filled with a deterministic pseudo-random tone
    plus noise, so it's neither silent nor perfectly periodic."""
    import random

    rng = random.Random(7)
    samples = [rng.randint(-32000, 32000) for _ in range(num_samples)]
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(frame_rate)
        f.writeframes(struct.pack(f"<{num_samples}h", *samples))


def build_and_encode(cover_path, stego_path, message, priv_pem, lsb_depth=LSB_DEPTH, secret=SECRET):
    encode_audio_file(
        input_path=cover_path,
        output_path=stego_path,
        media_id=MEDIA_ID,
        private_key_pem=priv_pem,
        start_secret=secret,
        lsb_depth=lsb_depth,
        note=message,
    )


def run_case(name, stego_path, pub_pem, secret=SECRET, lsb_depth=LSB_DEPTH, media_id=MEDIA_ID):
    result = decode_audio_file(
        stego_path, lsb_depth=lsb_depth, start_secret=secret,
        media_id=media_id, public_key_pem=pub_pem,
    )
    print(f"\n--- {name} ---")
    print(f"Verdict: {result.verdict}")
    if result.error:
        print(f"Error  : {type(result.error).__name__}: {result.error}")
    return result


def cleanup():
    """Delete the test WAVs this script generates, so repeated runs don't
    leave stray files lying around in the working directory."""
    for name in GENERATED_FILES:
        Path(name).unlink(missing_ok=True)


def main():
    priv_pem, pub_pem = generate_ed25519_keypair()
    make_cover("cover.wav")

    try:
        # AUD-P1: positive, short message
        build_and_encode("cover.wav", "stego_short.wav", SHORT_MSG, priv_pem)
        r1 = run_case("AUD-P1: Positive (short message)", "stego_short.wav", pub_pem)
        assert r1.verdict == Verdict.AUTHENTIC, r1.verdict

        # Positive, large message (Project Overview paragraph) at a
        # different LSB depth, per the spec's "various payload sizes" case
        build_and_encode("cover.wav", "stego_large.wav", LARGE_MSG, priv_pem, lsb_depth=4)
        r2 = run_case("Positive (large message, 4 LSBs)", "stego_large.wav", pub_pem, lsb_depth=4)
        assert r2.verdict == Verdict.AUTHENTIC, r2.verdict

        # AUD-N1: negative, corrupt a PCM sample's high bits only (leave the
        # low lsb_depth bits untouched so the embedded payload itself still
        # parses and verifies — isolates the FR9 media-hash check)
        wav = load_wav_pcm("stego_short.wav")
        samples = list(wav.samples)
        # Flip a high bit (bit 8) of a 16-bit signed sample, staying within
        # int16 range: work in unsigned 16-bit space, then convert back.
        unsigned = (samples[500] + 65536) % 65536
        unsigned ^= 1 << 8
        samples[500] = unsigned - 65536 if unsigned >= 32768 else unsigned
        with wave.open("stego_tampered.wav", "wb") as f:
            f.setnchannels(wav.channels)
            f.setsampwidth(wav.sample_width)
            f.setframerate(wav.frame_rate)
            f.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        r3 = run_case("AUD-N1: Negative (tampered sample after embedding)", "stego_tampered.wav", pub_pem)
        assert r3.verdict in (Verdict.TAMPERED, Verdict.SIGNATURE_INVALID), r3.verdict

        # AUD-N2: negative, wrong start-location secret at decode time
        r4 = run_case("AUD-N2: Negative (wrong start-location secret)", "stego_short.wav", pub_pem, secret=WRONG_SECRET)
        assert r4.verdict in (Verdict.WRONG_START_LOCATION, Verdict.PAYLOAD_MISSING, Verdict.CANNOT_VERIFY), r4.verdict

        # Negative: payload missing (no payload embedded at all)
        r5 = run_case("Negative (payload missing — cover, not stego)", "cover.wav", pub_pem)
        assert r5.verdict in (Verdict.PAYLOAD_MISSING, Verdict.WRONG_START_LOCATION, Verdict.CANNOT_VERIFY), r5.verdict

        print("\nAll cases produced expected verdicts.")
    finally:
        # Runs whether the test passed, failed, or errored — keeps the
        # working directory clean either way.
        cleanup()


if __name__ == "__main__":
    main()
