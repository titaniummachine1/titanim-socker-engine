"""Original synthesized 60s soundtrack for the Titanium showcase video.

No samples, no licensed material -- everything here is generated from sine/
saw oscillators with simple envelopes, so there is nothing to clear.

Mood arc, loosely following the showcase's own slide timeline (SLIDES in
make_titanium_showcase.py): calm pad under the title and engine/sim split,
a rising pulse under the ball-physics and intercept slides (the video's
longest, most technical stretch), a brighter accent under the threats/
challenge slides, and a settle back to the opening pad for the ending.

Run:  py -3.12 scripts/make_showcase_audio.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "titanium_showcase_audio.wav"

SR = 44100
DURATION = 60.0
N = int(SR * DURATION)

# A minor-ish, moody but not dark -- Am7 / Fmaj7 / Cmaj7 / G, two bars each.
CHORDS = [
    [220.00, 261.63, 329.63, 392.00],  # A C E G  (Am7)
    [174.61, 220.00, 261.63, 349.23],  # F A C F  (Fmaj7-ish)
    [130.81, 164.81, 196.00, 261.63],  # C E G C  (Cmaj7-ish)
    [196.00, 246.94, 293.66, 392.00],  # G B D G
]
BAR_S = 60.0 / 4  # ~4 bars/section at a slow 60bpm feel -> 15s per chord... too slow.


def env_adsr(n_samples, a, d, s_level, r, sr=SR):
    t = np.arange(n_samples) / sr
    total = n_samples / sr
    env = np.ones(n_samples) * s_level
    a_n = int(a * sr)
    d_n = int(d * sr)
    r_n = int(r * sr)
    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n)
    if d_n > 0 and a_n + d_n <= n_samples:
        env[a_n:a_n + d_n] = np.linspace(1, s_level, d_n)
    if r_n > 0 and r_n < n_samples:
        env[-r_n:] *= np.linspace(1, 0, r_n)
    return env


def pad_layer():
    """Slow, soft chord pad -- the bed for the whole track."""
    out = np.zeros(N)
    section_len = DURATION / len(CHORDS)
    for i, chord in enumerate(CHORDS):
        s0 = int(i * section_len * SR)
        s1 = int((i + 1) * section_len * SR)
        n = s1 - s0
        t = np.arange(n) / SR
        chunk = np.zeros(n)
        for freq in chord:
            # a couple of harmonics for warmth, soft low-pass via harmonic falloff
            chunk += 0.55 * np.sin(2 * math.pi * freq * t)
            chunk += 0.18 * np.sin(2 * math.pi * freq * 2 * t)
            chunk += 0.07 * np.sin(2 * math.pi * freq * 0.5 * t)
        chunk /= len(chord)
        chunk *= env_adsr(n, a=0.8, d=0.6, s_level=0.85, r=0.9)
        out[s0:s1] += chunk
    return out


def bass_pulse():
    """Slow rhythmic sub pulse, intensity rises through the technical middle
    section (ball physics + intercept, roughly 8s-42s) and settles for the end."""
    out = np.zeros(N)
    root_by_section = [55.00, 43.65, 32.70, 49.00]
    section_len = DURATION / len(root_by_section)
    beat = 0.5  # seconds per pulse
    t_axis = np.arange(N) / SR
    # Intensity envelope over the whole timeline: calm -> build -> bright -> settle.
    intensity = np.interp(
        t_axis,
        [0, 8, 20, 34, 42, 49, 55, 60],
        [0.25, 0.3, 0.55, 0.85, 0.7, 0.6, 0.4, 0.25],
    )
    for i, root in enumerate(root_by_section):
        s0 = int(i * section_len * SR)
        s1 = int((i + 1) * section_len * SR)
        n_beats = int((s1 - s0) / SR / beat) + 1
        for b in range(n_beats):
            bs = s0 + int(b * beat * SR)
            if bs >= s1:
                break
            bn = min(int(beat * SR), N - bs)
            if bn <= 0:
                continue
            t = np.arange(bn) / SR
            note = 0.9 * np.sin(2 * math.pi * root * t) + 0.3 * np.sin(2 * math.pi * root * 2 * t)
            note *= env_adsr(bn, a=0.01, d=0.15, s_level=0.15, r=0.25)
            out[bs:bs + bn] += note * intensity[bs:bs + bn]
    return out


def shimmer_arp():
    """Sparse high arpeggio accents under the threats/challenge slides
    (~42s-55s) so that stretch feels a touch brighter/more driven."""
    out = np.zeros(N)
    notes = [523.25, 659.25, 783.99, 880.00, 783.99, 659.25]  # C E G A G E
    start, end = 42.5, 55.0
    step = 0.28
    i = 0
    tcur = start
    while tcur < end:
        freq = notes[i % len(notes)]
        s0 = int(tcur * SR)
        n = int(0.5 * SR)
        if s0 + n > N:
            break
        t = np.arange(n) / SR
        note = 0.5 * np.sin(2 * math.pi * freq * t)
        note *= env_adsr(n, a=0.005, d=0.08, s_level=0.0, r=0.05)
        fade = np.interp(tcur, [start, start + 2, end - 2, end], [0.0, 1.0, 1.0, 0.0])
        out[s0:s0 + n] += note * fade
        tcur += step
        i += 1
    return out


def main() -> None:
    mix = pad_layer() * 0.55 + bass_pulse() * 0.5 + shimmer_arp() * 0.35

    # Overall fade in/out so it doesn't click at the video boundaries.
    fade_n = int(1.2 * SR)
    mix[:fade_n] *= np.linspace(0, 1, fade_n)
    mix[-fade_n:] *= np.linspace(1, 0, fade_n)

    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.9

    stereo = np.stack([mix, mix], axis=1)
    pcm = (stereo * 32767).astype(np.int16)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import wave

    with wave.open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
