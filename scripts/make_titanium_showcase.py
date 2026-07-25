"""Render a 60s Titanium showcase MP4 for AIA / pitch videos.

Exclusive slides: each scene fully fades out before the next fades in.
Run:  py -3.12 scripts/make_titanium_showcase.py
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "titanium_showcase_60s.mp4"

W, H = 1920, 1080
FPS = 60
DURATION = 60.0
N_FRAMES = int(DURATION * FPS)

BG = (12, 14, 18)
BG2 = (22, 28, 36)
STEEL = (176, 190, 204)
STEEL_DIM = (110, 124, 140)
ACCENT = (90, 200, 210)
ACCENT2 = (232, 180, 90)
WHITE = (236, 240, 244)
MUTED = (140, 150, 162)
PITCH = (34, 72, 52)
PITCH_LINE = (210, 225, 210)
DANGER = (220, 90, 80)
GOOD = (110, 200, 130)

# Exclusive windows: (start, end). Gaps between = fully blank.
# Short intro → ball bounce (fast then slow) → long intercept resim.
SLIDES = [
    (0.0, 3.5),    # 0 title
    (4.0, 7.5),    # 1 engine vs sim
    (8.2, 20.0),   # 2 45deg kick: accelerate to wall-5m, then slow bounce/land
    (20.8, 42.0),  # 3 intercept: walk fail + sprint resim
    (42.8, 49.0),  # 4 threats
    (49.8, 55.0),  # 5 challenge
    (55.8, 60.0),  # 6 end
]
FADE_IN = 0.4
FADE_OUT = 0.4


def load_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_HERO = load_font(110, bold=True)
FONT_H1 = load_font(56, bold=True)
FONT_H2 = load_font(40, bold=True)
FONT_BODY = load_font(30)
FONT_SMALL = load_font(22)
FONT_TINY = load_font(18)
FONT_EQ = load_font(26)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def ease_out(t: float) -> float:
    t = clamp01(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return 3 * t * t - 2 * t * t * t


def mix(c1, c2, t: float):
    t = clamp01(t)
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def slide_alpha(t: float, start: float, end: float) -> float:
    """Opacity for one exclusive slide. 0 outside; fade in then fade out inside."""
    if t < start or t > end:
        return 0.0
    fi = min(FADE_IN, (end - start) * 0.35)
    fo = min(FADE_OUT, (end - start) * 0.35)
    if t < start + fi:
        return ease_out((t - start) / fi)
    if t > end - fo:
        return ease_out((end - t) / fo)
    return 1.0


def slide_progress(t: float, start: float, end: float) -> float:
    """0..1 progress through the fully-visible middle of a slide."""
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    return (t - start) / (end - start)


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_text(draw, xy, text, font, fill, a: float):
    if a <= 0.01:
        return
    draw.text(xy, text, font=font, fill=mix(BG, fill, a))


def centered(draw, text, y, font, fill, a: float):
    if a <= 0.01:
        return
    tw, _ = text_size(draw, text, font)
    draw_text(draw, ((W - tw) // 2, y), text, font, fill, a)


def draw_bg(draw, t: float) -> None:
    for y in range(0, H, 4):
        k = 0.5 + 0.5 * math.sin((y / H) * math.pi + t * 0.12)
        draw.rectangle([0, y, W, y + 4], fill=mix(BG, BG2, 0.3 + 0.2 * k))


def draw_pitch(draw, cx, cy, pw, ph, a: float):
    line = mix(BG, PITCH_LINE, a)
    field = mix(BG, PITCH, a * 0.9)
    x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
    x1, y1 = int(cx + pw / 2), int(cy + ph / 2)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill=field, outline=line, width=3)
    mx = (x0 + x1) // 2
    draw.line([(mx, y0), (mx, y1)], fill=line, width=2)
    r = int(ph * 0.16)
    draw.ellipse([mx - r, cy - r, mx + r, cy + r], outline=line, width=2)
    gh = int(ph * 0.26)
    draw.rectangle([x0 - 10, cy - gh // 2, x0, cy + gh // 2], outline=line, width=3)
    draw.rectangle([x1, cy - gh // 2, x1 + 10, cy + gh // 2], outline=line, width=3)
    return x0, y0, x1, y1


def disc(draw, x, y, r, color, a, label=None):
    col = mix(BG, color, a)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=col)
    if label:
        tw, th = text_size(draw, label, FONT_TINY)
        draw_text(draw, (x - tw // 2, y - th // 2), label, FONT_TINY, BG, a)


_CURRENT_IMG = None  # set by render_frame each frame for localized blur compositing


def blurred_ring(cx, cy, r, color, width, a, blur=6):
    """Soft/blurred ring composited straight onto the current frame.

    A hard instantaneous swap between the walk-reach and sprint-reach circles
    read as a glitch cut. This lets the two states cross-fade through a
    genuinely blurred intermediate instead, so the handoff reads as one
    continuous wave settling into its answer.
    """
    if _CURRENT_IMG is None or a <= 0.01 or r <= 2:
        return
    pad = int(width + blur * 3 + 4)
    size = int(2 * (r + pad))
    if size <= 2 or size > 2400:
        return
    patch = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pd = ImageDraw.Draw(patch)
    col = (color[0], color[1], color[2], int(255 * clamp01(a)))
    pd.ellipse([pad, pad, pad + 2 * r, pad + 2 * r], outline=col, width=int(width))
    if blur > 0.3:
        patch = patch.filter(ImageFilter.GaussianBlur(blur))
    _CURRENT_IMG.paste(patch, (int(cx - r - pad), int(cy - r - pad)), patch)


# ---------------------------------------------------------------------------
# Ball paths: scripted kicks (this repo) — 45deg bounce demo + goal intercept demo
# ---------------------------------------------------------------------------

BOUNCE_PATH_JSON = ROOT / "out" / "kick_path_center_max_45.json"
GOAL_PATH_JSON = ROOT / "out" / "kick_path_center_max_goal.json"


def ensure_kick_paths() -> None:
    import subprocess
    import sys

    if BOUNCE_PATH_JSON.is_file() and GOAL_PATH_JSON.is_file():
        return
    recorder = ROOT / "scripts" / "showcase_kick_team.py"
    print("Recording scripted kick paths…")
    subprocess.check_call([sys.executable, str(recorder)], cwd=str(ROOT))


def load_kick_path(path: Path):
    import json

    doc = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    events = []
    prev = None
    for sm in doc["samples"]:
        kind = "coast"
        if prev is not None:
            if (not prev["grounded"]) and sm["grounded"]:
                kind = "land"
                events.append((sm["t"], "land", sm["x"], sm["z"]))
            near_wall = (
                abs(sm["x"] - doc["x_max"]) < 0.05
                or abs(sm["x"] - doc["x_min"]) < 0.05
                or abs(sm["z"] - doc["z_max"]) < 0.05
                or abs(sm["z"] - doc["z_min"]) < 0.05
            )
            vel_flip = prev["vx"] * sm["vx"] < 0 or prev["vz"] * sm["vz"] < 0
            if near_wall and vel_flip and (
                abs(sm["vx"]) + abs(sm["vz"]) > 0.05 or abs(prev["vx"]) + abs(prev["vz"]) > 0.5
            ):
                kind = "wall"
                events.append((sm["t"], "wall", sm["x"], sm["z"]))
            if sm["end"] != "none":
                kind = sm["end"]
                events.append((sm["t"], sm["end"], sm["x"], sm["z"]))
        samples.append((sm["t"], sm["x"], sm["z"], sm["height"], kind))
        prev = sm
    last = doc["samples"][-1]
    if last["vx"] == 0 and last["vz"] == 0 and last["grounded"]:
        events.append((last["t"], "stop", last["x"], last["z"]))
        t, x, z, h, _ = samples[-1]
        samples[-1] = (t, x, z, h, "stop")
    return doc, samples, events


ensure_kick_paths()
BOUNCE_DOC, BOUNCE_SAMPLES, BOUNCE_EVENTS = load_kick_path(BOUNCE_PATH_JSON)
GOAL_DOC, PATH_SAMPLES, PATH_EVENTS = load_kick_path(GOAL_PATH_JSON)
KICK_DOC = BOUNCE_DOC  # ball-physics slide captions
PITCH_X_MIN = float(BOUNCE_DOC["x_min"])
PITCH_X_MAX = float(BOUNCE_DOC["x_max"])
PITCH_Z_MIN = float(BOUNCE_DOC["z_min"])
PITCH_Z_MAX = float(BOUNCE_DOC["z_max"])


def sample_at_time(samples, t):
    best = samples[0]
    for sm in samples:
        if sm[0] <= t + 1e-9:
            best = sm
        else:
            break
    return best


# Engine event-leg model: one analytic solve until WALL or STOP, then teleport.
# Presentation skips land (height snap only) — XZ legs are wall/stop boundaries.
BOUNCE_LEGS = [("start", BOUNCE_SAMPLES[0][0], BOUNCE_SAMPLES[0][1], BOUNCE_SAMPLES[0][2], 0.0)]
for et, kind, ex, ez in BOUNCE_EVENTS:
    if kind == "land":
        continue
    sm = sample_at_time(BOUNCE_SAMPLES, et)
    BOUNCE_LEGS.append((kind, et, ex, ez, sm[3]))
if BOUNCE_LEGS[-1][0] != "stop":
    last = BOUNCE_SAMPLES[-1]
    BOUNCE_LEGS.append(("stop", last[0], last[1], last[2], last[3]))


def bounce_leg_index(p: float) -> int:
    """Discrete teleport: hold on each event-leg endpoint, then jump to the next."""
    p = clamp01(p)
    n = len(BOUNCE_LEGS)
    return min(n - 1, int(p * (n - 0.15)))


def pitch_map(wx, wz, x0, y0, x1, y1):
    """Map real sim meters (x,z) onto the drawn pitch rect. Straight XZ — no height bend."""
    px = lerp(x0 + 20, x1 - 20, (wx - PITCH_X_MIN) / (PITCH_X_MAX - PITCH_X_MIN))
    py = lerp(y1 - 20, y0 + 20, (wz - PITCH_Z_MIN) / (PITCH_Z_MAX - PITCH_Z_MIN))
    return int(px), int(py)


def find_intercept_index(player_x, player_z, speed, reach=1.75, before_t=None):
    """Earliest sample where reach covers the ball. None if never (before before_t)."""
    for i, (tt, bx, bz, _h, _k) in enumerate(PATH_SAMPLES):
        if tt <= 1e-6:
            continue
        if before_t is not None and tt > before_t + 1e-6:
            break
        dist = math.hypot(bx - player_x, bz - player_z)
        if dist <= speed * tt + reach:
            return i
    return None


def goal_sample_index():
    for i, (_t, _x, _z, _h, kind) in enumerate(PATH_SAMPLES):
        if kind == "goal_away" or kind == "goal_home":
            return i
    # fallback: last sample
    return len(PATH_SAMPLES) - 1


# Defender Z tuned so walk cannot cut before goal, but sprint can.
RIGHT_GOAL_X = 39.5
INTERCEPT_PLAYER = (RIGHT_GOAL_X - 10.0, 9.1)  # 10m from goal; z=9.1 => walk miss / sprint hit
WALK_SPEED = 7.0
SPRINT_SPEED = 8.0
INTERCEPT_REACH = 1.75
GOAL_IDX = goal_sample_index()
GOAL_T = PATH_SAMPLES[GOAL_IDX][0]
WALK_IDX = find_intercept_index(
    *INTERCEPT_PLAYER, WALK_SPEED, INTERCEPT_REACH, before_t=GOAL_T
)
SPRINT_IDX = find_intercept_index(
    *INTERCEPT_PLAYER, SPRINT_SPEED, INTERCEPT_REACH, before_t=GOAL_T
)
# Titanium pattern: try walk; if it cannot stop the goal, resimulate with sprint.
USE_SPRINT = WALK_IDX is None and SPRINT_IDX is not None
INTERCEPT_IDX = SPRINT_IDX if USE_SPRINT else (WALK_IDX if WALK_IDX is not None else GOAL_IDX)
INTERCEPT_SPEED = SPRINT_SPEED if USE_SPRINT else WALK_SPEED
INTERCEPT_MODE = "sprint" if USE_SPRINT else "walk"
INTERCEPT_POS = PATH_SAMPLES[INTERCEPT_IDX][1:3]
INTERCEPT_T = PATH_SAMPLES[INTERCEPT_IDX][0]


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_title(draw, t, a):
    if a <= 0:
        return
    p = slide_progress(t, *SLIDES[0])
    pulse = 0.5 + 0.5 * math.sin(t * 2.0)

    # Pitch-line grid sweeping in from the edges, like a stadium floodlight
    # snapping the markings into view rather than a static bordered box.
    reveal = ease_out(clamp01(p / 0.5))
    if reveal > 0.01:
        cx, cy = W // 2, H // 2
        half_w, half_h = int(820 * reveal), int(430 * reveal)
        line_col = mix(BG, ACCENT, a * 0.35 * reveal)
        draw.rectangle([cx - half_w, cy - half_h, cx + half_w, cy + half_h], outline=line_col, width=2)
        draw.line([(cx, cy - half_h), (cx, cy + half_h)], fill=line_col, width=2)
        r = int(140 * reveal)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=line_col, width=2)

    # Sweeping scan line — one clean pass left to right in the first ~0.6s.
    scan_p = clamp01((t - SLIDES[0][0]) / 0.6)
    if 0.0 < scan_p < 1.0:
        sx = int(lerp(0, W, ease_out(scan_p)))
        draw.rectangle([max(0, sx - 3), 0, sx, H], fill=mix(BG, ACCENT, a * 0.5))

    # Title materializes word-by-word instead of popping in whole.
    word_a1 = ease_out(clamp01((p - 0.10) / 0.22))
    centered(draw, "TITANIUM", H // 2 - 130, FONT_HERO, mix(STEEL, WHITE, pulse * 0.35), a * word_a1)

    underline_a = ease_out(clamp01((p - 0.30) / 0.15))
    if underline_a > 0.01:
        uw = int(560 * underline_a)
        draw.line(
            [(W // 2 - uw // 2, H // 2 - 30), (W // 2 + uw // 2, H // 2 - 30)],
            fill=mix(BG, ACCENT2, a * underline_a),
            width=3,
        )

    sub_a = ease_out(clamp01((p - 0.40) / 0.2))
    centered(draw, "How the decision engine works", H // 2 + 10, FONT_BODY, STEEL, a * sub_a)

    tag_a = ease_out(clamp01((p - 0.55) / 0.2))
    centered(draw, "60s showcase  ·  soccer policy brain", H // 2 + 66, FONT_SMALL, MUTED, a * tag_a)


def slide_boundary(draw, t, a):
    if a <= 0:
        return
    p = slide_progress(t, *SLIDES[1])
    centered(draw, "Not the match simulator.", 100, FONT_H1, WHITE, a)
    centered(draw, "The brain that decides every flick.", 170, FONT_BODY, STEEL, a)

    split = ease_out(clamp01((p - 0.08) / 0.25))
    left_x = int(lerp(W // 2 - 280, 260, split))
    right_x = int(lerp(W // 2 - 280, W - 260 - 560, split))
    box_y, bw, bh = 300, 560, 460

    panels = [
        (
            left_x,
            "aicomp-soccer-sim",
            "PHYSICS HARNESS",
            ["World physics", "Possession / kicks", "Headless matches", "AIA graph runtime"],
            STEEL_DIM,
        ),
        (
            right_x,
            "Titanium",
            "DECISION ENGINE",
            ["Event-leg ball predict", "Intercept solve", "Threat cover + free goals", "GK / attack / lackeys"],
            ACCENT,
        ),
    ]
    for x, title, tag, lines, color in panels:
        draw.rounded_rectangle(
            [x, box_y, x + bw, box_y + bh],
            radius=18,
            outline=mix(BG, color, a),
            width=3,
        )
        draw_text(draw, (x + 32, box_y + 28), title, FONT_SMALL, MUTED, a)
        draw_text(draw, (x + 32, box_y + 70), tag, FONT_H2, color, a)
        for i, line in enumerate(lines):
            draw_text(draw, (x + 32, box_y + 160 + i * 52), ">  " + line, FONT_BODY, WHITE, a)

    if split > 0.9:
        mid = W // 2
        draw.polygon(
            [(mid - 26, 500), (mid + 28, 520), (mid - 26, 540)],
            fill=mix(BG, ACCENT2, a),
        )


def slide_event_leg(draw, t, a):
    if a <= 0:
        return
    p = slide_progress(t, *SLIDES[2])
    leg = bounce_leg_index(p)
    kind, et, bx, bz, h = BOUNCE_LEGS[leg]

    titles = {
        "start": ("Iteration 0 — ball at center", "One analytic step solves until wall or stop"),
        "wall": (f"Iteration {leg} — WALL contact", "Bounce, then next event-leg instantly"),
        "stop": (f"Iteration {leg} — STOPPED", "Coulomb slide finished · event-leg chain done"),
    }
    title, sub = titles.get(kind, (f"Iteration {leg} — {kind}", "Event-leg teleport (engine model)"))
    centered(draw, title, 50, FONT_H1, WHITE if kind == "start" else (DANGER if kind == "wall" else GOOD), a)
    centered(draw, sub, 110, FONT_BODY, STEEL, a)

    cx, cy, pw, ph = W // 2 - 180, H // 2 + 40, 980, 560
    x0, y0, x1, y1 = draw_pitch(draw, cx, cy, pw, ph, a)

    # Only draw completed legs (no future path / no preview checkpoints).
    pts = [pitch_map(BOUNCE_LEGS[i][2], BOUNCE_LEGS[i][3], x0, y0, x1, y1) for i in range(leg + 1)]
    if len(pts) >= 2:
        draw.line(pts, fill=mix(BG, ACCENT, a), width=4)

    for i in range(leg + 1):
        lk, _lt, lx, lz, _lh = BOUNCE_LEGS[i]
        px, py = pitch_map(lx, lz, x0, y0, x1, y1)
        ecol = {"start": STEEL, "wall": DANGER, "stop": GOOD}.get(lk, STEEL)
        disc(draw, px, py, 10 if i < leg else 13, ecol if i < leg else WHITE, a)
        if i < leg or lk != "start":
            draw_text(draw, (px + 14, py - 22), f"{i}:{lk}", FONT_TINY, ecol, a)

    ball = pts[-1]
    disc(draw, ball[0], ball[1], 12, WHITE, a)
    note = {
        "start": "Solve until wall OR stop → one step",
        "wall": "Instant teleport to wall · next leg",
        "stop": "Rest — no more event legs",
    }.get(kind, kind)
    draw_text(draw, (ball[0] + 16, ball[1] + 8), f"t={et:.2f}s  h={h:+.2f}m", FONT_TINY, ACCENT2, a)
    draw_text(draw, (ball[0] + 16, ball[1] + 32), note, FONT_TINY, MUTED, a)

    meter_x, meter_y = 70, 280
    draw.rounded_rectangle(
        [meter_x, meter_y, meter_x + 70, meter_y + 320],
        radius=10,
        outline=mix(BG, STEEL_DIM, a),
        width=2,
    )
    draw_text(draw, (meter_x + 8, meter_y - 28), "Y", FONT_SMALL, MUTED, a)
    h_clamped_vis = max(-0.5, min(2.5, h))
    fill_h = int(((h_clamped_vis + 0.5) / 3.0) * 300)
    bar_bottom = meter_y + 310
    draw.rectangle(
        [meter_x + 18, bar_bottom - fill_h, meter_x + 52, bar_bottom],
        fill=mix(BG, ACCENT, a),
    )
    zero_y = bar_bottom - int((0.5 / 3.0) * 300)
    draw.line([(meter_x + 10, zero_y), (meter_x + 60, zero_y)], fill=mix(BG, WHITE, a), width=1)

    card_x = 1280
    draw.rounded_rectangle([card_x, 220, W - 60, 860], radius=16, outline=mix(BG, STEEL_DIM, a), width=2)
    lines = [
        "Engine event-leg model",
        "",
        "One step = solve until",
        "  WALL  or  STOP",
        "Then teleport to that event.",
        "",
        f"0  center   t={BOUNCE_LEGS[0][1]:.2f}s",
    ]
    for i, (lk, lt, *_rest) in enumerate(BOUNCE_LEGS[1:], start=1):
        mark = "→" if i == leg else " "
        lines.append(f"{mark}{i}  {lk:<5}  t={lt:.2f}s")
    lines += [
        "",
        "No continuous playback.",
        "No early checkpoints.",
        "Land = height snap only.",
    ]
    for i, line in enumerate(lines):
        draw_text(draw, (card_x + 28, 250 + i * 40), line, FONT_SMALL, WHITE if line else MUTED, a)


def slide_intercept(draw, t, a):
    if a <= 0:
        return
    p = slide_progress(t, *SLIDES[3])
    # Phase A: walk ONLY — ball reaches GOAL.
    # Phase B: resimulate with sprint ONLY — freeze at perfect intercept.
    phase_a = p < 0.42
    local_a = p / 0.42 if phase_a else 1.0
    local_b = (p - 0.42) / 0.58 if not phase_a else 0.0

    if phase_a:
        centered(draw, "1) Try WALK intercept", 50, FONT_H1, WHITE, a)
        centered(
            draw,
            f"Defender at ({INTERCEPT_PLAYER[0]:.1f}, {INTERCEPT_PLAYER[1]:.1f})  ·  walk {WALK_SPEED:.0f} m/s only",
            110,
            FONT_BODY,
            STEEL,
            a,
        )
    else:
        centered(draw, "2) Walk failed — RESIMULATE with sprint", 50, FONT_H1, ACCENT2, a)
        centered(
            draw,
            f"Same defender / same shot  ·  sprint {SPRINT_SPEED:.0f} m/s only",
            110,
            FONT_BODY,
            STEEL,
            a,
        )

    draw.rounded_rectangle([50, 180, 720, 920], radius=16, outline=mix(BG, ACCENT, a), width=2)
    draw_text(draw, (80, 210), "One speed at a time", FONT_H2, ACCENT, a)
    card = [
        "reach(t) = speed · t + r",
        "solve for t:  |ball(t) − me| ≤ reach(t)",
        "",
        f"z tuned = {INTERCEPT_PLAYER[1]:.1f}m",
        "  (sprint can cut, walk cannot)",
        f"Walk  {WALK_SPEED:.0f} m/s",
        f"Sprint {SPRINT_SPEED:.0f} m/s",
        f"Reach r = {INTERCEPT_REACH:.2f} m",
        "",
        f"Goal time  t={GOAL_T:.2f}s",
        f"Walk cut   {'NONE' if WALK_IDX is None else f't={PATH_SAMPLES[WALK_IDX][0]:.2f}s'}",
        f"Sprint cut {'NONE' if SPRINT_IDX is None else f't={PATH_SAMPLES[SPRINT_IDX][0]:.2f}s'}",
        "",
        "Titanium: walk first.",
        "If walk cannot beat the goal,",
        "sprint=true and solve again.",
        "",
        ("Phase: WALK only" if phase_a else f"Phase: SPRINT  t*={INTERCEPT_T:.2f}s"),
    ]
    for i, line in enumerate(card):
        hot = "NONE" in line or line.startswith("Phase")
        eq = i < 2
        col = DANGER if (phase_a and hot) else (GOOD if (not phase_a and hot) else (ACCENT if eq else WHITE))
        if not line.strip():
            col = MUTED
        draw_text(draw, (80, 275 + i * 36), line, FONT_EQ, col, a)

    cx, cy, pw, ph = 1300, 560, 820, 580
    x0, y0, x1, y1 = draw_pitch(draw, cx, cy, pw, ph, a)
    px_per_m = (x1 - x0 - 40) / (PITCH_X_MAX - PITCH_X_MIN)

    full_xy = [pitch_map(sx, sz, x0, y0, x1, y1) for (_, sx, sz, _, _) in PATH_SAMPLES]
    if len(full_xy) >= 2:
        draw.line(full_xy, fill=mix(BG, DANGER, a * 0.28), width=2)

    goal_pt = pitch_map(RIGHT_GOAL_X, 0.0, x0, y0, x1, y1)
    player = pitch_map(INTERCEPT_PLAYER[0], INTERCEPT_PLAYER[1], x0, y0, x1, y1)
    disc(draw, player[0], player[1], 14, ACCENT2, a, "D")
    draw_text(draw, (player[0] + 16, player[1] - 22), "defender", FONT_TINY, ACCENT2, a)

    # Soft cross-fade + blur through the walk -> sprint handoff. The circle
    # radius is the actual reach(t) equation shown on the card, so this is
    # the one growing from center that goes from a single (walk) answer to
    # a second (sprint) answer — blurred through the merge instead of an
    # instant swap, which is what read as a glitch cut before.
    trans_lo, trans_hi = 0.36, 0.48
    if trans_lo <= p <= trans_hi:
        mix_t = clamp01((p - trans_lo) / (trans_hi - trans_lo))
        blur_amt = 3 + 11 * (1 - abs(2 * mix_t - 1))
        r_walk_final = (WALK_SPEED * GOAL_T + INTERCEPT_REACH) * px_per_m
        r_sprint_start = (SPRINT_SPEED * (GOAL_T * 0.12) + INTERCEPT_REACH) * px_per_m
        blurred_ring(player[0], player[1], r_walk_final, ACCENT2, 3, a * (1 - mix_t) * 0.9, blur=blur_amt)
        blurred_ring(player[0], player[1], r_sprint_start, GOOD, 3, a * mix_t * 0.9, blur=blur_amt)

    if phase_a:
        n = max(2, int(1 + ease_in_out(local_a) * GOAL_IDX))
        n = min(n, GOAL_IDX + 1)
        scored = n >= GOAL_IDX
        path_xy = [pitch_map(PATH_SAMPLES[i][1], PATH_SAMPLES[i][2], x0, y0, x1, y1) for i in range(n)]
        if len(path_xy) >= 2:
            draw.line(path_xy, fill=mix(BG, DANGER if scored else ACCENT, a), width=4)
        ball = path_xy[-1]
        disc(draw, ball[0], ball[1], 10, WHITE, a)

        # WALK sphere only
        reach_t = PATH_SAMPLES[n - 1][0]
        r_walk = int((WALK_SPEED * reach_t + INTERCEPT_REACH) * px_per_m)
        if r_walk > 4:
            blurred_ring(player[0], player[1], r_walk, ACCENT2, 3, a * 0.95, blur=1.5)
            draw_text(draw, (player[0] + r_walk // 2, player[1] - r_walk - 10), "walk reach", FONT_TINY, ACCENT2, a)

        if scored:
            pulse = 0.5 + 0.5 * math.sin(t * 7.0)
            disc(draw, goal_pt[0], goal_pt[1], int(20 + 6 * pulse), DANGER, a)
            draw_text(draw, (goal_pt[0] - 110, goal_pt[1] - 42), "GOAL — walk cannot reach", FONT_BODY, DANGER, a)
            draw_text(draw, (goal_pt[0] - 110, goal_pt[1] + 8), "now resimulate with sprint", FONT_SMALL, ACCENT2, a)
        else:
            draw_text(draw, (goal_pt[0] - 40, goal_pt[1] - 24), "GOAL?", FONT_TINY, DANGER, a * 0.7)
    else:
        target_n = max(2, INTERCEPT_IDX + 1)
        play = ease_in_out(clamp01(local_b / 0.45))
        n = max(2, min(target_n, int(1 + play * (target_n - 1))))
        frozen = n >= target_n or local_b > 0.45

        path_xy = [pitch_map(PATH_SAMPLES[i][1], PATH_SAMPLES[i][2], x0, y0, x1, y1) for i in range(n)]
        if len(path_xy) >= 2:
            draw.line(path_xy, fill=mix(BG, ACCENT, a), width=4)

        # SPRINT sphere only
        reach_t = PATH_SAMPLES[n - 1][0]
        r_sprint = int((SPRINT_SPEED * reach_t + INTERCEPT_REACH) * px_per_m)
        if r_sprint > 4:
            blurred_ring(player[0], player[1], r_sprint, GOOD, 3, a * 0.95, blur=1.5)
            draw_text(draw, (player[0] + r_sprint // 2, player[1] - r_sprint - 10), "sprint reach", FONT_TINY, GOOD, a)

        ball = path_xy[-1]
        disc(draw, ball[0], ball[1], 10, WHITE, a)

        # Intercept marker only after the ball is actually cut — no "cut here" preview.
        if frozen:
            hit = pitch_map(INTERCEPT_POS[0], INTERCEPT_POS[1], x0, y0, x1, y1)
            draw.line([player, hit], fill=mix(BG, GOOD, a), width=3)
            pulse = 0.55 + 0.45 * math.sin(t * 6.0)
            disc(draw, hit[0], hit[1], int(18 + 5 * pulse), GOOD, a)
            draw.line([(hit[0] - 32, hit[1]), (hit[0] + 32, hit[1])], fill=mix(BG, GOOD, a), width=2)
            draw.line([(hit[0], hit[1] - 32), (hit[0], hit[1] + 32)], fill=mix(BG, GOOD, a), width=2)
            draw_text(draw, (hit[0] + 22, hit[1] - 48), "PERFECT INTERCEPT", FONT_SMALL, GOOD, a)
            draw_text(
                draw,
                (hit[0] + 22, hit[1] - 22),
                f"sprint t*={INTERCEPT_T:.2f}s  stop the goal",
                FONT_TINY,
                WHITE,
                a,
            )
            draw_text(draw, (goal_pt[0] - 90, goal_pt[1] + 16), "denied", FONT_BODY, GOOD, a)


def cover_wedge(draw, own_goal, post_l, post_r, opp, color, a):
    """One defender's sealed cone toward one opponent -- the same tangent
    construction the GK and each outfielder actually use (gk_cover_stand /
    threat_cover), just drawn for whichever body/opponent pair is passed in.
    A stand point plus a wedge to both posts, so the cone reads as *shut*,
    not just a dot near the goal.
    """
    stand = (int(lerp(own_goal[0], opp[0], 0.30)), int(lerp(own_goal[1], opp[1], 0.30)))
    draw.polygon([stand, post_l, post_r], fill=mix(BG, color, a * 0.22))
    draw.line([stand, post_l], fill=mix(BG, color, a * 0.6), width=2)
    draw.line([stand, post_r], fill=mix(BG, color, a * 0.6), width=2)
    return stand


def slide_threats(draw, t, a):
    if a <= 0:
        return
    p = slide_progress(t, *SLIDES[4])
    centered(draw, "The team plays as one four-body goalkeeper", 55, FONT_H1, WHITE, a)
    centered(
        draw,
        "Every opponent who could receive and shoot already has their lane sealed",
        115,
        FONT_BODY,
        STEEL,
        a,
    )

    cx, cy, pw, ph = W // 2, H // 2 + 50, 1200, 580
    x0, y0, x1, y1 = draw_pitch(draw, cx, cy, pw, ph, a)
    phase = ease_in_out(p)

    own_goal = ((x0 + x1) // 2 - int(pw * 0.42), cy)
    enemy_goal = ((x0 + x1) // 2 + int(pw * 0.42), cy)
    post_l = (own_goal[0], cy - 90)
    post_r = (own_goal[0], cy + 90)

    # 4 opponents: the carrier (pressed directly) + 3 receivers, each
    # assigned to one of OUR four bodies -- P1/P2/P3 seal a receiver's cone,
    # the GK presses the carrier. This mirrors the real per-slot assignment
    # in build_titanium.py (P1<->opponents[0], P2<->opponents[1],
    # P3<->opponents[2], GK<->carrier) instead of one blended average point.
    carrier = (cx + 210, cy - 10)
    receivers = [
        (cx - 30, cy - 150, "P1"),
        (cx + 40, cy + 130, "P2"),
        (cx - 190, cy + 30, "P3"),
    ]
    defenders = [(1, ACCENT2), (2, GOOD), (3, ACCENT)]

    reveal_each = clamp01((phase - 0.05) / 0.55) * (len(receivers) + 0.999)
    for i, ((rx, ry, tag), (_, color)) in enumerate(zip(receivers, defenders)):
        cone_a = clamp01(reveal_each - i)
        if cone_a <= 0.01:
            continue
        opp = (rx, ry)
        disc(draw, rx, ry, 12, DANGER, a * cone_a, tag.replace("P", "T"))
        stand = cover_wedge(draw, own_goal, post_l, post_r, opp, color, a * cone_a)
        draw.line([stand, opp], fill=mix(BG, color, a * cone_a * 0.5), width=1)
        disc(draw, stand[0], stand[1], 9, color, a * cone_a)
        draw_text(draw, (stand[0] + 12, stand[1] - 26), f"{tag} seals T{i + 1}", FONT_TINY, color, a * cone_a)

    # GK presses the carrier directly -- the same construction, fourth body.
    gk_a = clamp01((phase - 0.55) / 0.35)
    if gk_a > 0.01:
        gk_stand = cover_wedge(draw, own_goal, post_l, post_r, carrier, ACCENT, a * gk_a)
        disc(draw, gk_stand[0], gk_stand[1], 13, ACCENT, a * gk_a, "GK")
        draw_text(draw, (gk_stand[0] + 14, gk_stand[1] - 28), "GK presses carrier", FONT_TINY, ACCENT, a * gk_a)
        disc(draw, carrier[0], carrier[1], 13, DANGER, a * gk_a, "C")

    sealed_a = clamp01((phase - 0.75) / 0.2)
    if sealed_a > 0.01:
        centered(draw, "four lanes, four bodies, nothing open", H - 80, FONT_BODY, GOOD, a * sealed_a)
    elif phase < 0.75:
        centered(
            draw,
            "P1 → T1  ·  P2 → T2  ·  P3 → T3  ·  GK → carrier — one assignment each",
            H - 80,
            FONT_BODY,
            STEEL,
            a,
        )

    # caption swap
    if phase < 0.5:
        centered(draw, "Intercept enemy shooters — stand where every immediate threat is covered", H - 80, FONT_BODY, STEEL, a)
    else:
        centered(draw, "If our carrier has a clear lane now — take the free goal", H - 80, FONT_BODY, GOOD, a)


def slide_challenge(draw, t, a):
    if a <= 0:
        return
    p = slide_progress(t, *SLIDES[5])
    centered(draw, "Challenge protocol", 90, FONT_H1, WHITE, a)
    centered(draw, "main is the goat  ·  beat it to merge", 155, FONT_BODY, STEEL, a)

    stages = [("10", "smoke"), ("20", "check"), ("40-80", "solid"), ("100-200", "gate"), ("500", "cap")]
    active = min(len(stages) - 1, int(p * len(stages)))
    total_w = 1400
    x0 = (W - total_w) // 2
    y = 360
    for i, (games, label) in enumerate(stages):
        x = x0 + i * (total_w // len(stages))
        lit = i <= active
        col = ACCENT2 if lit else STEEL_DIM
        r = 46 if lit else 34
        cx = x + 90
        cy = y
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=mix(BG, col, a), width=4)
        tw, _ = text_size(draw, games, FONT_BODY)
        draw_text(draw, (cx - tw // 2, cy - 16), games, FONT_BODY, WHITE if lit else MUTED, a)
        tw2, _ = text_size(draw, label, FONT_SMALL)
        draw_text(draw, (cx - tw2 // 2, cy + 64), label, FONT_SMALL, col, a)
        if i < len(stages) - 1:
            draw.line([(cx + r + 6, cy), (x + total_w // len(stages) + 90 - 34, cy)], fill=mix(BG, STEEL_DIM, a), width=3)

    card_a = ease_out(clamp01((p - 0.35) / 0.25)) * a
    if card_a > 0:
        x, y = 380, 560
        draw.rounded_rectangle([x, y, W - x, y + 260], radius=16, outline=mix(BG, ACCENT2, card_a), width=3)
        for i, line in enumerate(
            [
                "Pass: >= 55% vs champion, or clear Elo gap",
                "Also vs AIA / baseline — no self-play overfitting",
                "Fail → close branch.  Pass → promote main.",
            ]
        ):
            draw_text(draw, (x + 40, y + 40 + i * 60), line, FONT_BODY, WHITE, card_a)


def slide_end(draw, t, a):
    if a <= 0:
        return
    pulse = 0.5 + 0.5 * math.sin(t * 2.8)
    centered(draw, "TITANIUM", H // 2 - 150, FONT_HERO, mix(STEEL, WHITE, pulse * 0.45), a)
    centered(draw, "Soccer decision engine", H // 2 - 20, FONT_H2, ACCENT, a)
    centered(
        draw,
        "Event-leg predict  ·  Intercept  ·  Threat cover  ·  Free goals  ·  Challenge",
        H // 2 + 50,
        FONT_BODY,
        STEEL,
        a,
    )
    centered(draw, "Publishing after the tournament", H // 2 + 120, FONT_BODY, ACCENT2, a)
    draw.rectangle([0, H - 8, int(W * slide_progress(t, *SLIDES[6])), H], fill=mix(BG, ACCENT, a))


SLIDE_FNS = [
    slide_title,
    slide_boundary,
    slide_event_leg,
    slide_intercept,
    slide_threats,
    slide_challenge,
    slide_end,
]


def render_frame(t: float) -> np.ndarray:
    global _CURRENT_IMG
    img = Image.new("RGB", (W, H), BG)
    _CURRENT_IMG = img
    draw = ImageDraw.Draw(img)
    draw_bg(draw, t)

    # Exactly one slide can be non-zero — windows never overlap.
    for (start, end), fn in zip(SLIDES, SLIDE_FNS):
        a = slide_alpha(t, start, end)
        if a > 0:
            fn(draw, t, a)
            break

    # global progress bar (always subtle)
    draw.rectangle([0, 0, int(W * (t / DURATION)), 3], fill=mix(BG, ACCENT, 0.7))
    return np.asarray(img)


SILENT_OUT = OUT.with_name(OUT.stem + "_silent.mp4")
AUDIO_WAV = ROOT / "out" / "titanium_showcase_audio.wav"


def ensure_audio() -> None:
    if AUDIO_WAV.is_file():
        return
    import subprocess
    import sys

    print("Synthesizing soundtrack…")
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "make_showcase_audio.py")], cwd=str(ROOT)
    )


def mux_audio() -> None:
    import subprocess

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.check_call(
        [
            ffmpeg, "-y",
            "-i", str(SILENT_OUT),
            "-i", str(AUDIO_WAV),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(OUT),
        ]
    )
    SILENT_OUT.unlink(missing_ok=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ensure_audio()
    print(f"Rendering {N_FRAMES} frames @ {FPS}fps -> {SILENT_OUT}")
    writer = imageio.get_writer(
        SILENT_OUT,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for i in range(N_FRAMES):
            t = i / FPS
            writer.append_data(render_frame(t))
            if i % 60 == 0:
                print(f"  {t:5.1f}s / {DURATION:.0f}s")
    finally:
        writer.close()
    print("Muxing soundtrack…")
    mux_audio()
    mb = OUT.stat().st_size / 1e6
    print(f"Done: {OUT}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
