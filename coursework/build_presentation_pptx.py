"""Build CS5478 final presentation PPTX with embedded recovery videos."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "media"
OUT = ROOT / "CS5478_Final_Presentation.pptx"

NAVY = RGBColor(0x0E, 0x1F, 0x34)
NAVY_B = RGBColor(0x19, 0x31, 0x4C)
TEAL = RGBColor(0x00, 0x97, 0x97)
CYAN = RGBColor(0x1D, 0xB4, 0xC2)
GOLD = RGBColor(0xDF, 0x9D, 0x34)
CORAL = RGBColor(0xD4, 0x64, 0x60)
MUTED = RGBColor(0x5F, 0x71, 0x82)
PALE = RGBColor(0xF5, 0xF8, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xDD, 0xE6, 0xEE)

# 16:9 widescreen
W, H = Inches(13.333), Inches(7.5)


def _set_run(run, *, size=18, bold=False, color=NAVY, font="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def _fill_solid(shape, color: RGBColor):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _add_textbox(slide, left, top, width, height, text, *, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT, font="Calibri"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _set_run(run, size=size, bold=bold, color=color, font=font)
    return box


def _add_bullets(slide, left, top, width, height, items, *, size=16, color=NAVY, gap_pt=10):
    from lxml import etree

    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    a_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        p.space_after = Pt(gap_pt)
        run = p.add_run()
        run.text = item
        _set_run(run, size=size, color=color)
        pPr = p._p.get_or_add_pPr()
        for child in list(pPr):
            if child.tag.endswith("buNone") or child.tag.endswith("buChar") or child.tag.endswith("buFont"):
                pPr.remove(child)
        buFont = etree.SubElement(pPr, f"{{{a_ns}}}buFont")
        buFont.set("typeface", "Arial")
        buChar = etree.SubElement(pPr, f"{{{a_ns}}}buChar")
        buChar.set("char", "•")
    return box


def _title_bar(slide, title: str, eyebrow: str = ""):
    _add_textbox(slide, Inches(0.45), Inches(0.22), Inches(12.4), Inches(0.35), eyebrow, size=11, bold=True, color=TEAL)
    _add_textbox(slide, Inches(0.45), Inches(0.45), Inches(12.4), Inches(0.55), title, size=28, bold=True, color=NAVY)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(1.05), Inches(12.4), Pt(1.2))
    _fill_solid(line, LINE)


def _footer(slide, page: int, total: int = 11):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, H - Pt(6), W, Pt(6))
    _fill_solid(bar, TEAL)
    _add_textbox(
        slide,
        Inches(0.45),
        H - Inches(0.38),
        Inches(8),
        Inches(0.3),
        "CS5478  |  Final Project Presentation",
        size=10,
        color=MUTED,
    )
    _add_textbox(
        slide,
        Inches(11.2),
        H - Inches(0.38),
        Inches(1.7),
        Inches(0.3),
        f"{page} / {total}",
        size=10,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


def slide_title(prs):
    s = _blank(prs)
    left = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(9.6), H)
    _fill_solid(left, NAVY)
    right = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.6), 0, W - Inches(9.6), H)
    _fill_solid(right, NAVY_B)
    _add_textbox(s, Inches(0.55), Inches(1.6), Inches(8.5), Inches(0.4), "CS5478  |  FINAL PROJECT  |  NOVEMBER 2026", size=14, bold=True, color=CYAN)
    _add_textbox(s, Inches(0.55), Inches(2.15), Inches(8.8), Inches(1.4), "One Policy, Multiple\nGrasp Recovery Skills", size=36, bold=True, color=WHITE)
    _add_textbox(s, Inches(0.55), Inches(3.75), Inches(8.8), Inches(0.45), "A Unified Transformer with Shared Action Primitives", size=18, color=RGBColor(0xCC, 0xDD, 0xE8))
    _add_textbox(s, Inches(0.55), Inches(5.1), Inches(8.8), Inches(0.35), "Team 5  |  Li Jiaqi", size=16, color=WHITE)
    _add_textbox(s, Inches(0.55), Inches(5.5), Inches(8.8), Inches(0.3), "National University of Singapore  |  13 November 2026", size=12, color=MUTED)


def slide_problem(prs):
    s = _blank(prs)
    _title_bar(s, "Grasp recovery is not a single behavior", "PROBLEM & MOTIVATION")
    _add_textbox(s, Inches(0.55), Inches(1.3), Inches(7.2), Inches(0.55), "A grasp can fail even after it looks secure.", size=22, bold=True, color=NAVY)
    _add_bullets(
        s,
        Inches(0.55),
        Inches(1.95),
        Inches(7.0),
        Inches(2.2),
        [
            "Uncertain friction, object dynamics and contact geometry",
            "Slip, rotation or complete loss of contact during lifting",
            "Excessive correction can destabilize an otherwise safe grasp",
        ],
        size=16,
    )
    # Emphasize scoped disturbance
    scope = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(4.35), Inches(7.0), Inches(1.55))
    _fill_solid(scope, RGBColor(0xF8, 0xE8, 0xE6))
    scope.line.color.rgb = CORAL
    _add_textbox(s, Inches(0.75), Inches(4.5), Inches(6.6), Inches(0.35), "Scope of this work", size=13, bold=True, color=CORAL)
    _add_textbox(
        s,
        Inches(0.75),
        Inches(4.9),
        Inches(6.6),
        Inches(0.85),
        "We currently study only one external disturbance family: a ballistic ball impact on the grasped object. Other failure modes (sensor noise, actuator fault, human push, etc.) are out of scope for now.",
        size=13,
        color=NAVY,
    )
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.5), Inches(4.4), Inches(4.4))
    _fill_solid(card, PALE)
    card.line.color.rgb = LINE
    _add_textbox(s, Inches(8.5), Inches(2.1), Inches(4.0), Inches(0.4), "External disturbance", size=14, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(8.5), Inches(2.7), Inches(4.0), Inches(0.7), "Ball impact\non the grasp", size=22, bold=True, color=CORAL, align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(8.5), Inches(3.7), Inches(4.0), Inches(0.5), "→ slip / detach / pose error", size=13, color=NAVY, align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(8.5), Inches(4.5), Inches(4.0), Inches(0.5), "Observe → Correct → Hold", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    _add_textbox(
        s,
        Inches(8.5),
        Inches(5.3),
        Inches(4.0),
        Inches(0.4),
        "MuJoCo · Franka two-finger",
        size=11,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    _footer(s, 2)


def slide_skills(prs):
    s = _blank(prs)
    _title_bar(s, "Four recovery families, one deployed policy", "MULTI-SKILL SETTING")
    cards = [
        (TEAL, "A  ·  Re-seat after ball impact", "Regain a secure bilateral grasp after impact destabilizes contact."),
        (CYAN, "B  ·  Re-grasp after ball impact", "Recover a disrupted grasp with a different corrective sequence."),
        (CORAL, "C  ·  Detach and recatch", "Recover after impact causes full loss of contact; re-establish grasp."),
        (GOLD, "D  ·  Precision correction", "Return pose/orientation after impact while keeping contact."),
    ]
    positions = [
        (Inches(0.5), Inches(1.35)),
        (Inches(0.5), Inches(3.35)),
        (Inches(6.9), Inches(1.35)),
        (Inches(6.9), Inches(3.35)),
    ]
    for (color, title, body), (left, top) in zip(cards, positions):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.9), Inches(1.75))
        _fill_solid(box, PALE)
        box.line.color.rgb = LINE
        accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Pt(6), Inches(1.75))
        _fill_solid(accent, color)
        _add_textbox(s, left + Inches(0.25), top + Inches(0.25), Inches(5.4), Inches(0.45), title, size=16, bold=True, color=NAVY)
        _add_textbox(s, left + Inches(0.25), top + Inches(0.75), Inches(5.4), Inches(0.8), body, size=13, color=MUTED)
    _add_textbox(
        s,
        Inches(0.5),
        Inches(5.35),
        Inches(12.3),
        Inches(0.55),
        "A–D classify expert demonstrations / eval scenarios only. The deployed Actor has no skill ID and does not select among A/B/C/D strategies.",
        size=13,
        bold=True,
        color=TEAL,
        align=PP_ALIGN.CENTER,
    )
    _footer(s, 3)


def slide_method(prs):
    s = _blank(prs)
    _title_bar(s, "From expert skills to a unified Transformer", "APPROACH & IMPLEMENTATION")
    steps = [
        ("Multi-skill experts", "families A / B / C / D"),
        ("Shared vocabulary", "111 motion primitives"),
        ("Causal Transformer", "history-conditioned Actor"),
        ("Token + residual", "7D corrective action"),
        ("Safe handoff", "return to nominal control"),
    ]
    x0 = Inches(0.45)
    for i, (h, sub) in enumerate(steps):
        left = x0 + Inches(i * 2.55)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.5), Inches(2.35), Inches(1.35))
        _fill_solid(box, PALE)
        box.line.color.rgb = LINE
        _add_textbox(s, left + Inches(0.1), Inches(1.65), Inches(2.15), Inches(0.55), h, size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        _add_textbox(s, left + Inches(0.1), Inches(2.25), Inches(2.15), Inches(0.4), sub, size=11, color=MUTED, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            _add_textbox(s, left + Inches(2.2), Inches(1.9), Inches(0.35), Inches(0.4), "→", size=18, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(0.55), Inches(3.3), Inches(5.8), Inches(0.4), "Actor input", size=16, bold=True, color=NAVY)
    _add_textbox(s, Inches(0.55), Inches(3.7), Inches(5.8), Inches(0.7), "Observation history plus the previous executed action.", size=14, color=MUTED)
    _add_textbox(s, Inches(7.0), Inches(3.3), Inches(5.8), Inches(0.4), "Actor output", size=16, bold=True, color=NAVY)
    _add_textbox(s, Inches(7.0), Inches(3.7), Inches(5.8), Inches(0.7), "Discrete motion primitive + continuous residual; or handoff.", size=14, color=MUTED)
    _add_textbox(
        s,
        Inches(0.55),
        Inches(5.0),
        Inches(12.2),
        Inches(0.5),
        "Primitives are shared across skills. No skill ID is provided at test time.",
        size=12,
        color=MUTED,
    )
    _footer(s, 4)


def slide_discrete(prs):
    s = _blank(prs)
    _title_bar(s, "Why discretize the action space?", "KEY DESIGN DECISION")
    _add_textbox(s, Inches(0.55), Inches(1.3), Inches(7.5), Inches(0.55), "Experts demonstrate multiple distinct skills.", size=22, bold=True, color=NAVY)
    _add_bullets(
        s,
        Inches(0.55),
        Inches(2.1),
        Inches(7.2),
        Inches(3.5),
        [
            "A single continuous regression target can blur different corrective behaviors.",
            "A shared discrete vocabulary keeps modes separable.",
            "A residual adds fine control without separate skill-specific policies.",
            "The Actor also learns when to stop correcting and hand back control.",
        ],
        size=15,
    )
    for i, (label, color) in enumerate(
        [("Select primitive z_t", TEAL), ("Predict residual r_t", CYAN), ("a_t = clip(p_z + r_t)", NAVY)]
    ):
        top = Inches(1.6) + Inches(i * 1.35)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.4), top, Inches(4.3), Inches(1.1))
        _fill_solid(box, PALE)
        box.line.color.rgb = color
        _add_textbox(s, Inches(8.55), top + Inches(0.3), Inches(4.0), Inches(0.5), label, size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    _footer(s, 5)


def slide_train(prs):
    s = _blank(prs)
    _title_bar(s, "Training and physical evaluation", "METHODOLOGY")
    _add_textbox(s, Inches(0.55), Inches(1.35), Inches(5.8), Inches(0.4), "Learning pipeline", size=18, bold=True, color=NAVY)
    _add_bullets(
        s,
        Inches(0.55),
        Inches(1.9),
        Inches(5.8),
        Inches(2.5),
        [
            "Generate and validate expert recovery demonstrations.",
            "Train a unified Actor with imitation learning.",
            "Refine physical closed-loop performance with reinforcement learning.",
        ],
        size=15,
    )
    pill = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(4.6), Inches(2.6), Inches(0.45))
    _fill_solid(pill, RGBColor(0xE0, 0xF4, 0xF4))
    pill.line.fill.background()
    _add_textbox(s, Inches(0.55), Inches(4.65), Inches(2.6), Inches(0.4), "Frozen final policy", size=12, bold=True, color=TEAL, align=PP_ALIGN.CENTER)

    _add_textbox(s, Inches(7.0), Inches(1.35), Inches(5.8), Inches(0.4), "Evaluation contract", size=18, bold=True, color=NAVY)
    _add_bullets(
        s,
        Inches(7.0),
        Inches(1.9),
        Inches(5.8),
        Inches(3.0),
        [
            "Same frozen policy across families A–D",
            "Official safe-handoff rule to the nominal controller",
            "12-second physical stability after handoff",
            "Closed-loop success matters more than offline action accuracy",
        ],
        size=15,
    )
    _add_textbox(
        s,
        Inches(0.55),
        Inches(5.5),
        Inches(12.2),
        Inches(0.4),
        "Primary metric: 12-second relaxed physical success after handoff.",
        size=12,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    _footer(s, 6)


def slide_results(prs):
    s = _blank(prs)
    _title_bar(s, "Closed-loop success across four families", "RESULTS")
    rows = [("A", 6, TEAL), ("B", 4, CYAN), ("C", 3, CORAL), ("D", 7, GOLD)]
    bar_left = Inches(1.5)
    bar_max = Inches(7.2)
    for i, (lab, n, color) in enumerate(rows):
        top = Inches(1.45) + Inches(i * 0.95)
        _add_textbox(s, Inches(0.55), top, Inches(0.8), Inches(0.5), lab, size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        track = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bar_left, top + Inches(0.08), bar_max, Inches(0.42))
        _fill_solid(track, LINE)
        width = int(bar_max * (n / 16.0))
        fill = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bar_left, top + Inches(0.08), width, Inches(0.42))
        _fill_solid(fill, color)
        _add_textbox(s, bar_left + bar_max + Inches(0.15), top, Inches(1.4), Inches(0.5), f"{n}/16", size=18, bold=True, color=NAVY)

    summary = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.3), Inches(1.7), Inches(2.5), Inches(2.2))
    _fill_solid(summary, NAVY)
    _add_textbox(s, Inches(10.3), Inches(1.95), Inches(2.5), Inches(0.7), "20/64", size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(10.3), Inches(2.7), Inches(2.5), Inches(0.4), "total successes", size=12, color=RGBColor(0xCC, 0xDD, 0xE8), align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(10.3), Inches(3.15), Inches(2.5), Inches(0.4), "31.25%", size=16, bold=True, color=CYAN, align=PP_ALIGN.CENTER)

    _add_textbox(s, Inches(10.3), Inches(4.2), Inches(2.5), Inches(0.35), "Key observation", size=12, bold=True, color=NAVY)
    _add_textbox(s, Inches(10.3), Inches(4.55), Inches(2.5), Inches(0.8), "Detach–recatch (C) remains the hardest family.", size=12, color=MUTED)
    _add_textbox(
        s,
        Inches(0.55),
        Inches(5.55),
        Inches(12.2),
        Inches(0.35),
        "Counts: successes / 16 trials per family. Same frozen Actor for A–D.",
        size=11,
        color=MUTED,
    )
    _footer(s, 7)


def _add_video(slide, mp4: Path, poster: Path, left, top, width, height):
    slide.shapes.add_movie(
        str(mp4),
        left,
        top,
        width,
        height,
        poster_frame_image=str(poster),
        mime_type="video/mp4",
    )


def slide_video_one(prs, page: int, *, family: str, title: str, mp4: str, poster: str, color, caption: str):
    """One full-width embedded clip per slide."""
    s = _blank(prs)
    _title_bar(s, title, f"DEMONSTRATION  ·  EXPERT FAMILY {family}")
    note = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(1.15), Inches(11.6), Inches(0.42))
    _fill_solid(note, RGBColor(0xE8, 0xF4, 0xF4))
    note.line.fill.background()
    _add_textbox(
        s,
        Inches(0.95),
        Inches(1.2),
        Inches(11.4),
        Inches(0.35),
        "A / B / C / D = expert demonstration taxonomy only — not a skill ID, and not the policy’s chosen strategy.",
        size=12,
        bold=True,
        color=TEAL,
        align=PP_ALIGN.CENTER,
    )
    left, top = Inches(0.85), Inches(1.7)
    vw, vh = Inches(11.6), Inches(4.35)
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Pt(6), vh)
    _fill_solid(accent, color)
    _add_video(s, MEDIA / mp4, MEDIA / poster, left, top, vw, vh)
    _add_textbox(
        s,
        Inches(0.85),
        Inches(6.15),
        Inches(11.6),
        Inches(0.4),
        caption,
        size=12,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    _footer(s, page)


def slide_lessons(prs, page: int = 12):
    s = _blank(prs)
    _title_bar(s, "What the experiments taught us", "DISCUSSION & LIMITATIONS")
    left = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(1.35), Inches(5.95), Inches(3.7))
    _fill_solid(left, PALE)
    left.line.color.rgb = LINE
    right = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.35), Inches(5.95), Inches(3.7))
    _fill_solid(right, PALE)
    right.line.color.rgb = LINE
    _add_textbox(s, Inches(0.7), Inches(1.55), Inches(5.4), Inches(0.4), "What worked", size=18, bold=True, color=NAVY)
    _add_bullets(
        s,
        Inches(0.7),
        Inches(2.15),
        Inches(5.4),
        Inches(2.6),
        [
            "A single Actor can execute multiple recovery behaviors.",
            "Token + residual supports varied expert skills without a skill router.",
            "Physical closed-loop testing exposes failures missed by imitation metrics.",
        ],
        size=14,
    )
    _add_textbox(s, Inches(7.15), Inches(1.55), Inches(5.4), Inches(0.4), "What remains difficult", size=18, bold=True, color=NAVY)
    _add_bullets(
        s,
        Inches(7.15),
        Inches(2.15),
        Inches(5.4),
        Inches(2.6),
        [
            "Recatch generalization and contact stability after full detachment.",
            "Distribution shift after the Actor's own actions.",
            "Only ball-impact disturbances studied so far — other failure modes remain open.",
        ],
        size=14,
    )
    _add_textbox(
        s,
        Inches(0.55),
        Inches(5.4),
        Inches(12.2),
        Inches(0.55),
        "Follow-up diagnostics: early contact prediction was not a useful warning signal; extra corrective imitation alone did not replace closed-loop gains.",
        size=12,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    _footer(s, page)


def slide_conclusion(prs, page: int = 13):
    s = _blank(prs)
    _title_bar(s, "Takeaways and next steps", "CONCLUSION")
    _add_textbox(
        s,
        Inches(0.55),
        Inches(1.3),
        Inches(12.2),
        Inches(0.8),
        "One unified policy can express different grasp recovery skills, but reliable generalization is not solved.",
        size=20,
        bold=True,
        color=NAVY,
    )
    left = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(2.3), Inches(5.95), Inches(3.0))
    _fill_solid(left, PALE)
    left.line.color.rgb = LINE
    right = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(2.3), Inches(5.95), Inches(3.0))
    _fill_solid(right, PALE)
    right.line.color.rgb = LINE
    pill_l = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(2.5), Inches(2.0), Inches(0.4))
    _fill_solid(pill_l, RGBColor(0xE0, 0xF4, 0xF4))
    pill_l.line.fill.background()
    _add_textbox(s, Inches(0.7), Inches(2.55), Inches(2.0), Inches(0.35), "Achievements", size=12, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    _add_bullets(
        s,
        Inches(0.7),
        Inches(3.15),
        Inches(5.4),
        Inches(1.9),
        [
            "MuJoCo recovery benchmark with multi-skill experts",
            "Shared primitives + residual unified Transformer Actor",
            "Physical closed-loop validation with a frozen checkpoint",
        ],
        size=13,
    )
    pill_r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.15), Inches(2.5), Inches(2.6), Inches(0.4))
    _fill_solid(pill_r, RGBColor(0xF8, 0xE8, 0xE6))
    pill_r.line.fill.background()
    _add_textbox(s, Inches(7.15), Inches(2.55), Inches(2.6), Inches(0.35), "Next research questions", size=12, bold=True, color=CORAL, align=PP_ALIGN.CENTER)
    _add_bullets(
        s,
        Inches(7.15),
        Inches(3.15),
        Inches(5.4),
        Inches(1.9),
        [
            "Better coverage of recoverable physical states",
            "Stable sequencing through contact transitions",
            "Extend beyond ball-impact to other disturbance types",
        ],
        size=13,
    )
    _add_textbox(s, Inches(0.55), Inches(5.7), Inches(12.2), Inches(0.45), "Thank you!   Questions?", size=22, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    _footer(s, page)


def build():
    required = [
        "recover_A1.mp4",
        "recover_B1.mp4",
        "recover_C.mp4",
        "recover_D.mp4",
        "recover_A1_poster.png",
        "recover_B1_poster.png",
        "recover_C_poster.png",
        "recover_D_poster.png",
    ]
    for name in required:
        path = MEDIA / name
        if not path.is_file():
            short = MEDIA / name.replace(".mp4", "_short.mp4")
            if name.endswith(".mp4") and short.is_file():
                path.write_bytes(short.read_bytes())
            else:
                raise FileNotFoundError(path)

    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    total = 13

    global _footer

    def _footer_fixed(slide, page: int, total: int = total):
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, H - Pt(6), W, Pt(6))
        _fill_solid(bar, TEAL)
        _add_textbox(
            slide,
            Inches(0.45),
            H - Inches(0.38),
            Inches(8),
            Inches(0.3),
            "CS5478  |  Final Project Presentation",
            size=10,
            color=MUTED,
        )
        _add_textbox(
            slide,
            Inches(11.2),
            H - Inches(0.38),
            Inches(1.7),
            Inches(0.3),
            f"{page} / {total}",
            size=10,
            color=MUTED,
            align=PP_ALIGN.RIGHT,
        )

    _footer = _footer_fixed

    slide_title(prs)
    slide_problem(prs)
    slide_skills(prs)
    slide_method(prs)
    slide_discrete(prs)
    slide_train(prs)
    slide_results(prs)
    slide_video_one(
        prs,
        8,
        family="A",
        title="Expert family A — example closed-loop recovery",
        mp4="recover_A1.mp4",
        poster="recover_A1_poster.png",
        color=TEAL,
        caption="Eval scenario from expert family A. Left = unified policy; right = do-nothing hold (drops). Click to play.",
    )
    slide_video_one(
        prs,
        9,
        family="B",
        title="Expert family B — example closed-loop recovery",
        mp4="recover_B1.mp4",
        poster="recover_B1_poster.png",
        color=CYAN,
        caption="Eval scenario from expert family B. Same frozen Actor — no B-specific strategy switch. Click to play.",
    )
    slide_video_one(
        prs,
        10,
        family="C",
        title="Expert family C — example closed-loop recovery",
        mp4="recover_C.mp4",
        poster="recover_C_poster.png",
        color=CORAL,
        caption="Eval scenario from expert family C. Behavior emerges from the unified policy, not a C skill label.",
    )
    slide_video_one(
        prs,
        11,
        family="D",
        title="Expert family D — observed: release + catch",
        mp4="recover_D.mp4",
        poster="recover_D_poster.png",
        color=GOLD,
        caption="Still an expert-family-D scenario label only. In this clip the policy actually releases and recatches — not a separate “D strategy”.",
    )
    slide_lessons(prs, 12)
    slide_conclusion(prs, 13)

    prs.save(OUT)
    mb = OUT.stat().st_size / 1e6
    n = len(prs.slides)
    print(f"Wrote {OUT} ({mb:.1f} MB, {n} slides)")


if __name__ == "__main__":
    build()
